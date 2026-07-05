# Databricks notebook source
# MAGIC %md
# MAGIC # Gold — Cash Deposit Reconciliation
# MAGIC
# MAGIC Matches Toast register close-out declarations against physical bank deposits
# MAGIC pulled from Monarch, producing a persistent audit ledger.
# MAGIC
# MAGIC **Reads from:**
# MAGIC - `bronze.toast_cash_entries_raw` — drawer events per business date
# MAGIC - `bronze.toast_cash_deposits_raw` — employee-declared deposit amounts
# MAGIC - `bronze.monarch_bank_deposits_raw` — candidate bank deposits
# MAGIC
# MAGIC **Writes to:**
# MAGIC - `gold.cash_register_closes` — one row per business date (declared close amounts)
# MAGIC - `gold.cash_audit_matches` — reconciliation ledger (close date ↔ bank deposit)
# MAGIC
# MAGIC **Matching algorithm:**
# MAGIC For each unmatched close date, scan forward up to MATCH_WINDOW_DAYS for bank deposits.
# MAGIC A match is found when one deposit or a combination of deposits sums to within
# MAGIC MATCH_TOLERANCE dollars of the declared amount.
# MAGIC Deposits confirmed by user as 'Cash Deposit' in Monarch are prioritized.
# MAGIC
# MAGIC **Match statuses:**
# MAGIC - `MATCHED`      — amount ties exactly (within tolerance) to declared close
# MAGIC - `PARTIAL`      — closest match found but variance exceeds tolerance
# MAGIC - `UNMATCHED`    — no bank deposit found within MATCH_WINDOW_DAYS
# MAGIC - `NEEDS_REVIEW` — deposit candidate exists but user hasn't confirmed in Monarch yet
# MAGIC - `NO_CASH`      — close date had zero declared cash (card-only day)

# COMMAND ----------

import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import DateType

# COMMAND ----------

CATALOG = "YOUR_CATALOG"

CLOSES_TABLE  = f"{CATALOG}.gold.cash_register_closes"
MATCHES_TABLE = f"{CATALOG}.gold.cash_audit_matches"

MATCH_WINDOW_DAYS = 7    # Look forward this many days for a matching deposit
MATCH_TOLERANCE   = 5.0  # Dollars — within this amount is considered a match

# COMMAND ----------

# ── TABLE CREATION ────────────────────────────────────────────────────────────

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CLOSES_TABLE} (
        business_date           DATE    NOT NULL,
        declared_deposit_amount DOUBLE,
        cash_out_total          DOUBLE,
        close_out_type          STRING,
        close_out_amount        DOUBLE,
        entry_count             INT,
        has_close_out_entry     BOOLEAN,
        computed_at             TIMESTAMP NOT NULL
    )
    USING DELTA
    COMMENT 'One row per Toast business date. Declared cash amounts from /cashmgmt/v1/deposits and computed totals from entries.'
""")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {MATCHES_TABLE} (
        close_date              DATE    NOT NULL,
        declared_amount         DOUBLE,
        match_status            STRING  NOT NULL,
        bank_deposit_id         STRING,
        bank_deposit_date       DATE,
        bank_deposit_amount     DOUBLE,
        account_name            STRING,
        original_statement      STRING,
        monarch_category        STRING,
        variance_dollars        DOUBLE,
        days_to_deposit         INT,
        is_confirmed_deposit    BOOLEAN,
        notes                   STRING,
        last_evaluated_at       TIMESTAMP NOT NULL
    )
    USING DELTA
    COMMENT 'Cash audit reconciliation ledger. One row per close date. Updated daily as new bank deposits arrive.'
""")

print(f"✓ Tables ready")

# COMMAND ----------

# ── STEP 1: BUILD cash_register_closes ────────────────────────────────────────
# Aggregate Toast entries into one row per business date.

spark.sql(f"""
    CREATE OR REPLACE TEMPORARY VIEW v_closes AS
    SELECT
        e.business_date,

        -- Sum of all entries that represent cash physically removed from the drawer
        SUM(CASE
            WHEN e.entry_type IN ('CASH_OUT', 'PAY_OUT', 'TIP_OUT', 'DRIVER_REIMBURSEMENT')
            THEN ABS(e.amount) ELSE 0
        END) AS cash_out_total,

        -- Close-out declaration type for the night
        MAX(CASE WHEN e.entry_type IN ('CLOSE_OUT_EXACT','CLOSE_OUT_OVERAGE','CLOSE_OUT_SHORTAGE')
            THEN e.entry_type END) AS close_out_type,

        -- Amount declared at close-out
        MAX(CASE WHEN e.entry_type IN ('CLOSE_OUT_EXACT','CLOSE_OUT_OVERAGE','CLOSE_OUT_SHORTAGE')
            THEN e.amount END) AS close_out_amount,

        COUNT(*) AS entry_count,
        MAX(CASE WHEN e.entry_type IN ('CLOSE_OUT_EXACT','CLOSE_OUT_OVERAGE','CLOSE_OUT_SHORTAGE')
            THEN TRUE ELSE FALSE END) AS has_close_out_entry

    FROM {CATALOG}.bronze.toast_cash_entries_raw e
    GROUP BY e.business_date
""")

# Join declared deposit amounts from the deposits endpoint
spark.sql(f"""
    CREATE OR REPLACE TEMPORARY VIEW v_closes_with_declared AS
    SELECT
        c.*,
        COALESCE(d.declared_deposit_amount, c.close_out_amount) AS declared_deposit_amount
    FROM v_closes c
    LEFT JOIN (
        SELECT business_date, SUM(amount) AS declared_deposit_amount
        FROM {CATALOG}.bronze.toast_cash_deposits_raw
        GROUP BY business_date
    ) d ON c.business_date = d.business_date
""")

# Merge into persistent closes table
spark.sql(f"""
    MERGE INTO {CLOSES_TABLE} t
    USING (
        SELECT *, current_timestamp() AS computed_at
        FROM v_closes_with_declared
    ) s
    ON t.business_date = s.business_date
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

closes_count = spark.sql(f"SELECT COUNT(*) FROM {CLOSES_TABLE}").collect()[0][0]
print(f"✓ cash_register_closes: {closes_count} business dates")

# COMMAND ----------

# ── STEP 2: RUN MATCHING ALGORITHM ────────────────────────────────────────────
# For each close date, find bank deposits within the match window.
# We run this as a Python loop rather than pure SQL because the matching logic
# involves amount-proximity search across a time window.

closes_df = spark.sql(f"""
    SELECT business_date, declared_deposit_amount
    FROM {CLOSES_TABLE}
    WHERE declared_deposit_amount IS NOT NULL
      AND declared_deposit_amount > 0
    ORDER BY business_date
""").collect()

deposits_df = spark.sql(f"""
    SELECT
        monarch_id,
        bank_date,
        amount,
        account_name,
        original_statement,
        monarch_category,
        pattern_matched,
        CASE WHEN monarch_category = 'Cash Deposit' THEN TRUE ELSE FALSE END AS is_confirmed_deposit
    FROM {CATALOG}.bronze.monarch_bank_deposits_raw
    ORDER BY bank_date
""").collect()

# Index deposits by date for fast lookup
from collections import defaultdict
deposits_by_date = defaultdict(list)
for d in deposits_df:
    deposits_by_date[d["bank_date"]].append(dict(d))

print(f"Close dates to evaluate: {len(closes_df)}")
print(f"Bank deposit candidates: {len(deposits_df)}")

# COMMAND ----------

match_rows = []
evaluated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

for close in closes_df:
    close_date      = close["business_date"]
    declared_amount = float(close["declared_deposit_amount"] or 0)

    if declared_amount == 0:
        match_rows.append({
            "close_date":           close_date.isoformat(),
            "declared_amount":      declared_amount,
            "match_status":         "NO_CASH",
            "bank_deposit_id":      None,
            "bank_deposit_date":    None,
            "bank_deposit_amount":  None,
            "account_name":         None,
            "original_statement":   None,
            "monarch_category":     None,
            "variance_dollars":     0.0,
            "days_to_deposit":      None,
            "is_confirmed_deposit": None,
            "notes":                "No cash declared for this business date",
            "last_evaluated_at":    evaluated_at,
        })
        continue

    # Search forward MATCH_WINDOW_DAYS for a matching deposit
    best_match   = None
    best_variance = None

    for delta in range(MATCH_WINDOW_DAYS + 1):
        check_date = close_date + datetime.timedelta(days=delta)
        day_deposits = deposits_by_date.get(check_date, [])

        # Prioritize confirmed deposits
        confirmed = [d for d in day_deposits if d["is_confirmed_deposit"]]
        candidates = confirmed + [d for d in day_deposits if not d["is_confirmed_deposit"]]

        for dep in candidates:
            variance = declared_amount - dep["amount"]
            if best_variance is None or abs(variance) < abs(best_variance):
                best_match   = (dep, delta)
                best_variance = variance

            # Exact match — stop searching
            if abs(variance) <= MATCH_TOLERANCE:
                break

        if best_match and abs(best_variance) <= MATCH_TOLERANCE:
            break

    if best_match is None:
        # No deposit found in window at all
        match_rows.append({
            "close_date":           close_date.isoformat(),
            "declared_amount":      declared_amount,
            "match_status":         "UNMATCHED",
            "bank_deposit_id":      None,
            "bank_deposit_date":    None,
            "bank_deposit_amount":  None,
            "account_name":         None,
            "original_statement":   None,
            "monarch_category":     None,
            "variance_dollars":     None,
            "days_to_deposit":      None,
            "is_confirmed_deposit": None,
            "notes":                f"No deposit found within {MATCH_WINDOW_DAYS} days",
            "last_evaluated_at":    evaluated_at,
        })
    else:
        dep, days = best_match
        variance  = declared_amount - dep["amount"]

        if abs(variance) <= MATCH_TOLERANCE:
            status = "MATCHED"
        elif dep["is_confirmed_deposit"]:
            status = "PARTIAL"
        else:
            status = "NEEDS_REVIEW"

        match_rows.append({
            "close_date":           close_date.isoformat(),
            "declared_amount":      declared_amount,
            "match_status":         status,
            "bank_deposit_id":      dep["monarch_id"],
            "bank_deposit_date":    (close_date + datetime.timedelta(days=days)).isoformat(),
            "bank_deposit_amount":  dep["amount"],
            "account_name":         dep["account_name"],
            "original_statement":   dep["original_statement"],
            "monarch_category":     dep["monarch_category"],
            "variance_dollars":     round(variance, 2),
            "days_to_deposit":      days,
            "is_confirmed_deposit": dep["is_confirmed_deposit"],
            "notes":                dep["pattern_matched"],
            "last_evaluated_at":    evaluated_at,
        })

print(f"Evaluated {len(match_rows)} close dates")

# COMMAND ----------

# Write results to persistent matches table
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, BooleanType, TimestampType, DateType
)

schema = StructType([
    StructField("close_date",           StringType(),  True),
    StructField("declared_amount",      DoubleType(),  True),
    StructField("match_status",         StringType(),  False),
    StructField("bank_deposit_id",      StringType(),  True),
    StructField("bank_deposit_date",    StringType(),  True),
    StructField("bank_deposit_amount",  DoubleType(),  True),
    StructField("account_name",         StringType(),  True),
    StructField("original_statement",   StringType(),  True),
    StructField("monarch_category",     StringType(),  True),
    StructField("variance_dollars",     DoubleType(),  True),
    StructField("days_to_deposit",      IntegerType(), True),
    StructField("is_confirmed_deposit", BooleanType(), True),
    StructField("notes",                StringType(),  True),
    StructField("last_evaluated_at",    StringType(),  False),
])

df_matches = spark.createDataFrame(match_rows, schema=schema) \
    .withColumn("close_date",        F.col("close_date").cast(DateType())) \
    .withColumn("bank_deposit_date", F.col("bank_deposit_date").cast(DateType())) \
    .withColumn("last_evaluated_at", F.col("last_evaluated_at").cast(TimestampType()))

from delta.tables import DeltaTable
DeltaTable.forName(spark, MATCHES_TABLE).alias("t").merge(
    df_matches.alias("s"),
    "t.close_date = s.close_date"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

print(f"✓ Wrote {len(match_rows)} rows to {MATCHES_TABLE}")

# COMMAND ----------

# ── STEP 3: SUMMARY STATS ────────────────────────────────────────────────────

summary = spark.sql(f"""
    SELECT
        match_status,
        COUNT(*)                    AS count,
        ROUND(SUM(declared_amount), 2) AS total_declared,
        ROUND(SUM(variance_dollars), 2) AS total_variance
    FROM {MATCHES_TABLE}
    GROUP BY match_status
    ORDER BY match_status
""")
summary.show()

cumulative = spark.sql(f"""
    SELECT
        ROUND(SUM(declared_amount), 2)    AS total_declared_inception,
        ROUND(SUM(bank_deposit_amount), 2) AS total_deposited_inception,
        ROUND(SUM(
            CASE WHEN match_status IN ('MATCHED','PARTIAL')
            THEN variance_dollars ELSE 0 END
        ), 2) AS cumulative_variance_confirmed,
        ROUND(SUM(
            CASE WHEN match_status = 'UNMATCHED'
            THEN declared_amount ELSE 0 END
        ), 2) AS total_unmatched_declared
    FROM {MATCHES_TABLE}
    WHERE match_status != 'NO_CASH'
""")
cumulative.show()
