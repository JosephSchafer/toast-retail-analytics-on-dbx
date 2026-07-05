# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze — Ingest Monarch Bank Deposits
# MAGIC
# MAGIC Pulls bank transaction data from Monarch Money (via MCP/API) for the three
# MAGIC business checking accounts and writes candidate cash deposit transactions to
# MAGIC `bronze.monarch_bank_deposits_raw`.
# MAGIC
# MAGIC **Candidate filter logic (inclusive by design):**
# MAGIC - Keeps ALL credits matching raw deposit patterns: `DEPOSIT ID NUMBER` (Chase), `MDEPOSIT` (Rockland)
# MAGIC - Also keeps any transaction the user has categorized as `Cash Deposit` in Monarch
# MAGIC - Excludes known non-cash patterns: Toast ACH, Citizens ACH, payroll, wire transfers, card payments
# MAGIC - When in doubt, keeps the transaction — the reconciliation notebook decides relevance
# MAGIC
# MAGIC **Match status progression (set by reconciliation notebook, not here):**
# MAGIC - `NEEDS_REVIEW` — deposit candidate surfaced, not yet confirmed
# MAGIC - `MATCHED` — ties to a declared close within 7 days
# MAGIC - `PARTIAL` — partial amount match or consolidated multi-day
# MAGIC - `EXCLUDED` — human confirmed not a store deposit (set in reconciliation)
# MAGIC
# MAGIC **Accounts monitored:**
# MAGIC - Chase BUS COMPLETE CHK (...2062) — primary store deposit account
# MAGIC - Chase Shared Total Chk (...3255) — secondary, used occasionally
# MAGIC - Rockland Checking (Shared) (...4372) — personal/household but monitored for store deposits

# COMMAND ----------

dbutils.widgets.dropdown(
    name="run_mode",
    defaultValue="incremental",
    choices=["incremental", "backfill"],
    label="Run Mode"
)

dbutils.widgets.text(
    name="backfill_start_date",
    defaultValue="YOUR_TOAST_GOLIVE_DATE",
    label="Backfill Start Date (YYYY-MM-DD)"
)

dbutils.widgets.text(
    name="backfill_end_date",
    defaultValue="",
    label="Backfill End Date (YYYY-MM-DD) — leave blank for yesterday"
)

# COMMAND ----------

import datetime
import json
import uuid
import requests

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, TimestampType, DoubleType, BooleanType

# COMMAND ----------

RUN_MODE           = dbutils.widgets.get("run_mode")
BACKFILL_START_RAW = dbutils.widgets.get("backfill_start_date")
BACKFILL_END_RAW   = dbutils.widgets.get("backfill_end_date")

CATALOG = "YOUR_CATALOG"
SCHEMA  = "bronze"
TABLE   = f"{CATALOG}.{SCHEMA}.monarch_bank_deposits_raw"

WATERMARK_TABLE = f"{CATALOG}.{SCHEMA}.ingestion_watermark"
WATERMARK_KEY   = "monarch_deposits"

# Monarch account IDs (stable Monarch internal IDs)
MONITORED_ACCOUNTS = [
    "241668341029386185",  # Chase BUS COMPLETE CHK (...2062)
    "241668341112223690",  # Chase Shared Total Chk (...3255)
    "171464037038798261",  # Rockland Checking (Shared) (...4372)
]

# Raw statement patterns that indicate a physical cash/check deposit at the teller
DEPOSIT_PATTERNS = [
    "DEPOSIT ID NUMBER",  # Chase teller/ATM deposits
    "MDEPOSIT",           # Rockland mobile/teller deposits
]

# Statement substrings that definitively identify non-cash-deposit credits
# Anything matching these is excluded regardless of amount or category
EXCLUSION_PATTERNS = [
    "ORIG CO NAME:TOAST",          # Toast card settlement ACH
    "ORIG CO NAME:Citizens",       # Citizens Bank card settlement ACH
    "INDICIUM",                    # Payroll direct deposits
    "WT CR",                       # Wire transfers (payroll wires)
    "AUTOMATIC PAYMENT",           # Credit card autopays
    "AUTOPAY PAYMENT",             # Amex autopay
    "Online Transfer",             # Internal Chase transfers
]

# Monarch categories that indicate a transaction is definitely not a store cash deposit
EXCLUDED_CATEGORIES = [
    "Paychecks",
    "Credit Card Payment",
    "Transfer",
    "Dividends & Capital Gains",
    "Sell",
]

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        monarch_id          STRING    NOT NULL,
        bank_date           DATE      NOT NULL,
        amount              DOUBLE    NOT NULL,
        account_id          STRING    NOT NULL,
        account_name        STRING,
        original_statement  STRING,
        monarch_category    STRING,
        monarch_merchant    STRING,
        is_whole_dollar     BOOLEAN,
        pattern_matched     STRING,
        candidate_reason    STRING,
        raw_json            STRING    NOT NULL,
        ingested_at         TIMESTAMP NOT NULL,
        ingestion_batch_id  STRING    NOT NULL
    )
    USING DELTA
    PARTITIONED BY (bank_date)
    COMMENT 'Monarch bank deposit candidates — physical cash/check deposits across monitored checking accounts'
""")

print(f"✓ Table ready: {TABLE}")

# COMMAND ----------

def get_watermark() -> datetime.date | None:
    try:
        row = spark.sql(f"""
            SELECT last_ingested_date FROM {WATERMARK_TABLE}
            WHERE source_key = '{WATERMARK_KEY}'
        """).collect()
        return row[0][0] if row else None
    except Exception:
        return None


def set_watermark(date: datetime.date):
    spark.sql(f"""
        MERGE INTO {WATERMARK_TABLE} t
        USING (SELECT '{WATERMARK_KEY}' AS source_key,
                      DATE('{date.isoformat()}') AS last_ingested_date,
                      current_timestamp() AS updated_at) s
        ON t.source_key = s.source_key
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


def is_excluded(statement: str, category: str) -> bool:
    """True if this transaction is definitively not a store cash deposit."""
    stmt_upper = (statement or "").upper()
    for pattern in EXCLUSION_PATTERNS:
        if pattern.upper() in stmt_upper:
            return True
    if category in EXCLUDED_CATEGORIES:
        return True
    return False


def classify_candidate(statement: str, category: str, amount: float) -> tuple[str | None, str | None]:
    """
    Returns (pattern_matched, candidate_reason) if this transaction should be kept,
    or (None, None) if it should be dropped.

    Priority:
    1. Confirmed by user in Monarch as 'Cash Deposit' — always keep
    2. Matches a known raw deposit statement pattern — keep
    3. Everything else — drop
    """
    if is_excluded(statement, category):
        return None, None

    # User has explicitly confirmed this is a cash deposit in Monarch
    if category == "Cash Deposit":
        return "MONARCH_CONFIRMED", "User categorized as Cash Deposit in Monarch"

    # Matches a known teller/ATM deposit statement pattern
    stmt_upper = (statement or "").upper()
    for pattern in DEPOSIT_PATTERNS:
        if pattern.upper() in stmt_upper:
            reason = f"Matches deposit pattern '{pattern}'"
            if amount == int(amount):
                reason += " + whole dollar amount"
            return pattern, reason

    return None, None


def fetch_monarch_transactions(start_date: str, end_date: str) -> list:
    """
    Fetch transactions from Monarch via the MCP tool.
    This notebook runs on Databricks — we call the Monarch MCP endpoint
    via the Databricks secret-backed HTTP proxy pattern.

    NOTE: In practice this notebook is called by Claude Code / the MCP harness
    which has Monarch credentials. For fully automated Databricks job execution,
    the Monarch session token needs to be stored as a Databricks secret and
    refreshed periodically (Monarch sessions expire).

    For now: fetch via stored session token in Databricks secrets.
    Falls back to a clear error if the token is missing or expired.
    """
    try:
        monarch_token = dbutils.secrets.get(scope="monarch_api", key="session_token")
    except Exception:
        raise RuntimeError(
            "Monarch session token not found. Store it at scope='monarch_api', key='session_token'.\n"
            "The token can be extracted from browser devtools after logging into monarchmoney.com.\n"
            "See SETUP.md for instructions."
        )

    url = "https://api.monarchmoney.com/graphql"
    headers = {
        "Authorization": f"Token {monarch_token}",
        "Content-Type": "application/json",
    }

    # GraphQL query matching what the Monarch MCP server uses
    query = """
    query GetTransactions($filters: TransactionFilterInput, $limit: Int, $offset: Int) {
      allTransactions(filters: $filters, limit: $limit, offset: $offset) {
        totalCount
        results {
          id
          date
          amount
          pending
          account { id displayName }
          merchant { name }
          category { name }
          originalStatement
          hideFromReports
        }
      }
    }
    """

    variables = {
        "filters": {
            "startDate": start_date,
            "endDate": end_date,
            "accountIds": MONITORED_ACCOUNTS,
            "transactionType": "CREDIT",
        },
        "limit": 500,
        "offset": 0,
    }

    all_txns = []
    while True:
        resp = requests.post(url, headers=headers, json={"query": query, "variables": variables}, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "errors" in data:
            raise RuntimeError(f"Monarch GraphQL error: {data['errors']}")

        results = data["data"]["allTransactions"]["results"]
        total   = data["data"]["allTransactions"]["totalCount"]
        all_txns.extend(results)

        if len(all_txns) >= total:
            break
        variables["offset"] += 500

    return all_txns

# COMMAND ----------

# ── DATE RANGE SETUP ──────────────────────────────────────────────────────────

yesterday = datetime.date.today() - datetime.timedelta(days=1)

if RUN_MODE == "incremental":
    # Pull a 14-day window on each incremental run to catch late-arriving deposits
    # (bags held at home for several days before being brought to the bank)
    start_date = (yesterday - datetime.timedelta(days=14)).isoformat()
    end_date   = yesterday.isoformat()
    print(f"Mode: INCREMENTAL — fetching {start_date} to {end_date} (14-day window)")

elif RUN_MODE == "backfill":
    backfill_start = datetime.date.fromisoformat(BACKFILL_START_RAW)
    backfill_end   = datetime.date.fromisoformat(BACKFILL_END_RAW) if BACKFILL_END_RAW else yesterday

    watermark = get_watermark()
    if watermark and watermark >= backfill_start:
        backfill_start = watermark + datetime.timedelta(days=1)
        print(f"Watermark: {watermark}. Resuming from {backfill_start}")

    if backfill_start > backfill_end:
        print("All dates already ingested. Nothing to do.")
        dbutils.notebook.exit("SKIPPED: all dates already ingested")

    start_date = backfill_start.isoformat()
    end_date   = backfill_end.isoformat()
    print(f"Mode: BACKFILL — {start_date} to {end_date}")

else:
    raise ValueError(f"Unknown run_mode: '{RUN_MODE}'")

# COMMAND ----------

# ── FETCH AND FILTER ──────────────────────────────────────────────────────────

batch_id    = str(uuid.uuid4())
ingested_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

print(f"Batch ID: {batch_id}")
print(f"Fetching Monarch transactions {start_date} → {end_date}...")

all_txns = fetch_monarch_transactions(start_date, end_date)
print(f"  Raw credits fetched: {len(all_txns)}")

# Filter to candidates
rows = []
excluded_count = 0
for txn in all_txns:
    stmt     = txn.get("originalStatement") or ""
    category = (txn.get("category") or {}).get("name") or ""
    amount   = abs(float(txn.get("amount") or 0))

    pattern_matched, candidate_reason = classify_candidate(stmt, category, amount)

    if pattern_matched is None:
        excluded_count += 1
        continue

    rows.append({
        "monarch_id":         txn["id"],
        "bank_date":          txn["date"],
        "amount":             amount,
        "account_id":         (txn.get("account") or {}).get("id", ""),
        "account_name":       (txn.get("account") or {}).get("displayName", ""),
        "original_statement": stmt,
        "monarch_category":   category,
        "monarch_merchant":   (txn.get("merchant") or {}).get("name", ""),
        "is_whole_dollar":    (amount == int(amount)),
        "pattern_matched":    pattern_matched,
        "candidate_reason":   candidate_reason,
        "raw_json":           json.dumps(txn),
        "ingested_at":        ingested_at,
        "ingestion_batch_id": batch_id,
    })

print(f"  Candidates kept:    {len(rows)}")
print(f"  Excluded:           {excluded_count}")

# COMMAND ----------

if rows:
    df = spark.createDataFrame(rows) \
        .withColumn("bank_date",    F.col("bank_date").cast(DateType())) \
        .withColumn("ingested_at",  F.col("ingested_at").cast(TimestampType())) \
        .withColumn("is_whole_dollar", F.col("is_whole_dollar").cast(BooleanType()))

    DeltaTable.forName(spark, TABLE).alias("t").merge(
        df.alias("s"),
        "t.monarch_id = s.monarch_id"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

    print(f"✓ Wrote {len(rows)} candidate deposits to {TABLE}")
else:
    print("No candidate deposits found in this date range.")

if RUN_MODE == "backfill":
    set_watermark(datetime.date.fromisoformat(end_date))

# COMMAND ----------

print(f"\nSummary:")
print(f"  Batch:      {batch_id}")
print(f"  Date range: {start_date} → {end_date}")
print(f"  Candidates: {len(rows)}")
print(f"  Excluded:   {excluded_count}")
