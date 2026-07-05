# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze — Ingest Toast Cash Management
# MAGIC
# MAGIC Pulls cash drawer entries and declared deposits from the Toast Cash Management API
# MAGIC and writes them to Delta tables in `YOUR_CATALOG.bronze`.
# MAGIC
# MAGIC **Two modes — controlled by the `run_mode` widget:**
# MAGIC
# MAGIC | Mode | What it does |
# MAGIC |---|---|
# MAGIC | `incremental` | Pulls yesterday's cash entries. Run this on the daily schedule. |
# MAGIC | `backfill` | Loops day-by-day from `backfill_start_date` to yesterday. Run once manually. Resumable. |
# MAGIC
# MAGIC **Writes to:**
# MAGIC - `bronze.toast_cash_entries_raw` — one row per drawer event (CASH_OUT, PAY_OUT, CLOSE_OUT_EXACT, etc.)
# MAGIC - `bronze.toast_cash_deposits_raw` — one row per declared deposit per business date
# MAGIC
# MAGIC **Design:**
# MAGIC - MERGE on `entry_guid` / `deposit_guid` — reruns are idempotent
# MAGIC - Full raw JSON preserved for forward compatibility
# MAGIC - Shares watermark table with NB1 (`bronze.ingestion_watermark`, key `cash_management`)

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

import requests
import json
import datetime
import uuid

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, DateType

# COMMAND ----------

RUN_MODE           = dbutils.widgets.get("run_mode")
BACKFILL_START_RAW = dbutils.widgets.get("backfill_start_date")
BACKFILL_END_RAW   = dbutils.widgets.get("backfill_end_date")

TOAST_AUTH_URL     = "https://ws.toasttab.com/authentication/v1/authentication/login"
TOAST_CASHMGMT_URL = "https://ws.toasttab.com/cashmgmt/v1"

CATALOG    = "YOUR_CATALOG"
SCHEMA     = "bronze"
ENTRIES_TABLE  = f"{CATALOG}.{SCHEMA}.toast_cash_entries_raw"
DEPOSITS_TABLE = f"{CATALOG}.{SCHEMA}.toast_cash_deposits_raw"
WATERMARK_TABLE = f"{CATALOG}.{SCHEMA}.ingestion_watermark"
WATERMARK_KEY   = "cash_management"

BACKFILL_MAX_DAYS = 400

# COMMAND ----------

try:
    TOAST_CLIENT_ID       = dbutils.secrets.get(scope="toast_api", key="toast_client_id")
    TOAST_CLIENT_SECRET   = dbutils.secrets.get(scope="toast_api", key="toast_client_secret")
    TOAST_RESTAURANT_GUID = dbutils.secrets.get(scope="toast_api", key="restaurant_guid")
    print("✓ Secrets loaded")
except Exception as e:
    print("✗ Secret retrieval failed. Check scope='toast_api' and key names.")
    raise e

# COMMAND ----------

def get_access_token() -> str:
    resp = requests.post(
        TOAST_AUTH_URL,
        json={
            "clientId": TOAST_CLIENT_ID,
            "clientSecret": TOAST_CLIENT_SECRET,
            "userAccessType": "TOAST_MACHINE_CLIENT"
        },
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    resp.raise_for_status()
    token = resp.json()["token"]["accessToken"]
    print("✓ Toast authentication successful")
    return token


def fetch_cash_entries(business_date: datetime.date, token: str) -> list:
    """Fetch cash drawer entries for a single business date."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Toast-Restaurant-External-ID": TOAST_RESTAURANT_GUID
    }
    params = {"businessDate": business_date.strftime("%Y%m%d")}
    resp = requests.get(
        f"{TOAST_CASHMGMT_URL}/entries",
        headers=headers,
        params=params,
        timeout=30
    )
    if resp.status_code == 404:
        return []  # No cash management data for this date
    resp.raise_for_status()
    return resp.json() if resp.json() else []


def fetch_cash_deposits(business_date: datetime.date, token: str) -> list:
    """Fetch declared deposit entries for a single business date."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Toast-Restaurant-External-ID": TOAST_RESTAURANT_GUID
    }
    params = {"businessDate": business_date.strftime("%Y%m%d")}
    resp = requests.get(
        f"{TOAST_CASHMGMT_URL}/deposits",
        headers=headers,
        params=params,
        timeout=30
    )
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json() if resp.json() else []


def get_watermark() -> datetime.date | None:
    try:
        row = spark.sql(f"""
            SELECT last_date_done FROM {WATERMARK_TABLE}
            WHERE pipeline = '{WATERMARK_KEY}'
        """).collect()
        return row[0][0] if row else None
    except Exception:
        return None


def set_watermark(date: datetime.date):
    spark.sql(f"""
        MERGE INTO {WATERMARK_TABLE} t
        USING (SELECT '{WATERMARK_KEY}' AS pipeline,
                      DATE('{date.isoformat()}') AS last_date_done,
                      current_timestamp() AS completed_at) s
        ON t.pipeline = s.pipeline
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

# COMMAND ----------

# ── TABLE CREATION ────────────────────────────────────────────────────────────

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {ENTRIES_TABLE} (
        entry_guid          STRING NOT NULL,
        business_date       DATE   NOT NULL,
        entry_type          STRING,
        amount              DOUBLE,
        reason_guid         STRING,
        no_sale_reason      STRING,
        cash_drawer_guid    STRING,
        employee1_guid      STRING,
        employee2_guid      STRING,
        undoes_guid         STRING,
        entry_date          TIMESTAMP,
        raw_json            STRING NOT NULL,
        ingested_at         TIMESTAMP NOT NULL,
        ingestion_batch_id  STRING NOT NULL
    )
    USING DELTA
    PARTITIONED BY (business_date)
    COMMENT 'Raw Toast cash drawer entries from /cashmgmt/v1/entries'
""")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {DEPOSITS_TABLE} (
        deposit_guid        STRING NOT NULL,
        business_date       DATE   NOT NULL,
        amount              DOUBLE,
        employee_guid       STRING,
        undoes_guid         STRING,
        deposit_date        TIMESTAMP,
        raw_json            STRING NOT NULL,
        ingested_at         TIMESTAMP NOT NULL,
        ingestion_batch_id  STRING NOT NULL
    )
    USING DELTA
    PARTITIONED BY (business_date)
    COMMENT 'Raw Toast declared deposit entries from /cashmgmt/v1/deposits'
""")

print(f"✓ Tables ready: {ENTRIES_TABLE}, {DEPOSITS_TABLE}")

# COMMAND ----------

# ── DATE RANGE SETUP ──────────────────────────────────────────────────────────

yesterday = datetime.date.today() - datetime.timedelta(days=1)

if RUN_MODE == "incremental":
    dates_to_process = [yesterday]
    print(f"Mode: INCREMENTAL — fetching {yesterday}")

elif RUN_MODE == "backfill":
    backfill_start = datetime.date.fromisoformat(BACKFILL_START_RAW)
    backfill_end   = datetime.date.fromisoformat(BACKFILL_END_RAW) if BACKFILL_END_RAW else yesterday

    watermark = get_watermark()
    if watermark and watermark >= backfill_start:
        backfill_start = watermark + datetime.timedelta(days=1)
        print(f"Watermark found: {watermark}. Resuming from {backfill_start}")

    dates_to_process = []
    cursor = backfill_start
    while cursor <= backfill_end:
        dates_to_process.append(cursor)
        cursor += datetime.timedelta(days=1)

    if len(dates_to_process) == 0:
        print(f"All dates already ingested (watermark={watermark}). Nothing to do.")
        dbutils.notebook.exit("SKIPPED: all dates already ingested")

    if len(dates_to_process) > BACKFILL_MAX_DAYS:
        raise ValueError(f"Backfill range {len(dates_to_process)} days exceeds BACKFILL_MAX_DAYS={BACKFILL_MAX_DAYS}")

    print(f"Mode: BACKFILL — {len(dates_to_process)} dates from {dates_to_process[0]} to {dates_to_process[-1]}")

else:
    raise ValueError(f"Unknown run_mode: '{RUN_MODE}'")

# COMMAND ----------

# ── INGEST LOOP ───────────────────────────────────────────────────────────────

batch_id    = str(uuid.uuid4())
token       = get_access_token()
ingested_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

total_entries  = 0
total_deposits = 0
failed_dates   = []

print(f"\nBatch ID: {batch_id}")
print(f"Processing {len(dates_to_process)} date(s)...\n")

for business_date in dates_to_process:
    date_str = business_date.isoformat()
    try:
        entries  = fetch_cash_entries(business_date, token)
        deposits = fetch_cash_deposits(business_date, token)

        # ── Write entries ──────────────────────────────────────────────────────
        if entries:
            entry_rows = []
            for e in entries:
                entry_rows.append({
                    "entry_guid":         e.get("guid", f"MISSING_{date_str}_{uuid.uuid4()}"),
                    "business_date":      date_str,
                    "entry_type":         e.get("type"),
                    "amount":             e.get("amount"),
                    "reason_guid":        (e.get("payoutReason") or {}).get("guid"),
                    "no_sale_reason":     (e.get("noSaleReason") or {}).get("guid") if isinstance(e.get("noSaleReason"), dict) else e.get("noSaleReason"),
                    "cash_drawer_guid":   (e.get("cashDrawer") or {}).get("guid"),
                    "employee1_guid":     (e.get("employee1") or {}).get("guid"),
                    "employee2_guid":     (e.get("employee2") or {}).get("guid"),
                    "undoes_guid":        (e.get("undoes") or {}).get("guid"),
                    "entry_date":         e.get("date"),
                    "raw_json":           json.dumps(e),
                    "ingested_at":        ingested_at,
                    "ingestion_batch_id": batch_id
                })

            from pyspark.sql.types import StructType, StructField, StringType, DoubleType
            entry_schema = StructType([
                StructField("entry_guid",         StringType(), False),
                StructField("business_date",       StringType(), False),
                StructField("entry_type",          StringType(), True),
                StructField("amount",              DoubleType(), True),
                StructField("reason_guid",         StringType(), True),
                StructField("no_sale_reason",      StringType(), True),
                StructField("cash_drawer_guid",    StringType(), True),
                StructField("employee1_guid",      StringType(), True),
                StructField("employee2_guid",      StringType(), True),
                StructField("undoes_guid",         StringType(), True),
                StructField("entry_date",          StringType(), True),
                StructField("raw_json",            StringType(), False),
                StructField("ingested_at",         StringType(), False),
                StructField("ingestion_batch_id",  StringType(), False),
            ])
            df_entries = spark.createDataFrame(entry_rows, schema=entry_schema) \
                .withColumn("business_date", F.col("business_date").cast(DateType())) \
                .withColumn("ingested_at",   F.col("ingested_at").cast(TimestampType())) \
                .withColumn("entry_date",    F.col("entry_date").cast(TimestampType()))

            DeltaTable.forName(spark, ENTRIES_TABLE).alias("t").merge(
                df_entries.alias("s"),
                "t.entry_guid = s.entry_guid"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
            total_entries += len(entry_rows)

        # ── Write deposits ─────────────────────────────────────────────────────
        if deposits:
            deposit_rows = []
            for d in deposits:
                deposit_rows.append({
                    "deposit_guid":       d.get("guid", f"MISSING_{date_str}_{uuid.uuid4()}"),
                    "business_date":      date_str,
                    "amount":             d.get("amount"),
                    "employee_guid":      (d.get("employee") or {}).get("guid"),
                    "undoes_guid":        (d.get("undoes") or {}).get("guid"),
                    "deposit_date":       d.get("date"),
                    "raw_json":           json.dumps(d),
                    "ingested_at":        ingested_at,
                    "ingestion_batch_id": batch_id
                })

            deposit_schema = StructType([
                StructField("deposit_guid",        StringType(), False),
                StructField("business_date",       StringType(), False),
                StructField("amount",              DoubleType(), True),
                StructField("employee_guid",       StringType(), True),
                StructField("undoes_guid",         StringType(), True),
                StructField("deposit_date",        StringType(), True),
                StructField("raw_json",            StringType(), False),
                StructField("ingested_at",         StringType(), False),
                StructField("ingestion_batch_id",  StringType(), False),
            ])
            df_deposits = spark.createDataFrame(deposit_rows, schema=deposit_schema) \
                .withColumn("business_date", F.col("business_date").cast(DateType())) \
                .withColumn("ingested_at",   F.col("ingested_at").cast(TimestampType())) \
                .withColumn("deposit_date",  F.col("deposit_date").cast(TimestampType()))

            DeltaTable.forName(spark, DEPOSITS_TABLE).alias("t").merge(
                df_deposits.alias("s"),
                "t.deposit_guid = s.deposit_guid"
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
            total_deposits += len(deposit_rows)

        print(f"  {date_str}: {len(entries)} entries, {len(deposits)} deposits")

        if RUN_MODE == "backfill":
            set_watermark(business_date)

    except Exception as e:
        print(f"  ✗ {date_str}: {e}")
        failed_dates.append(date_str)

# COMMAND ----------

print(f"\n{'='*60}")
print(f"Batch complete: {batch_id}")
print(f"  Entries written:  {total_entries}")
print(f"  Deposits written: {total_deposits}")
print(f"  Failed dates:     {len(failed_dates)}")
if failed_dates:
    print(f"  Failed: {failed_dates}")
    raise RuntimeError(f"{len(failed_dates)} date(s) failed ingestion: {failed_dates}")
