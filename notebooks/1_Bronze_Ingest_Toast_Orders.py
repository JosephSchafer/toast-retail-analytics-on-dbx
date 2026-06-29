# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze — Ingest Toast Orders
# MAGIC
# MAGIC Writes raw order data from the Toast `/orders/v2/ordersBulk` API into a Delta table
# MAGIC at `3sp_analytics_workspace.bronze.toast_orders_raw`.
# MAGIC
# MAGIC **Two modes — controlled by the `run_mode` widget:**
# MAGIC
# MAGIC | Mode | What it does |
# MAGIC |---|---|
# MAGIC | `incremental` | Pulls yesterday's orders. Run this on the daily schedule. |
# MAGIC | `backfill` | Loops day-by-day from `BACKFILL_START_DATE` to yesterday. Run once manually. Resumable. |
# MAGIC
# MAGIC **Key design decisions:**
# MAGIC - All writes use **MERGE INTO** on `order_guid` — reruns are always idempotent, no duplicates
# MAGIC - Raw JSON is preserved in full in the `raw_json` column — Silver transforms parse it later
# MAGIC - A `watermark` Delta table tracks the last successfully ingested date for safe resumption
# MAGIC - No cloud-specific libraries — only `requests`, `json`, `datetime`, and PySpark/Delta

# COMMAND ----------

# ── WIDGET SETUP ──────────────────────────────────────────────────────────────
# Widgets appear as dropdowns at the top of the notebook in Databricks UI.
# When run as a Job, pass these as job parameters.

dbutils.widgets.dropdown(
    name="run_mode",
    defaultValue="incremental",
    choices=["incremental", "backfill"],
    label="Run Mode"
)

dbutils.widgets.text(
    name="backfill_start_date",
    defaultValue="2025-07-01",
    label="Backfill Start Date (YYYY-MM-DD)"
)

dbutils.widgets.text(
    name="backfill_end_date",
    defaultValue="",
    label="Backfill End Date (YYYY-MM-DD) — leave blank for yesterday"
)

# COMMAND ----------

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

import requests
import json
import datetime
import uuid
import time

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

# COMMAND ----------

# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────

# --- Run parameters ---
RUN_MODE            = dbutils.widgets.get("run_mode")           # "incremental" or "backfill"
BACKFILL_START_RAW  = dbutils.widgets.get("backfill_start_date")
BACKFILL_END_RAW    = dbutils.widgets.get("backfill_end_date")

# --- Toast API ---
# Confirmed working endpoint (firewall-friendly ws subdomain)
TOAST_AUTH_URL   = "https://ws.toasttab.com/authentication/v1/authentication/login"
TOAST_ORDERS_URL = "https://ws.toasttab.com/orders/v2/ordersBulk"
TOAST_PAGE_SIZE  = 100

# --- Databricks ---
CATALOG   = "3sp_analytics_workspace"
SCHEMA    = "bronze"
TABLE     = "toast_orders_raw"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}"

WATERMARK_TABLE = f"{CATALOG}.{SCHEMA}.ingestion_watermark"

# --- Backfill safety: max days per run to avoid cluster timeouts ---
# At ~1 API call per day, 365 days ≈ 5-10 minutes. Safe to run all at once.
# Raise this if you ever need to backfill multiple years.
BACKFILL_MAX_DAYS = 400

# COMMAND ----------

# ── 3. SECRETS ────────────────────────────────────────────────────────────────
# Scope: toast_api  Keys: toast_client_id, toast_client_secret, restaurant_guid
# These match the secrets already set up in your workspace.

try:
    TOAST_CLIENT_ID       = dbutils.secrets.get(scope="toast_api", key="toast_client_id")
    TOAST_CLIENT_SECRET   = dbutils.secrets.get(scope="toast_api", key="toast_client_secret")
    TOAST_RESTAURANT_GUID = dbutils.secrets.get(scope="toast_api", key="restaurant_guid")
    print("✓ Secrets loaded")
except Exception as e:
    print("✗ Secret retrieval failed. Check scope='toast_api' and key names.")
    raise e

# COMMAND ----------

# ── 4. SETUP: SCHEMA AND TABLES ───────────────────────────────────────────────
# Creates the bronze schema and both Delta tables if they don't already exist.
# Safe to re-run — IF NOT EXISTS guards everything.

spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")

# Bronze raw table — one row per order, keyed on order_guid
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME} (
        order_guid          STRING      NOT NULL  COMMENT 'Toast order GUID — primary dedup key',
        business_date       DATE                  COMMENT 'The local business date of the order (YYYY-MM-DD)',
        raw_json            STRING      NOT NULL  COMMENT 'Full raw JSON object from Toast API — never modified',
        ingested_at         TIMESTAMP   NOT NULL  COMMENT 'UTC timestamp when this row was written by the pipeline',
        ingestion_batch_id  STRING      NOT NULL  COMMENT 'UUID for the ingestion run — links all rows from same execution',
        source_date         DATE        NOT NULL  COMMENT 'The startDate filter used in the API call that fetched this row',
        restaurant_guid     STRING      NOT NULL  COMMENT 'Toast Restaurant External ID'
    )
    USING DELTA
    PARTITIONED BY (source_date)
    COMMENT 'Bronze: raw Toast order data. Append/merge only. No transforms.'
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'quality' = 'bronze'
    )
""")

# Watermark table — tracks last successfully completed ingestion date
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {WATERMARK_TABLE} (
        pipeline        STRING    NOT NULL  COMMENT 'Pipeline identifier',
        last_date_done  DATE      NOT NULL  COMMENT 'Last business date successfully ingested',
        completed_at    TIMESTAMP NOT NULL  COMMENT 'UTC time the run completed'
    )
    USING DELTA
    COMMENT 'Ingestion watermarks for resumable backfills'
""")

print(f"✓ Tables ready: {FULL_TABLE_NAME}")
print(f"✓ Watermark table ready: {WATERMARK_TABLE}")

# COMMAND ----------

# ── 5. HELPER FUNCTIONS ───────────────────────────────────────────────────────

def get_access_token() -> str:
    """Authenticate with Toast and return a bearer token.
    Uses the ws.toasttab.com endpoint which bypasses the firewall block
    on token.toasttab.com (confirmed working in your existing notebooks).
    """
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


def fetch_orders_for_date(business_date: datetime.date, token: str) -> list:
    """Fetch all orders for a single calendar day.

    Toast's ordersBulk endpoint uses page-number pagination (not a cursor token).
    We stop when we get a partial page (< pageSize), which means we've hit the end.

    Args:
        business_date: The date to fetch. We query midnight-to-midnight UTC.
        token: Valid Toast bearer token.

    Returns:
        List of raw order dicts from the API.
    """
    start_str = business_date.strftime("%Y-%m-%dT00:00:00.000Z")
    end_str   = (business_date + datetime.timedelta(days=1)).strftime("%Y-%m-%dT00:00:00.000Z")

    headers = {
        "Authorization": f"Bearer {token}",
        "Toast-Restaurant-External-ID": TOAST_RESTAURANT_GUID
    }
    params = {
        "startDate": start_str,
        "endDate":   end_str,
        "pageSize":  TOAST_PAGE_SIZE,
        "page":      1
    }

    all_orders = []
    while True:
        resp = requests.get(TOAST_ORDERS_URL, headers=headers, params=params, timeout=60)

        # Surface API errors clearly rather than silently returning empty
        if resp.status_code != 200:
            raise RuntimeError(
                f"Toast API error on {business_date}: HTTP {resp.status_code} — {resp.text[:500]}"
            )

        page_data = resp.json()

        # Empty list = no more pages
        if not page_data:
            break

        all_orders.extend(page_data)

        # Partial page = last page
        if len(page_data) < TOAST_PAGE_SIZE:
            break

        params["page"] += 1

    return all_orders


def merge_orders_to_bronze(orders: list, business_date: datetime.date, batch_id: str):
    """Write a list of orders into the Bronze Delta table using MERGE INTO.

    MERGE on order_guid ensures this is fully idempotent — running the same
    date twice will update existing rows rather than creating duplicates.
    The raw_json is always updated on match so corrections from Toast propagate.

    Args:
        orders: List of order dicts from the API.
        business_date: The business date these orders belong to.
        batch_id: UUID string for this ingestion run.
    """
    if not orders:
        print(f"  No orders for {business_date} — skipping merge")
        return

    # Build rows for the DataFrame
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    rows = [
        {
            "order_guid":         order.get("guid", "MISSING_GUID"),
            "business_date":      business_date.isoformat(),
            "raw_json":           json.dumps(order),
            "ingested_at":        now_utc.isoformat(),
            "ingestion_batch_id": batch_id,
            "source_date":        business_date.isoformat(),
            "restaurant_guid":    TOAST_RESTAURANT_GUID
        }
        for order in orders
    ]

    incoming_df = (
        spark.createDataFrame(rows)
        .withColumn("business_date", F.col("business_date").cast("date"))
        .withColumn("ingested_at",   F.col("ingested_at").cast("timestamp"))
        .withColumn("source_date",   F.col("source_date").cast("date"))
    )

    # MERGE — matched rows update raw_json + audit cols; new rows are inserted
    bronze_table = DeltaTable.forName(spark, FULL_TABLE_NAME)
    bronze_table.alias("target").merge(
        incoming_df.alias("source"),
        "target.order_guid = source.order_guid"
    ).whenMatchedUpdate(set={
        "raw_json":           "source.raw_json",
        "ingested_at":        "source.ingested_at",
        "ingestion_batch_id": "source.ingestion_batch_id"
    }).whenNotMatchedInsertAll().execute()

    print(f"  ✓ {business_date} — merged {len(orders)} orders → {FULL_TABLE_NAME}")


def update_watermark(business_date: datetime.date):
    """Record the last successfully completed date in the watermark table."""
    spark.createDataFrame([{
        "pipeline":       FULL_TABLE_NAME,
        "last_date_done": business_date.isoformat(),
        "completed_at":   datetime.datetime.now(datetime.timezone.utc).isoformat()
    }]).withColumn("last_date_done", F.col("last_date_done").cast("date")) \
       .withColumn("completed_at",   F.col("completed_at").cast("timestamp")) \
       .write.format("delta").mode("append").saveAsTable(WATERMARK_TABLE)


def get_watermark() -> datetime.date | None:
    """Return the last successfully ingested date, or None if no watermark exists."""
    wm_df = spark.table(WATERMARK_TABLE).filter(F.col("pipeline") == FULL_TABLE_NAME)
    if wm_df.count() == 0:
        return None
    row = wm_df.orderBy(F.col("last_date_done").desc()).limit(1).collect()[0]
    return row["last_date_done"]


# COMMAND ----------

# ── 6. BUILD DATE RANGE ───────────────────────────────────────────────────────

now_utc   = datetime.datetime.now(datetime.timezone.utc)
yesterday = (now_utc - datetime.timedelta(days=1)).date()

if RUN_MODE == "incremental":
    # Always pull yesterday — the most recently closed business day
    dates_to_process = [yesterday]
    print(f"Mode: INCREMENTAL — fetching {yesterday}")

elif RUN_MODE == "backfill":
    # Parse start date
    backfill_start = datetime.date.fromisoformat(BACKFILL_START_RAW)

    # Parse end date — default to yesterday if blank
    if BACKFILL_END_RAW.strip():
        backfill_end = datetime.date.fromisoformat(BACKFILL_END_RAW)
    else:
        backfill_end = yesterday

    # Resume from watermark if one exists — avoids re-processing already-done dates
    watermark = get_watermark()
    if watermark and watermark >= backfill_start:
        resume_from = watermark + datetime.timedelta(days=1)
        print(f"Watermark found: last completed date was {watermark}")
        print(f"Resuming backfill from {resume_from} (skipping already-ingested dates)")
        backfill_start = resume_from

    # Build list of every date in range
    dates_to_process = []
    cursor = backfill_start
    while cursor <= backfill_end:
        dates_to_process.append(cursor)
        cursor += datetime.timedelta(days=1)

    # Nothing to do — watermark already covers the entire requested range
    if len(dates_to_process) == 0:
        print(f"Mode: BACKFILL — all dates already ingested (watermark={watermark}). Nothing to do.")
        dbutils.notebook.exit("SKIPPED: all dates already ingested")

    # Safety cap
    if len(dates_to_process) > BACKFILL_MAX_DAYS:
        raise ValueError(
            f"Backfill range has {len(dates_to_process)} days, which exceeds "
            f"BACKFILL_MAX_DAYS={BACKFILL_MAX_DAYS}. "
            f"Narrow the date range or raise the limit."
        )

    print(f"Mode: BACKFILL — {len(dates_to_process)} dates from {dates_to_process[0]} to {dates_to_process[-1]}")

else:
    raise ValueError(f"Unknown run_mode: '{RUN_MODE}'. Must be 'incremental' or 'backfill'.")

# COMMAND ----------

# ── 7. MAIN INGESTION LOOP ────────────────────────────────────────────────────
# One batch_id per full run — all rows from this execution share the same ID
# so you can always trace "what did this specific run write?"

batch_id     = str(uuid.uuid4())
token        = get_access_token()
token_issued = datetime.datetime.now(datetime.timezone.utc)

total_dates    = len(dates_to_process)
total_orders   = 0
failed_dates   = []

print(f"\nBatch ID: {batch_id}")
print(f"Processing {total_dates} date(s)...\n")

for i, business_date in enumerate(dates_to_process, start=1):
    # Toast tokens expire after ~1 hour — refresh proactively every 50 minutes
    elapsed_minutes = (datetime.datetime.now(datetime.timezone.utc) - token_issued).seconds / 60
    if elapsed_minutes > 50:
        print("  ↻ Refreshing Toast token (50-min threshold)...")
        token        = get_access_token()
        token_issued = datetime.datetime.now(datetime.timezone.utc)

    print(f"[{i}/{total_dates}] {business_date}")

    try:
        orders = fetch_orders_for_date(business_date, token)
        merge_orders_to_bronze(orders, business_date, batch_id)
        total_orders += len(orders)

        # Only update watermark after a successful write
        # This is what makes the backfill resumable — if the cluster dies at
        # day 200, the next run picks up from day 201, not day 1.
        if RUN_MODE == "backfill":
            update_watermark(business_date)

        # Small pause to be a polite API consumer during backfills
        if RUN_MODE == "backfill" and i < total_dates:
            time.sleep(0.25)

    except Exception as e:
        print(f"  ✗ FAILED for {business_date}: {e}")
        failed_dates.append((business_date, str(e)))
        # Continue to next date rather than aborting the entire backfill
        # Failed dates are reported in the summary below

# For incremental runs, update watermark once at the end
if RUN_MODE == "incremental" and not failed_dates:
    update_watermark(yesterday)

# COMMAND ----------

# ── 8. RUN SUMMARY ────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("INGESTION COMPLETE")
print("="*60)
print(f"  Mode:          {RUN_MODE}")
print(f"  Batch ID:      {batch_id}")
print(f"  Dates run:     {total_dates}")
print(f"  Total orders:  {total_orders}")
print(f"  Failures:      {len(failed_dates)}")

if failed_dates:
    print("\n  ⚠ FAILED DATES (re-run backfill to retry these):")
    for d, err in failed_dates:
        print(f"    {d}: {err}")
    raise RuntimeError(
        f"Ingestion completed with {len(failed_dates)} failed date(s). "
        f"Check output above. Re-run in backfill mode to retry."
    )
else:
    print("\n  ✓ All dates ingested successfully")

# COMMAND ----------

# ── 9. QUICK VALIDATION ───────────────────────────────────────────────────────
# Spot-check the table after every run so failures are visible immediately.

print(f"\nRow counts by source_date (last 7 days ingested):")
spark.sql(f"""
    SELECT
        source_date,
        COUNT(*)            AS order_count,
        MAX(ingested_at)    AS last_ingested_at,
        ingestion_batch_id
    FROM {FULL_TABLE_NAME}
    GROUP BY source_date, ingestion_batch_id
    ORDER BY source_date DESC
    LIMIT 7
""").show(truncate=False)

print(f"\nCurrent watermark:")
spark.sql(f"""
    SELECT pipeline, last_date_done, completed_at
    FROM {WATERMARK_TABLE}
    WHERE pipeline = '{FULL_TABLE_NAME}'
    ORDER BY last_date_done DESC
    LIMIT 1
""").show(truncate=False)