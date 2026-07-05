# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze — Ingest Toast Retail Purchase Orders
# MAGIC
# MAGIC Writes raw purchase order data from the Toast Retail API
# MAGIC `/v1/purchaseOrders/search` into `YOUR_CATALOG.bronze.toast_purchase_orders_raw`.
# MAGIC
# MAGIC | Mode | What it does |
# MAGIC |---|---|
# MAGIC | `incremental` | Pulls POs updated yesterday. Run on the nightly schedule. |
# MAGIC | `backfill` | Fetches in weekly chunks from `backfill_start_date` to yesterday. Resumable via watermark. |
# MAGIC
# MAGIC **Key design decisions:**
# MAGIC - MERGE on `po_id` — idempotent, reruns are safe
# MAGIC - Full raw JSON preserved — Silver parses it later
# MAGIC - Backfill uses weekly date chunks (not day-by-day) — reduces API calls ~7× vs daily loop
# MAGIC - One MERGE per weekly chunk — not per day — reduces Delta transaction overhead ~7×
# MAGIC - `source_date` is set on first insert and never updated — partition key stays stable

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

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

import requests
import json
import datetime
import uuid
import time

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, TimestampType, DateType
)

# COMMAND ----------

# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────

RUN_MODE           = dbutils.widgets.get("run_mode")
BACKFILL_START_RAW = dbutils.widgets.get("backfill_start_date")
BACKFILL_END_RAW   = dbutils.widgets.get("backfill_end_date")

TOAST_AUTH_URL = "https://ws.toasttab.com/authentication/v1/authentication/login"
TOAST_PO_URL   = "https://ws.toasttab.com/retail/v1/purchaseOrders/search"

CATALOG         = "YOUR_CATALOG"
FULL_TABLE_NAME = f"{CATALOG}.bronze.toast_purchase_orders_raw"
WATERMARK_TABLE = f"{CATALOG}.bronze.ingestion_watermark"

CHUNK_DAYS        = 7    # API calls cover one week at a time
BACKFILL_MAX_DAYS = 400
MAX_RETRIES       = 3
TOKEN_TTL_MINUTES = 50   # Refresh token before Toast's ~60 min expiry

# Explicit schema — avoids Spark type inference and post-creation casts
PO_SCHEMA = StructType([
    StructField("po_id",              StringType(),    False),
    StructField("po_number",          LongType(),      True),
    StructField("status",             StringType(),    True),
    StructField("supplier_id",        StringType(),    True),
    StructField("supplier_name",      StringType(),    True),
    StructField("updated_date",       TimestampType(), True),
    StructField("raw_json",           StringType(),    False),
    StructField("ingested_at",        TimestampType(), False),
    StructField("ingestion_batch_id", StringType(),    False),
    StructField("source_date",        DateType(),      False),
    StructField("restaurant_guid",    StringType(),    False),
])

# COMMAND ----------

# ── 3. SECRETS ────────────────────────────────────────────────────────────────

try:
    TOAST_CLIENT_ID       = dbutils.secrets.get(scope="toast_api", key="toast_client_id")
    TOAST_CLIENT_SECRET   = dbutils.secrets.get(scope="toast_api", key="toast_client_secret")
    TOAST_RESTAURANT_GUID = dbutils.secrets.get(scope="toast_api", key="restaurant_guid")
    print("✓ Secrets loaded")
except Exception as e:
    print("✗ Secret retrieval failed. Check scope='toast_api' and key names.")
    raise e

# COMMAND ----------

# ── 4. TABLE SETUP ────────────────────────────────────────────────────────────

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME} (
        po_id               STRING      NOT NULL  COMMENT 'Toast PO GUID — primary dedup key',
        po_number           BIGINT                COMMENT 'Toast-generated purchase order number',
        status              STRING                COMMENT 'DRAFT, ORDERED, PARTIALLY_RECEIVED, RECEIVED, VOIDED',
        supplier_id         STRING                COMMENT 'Toast supplier GUID',
        supplier_name       STRING                COMMENT 'Supplier name — denormalized for quick filtering',
        updated_date        TIMESTAMP             COMMENT 'Last modified timestamp from Toast',
        raw_json            STRING      NOT NULL  COMMENT 'Full raw JSON from Toast API — never modified',
        ingested_at         TIMESTAMP   NOT NULL  COMMENT 'UTC timestamp when this row was first written',
        ingestion_batch_id  STRING      NOT NULL  COMMENT 'UUID for this ingestion run',
        source_date         DATE        NOT NULL  COMMENT 'First ingestion date — set on insert, never updated on merge',
        restaurant_guid     STRING      NOT NULL  COMMENT 'Toast Restaurant External ID'
    )
    USING DELTA
    PARTITIONED BY (source_date)
    COMMENT 'Bronze: raw Toast purchase order data from Retail API. Append/merge only. No transforms.'
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true', 'quality' = 'bronze')
""")

print(f"✓ Table ready: {FULL_TABLE_NAME}")

# COMMAND ----------

# ── 5. HELPER FUNCTIONS ───────────────────────────────────────────────────────

def get_access_token(session: requests.Session) -> str:
    """Authenticate with Toast and return a bearer token."""
    resp = session.post(
        TOAST_AUTH_URL,
        json={
            "clientId":       TOAST_CLIENT_ID,
            "clientSecret":   TOAST_CLIENT_SECRET,
            "userAccessType": "TOAST_MACHINE_CLIENT"
        },
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    resp.raise_for_status()
    token = resp.json()["token"]["accessToken"]
    print("✓ Toast authentication successful")
    return token


def api_post_with_retry(session: requests.Session, url: str, headers: dict,
                        body: dict, params: dict = None) -> requests.Response:
    """POST with exponential backoff retry.

    - 401: raises immediately — re-authentication is the caller's responsibility
    - 429: backs off and retries
    - Other non-2xx: retries up to MAX_RETRIES times
    """
    for attempt in range(1, MAX_RETRIES + 1):
        resp = session.post(url, headers=headers, json=body, params=params or {}, timeout=60)

        if resp.status_code == 401:
            raise RuntimeError(f"Toast API returned 401 — token may have expired: {resp.text[:200]}")

        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"  Rate limited (429) — waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp

    raise RuntimeError(f"API call failed after {MAX_RETRIES} attempts")


def fetch_purchase_orders(start_date: datetime.date, end_date: datetime.date,
                          session: requests.Session, token: str) -> list:
    """Fetch all POs with updatedDate in [start_date, end_date).

    Uses cursor-based pagination — fetches all pages before returning.
    """
    headers = {
        "Authorization":                f"Bearer {token}",
        "Toast-Restaurant-External-ID": TOAST_RESTAURANT_GUID,
        "Content-Type":                 "application/json"
    }
    body = {
        "updatedDateRange": {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate":   end_date.strftime("%Y-%m-%d")
        }
    }

    all_pos    = []
    page_token = None

    while True:
        params = {"pageToken": page_token} if page_token else {}
        resp      = api_post_with_retry(session, TOAST_PO_URL, headers, body, params)
        payload   = resp.json()
        page_data = payload.get("data", [])
        all_pos.extend(page_data)

        page_token = payload.get("nextPageToken")
        if not page_token or not page_data:
            break

    return all_pos


def build_rows(pos: list, source_date: datetime.date,
               batch_id: str, now_utc: datetime.datetime) -> list:
    """Convert API dicts to typed Python dicts matching PO_SCHEMA."""
    rows = []
    for po in pos:
        supplier     = po.get("supplier") or {}
        modified_raw = po.get("modifiedDate")
        rows.append({
            "po_id":               po.get("id", "MISSING_ID"),
            "po_number":           po.get("toastGeneratedPurchaseOrderNumber"),
            "status":              po.get("status"),
            "supplier_id":         supplier.get("id"),
            "supplier_name":       supplier.get("name"),
            "updated_date":        datetime.datetime.fromisoformat(modified_raw.replace("Z", "+00:00")) if modified_raw else None,
            "raw_json":            json.dumps(po),
            "ingested_at":         now_utc,
            "ingestion_batch_id":  batch_id,
            "source_date":         source_date,
            "restaurant_guid":     TOAST_RESTAURANT_GUID
        })
    return rows


def merge_to_bronze(rows: list):
    """Merge a list of rows into the Bronze Delta table.

    source_date is intentionally excluded from the update set — it records
    when the PO was first seen and should not shift as the PO status changes.
    """
    if not rows:
        return

    df = spark.createDataFrame(rows, schema=PO_SCHEMA)

    DeltaTable.forName(spark, FULL_TABLE_NAME).alias("t").merge(
        df.alias("s"), "t.po_id = s.po_id"
    ).whenMatchedUpdate(set={
        "status":              "s.status",
        "supplier_name":       "s.supplier_name",
        "updated_date":        "s.updated_date",
        "raw_json":            "s.raw_json",
        "ingested_at":         "s.ingested_at",
        "ingestion_batch_id":  "s.ingestion_batch_id"
        # source_date intentionally omitted — stable partition key
    }).whenNotMatchedInsertAll().execute()


def update_watermark(up_to_date: datetime.date):
    """Upsert watermark — one row per pipeline, never grows unbounded."""
    spark.sql(f"""
        MERGE INTO {WATERMARK_TABLE} AS t
        USING (
            SELECT
                '{FULL_TABLE_NAME}'    AS pipeline,
                DATE '{up_to_date}'    AS last_date_done,
                CURRENT_TIMESTAMP()    AS completed_at
        ) AS s ON t.pipeline = s.pipeline
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)


def get_watermark() -> datetime.date | None:
    """Return last successfully completed date, or None."""
    row = (
        spark.table(WATERMARK_TABLE)
        .filter(F.col("pipeline") == FULL_TABLE_NAME)
        .orderBy(F.col("last_date_done").desc())
        .first()
    )
    return row["last_date_done"] if row else None


def weekly_chunks(start: datetime.date, end: datetime.date) -> list[tuple]:
    """Split [start, end] into (chunk_start, chunk_end) pairs of up to CHUNK_DAYS."""
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + datetime.timedelta(days=CHUNK_DAYS - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + datetime.timedelta(days=1)
    return chunks

# COMMAND ----------

# ── 6. BUILD PROCESSING PLAN ──────────────────────────────────────────────────

now_utc   = datetime.datetime.now(datetime.timezone.utc)
yesterday = (now_utc - datetime.timedelta(days=1)).date()

if RUN_MODE == "incremental":
    # Two-day lookback to catch late-arriving PO updates
    start = yesterday - datetime.timedelta(days=1)
    chunks = [(start, yesterday)]
    print(f"Mode: INCREMENTAL — fetching {start} → {yesterday}")

elif RUN_MODE == "backfill":
    backfill_start = datetime.date.fromisoformat(BACKFILL_START_RAW)
    backfill_end   = datetime.date.fromisoformat(BACKFILL_END_RAW) if BACKFILL_END_RAW.strip() else yesterday

    watermark = get_watermark()
    if watermark and watermark >= backfill_start:
        backfill_start = watermark + datetime.timedelta(days=1)
        print(f"Resuming from watermark — starting at {backfill_start}")

    total_days = (backfill_end - backfill_start).days + 1
    if total_days > BACKFILL_MAX_DAYS:
        raise ValueError(f"Backfill range {total_days} days exceeds BACKFILL_MAX_DAYS={BACKFILL_MAX_DAYS}")

    chunks = weekly_chunks(backfill_start, backfill_end)
    print(f"Mode: BACKFILL — {total_days} days in {len(chunks)} weekly chunks ({backfill_start} → {backfill_end})")

else:
    raise ValueError(f"Unknown run_mode: '{RUN_MODE}'")

# COMMAND ----------

# ── 7. MAIN INGESTION LOOP ────────────────────────────────────────────────────

batch_id      = str(uuid.uuid4())
token_issued  = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)  # force immediate auth
token         = None
session       = requests.Session()

total_pos     = 0
failed_chunks = []

print(f"\nBatch ID: {batch_id}")
print(f"Processing {len(chunks)} chunk(s)...\n")

try:
    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):

        # Refresh token proactively using total_seconds() — not .seconds
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - token_issued).total_seconds() / 60
        if elapsed >= TOKEN_TTL_MINUTES:
            print("  ↻ Refreshing Toast token...")
            token        = get_access_token(session)
            token_issued = datetime.datetime.now(datetime.timezone.utc)

        print(f"[{i}/{len(chunks)}] {chunk_start} → {chunk_end}")

        try:
            pos  = fetch_purchase_orders(chunk_start, chunk_end, session, token)
            rows = build_rows(pos, chunk_start, batch_id, now_utc)
            merge_to_bronze(rows)
            total_pos += len(pos)
            print(f"  ✓ {len(pos)} POs merged")

            if RUN_MODE == "backfill":
                update_watermark(chunk_end)

        except RuntimeError as e:
            if "401" in str(e):
                # Auth failure affects all remaining chunks — abort immediately
                raise
            print(f"  ✗ FAILED: {e}")
            failed_chunks.append((chunk_start, chunk_end, str(e)))

finally:
    session.close()

if RUN_MODE == "incremental" and not failed_chunks:
    update_watermark(yesterday)

# COMMAND ----------

# ── 8. RUN SUMMARY ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("INGESTION COMPLETE — Purchase Orders")
print("=" * 60)
print(f"  Batch ID:         {batch_id}")
print(f"  Chunks processed: {len(chunks)}")
print(f"  Total POs merged: {total_pos}")
print(f"  Failures:         {len(failed_chunks)}")

if failed_chunks:
    for cs, ce, err in failed_chunks:
        print(f"  ✗ {cs} → {ce}: {err}")
    raise RuntimeError(f"Ingestion completed with {len(failed_chunks)} failed chunk(s).")
else:
    print("\n  ✓ All chunks ingested successfully")

# COMMAND ----------

# ── 9. QUICK VALIDATION ───────────────────────────────────────────────────────

spark.sql(f"""
    SELECT
        source_date,
        COUNT(*)                      AS po_count,
        COUNT(DISTINCT supplier_name) AS supplier_count,
        MAX(ingested_at)              AS last_ingested_at
    FROM {FULL_TABLE_NAME}
    GROUP BY source_date
    ORDER BY source_date DESC
    LIMIT 7
""").show(truncate=False)
