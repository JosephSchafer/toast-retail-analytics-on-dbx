# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze — Ingest Toast Retail Inventory History
# MAGIC
# MAGIC Writes raw inventory adjustment events from the Toast Retail API
# MAGIC `/v1/inventoryHistory/search` into `3sp_analytics_workspace.bronze.toast_inventory_history_raw`.
# MAGIC
# MAGIC | Mode | What it does |
# MAGIC |---|---|
# MAGIC | `incremental` | Pulls events updated yesterday. Run on the nightly schedule. |
# MAGIC | `backfill` | Fetches in weekly chunks from `backfill_start_date` to yesterday. Resumable via watermark. |
# MAGIC
# MAGIC **Key design decisions:**
# MAGIC - MERGE on `inventory_history_log_id` — idempotent, reruns are safe
# MAGIC - Full raw JSON preserved — Silver parses it later
# MAGIC - `quantity_on_hand` denormalized at ingest for fast position queries
# MAGIC - Backfill uses weekly date chunks — reduces API calls ~7× vs daily loop
# MAGIC - One MERGE per weekly chunk — reduces Delta transaction overhead ~7×
# MAGIC - `source_date` set on first insert, never updated on merge — stable partition key

# COMMAND ----------

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
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, TimestampType, DateType
)

# COMMAND ----------

# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────

RUN_MODE           = dbutils.widgets.get("run_mode")
BACKFILL_START_RAW = dbutils.widgets.get("backfill_start_date")
BACKFILL_END_RAW   = dbutils.widgets.get("backfill_end_date")

TOAST_AUTH_URL     = "https://ws.toasttab.com/authentication/v1/authentication/login"
TOAST_INV_HIST_URL = "https://ws.toasttab.com/retail/v1/inventoryHistory/search"

CATALOG         = "3sp_analytics_workspace"
FULL_TABLE_NAME = f"{CATALOG}.bronze.toast_inventory_history_raw"
WATERMARK_TABLE = f"{CATALOG}.bronze.ingestion_watermark"

CHUNK_DAYS        = 7
BACKFILL_MAX_DAYS = 400
MAX_RETRIES       = 3
TOKEN_TTL_MINUTES = 50

INV_SCHEMA = StructType([
    StructField("inventory_history_log_id", StringType(),    False),
    StructField("item_version_id",          StringType(),    True),
    StructField("adjustment_type",          StringType(),    True),
    StructField("quantity_on_hand",         DoubleType(),    True),
    StructField("quantity_variance",        DoubleType(),    True),
    StructField("created_date",             TimestampType(), True),
    StructField("modified_date",            TimestampType(), True),
    StructField("purchase_order_id",        StringType(),    True),
    StructField("supplier_id",              StringType(),    True),
    StructField("supplier_name",            StringType(),    True),
    StructField("raw_json",                 StringType(),    False),
    StructField("ingested_at",              TimestampType(), False),
    StructField("ingestion_batch_id",       StringType(),    False),
    StructField("source_date",              DateType(),      False),
    StructField("restaurant_guid",          StringType(),    False),
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
        inventory_history_log_id  STRING      NOT NULL  COMMENT 'Toast inventory history log GUID — primary dedup key',
        item_version_id           STRING                COMMENT 'Toast item version ID of the adjusted item',
        adjustment_type           STRING                COMMENT 'RECEIVE, COUNT, DAMAGE, LOSS, THEFT, WASTED, etc.',
        quantity_on_hand          DOUBLE                COMMENT 'Quantity on hand after this adjustment — denormalized for fast position queries',
        quantity_variance         DOUBLE                COMMENT 'Variance between actual and expected quantity — populated for COUNT events',
        created_date              TIMESTAMP             COMMENT 'When this adjustment was created in Toast',
        modified_date             TIMESTAMP             COMMENT 'Last modified timestamp — used for watermark filtering',
        purchase_order_id         STRING                COMMENT 'Linked PO GUID for RECEIVE adjustments — null for all other types',
        supplier_id               STRING                COMMENT 'Supplier GUID if present on the event',
        supplier_name             STRING                COMMENT 'Supplier name — denormalized for quick filtering',
        raw_json                  STRING      NOT NULL  COMMENT 'Full raw JSON from Toast API — never modified',
        ingested_at               TIMESTAMP   NOT NULL  COMMENT 'UTC timestamp when this row was first written',
        ingestion_batch_id        STRING      NOT NULL  COMMENT 'UUID for this ingestion run',
        source_date               DATE        NOT NULL  COMMENT 'First ingestion date — set on insert, never updated on merge',
        restaurant_guid           STRING      NOT NULL  COMMENT 'Toast Restaurant External ID'
    )
    USING DELTA
    PARTITIONED BY (source_date)
    COMMENT 'Bronze: raw Toast inventory history adjustment events from Retail API. Append/merge only. No transforms.'
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

    - 401: raises immediately — caller must re-authenticate
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


def fetch_inventory_history(start_date: datetime.date, end_date: datetime.date,
                            session: requests.Session, token: str) -> list:
    """Fetch all inventory history events with modifiedDate in [start_date, end_date).

    Fetches all adjustment types — Silver will filter by type for analytics.
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

    all_entries = []
    page_token  = None

    while True:
        params    = {"pageToken": page_token} if page_token else {}
        resp      = api_post_with_retry(session, TOAST_INV_HIST_URL, headers, body, params)
        payload   = resp.json()
        page_data = payload.get("data", [])
        all_entries.extend(page_data)

        page_token = payload.get("nextPageToken")
        if not page_token or not page_data:
            break

    return all_entries


def _parse_ts(raw: str | None) -> datetime.datetime | None:
    if not raw:
        return None
    return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))


def build_rows(entries: list, source_date: datetime.date,
               batch_id: str, now_utc: datetime.datetime) -> list:
    """Convert API dicts to typed Python dicts matching INV_SCHEMA."""
    rows = []
    for entry in entries:
        qty_info = entry.get("quantityInfo") or {}
        supplier = entry.get("supplier") or {}
        rows.append({
            "inventory_history_log_id": entry.get("inventoryHistoryLogId", "MISSING_ID"),
            "item_version_id":          entry.get("itemVersionId"),
            "adjustment_type":          entry.get("adjustmentType"),
            "quantity_on_hand":         qty_info.get("quantityOnHand"),
            "quantity_variance":        qty_info.get("quantityVariance"),
            "created_date":             _parse_ts(entry.get("createdDate")),
            "modified_date":            _parse_ts(entry.get("modifiedDate")),
            "purchase_order_id":        entry.get("purchaseOrderId"),
            "supplier_id":              supplier.get("id"),
            "supplier_name":            supplier.get("name"),
            "raw_json":                 json.dumps(entry),
            "ingested_at":              now_utc,
            "ingestion_batch_id":       batch_id,
            "source_date":              source_date,
            "restaurant_guid":          TOAST_RESTAURANT_GUID
        })
    return rows


def merge_to_bronze(rows: list):
    """Merge inventory history rows into the Bronze Delta table.

    source_date is excluded from the update set — it records when the entry
    was first ingested and should not shift on subsequent merges.
    """
    if not rows:
        return

    df = spark.createDataFrame(rows, schema=INV_SCHEMA)

    DeltaTable.forName(spark, FULL_TABLE_NAME).alias("t").merge(
        df.alias("s"), "t.inventory_history_log_id = s.inventory_history_log_id"
    ).whenMatchedUpdate(set={
        "adjustment_type":    "s.adjustment_type",
        "quantity_on_hand":   "s.quantity_on_hand",
        "quantity_variance":  "s.quantity_variance",
        "modified_date":      "s.modified_date",
        "raw_json":           "s.raw_json",
        "ingested_at":        "s.ingested_at",
        "ingestion_batch_id": "s.ingestion_batch_id"
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
    start  = yesterday - datetime.timedelta(days=1)
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

batch_id     = str(uuid.uuid4())
token_issued = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
token        = None
session      = requests.Session()

total_entries = 0
failed_chunks = []

print(f"\nBatch ID: {batch_id}")
print(f"Processing {len(chunks)} chunk(s)...\n")

try:
    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):

        # total_seconds() — not .seconds — gives correct elapsed time across hour boundaries
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - token_issued).total_seconds() / 60
        if elapsed >= TOKEN_TTL_MINUTES:
            print("  ↻ Refreshing Toast token...")
            token        = get_access_token(session)
            token_issued = datetime.datetime.now(datetime.timezone.utc)

        print(f"[{i}/{len(chunks)}] {chunk_start} → {chunk_end}")

        try:
            entries = fetch_inventory_history(chunk_start, chunk_end, session, token)
            rows    = build_rows(entries, chunk_start, batch_id, now_utc)
            merge_to_bronze(rows)
            total_entries += len(entries)
            print(f"  ✓ {len(entries)} entries merged")

            if RUN_MODE == "backfill":
                update_watermark(chunk_end)

        except RuntimeError as e:
            if "401" in str(e):
                raise  # Auth failures abort immediately — no point continuing
            print(f"  ✗ FAILED: {e}")
            failed_chunks.append((chunk_start, chunk_end, str(e)))

finally:
    session.close()

if RUN_MODE == "incremental" and not failed_chunks:
    update_watermark(yesterday)

# COMMAND ----------

# ── 8. RUN SUMMARY ────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("INGESTION COMPLETE — Inventory History")
print("=" * 60)
print(f"  Batch ID:              {batch_id}")
print(f"  Chunks processed:      {len(chunks)}")
print(f"  Total entries merged:  {total_entries}")
print(f"  Failures:              {len(failed_chunks)}")

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
        adjustment_type,
        COUNT(*)                        AS event_count,
        COUNT(DISTINCT item_version_id) AS distinct_items,
        MAX(ingested_at)                AS last_ingested_at
    FROM {FULL_TABLE_NAME}
    GROUP BY source_date, adjustment_type
    ORDER BY source_date DESC, event_count DESC
    LIMIT 20
""").show(truncate=False)
