# Databricks notebook source
# MAGIC %md
# MAGIC # Reference — Supplier Schedules
# MAGIC
# MAGIC Creates and seeds `3sp_analytics_workspace.reference.supplier_schedules`.
# MAGIC
# MAGIC This table is **manually maintained** — it is not updated by any automated pipeline.
# MAGIC It tells the reorder intelligence layer when each supplier normally delivers,
# MAGIC how far in advance an order must be placed, and any notes about their schedule.
# MAGIC
# MAGIC **Run this notebook manually** whenever supplier schedules change.
# MAGIC It is fully idempotent — re-running updates existing rows and inserts new ones.
# MAGIC
# MAGIC ## Column guide
# MAGIC
# MAGIC | Column | Description |
# MAGIC |---|---|
# MAGIC | `supplier_name` | Must match `reference.item_catalog.supplier` exactly (case-sensitive) |
# MAGIC | `delivery_days` | Comma-separated day names: `Monday`, `Tuesday`, ..., `Saturday` |
# MAGIC | `order_cutoff_days_ahead` | How many days before delivery the order must be placed |
# MAGIC | `order_cutoff_hour_et` | Hour of day (ET, 24h) by which the order must be submitted |
# MAGIC | `min_order_days` | Minimum days between order date and expected delivery (lead time) |
# MAGIC | `max_order_days` | Maximum days — upper bound on lead time when schedule slips |
# MAGIC | `is_active` | False = supplier paused or discontinued, exclude from reorder signals |
# MAGIC | `notes` | Freeform — holiday schedules, weather policies, rep contact, etc. |
# MAGIC
# MAGIC ## How to add or update a supplier
# MAGIC Edit the `SUPPLIER_SCHEDULES` list below and re-run this notebook.

# COMMAND ----------

CATALOG = "3sp_analytics_workspace"
TABLE   = f"{CATALOG}.reference.supplier_schedules"

# COMMAND ----------

# ── TABLE DDL ─────────────────────────────────────────────────────────────────

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        supplier_name             STRING    NOT NULL  COMMENT 'Must match reference.item_catalog.supplier exactly',
        delivery_days             STRING              COMMENT 'Comma-separated delivery day names, e.g. Monday,Thursday',
        order_cutoff_days_ahead   INT                 COMMENT 'Days before delivery that order must be placed',
        order_cutoff_hour_et      INT                 COMMENT 'Hour of day ET (24h) by which order must be submitted',
        min_order_days            INT                 COMMENT 'Minimum days between order and delivery (best case lead time)',
        max_order_days            INT                 COMMENT 'Maximum days between order and delivery (worst case / weather / holiday)',
        is_active                 BOOLEAN   NOT NULL  COMMENT 'False if supplier is paused or discontinued',
        notes                     STRING              COMMENT 'Holiday schedules, weather policy, rep contact info, special instructions',
        last_updated_at           TIMESTAMP NOT NULL  COMMENT 'When this row was last edited in this notebook'
    )
    USING DELTA
    COMMENT 'Reference: supplier delivery schedules and lead times. Manually maintained. Powers the reorder intelligence layer.'
    TBLPROPERTIES ('quality' = 'reference', 'delta.enableChangeDataFeed' = 'true')
""")

print(f"✓ Table ready: {TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Supplier Schedule Data
# MAGIC
# MAGIC Edit the list below. Re-run the notebook to apply changes.
# MAGIC
# MAGIC **Known suppliers from `reference.item_catalog`:**
# MAGIC - Faire (marketplace — 355 SKUs, many underlying vendors — add individual vendors as needed)
# MAGIC - Rainforest (180 SKUs)
# MAGIC - Baldor (49 SKUs)
# MAGIC - Lisa Schafer - Fine Art (26 SKUs)
# MAGIC - Nobl (25 SKUs)
# MAGIC - Friend (14 SKUs)
# MAGIC - Burke's (6 SKUs)
# MAGIC - Kneady Mama (4 SKUs — Tue/Fri bread delivery, confirmed in sales data)
# MAGIC - Snowy Owl Coffee Roasters (4 SKUs)
# MAGIC - Simple Ecology (2 SKUs)
# MAGIC - George (1 SKU)
# MAGIC - ~402 SKUs have no supplier assigned — update item_catalog in Toast BOH

# COMMAND ----------

import datetime
from delta.tables import DeltaTable
from pyspark.sql import functions as F

NOW_TS = datetime.datetime.now(datetime.timezone.utc).isoformat()

# ── Edit this list to add or update suppliers ──────────────────────────────────
# Leave delivery_days, cutoff, and lead times as None if unknown — the reorder
# model will skip those suppliers until the data is filled in.

SUPPLIER_SCHEDULES = [
    {
        "supplier_name":           "Kneady Mama",
        "delivery_days":           "Tuesday,Friday",
        "order_cutoff_days_ahead": 1,
        "order_cutoff_hour_et":    12,
        "min_order_days":          1,
        "max_order_days":          2,
        "is_active":               True,
        "notes":                   "Bread delivery Tue and Fri. Order by noon the day before. Weather can delay — check day-of if snow forecast."
    },
    {
        "supplier_name":           "Rainforest",
        "delivery_days":           None,
        "order_cutoff_days_ahead": None,
        "order_cutoff_hour_et":    None,
        "min_order_days":          None,
        "max_order_days":          None,
        "is_active":               True,
        "notes":                   "Delivery schedule not yet confirmed. Update this row once known."
    },
    {
        "supplier_name":           "Baldor",
        "delivery_days":           None,
        "order_cutoff_days_ahead": None,
        "order_cutoff_hour_et":    None,
        "min_order_days":          None,
        "max_order_days":          None,
        "is_active":               True,
        "notes":                   "Specialty food distributor. Delivery schedule not yet confirmed."
    },
    {
        "supplier_name":           "Faire",
        "delivery_days":           None,
        "order_cutoff_days_ahead": None,
        "order_cutoff_hour_et":    None,
        "min_order_days":          3,
        "max_order_days":          10,
        "is_active":               True,
        "notes":                   "Faire is a marketplace — lead times vary widely by individual brand/vendor. This row is a fallback. Add individual Faire vendors as separate rows (e.g. 'Faire - Kolsvart') with their specific schedules."
    },
    {
        "supplier_name":           "Nobl",
        "delivery_days":           None,
        "order_cutoff_days_ahead": None,
        "order_cutoff_hour_et":    None,
        "min_order_days":          None,
        "max_order_days":          None,
        "is_active":               True,
        "notes":                   "Delivery schedule not yet confirmed."
    },
    {
        "supplier_name":           "Friend",
        "delivery_days":           None,
        "order_cutoff_days_ahead": None,
        "order_cutoff_hour_et":    None,
        "min_order_days":          None,
        "max_order_days":          None,
        "is_active":               True,
        "notes":                   "Delivery schedule not yet confirmed."
    },
    {
        "supplier_name":           "Burke's",
        "delivery_days":           None,
        "order_cutoff_days_ahead": None,
        "order_cutoff_hour_et":    None,
        "min_order_days":          None,
        "max_order_days":          None,
        "is_active":               True,
        "notes":                   "Delivery schedule not yet confirmed."
    },
    {
        "supplier_name":           "Snowy Owl Coffee Roasters",
        "delivery_days":           None,
        "order_cutoff_days_ahead": None,
        "order_cutoff_hour_et":    None,
        "min_order_days":          None,
        "max_order_days":          None,
        "is_active":               True,
        "notes":                   "Cafe coffee supplier. Delivery schedule not yet confirmed."
    },
    {
        "supplier_name":           "Simple Ecology",
        "delivery_days":           None,
        "order_cutoff_days_ahead": None,
        "order_cutoff_hour_et":    None,
        "min_order_days":          None,
        "max_order_days":          None,
        "is_active":               True,
        "notes":                   "Delivery schedule not yet confirmed."
    },
    {
        "supplier_name":           "Lisa Schafer - Fine Art",
        "delivery_days":           None,
        "order_cutoff_days_ahead": None,
        "order_cutoff_hour_et":    None,
        "min_order_days":          None,
        "max_order_days":          None,
        "is_active":               True,
        "notes":                   "Art/housewares items. Consignment or direct — confirm ordering process."
    },
    {
        "supplier_name":           "George",
        "delivery_days":           None,
        "order_cutoff_days_ahead": None,
        "order_cutoff_hour_et":    None,
        "min_order_days":          None,
        "max_order_days":          None,
        "is_active":               True,
        "notes":                   "Single-SKU supplier. Delivery schedule not yet confirmed."
    },
]

# Add audit timestamp
for row in SUPPLIER_SCHEDULES:
    row["last_updated_at"] = NOW_TS

# COMMAND ----------

# ── UPSERT ────────────────────────────────────────────────────────────────────

df = (
    spark.createDataFrame(SUPPLIER_SCHEDULES)
    .withColumn("last_updated_at", F.col("last_updated_at").cast("timestamp"))
)

DeltaTable.forName(spark, TABLE).alias("t").merge(
    df.alias("s"), "t.supplier_name = s.supplier_name"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

print(f"✓ Upserted {len(SUPPLIER_SCHEDULES)} supplier schedule rows into {TABLE}")

# COMMAND ----------

# ── REVIEW ────────────────────────────────────────────────────────────────────
# MAGIC %md
# MAGIC ## Current Supplier Schedules

spark.sql(f"""
    SELECT
        supplier_name,
        delivery_days,
        order_cutoff_days_ahead,
        min_order_days,
        max_order_days,
        is_active,
        notes
    FROM {TABLE}
    ORDER BY
        CASE WHEN delivery_days IS NOT NULL THEN 0 ELSE 1 END,
        supplier_name
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Suppliers in item_catalog with no schedule row yet
# MAGIC
# MAGIC These suppliers have active SKUs but no matching row in this table.
# MAGIC Add them above and re-run.

spark.sql(f"""
    SELECT
        COALESCE(c.supplier, '(no supplier assigned)') AS supplier,
        COUNT(*) AS active_sku_count
    FROM {CATALOG}.reference.item_catalog c
    LEFT JOIN {TABLE} s ON c.supplier = s.supplier_name
    WHERE (c.is_discontinued = false OR c.is_discontinued IS NULL)
      AND s.supplier_name IS NULL
    GROUP BY COALESCE(c.supplier, '(no supplier assigned)')
    ORDER BY active_sku_count DESC
""").show(truncate=False)
