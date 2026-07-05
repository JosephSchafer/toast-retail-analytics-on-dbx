# Databricks notebook source
# MAGIC %md
# MAGIC # Reference — Item Catalog (Toast Export)
# MAGIC
# MAGIC Ingests a Toast item library CSV export into a reference Delta table at
# MAGIC `YOUR_CATALOG.reference.item_catalog`.
# MAGIC
# MAGIC ## How to use
# MAGIC
# MAGIC **1. Export from Toast Back-of-House**
# MAGIC
# MAGIC Navigate to **Back of House → Menu → Items → Export CSV**. Toast will download
# MAGIC a file named like `retail-items-YYYY-MM-DD-HHMMSS.csv`. Do not rename it —
# MAGIC the timestamp in the filename is used to track when the snapshot was taken.
# MAGIC
# MAGIC **2. Upload the CSV to the Databricks Volume**
# MAGIC
# MAGIC The notebook reads from `/Volumes/YOUR_CATALOG/default/raw_toast_data/`.
# MAGIC To upload a file there:
# MAGIC
# MAGIC 1. In the Databricks left sidebar, click **Catalog**
# MAGIC 2. Navigate to `YOUR_CATALOG` → `default` → **Volumes** → `raw_toast_data`
# MAGIC 3. Click **Upload to this volume** (top right of the volume view)
# MAGIC 4. Drag your CSV into the upload dialog, or click to browse, then click **Upload**
# MAGIC 5. Copy the filename — you'll need it in the next step
# MAGIC
# MAGIC It's fine to leave previous exports in the folder. Old files don't interfere;
# MAGIC the notebook only reads the file specified in the widget.
# MAGIC
# MAGIC **3. Set the `csv_filename` widget** to the filename you just uploaded (e.g. `retail-items-2026-03-21-055413.csv`)
# MAGIC
# MAGIC **4. Run the notebook** — it will overwrite the reference table with the latest snapshot and run a category alignment report
# MAGIC
# MAGIC ## Design notes
# MAGIC - This is a **full replace** on every run, not a MERGE. The item catalog is a
# MAGIC   current-state reference snapshot, not an append ledger. Re-running is always safe.
# MAGIC - The join key to `silver_sales.order_items_silver` is `item_id` = `item_guid`
# MAGIC - `sales_category` is a text name (not a GUID) in the Toast export — this is
# MAGIC   intentional; Toast does not expose sales category GUIDs in the item library CSV
# MAGIC - `category` is the primary reporting dimension used in Gold tables and Genie queries.
# MAGIC   `sales_category` should always match it — the alignment report at the end of this
# MAGIC   notebook flags any items where they differ.
# MAGIC - `contains_alcohol`, `tax_inclusion`, and `applicable_taxes` support tax-aware
# MAGIC   Gold aggregations across your Grocery / Alcohol / Cafe Menu product mix

# COMMAND ----------

# ── WIDGET ────────────────────────────────────────────────────────────────────

dbutils.widgets.text(
    name="csv_filename",
    defaultValue="retail-items-2026-03-21-055413.csv",
    label="CSV filename (in /Volumes/.../raw_toast_data/)"
)

# COMMAND ----------

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, TimestampType
import datetime

# COMMAND ----------

# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────

CSV_FILENAME  = dbutils.widgets.get("csv_filename")
VOLUME_PATH   = f"/Volumes/YOUR_CATALOG/default/raw_toast_data/{CSV_FILENAME}"

CATALOG       = "YOUR_CATALOG"
REF_SCHEMA    = "reference"
REF_TABLE     = f"{CATALOG}.{REF_SCHEMA}.item_catalog"

# COMMAND ----------

# ── 3. SETUP ──────────────────────────────────────────────────────────────────

spark.sql(f"CREATE CATALOG IF NOT EXISTS `{CATALOG}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{REF_SCHEMA}`")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {REF_TABLE} (
        -- Join key
        item_id                 STRING      COMMENT 'Toast item GUID — joins to order_items_silver.item_guid',
        item_multi_location_id  STRING      COMMENT 'Toast multi-location item ID',

        -- Item identity
        name                    STRING      COMMENT 'Item display name',
        pos_name                STRING      COMMENT 'Name shown on POS if different from display name',
        brand                   STRING      COMMENT 'Brand/producer name',
        description             STRING      COMMENT 'Item description',
        item_type               STRING      COMMENT 'DEFAULT, OPEN_ITEM, etc.',

        -- Three-level category hierarchy
        category_group          STRING      COMMENT 'Top-level menu grouping: Grocery, Alcohol, Cafe Menu, Housewares & Decor, etc.',
        category                STRING      COMMENT 'Mid-level category within group: Candy, Beer, Sandwiches, etc.',
        subcategory             STRING      COMMENT 'Optional subcategory',
        sales_category          STRING      COMMENT 'Toast reporting sales category (33 values): Cafe Food, Beer, Candy, etc. — joins to transaction sales_category_guid by item_guid',

        -- Tax and alcohol flags (critical for MA tax logic)
        applicable_taxes        STRING      COMMENT 'Tax names applied to this item, e.g. MA State Sales Tax',
        tax_rates               STRING      COMMENT 'Tax rate string, e.g. 6.25%',
        tax_inclusion           STRING      COMMENT 'TAX_NOT_INCLUDED, TAX_INCLUDED',
        contains_alcohol        BOOLEAN     COMMENT 'True if item is alcoholic — beer, wine, cider',
        takeout_delivery_tax_exempt STRING  COMMENT 'Takeout/delivery tax exemption flag',

        -- Pricing
        price                   DOUBLE      COMMENT 'Current retail price',
        cost                    DOUBLE      COMMENT 'Current unit cost',
        gross_margin            STRING      COMMENT 'Gross margin % as string from export',

        -- Supplier / receiving
        supplier                STRING      COMMENT 'Primary supplier name',
        supplier_item_id        STRING      COMMENT 'Supplier SKU',
        barcode                 STRING      COMMENT 'UPC/barcode',
        plu                     STRING      COMMENT 'PLU code',
        size                    STRING      COMMENT 'Item size description',
        package_unit            STRING      COMMENT 'Unit of packaging',
        receiving_units         STRING      COMMENT 'How item is received (e.g. Case, Each)',
        receiving_unit_quantities STRING    COMMENT 'Quantity per receiving unit',
        unit_of_measure         STRING      COMMENT 'UOM for inventory tracking',

        -- Inventory state
        inventory_status        STRING      COMMENT 'in stock, out of stock, etc.',
        inventory_quantity      STRING      COMMENT 'Current on-hand quantity (string — may be blank)',
        par_min                 STRING      COMMENT 'Minimum par level',
        par_max                 STRING      COMMENT 'Maximum par level',

        -- Availability and status
        visibility              STRING      COMMENT 'Where item appears: POS, Kiosk, Toast Online Orders',
        discount_eligible       STRING      COMMENT 'Yes/No — whether discounts can be applied',
        rewards_eligible        STRING      COMMENT 'Yes/No — whether item earns loyalty rewards',
        is_discontinued         BOOLEAN     COMMENT 'True if item has been discontinued',
        discontinued_reason     STRING      COMMENT 'Reason for discontinuation if applicable',

        -- Sales velocity (snapshot at export time — not historical)
        last_7_day_sales        STRING      COMMENT 'Revenue in last 7 days at export time',
        last_30_day_sales       STRING      COMMENT 'Revenue in last 30 days at export time',
        last_90_day_sales       STRING      COMMENT 'Revenue in last 90 days at export time',

        -- Dates
        created                 STRING      COMMENT 'Item created timestamp from Toast',
        last_updated            STRING      COMMENT 'Item last updated timestamp from Toast',

        -- Data quality
        category_drift          BOOLEAN     COMMENT 'True when sales_category != category — means Toast reporting will differ from menu hierarchy. Should always be false. Fix in Toast BOH.',

        -- Audit
        _catalog_exported_at    TIMESTAMP   COMMENT 'When this snapshot was loaded (derived from filename if parseable, else now)',
        _source_filename        STRING      COMMENT 'Source CSV filename for lineage'
    )
    USING DELTA
    COMMENT 'Reference: Toast item catalog snapshot. Full replace on each import. Join to order_items_silver on item_id = item_guid.'
    TBLPROPERTIES ('quality' = 'reference')
""")

print(f"✓ {REF_TABLE} ready")

# COMMAND ----------

# ── 4. READ CSV ───────────────────────────────────────────────────────────────
# Read with multiLine=true to handle description fields that contain newlines
# (confirmed in the export — item descriptions span multiple lines)

# Spark's multiLine CSV reader can misalign columns when description fields
# contain both embedded newlines AND commas. Using pandas to read the CSV first
# is more robust for this shape of data, then converting to a Spark DataFrame.

import pandas as pd

pandas_df = pd.read_csv(
    VOLUME_PATH.replace("/Volumes/", "/Volumes/"),  # path is already correct
    dtype=str,          # read everything as string — we cast in the transform
    keep_default_na=False,  # don't auto-convert blanks to NaN — we handle nulls explicitly
    na_values=[]        # no auto-null values
)

# Normalise column names: strip whitespace
pandas_df.columns = [c.strip() for c in pandas_df.columns]

raw_df = spark.createDataFrame(pandas_df)


print(f"Raw rows read: {raw_df.count()}")
print(f"Columns found: {len(raw_df.columns)}")

# COMMAND ----------

# ── 5. TRANSFORM ──────────────────────────────────────────────────────────────

def clean_currency(col):
    """Strip $ and commas from currency strings like '$8.75' or '$2,331.61', cast to double.
    Blank strings and unparseable values return null instead of raising a cast error."""
    stripped = F.regexp_replace(F.regexp_replace(col, r'\$', ''), ',', '')
    return F.when(
        stripped.isNull() | (F.trim(stripped) == ''), F.lit(None).cast('double')
    ).otherwise(stripped.cast('double'))

def parse_bool_yn(col):
    """Convert 'Yes'/'No'/blank to boolean. Blank → False (not null) so
    SUM aggregations work correctly — a blank means 'No' in Toast exports."""
    return F.when(F.upper(F.trim(col)) == 'YES', F.lit(True)) \
             .otherwise(F.lit(False))

# Try to extract export date from filename (format: retail-items-YYYY-MM-DD-HHmmss.csv)
try:
    date_part = CSV_FILENAME.replace('retail-items-', '').replace('.csv', '')[:10]
    exported_at = datetime.datetime.strptime(date_part, '%Y-%m-%d').isoformat()
    print(f"Parsed export date from filename: {date_part}")
except Exception:
    exported_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"Could not parse date from filename — using current time")

catalog_df = raw_df.select(

    # ── Join key ──────────────────────────────────────────────────────────────
    F.col("`item id`")                          .alias("item_id"),
    F.col("`item multi location id`")           .alias("item_multi_location_id"),

    # ── Item identity ─────────────────────────────────────────────────────────
    F.col("name"),
    F.col("`pos name`")                         .alias("pos_name"),
    F.col("brand"),
    F.col("description"),
    F.col("`item type`")                        .alias("item_type"),

    # ── Three-level category hierarchy ────────────────────────────────────────
    F.col("`category group`")                   .alias("category_group"),
    F.col("category"),
    F.col("subcategory"),
    F.col("`sales category`")                   .alias("sales_category"),

    # ── Tax and alcohol ───────────────────────────────────────────────────────
    F.col("`applicable taxes`")                 .alias("applicable_taxes"),
    F.col("`tax rates`")                        .alias("tax_rates"),
    F.col("`tax inclusion`")                    .alias("tax_inclusion"),
    # contains alcohol: "Yes" → True, blank → False for retail alcohol items
    parse_bool_yn(F.col("`contains alcohol`"))  .alias("contains_alcohol"),
    F.col("`takeout and delivery tax exemption`").alias("takeout_delivery_tax_exempt"),

    # ── Pricing ───────────────────────────────────────────────────────────────
    clean_currency(F.col("price"))              .alias("price"),
    clean_currency(F.col("cost"))               .alias("cost"),
    F.col("`gross margin`")                     .alias("gross_margin"),

    # ── Supplier / receiving ──────────────────────────────────────────────────
    F.col("supplier"),
    F.col("`supplier item id`")                 .alias("supplier_item_id"),
    F.col("barcode"),
    F.col("plu"),
    F.col("size"),
    F.col("`package unit`")                     .alias("package_unit"),
    F.col("`receiving units`")                  .alias("receiving_units"),
    F.col("`receiving unit quantities`")        .alias("receiving_unit_quantities"),
    F.col("`unit of measure`")                  .alias("unit_of_measure"),

    # ── Inventory state ───────────────────────────────────────────────────────
    F.col("`inventory status`")                 .alias("inventory_status"),
    F.col("`inventory quantity`")               .alias("inventory_quantity"),
    F.col("`par min`")                          .alias("par_min"),
    F.col("`par max`")                          .alias("par_max"),

    # ── Availability ─────────────────────────────────────────────────────────
    F.col("visibility"),
    F.col("`discount eligible`")                .alias("discount_eligible"),
    F.col("`rewards eligible`")                 .alias("rewards_eligible"),
    # Discontinued: blank = active, any value = discontinued
    F.when(
        F.col("Discontinued").isNotNull() & (F.trim(F.col("Discontinued")) != ""),
        F.lit(True)
    ).otherwise(F.lit(False))                   .alias("is_discontinued"),
    F.col("`Discontinued Reason`")              .alias("discontinued_reason"),

    # ── Sales velocity snapshot ───────────────────────────────────────────────
    F.col("`last 7 day sales`")                 .alias("last_7_day_sales"),
    F.col("`last 30 day sales`")                .alias("last_30_day_sales"),
    F.col("`last 90 day sales`")                .alias("last_90_day_sales"),

    # ── Dates ─────────────────────────────────────────────────────────────────
    F.col("created"),
    F.col("`last updated`")                     .alias("last_updated"),

    # ── Audit ─────────────────────────────────────────────────────────────────
    F.lit(exported_at).cast("timestamp")        .alias("_catalog_exported_at"),
    F.lit(CSV_FILENAME)                         .alias("_source_filename")
)

# Compute category_drift after the main select so both columns are available
# Comparison is case-insensitive and trims whitespace to avoid false positives
catalog_df = catalog_df.withColumn(
    "category_drift",
    F.when(
        F.col("sales_category").isNull() | F.col("category").isNull(),
        F.lit(None).cast("boolean")             # can't compare if either is null
    ).otherwise(
        F.lower(F.trim(F.col("sales_category"))) != F.lower(F.trim(F.col("category")))
    )
)

# Drop rows with no item_id (blank trailing rows from CSV export)
catalog_df = catalog_df.filter(
    F.col("item_id").isNotNull() & (F.trim(F.col("item_id")) != "")
)

print(f"Clean rows after filtering: {catalog_df.count()}")

# COMMAND ----------

# ── 6. WRITE — FULL REPLACE ───────────────────────────────────────────────────
# Reference tables are current-state snapshots. We overwrite completely on each
# import so the table always reflects the latest export with no stale rows.

catalog_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(REF_TABLE)

print(f"✓ Written {catalog_df.count()} rows to {REF_TABLE}")

# COMMAND ----------

# ── 6b. APPLY UNITY CATALOG COMMENTS AND TAGS ─────────────────────────────────
# Comments and tags must be set via ALTER TABLE after every write.
# saveAsTable(overwrite) replaces data but does not re-apply DDL metadata,
# so we stamp them explicitly here to ensure Genie, the Catalog UI, and
# any downstream AI tools always have accurate descriptions.
#
# TABLE-LEVEL TAGS — key/value labels used for governance and discovery:
#   domain    : which business domain owns this table
#   source    : where the data originated
#   pii       : whether the table contains personally identifiable information
#   layer     : medallion layer (bronze / silver / gold / reference)
#   refresh   : how often this table is updated

spark.sql(f"""
    ALTER TABLE {REF_TABLE}
    SET TAGS (
        'domain'  = 'retail',
        'source'  = 'toast_item_export',
        'pii'     = 'false',
        'layer'   = 'reference',
        'refresh' = 'manual_on_export'
    )
""")

spark.sql(f"""
    COMMENT ON TABLE {REF_TABLE} IS
    'Reference: Toast item catalog snapshot. Loaded from Toast Back-of-House CSV export.
Full replace on each import — always reflects the latest export, not historical state.
Join to silver_sales.order_items_silver on item_id = item_guid to enrich transactions
with current item metadata: category, sales category, brand, tax flags, alcohol status.
category_drift flag identifies items where Toast sales_category differs from menu category —
these should be corrected in Toast BOH to keep analytics consistent.'
""")

# ── Column comments ────────────────────────────────────────────────────────────
column_comments = {
    "item_id":                  "Toast item GUID. Primary join key to order_items_silver.item_guid.",
    "item_multi_location_id":   "Toast multi-location item ID. Used when syncing items across locations.",
    "name":                     "Item display name as shown in the Toast item library.",
    "pos_name":                 "Shorter name shown on the POS screen if different from the display name.",
    "brand":                    "Brand or producer name (e.g. [your store name], Kolsvart).",
    "description":              "Full item description from the Toast item library.",
    "item_type":                "Toast item type. DEFAULT for standard items, OPEN_ITEM for variable-price items.",
    "category_group":           "Top level of the menu/inventory hierarchy. Values: Grocery, Alcohol, Cafe Menu, Housewares & Decor, Snacks, Bags, Experiences, Breakfast, Thanksgiving.",
    "category":                 "Mid level of the menu/inventory hierarchy (e.g. Candy, Beer, Sandwiches). This is the primary reporting dimension used in Gold tables and Genie queries.",
    "subcategory":              "Optional third level of the menu hierarchy for finer item grouping.",
    "sales_category":           "Toast system reporting category (e.g. Cafe Food, Beer, Candy). Maintained separately from the menu hierarchy. Should always match category — see category_drift flag.",
    "applicable_taxes":         "Name of the tax applied to this item (e.g. MA State Sales Tax). Blank for non-taxable items such as most groceries and alcohol.",
    "tax_rates":                "Tax rate as a percentage string (e.g. 6.25%). Blank for non-taxable items.",
    "tax_inclusion":            "Whether tax is included in the displayed price. TAX_NOT_INCLUDED is standard for retail.",
    "contains_alcohol":         "True if this item is an alcoholic beverage (beer, wine, cider). Used for alcohol-specific reporting and compliance.",
    "takeout_delivery_tax_exempt": "Whether the item is exempt from tax on takeout and delivery orders.",
    "price":                    "Current retail selling price in USD. Null for open-price or variable items.",
    "cost":                     "Current unit cost in USD. Used for margin and COGS calculations.",
    "gross_margin":             "Gross margin percentage as a string from the Toast export (e.g. 54.69%). Cast to numeric in Gold if needed.",
    "supplier":                 "Primary supplier or vendor name for this item.",
    "supplier_item_id":         "Supplier SKU or item code. Useful for purchase order matching.",
    "barcode":                  "UPC or EAN barcode. Used for barcode scanner checkout.",
    "plu":                      "Price Look-Up code. Used for quick POS entry.",
    "size":                     "Item size description (e.g. 4.2oz, 750ml).",
    "package_unit":             "Unit of packaging as received from the supplier.",
    "receiving_units":          "How the item is ordered and received (e.g. Case, Each, Bottle).",
    "receiving_unit_quantities":"Quantity per receiving unit (e.g. 12 if a case contains 12 bottles).",
    "unit_of_measure":          "Unit of measure for inventory tracking (e.g. Each, oz, lb).",
    "inventory_status":         "Current inventory status from Toast: in stock, out of stock, on order.",
    "inventory_quantity":       "Current on-hand quantity at time of export. String — may be blank if not tracked.",
    "par_min":                  "Minimum par level for reorder triggering. Blank if not configured.",
    "par_max":                  "Maximum par level for reorder quantity. Blank if not configured.",
    "visibility":               "Where the item is available: POS, Kiosk, Toast Online Orders, or combinations.",
    "discount_eligible":        "Whether discounts can be applied to this item at the POS.",
    "rewards_eligible":         "Whether this item earns loyalty rewards points.",
    "is_discontinued":          "True if the item has been discontinued in Toast. Discontinued items are excluded from most reports but retained for historical transaction matching.",
    "discontinued_reason":      "Reason the item was discontinued, if provided.",
    "last_7_day_sales":         "Revenue in the 7 days prior to export. Point-in-time snapshot — not updated by the pipeline.",
    "last_30_day_sales":        "Revenue in the 30 days prior to export. Point-in-time snapshot.",
    "last_90_day_sales":        "Revenue in the 90 days prior to export. Point-in-time snapshot.",
    "created":                  "Timestamp when the item was created in Toast.",
    "last_updated":             "Timestamp when the item was last modified in Toast.",
    "category_drift":           "True when sales_category does not match category (case-insensitive). Indicates a discrepancy between the menu hierarchy and Toast reporting category. Should always be false — fix in Toast BOH: Items → Edit → Sales Category.",
    "_catalog_exported_at":     "Timestamp of the Toast export this snapshot was loaded from. Parsed from the CSV filename.",
    "_source_filename":         "Original CSV filename loaded into this table. Provides lineage back to the source export.",
}

for col_name, comment in column_comments.items():
    safe_comment = comment.replace("'", "\\'")
    spark.sql(f"""
        ALTER TABLE {REF_TABLE}
        ALTER COLUMN `{col_name}`
        COMMENT '{safe_comment}'
    """)

print(f"✓ Comments and tags applied to {REF_TABLE}")
print(f"  {len(column_comments)} column comments set")


# COMMAND ----------

# ── 7. VALIDATION ─────────────────────────────────────────────────────────────

print("\n── Items by category_group ──")
spark.sql(f"""
    SELECT
        category_group,
        COUNT(*)                                AS item_count,
        SUM(CAST(contains_alcohol AS INT))      AS alcohol_items,
        SUM(CAST(is_discontinued AS INT))       AS discontinued_items
    FROM {REF_TABLE}
    GROUP BY category_group
    ORDER BY item_count DESC
""").show(truncate=False)

print("\n── Items by category_group → category → sales_category ──")
spark.sql(f"""
    SELECT
        category_group,
        category,
        sales_category,
        COUNT(*)                                AS item_count,
        ROUND(AVG(price), 2)                    AS avg_price
    FROM {REF_TABLE}
    WHERE is_discontinued = false
    GROUP BY category_group, category, sales_category
    ORDER BY category_group, category, sales_category
""").show(100, truncate=False)

print("\n── Tax breakdown ──")
spark.sql(f"""
    SELECT
        CASE
            WHEN contains_alcohol = true            THEN 'Alcohol'
            WHEN applicable_taxes IS NOT NULL
             AND applicable_taxes != ''             THEN 'Taxable (non-alcohol)'
            ELSE 'Non-taxable'
        END                                         AS tax_class,
        category_group,
        COUNT(*)                                    AS item_count
    FROM {REF_TABLE}
    WHERE is_discontinued = false
    GROUP BY tax_class, category_group
    ORDER BY tax_class, category_group
""").show(truncate=False)

print("\n── Join coverage: items in catalog vs items seen in transactions ──")
spark.sql(f"""
    SELECT
        COUNT(DISTINCT i.item_guid)                 AS distinct_items_in_transactions,
        COUNT(DISTINCT c.item_id)                   AS distinct_items_in_catalog,
        COUNT(DISTINCT CASE WHEN c.item_id IS NOT NULL
            THEN i.item_guid END)                   AS items_matched,
        COUNT(DISTINCT CASE WHEN c.item_id IS NULL
            THEN i.item_guid END)                   AS items_unmatched
    FROM YOUR_CATALOG.silver_sales.order_items_silver i
    LEFT JOIN {REF_TABLE} c ON i.item_guid = c.item_id
""").show(truncate=False)

# COMMAND ----------

# ── 8. CATEGORY ALIGNMENT REPORT ──────────────────────────────────────────────
# Two separate issues, each needing a different action:
#
#   MISSING: sales_category is blank — item has no Toast reporting category at all.
#            Action: set it in Toast BOH to match the menu category.
#
#   MISMATCH: sales_category is set but doesn't match category.
#             Action: either fix in Toast BOH, or document a business rule exception
#             (e.g. Sandwiches → Cafe Food is intentional because of menu operations).
#
# Neither type causes pipeline failures — the Gold layer uses `category` as the
# primary reporting dimension. But MISSING items will appear as null in Toast's
# own in-app analytics, and MISMATCH items will aggregate differently there
# vs your Gold tables.

missing_df = spark.sql(f"""
    SELECT
        item_id                                     AS sku,
        name,
        category_group,
        category                                    AS menu_category,
        sales_category                              AS toast_sales_category,
        supplier
    FROM {REF_TABLE}
    WHERE is_discontinued = false
      AND (sales_category IS NULL OR TRIM(sales_category) = '')
    ORDER BY category_group, category, name
""")

mismatch_df = spark.sql(f"""
    SELECT
        item_id                                     AS sku,
        name,
        category_group,
        category                                    AS menu_category,
        sales_category                              AS toast_sales_category,
        supplier
    FROM {REF_TABLE}
    WHERE is_discontinued = false
      AND category_drift = true
    ORDER BY category_group, category, name
""")

missing_count  = missing_df.count()
mismatch_count = mismatch_df.count()

print()
print("=" * 70)
print("CATEGORY ALIGNMENT REPORT")
print("=" * 70)

# ── Missing sales_category ─────────────────────────────────────────────────
print(f"\n  MISSING sales_category:  {missing_count} active item(s)")
if missing_count == 0:
    print("  All active items have a sales_category set.")
else:
    print("  These items have no Toast sales_category — they will appear as")
    print("  null/blank in Toast in-app analytics.")
    print("  Fix: Toast BOH → Items → Edit → Sales Category → set to match menu category")
    print()
    missing_df.show(200, truncate=False)

# ── Mismatched sales_category ──────────────────────────────────────────────
print(f"\n  MISMATCHED sales_category: {mismatch_count} active item(s)")
if mismatch_count == 0:
    print("  All active items with a sales_category have it aligned to menu category.")
else:
    print("  These items have sales_category set but it differs from menu category.")
    print("  Review each one: fix in Toast BOH, or document a business rule exception.")
    print("  (e.g. if 'Sandwiches → Cafe Food' is intentional, note that here.)")
    print()
    mismatch_df.show(200, truncate=False)

print()
if missing_count == 0 and mismatch_count == 0:
    print("  RESULT: PASSED — all active items are aligned.")
else:
    total = missing_count + mismatch_count
    print(f"  RESULT: {total} item(s) need review (see above).")
print("=" * 70)
