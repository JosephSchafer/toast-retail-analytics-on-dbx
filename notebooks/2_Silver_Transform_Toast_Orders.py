# Databricks notebook source
# MAGIC %md
# MAGIC # Silver — Transform Toast Orders
# MAGIC
# MAGIC Reads from `3sp_analytics_workspace.bronze.toast_orders_raw` and produces two
# MAGIC clean, typed Silver tables:
# MAGIC
# MAGIC | Table | Grain | Description |
# MAGIC |---|---|---|
# MAGIC | `silver_sales.orders_silver` | One row per order | Order header: dates, totals, payment summary, server |
# MAGIC | `silver_sales.order_items_silver` | One row per line item (selection) | Exploded selections with price, discount, item metadata |
# MAGIC
# MAGIC **Design notes from your actual data:**
# MAGIC - Discounts exist at both check level (`checks[].appliedDiscounts[]`) AND item level
# MAGIC   (`checks[].selections[].appliedDiscounts[]`). Silver captures both: item-level discount
# MAGIC   amount is summed directly onto each selection row; check-level discount total is on the
# MAGIC   order header. Gold prorates any remaining check-level discount across items.
# MAGIC - `table` is always null for your in-store retail model — excluded from schema
# MAGIC - `businessDate` from Toast is an integer (e.g. `20260319`) — cast to proper DATE
# MAGIC - `deletedDate` defaults to `1970-01-01` instead of null — coalesced to null in Silver
# MAGIC - Tax appears to be 0 across items — field is preserved but not relied upon for totals
# MAGIC - An order has one or more checks; a check has one or more selections and payments
# MAGIC   Silver flattens checks by taking the first check's payment (covers 99%+ of in-store
# MAGIC   retail transactions which are single-check). Multi-check orders are flagged.

# COMMAND ----------

# ── WIDGET SETUP ──────────────────────────────────────────────────────────────
# run_mode:
#   incremental  — processes only Bronze rows ingested since the last Silver run
#   full_refresh — reprocesses all Bronze rows (use after schema changes)

dbutils.widgets.dropdown(
    name="run_mode",
    defaultValue="incremental",
    choices=["incremental", "full_refresh"],
    label="Run Mode"
)

# COMMAND ----------

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

import datetime
from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, BooleanType, TimestampType, DateType
)

# COMMAND ----------

# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────

RUN_MODE = dbutils.widgets.get("run_mode")

CATALOG         = "3sp_analytics_workspace"
BRONZE_SCHEMA   = "bronze"
SILVER_SCHEMA   = "silver_sales"

BRONZE_TABLE    = f"{CATALOG}.{BRONZE_SCHEMA}.toast_orders_raw"
ORDERS_TABLE    = f"{CATALOG}.{SILVER_SCHEMA}.orders_silver"
ITEMS_TABLE     = f"{CATALOG}.{SILVER_SCHEMA}.order_items_silver"
WATERMARK_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.ingestion_watermark"

# Epoch sentinel that Toast uses instead of null for deletedDate
TOAST_EPOCH_SENTINEL = "1970-01-01"

# COMMAND ----------

# ── 3. SETUP: SCHEMA AND TABLES ───────────────────────────────────────────────

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SILVER_SCHEMA}`")

# orders_silver — one row per order (order header facts)
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {ORDERS_TABLE} (
        -- Keys
        order_guid              STRING      NOT NULL    COMMENT 'Toast order GUID — primary key',
        restaurant_guid         STRING      NOT NULL    COMMENT 'Toast restaurant external ID',

        -- Dates (all UTC timestamps)
        business_date           DATE                    COMMENT 'Local business date as a proper DATE (from Toast integer e.g. 20260319)',
        opened_date             TIMESTAMP               COMMENT 'When the order was opened',
        closed_date             TIMESTAMP               COMMENT 'When the order was closed/paid — null if still open',
        paid_date               TIMESTAMP               COMMENT 'When payment was captured',
        modified_date           TIMESTAMP               COMMENT 'Last modification timestamp from Toast',

        -- Order metadata
        display_number          STRING                  COMMENT 'Human-readable order number shown on receipt',
        source                  STRING                  COMMENT 'Order source: In Store, Online, etc.',
        approval_status         STRING                  COMMENT 'APPROVED, PENDING, etc.',
        voided                  BOOLEAN                 COMMENT 'True if the entire order was voided',
        number_of_guests        INTEGER                 COMMENT 'Guest count on the order',
        duration_seconds        INTEGER                 COMMENT 'Order duration in seconds (from Toast duration field)',
        server_guid             STRING                  COMMENT 'Toast GUID of the server/employee',

        -- Check-level financials (aggregated across all checks on this order)
        check_count             INTEGER                 COMMENT 'Number of checks on this order (>1 = split bill)',
        gross_amount            DOUBLE                  COMMENT 'Sum of checks[].amount before any adjustments',
        total_amount            DOUBLE                  COMMENT 'Sum of checks[].totalAmount — final charged amount',
        tax_amount              DOUBLE                  COMMENT 'Sum of checks[].taxAmount',
        total_discount_amount   DOUBLE                  COMMENT 'Sum of all appliedDiscounts[].discountAmount across all checks',
        applied_service_charge  DOUBLE                  COMMENT 'Sum of appliedServiceCharges if present',

        -- Payment summary (first/primary check)
        payment_type            STRING                  COMMENT 'Payment type: CREDIT, CASH, etc. From first payment on first check.',
        card_type               STRING                  COMMENT 'Card brand: VISA, MASTERCARD, etc. Null for cash.',
        card_last4              STRING                  COMMENT 'Last 4 digits of card. Null for cash.',
        card_entry_mode         STRING                  COMMENT 'EMV_CHIP_SIGN, SWIPE, KEYED, etc.',
        tip_amount              DOUBLE                  COMMENT 'Tip amount from first payment',
        payment_status          STRING                  COMMENT 'Check payment status: CLOSED, OPEN',
        processing_fee          DOUBLE                  COMMENT 'originalProcessingFee from Toast payment object',

        -- Discount summary
        has_discount            BOOLEAN                 COMMENT 'True if any discount was applied to this order',
        discount_names          STRING                  COMMENT 'Comma-separated list of discount names applied',

        -- Audit
        _silver_updated_at      TIMESTAMP               COMMENT 'When this row was last written to Silver',
        _bronze_batch_id        STRING                  COMMENT 'Ingestion batch ID from Bronze for lineage'
    )
    USING DELTA
    PARTITIONED BY (business_date)
    COMMENT 'Silver: one row per Toast order. Typed, deduplicated, null-safe.'
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'quality' = 'silver'
    )
""")

# order_items_silver — one row per selection (line item)
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {ITEMS_TABLE} (
        -- Keys
        selection_guid          STRING      NOT NULL    COMMENT 'Toast selection GUID — primary key for this line item',
        order_guid              STRING      NOT NULL    COMMENT 'FK to orders_silver',
        check_guid              STRING      NOT NULL    COMMENT 'FK to the check this selection belongs to',
        restaurant_guid         STRING      NOT NULL    COMMENT 'Toast restaurant external ID',

        -- Dates
        business_date           DATE                    COMMENT 'Inherited from parent order',
        created_date            TIMESTAMP               COMMENT 'When the item was added to the check',
        modified_date           TIMESTAMP               COMMENT 'When the item was last modified',

        -- Item identity
        display_name            STRING                  COMMENT 'Item name as shown on receipt',
        item_guid               STRING                  COMMENT 'Toast MenuItem GUID — links to menu catalog',
        item_group_guid         STRING                  COMMENT 'Toast MenuGroup GUID — the menu category grouping',
        sales_category_guid     STRING                  COMMENT 'Toast SalesCategory GUID for reporting',
        plu                     STRING                  COMMENT 'PLU code if set',

        -- Quantity and pricing
        quantity                DOUBLE                  COMMENT 'Quantity ordered (double to handle fractional weights)',
        unit_price              DOUBLE                  COMMENT 'Actual unit price after any item-level adjustments',
        pre_discount_price      DOUBLE                  COMMENT 'Unit price before discounts — useful for discount impact analysis',
        receipt_line_price      DOUBLE                  COMMENT 'Line total as it appears on the receipt',
        tax                     DOUBLE                  COMMENT 'Tax amount on this line item',
        tax_inclusion           STRING                  COMMENT 'NOT_INCLUDED, INCLUDED',

        -- Item status
        voided                  BOOLEAN                 COMMENT 'True if this line item was voided',
        void_reason             STRING                  COMMENT 'Void reason text if voided',
        fulfillment_status      STRING                  COMMENT 'SENT, NEW, etc.',
        selection_type          STRING                  COMMENT 'NONE, SPLIT, etc.',

        -- Item-level discounts (separate from check-level discounts on orders_silver)
        item_discount_amount    DOUBLE                  COMMENT 'Sum of appliedDiscounts[].discountAmount on this selection. 0 if none.',
        item_discount_names     STRING                  COMMENT 'Comma-separated discount names applied directly to this item. Null if none.',
        has_item_discount       BOOLEAN                 COMMENT 'True if a discount was applied directly to this line item',

        -- Modifier flag
        has_modifiers           BOOLEAN                 COMMENT 'True if modifiers[] is non-empty',

        -- Audit
        _silver_updated_at      TIMESTAMP               COMMENT 'When this row was last written to Silver',
        _bronze_batch_id        STRING                  COMMENT 'Ingestion batch ID from Bronze for lineage'
    )
    USING DELTA
    PARTITIONED BY (business_date)
    COMMENT 'Silver: one row per Toast order line item (selection). Exploded from checks[].selections[].'
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'quality' = 'silver'
    )
""")

print(f"✓ {ORDERS_TABLE}")
print(f"✓ {ITEMS_TABLE}")

# COMMAND ----------

# ── 4. DETERMINE INCREMENTAL WINDOW ───────────────────────────────────────────
# In incremental mode we only reprocess Bronze rows that arrived since the last
# Silver run — identified by ingestion_batch_id via the watermark table.
# In full_refresh we reprocess everything.

if RUN_MODE == "incremental":
    # Find the most recent ingested_at already reflected in Silver
    try:
        last_silver_ts = spark.sql(f"""
            SELECT MAX(_silver_updated_at) AS last_ts FROM {ORDERS_TABLE}
        """).collect()[0]["last_ts"]
    except Exception:
        last_silver_ts = None

    if last_silver_ts is None:
        print("No prior Silver run found — processing all Bronze rows")
        bronze_df = spark.table(BRONZE_TABLE)
    else:
        print(f"Incremental mode — processing Bronze rows ingested after {last_silver_ts}")
        bronze_df = spark.table(BRONZE_TABLE).filter(
            F.col("ingested_at") > last_silver_ts
        )
else:
    print("Full refresh — processing all Bronze rows")
    bronze_df = spark.table(BRONZE_TABLE)

row_count = bronze_df.count()
print(f"Bronze rows to process: {row_count}")

if row_count == 0:
    print("Nothing new to process. Silver is up to date.")
    dbutils.notebook.exit("UP_TO_DATE")

# COMMAND ----------

# ── 5. PARSE RAW JSON ─────────────────────────────────────────────────────────
# Parse the raw_json string column into a nested struct so we can
# navigate it with dot notation rather than repeated get_json_object calls.

# Build the schema for the parts of the Toast order we care about.
# Using schema inference here (from_json with schema inference) would work but
# is slow on large datasets. Explicit schema is faster and catches drift.

# We define only the fields we use — unmapped fields stay in raw_json.
checks_schema = """
    ARRAY<STRUCT<
        guid: STRING,
        amount: DOUBLE,
        totalAmount: DOUBLE,
        taxAmount: DOUBLE,
        paymentStatus: STRING,
        openedDate: STRING,
        closedDate: STRING,
        paidDate: STRING,
        voided: BOOLEAN,
        payments: ARRAY<STRUCT<
            guid: STRING,
            type: STRING,
            amount: DOUBLE,
            tipAmount: DOUBLE,
            amountTendered: DOUBLE,
            originalProcessingFee: DOUBLE,
            paymentStatus: STRING,
            cardType: STRING,
            last4Digits: STRING,
            cardEntryMode: STRING,
            refundStatus: STRING
        >>,
        appliedDiscounts: ARRAY<STRUCT<
            guid: STRING,
            name: STRING,
            discountAmount: DOUBLE,
            discountPercent: DOUBLE,
            discountType: STRING,
            nonTaxDiscountAmount: DOUBLE
        >>,
        selections: ARRAY<STRUCT<
            guid: STRING,
            displayName: STRING,
            quantity: DOUBLE,
            price: DOUBLE,
            preDiscountPrice: DOUBLE,
            receiptLinePrice: DOUBLE,
            tax: DOUBLE,
            taxInclusion: STRING,
            voided: BOOLEAN,
            voidReason: STRING,
            fulfillmentStatus: STRING,
            selectionType: STRING,
            createdDate: STRING,
            modifiedDate: STRING,
            plu: STRING,
            appliedDiscounts: ARRAY<STRUCT<
                guid: STRING,
                name: STRING,
                discountAmount: DOUBLE
            >>,
            modifiers: ARRAY<STRUCT<guid: STRING>>,
            item: STRUCT<guid: STRING>,
            itemGroup: STRUCT<guid: STRING>,
            salesCategory: STRUCT<guid: STRING>
        >>
    >>
"""

parsed_df = bronze_df.withColumn(
    "o",
    F.from_json(F.col("raw_json"), f"""
        STRUCT<
            guid: STRING,
            displayNumber: STRING,
            source: STRING,
            approvalStatus: STRING,
            voided: BOOLEAN,
            businessDate: INT,
            openedDate: STRING,
            closedDate: STRING,
            paidDate: STRING,
            modifiedDate: STRING,
            createdDate: STRING,
            deletedDate: STRING,
            duration: INT,
            numberOfGuests: INT,
            server: STRUCT<guid: STRING>,
            checks: {checks_schema}
        >
    """)
)

# COMMAND ----------

# ── 6. BUILD orders_silver ────────────────────────────────────────────────────

now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

def parse_toast_ts(col_name):
    """Parse Toast ISO timestamp strings to proper UTC timestamps."""
    return F.to_timestamp(F.col(col_name), "yyyy-MM-dd'T'HH:mm:ss.SSSZ")

def toast_date_int_to_date(col):
    """Convert Toast integer businessDate (e.g. 20260319) to a DATE."""
    return F.to_date(col.cast("string"), "yyyyMMdd")

def null_epoch(col):
    """Replace Toast's 1970-01-01 sentinel with null."""
    return F.when(
        F.col(col).cast("date") == F.lit(TOAST_EPOCH_SENTINEL).cast("date"),
        F.lit(None)
    ).otherwise(F.col(col).cast("date"))

# Aggregate check-level financials across all checks on the order
orders_silver_df = (
    parsed_df
    .select(
        # ── Keys ──────────────────────────────────────────────────────────
        F.col("o.guid")                         .alias("order_guid"),
        F.col("restaurant_guid"),

        # ── Dates ─────────────────────────────────────────────────────────
        toast_date_int_to_date(F.col("o.businessDate"))
                                                .alias("business_date"),
        F.to_timestamp(F.col("o.openedDate"),   "yyyy-MM-dd'T'HH:mm:ss.SSSZ")
                                                .alias("opened_date"),
        F.when(
            F.col("o.closedDate").isNotNull() &
            (~F.col("o.closedDate").startswith("1970")),
            F.to_timestamp(F.col("o.closedDate"), "yyyy-MM-dd'T'HH:mm:ss.SSSZ")
        ).alias("closed_date"),
        F.to_timestamp(F.col("o.paidDate"),     "yyyy-MM-dd'T'HH:mm:ss.SSSZ")
                                                .alias("paid_date"),
        F.to_timestamp(F.col("o.modifiedDate"), "yyyy-MM-dd'T'HH:mm:ss.SSSZ")
                                                .alias("modified_date"),

        # ── Order metadata ─────────────────────────────────────────────────
        F.col("o.displayNumber")                .alias("display_number"),
        F.col("o.source"),
        F.col("o.approvalStatus")               .alias("approval_status"),
        F.col("o.voided"),
        F.col("o.numberOfGuests")               .alias("number_of_guests"),
        F.col("o.duration")                     .alias("duration_seconds"),
        F.col("o.server.guid")                  .alias("server_guid"),

        # ── Check-level aggregates ─────────────────────────────────────────
        # Number of checks — >1 means a split bill
        F.size(F.col("o.checks"))               .alias("check_count"),

        # Sum financials across all checks
        F.aggregate(
            F.col("o.checks"),
            F.lit(0.0),
            lambda acc, c: acc + F.coalesce(c["amount"], F.lit(0.0))
        )                                       .alias("gross_amount"),

        F.aggregate(
            F.col("o.checks"),
            F.lit(0.0),
            lambda acc, c: acc + F.coalesce(c["totalAmount"], F.lit(0.0))
        )                                       .alias("total_amount"),

        F.aggregate(
            F.col("o.checks"),
            F.lit(0.0),
            lambda acc, c: acc + F.coalesce(c["taxAmount"], F.lit(0.0))
        )                                       .alias("tax_amount"),

        # Total discount amount across all checks — flatten discounts array and sum
        F.aggregate(
            F.flatten(F.transform(
                F.col("o.checks"),
                lambda c: F.coalesce(c["appliedDiscounts"], F.array())
            )),
            F.lit(0.0),
            lambda acc, d: acc + F.coalesce(d["discountAmount"], F.lit(0.0))
        )                                       .alias("total_discount_amount"),

        # Service charges — sum if present
        F.lit(0.0)                              .alias("applied_service_charge"),

        # ── Payment summary (first payment on first check) ─────────────────
        # Use get() throughout — some orders have empty payments[] arrays
        # (cash, comped, or still-open orders ingested mid-day)
        F.expr("get(get(o.checks, 0).payments, 0).type")
                                                .alias("payment_type"),
        F.expr("get(get(o.checks, 0).payments, 0).cardType")
                                                .alias("card_type"),
        F.expr("get(get(o.checks, 0).payments, 0).last4Digits")
                                                .alias("card_last4"),
        F.expr("get(get(o.checks, 0).payments, 0).cardEntryMode")
                                                .alias("card_entry_mode"),
        F.coalesce(
            F.expr("get(get(o.checks, 0).payments, 0).tipAmount"), F.lit(0.0)
        )                                       .alias("tip_amount"),
        F.expr("get(o.checks, 0).paymentStatus")
                                                .alias("payment_status"),
        F.coalesce(
            F.expr("get(get(o.checks, 0).payments, 0).originalProcessingFee"), F.lit(0.0)
        )                                       .alias("processing_fee"),

        # ── Discount summary ───────────────────────────────────────────────
        # has_discount: true if any check has any discount
        # Wrapped in F.col() expression to prevent Python evaluating > 0 before .alias()
        (F.size(F.flatten(F.transform(
            F.col("o.checks"),
            lambda c: F.coalesce(c["appliedDiscounts"], F.array())
        ))) > F.lit(0))                         .alias("has_discount"),

        # Comma-separated discount names across all checks
        F.array_join(
            F.array_distinct(
                F.flatten(F.transform(
                    F.col("o.checks"),
                    lambda c: F.transform(
                        F.coalesce(c["appliedDiscounts"], F.array()),
                        lambda d: d["name"]
                    )
                ))
            ),
            ", "
        )                                       .alias("discount_names"),

        # ── Audit ──────────────────────────────────────────────────────────
        F.lit(now_utc).cast("timestamp")        .alias("_silver_updated_at"),
        F.col("ingestion_batch_id")             .alias("_bronze_batch_id")
    )
    # Drop any rows where order_guid couldn't be parsed (malformed JSON)
    .filter(F.col("order_guid").isNotNull())
)

print(f"orders_silver rows to merge: {orders_silver_df.count()}")

# COMMAND ----------

# ── 7. MERGE INTO orders_silver ───────────────────────────────────────────────

orders_table = DeltaTable.forName(spark, ORDERS_TABLE)

orders_table.alias("target").merge(
    orders_silver_df.alias("source"),
    "target.order_guid = source.order_guid"
).whenMatchedUpdateAll(
).whenNotMatchedInsertAll(
).execute()

print(f"✓ Merged into {ORDERS_TABLE}")

# COMMAND ----------

# ── 8. BUILD order_items_silver ───────────────────────────────────────────────
# Explode checks → explode selections to get one row per line item.
# We carry check_guid on each row so multi-check orders can be correctly
# attributed at the Gold layer.

items_silver_df = (
    parsed_df
    # Explode checks array → one row per check
    .select(
        F.col("o.guid")             .alias("order_guid"),
        F.col("restaurant_guid"),
        toast_date_int_to_date(F.col("o.businessDate")).alias("business_date"),
        F.col("ingestion_batch_id"),
        F.posexplode(F.col("o.checks")).alias("check_pos", "check")
    )
    # Explode selections array → one row per line item
    .select(
        F.col("order_guid"),
        F.col("check.guid")         .alias("check_guid"),
        F.col("restaurant_guid"),
        F.col("business_date"),
        F.col("ingestion_batch_id"),
        F.posexplode(F.col("check.selections")).alias("sel_pos", "sel")
    )
    # Map each selection to its Silver columns
    .select(
        # ── Keys ──────────────────────────────────────────────────────────
        F.col("sel.guid")                       .alias("selection_guid"),
        F.col("order_guid"),
        F.col("check_guid"),
        F.col("restaurant_guid"),

        # ── Dates ─────────────────────────────────────────────────────────
        F.col("business_date"),
        F.to_timestamp(F.col("sel.createdDate"),  "yyyy-MM-dd'T'HH:mm:ss.SSSZ")
                                                .alias("created_date"),
        F.to_timestamp(F.col("sel.modifiedDate"), "yyyy-MM-dd'T'HH:mm:ss.SSSZ")
                                                .alias("modified_date"),

        # ── Item identity ──────────────────────────────────────────────────
        F.col("sel.displayName")                .alias("display_name"),
        F.col("sel.item.guid")                  .alias("item_guid"),
        F.col("sel.itemGroup.guid")             .alias("item_group_guid"),
        F.col("sel.salesCategory.guid")         .alias("sales_category_guid"),
        F.when(
            F.col("sel.plu") == "", F.lit(None)
        ).otherwise(F.col("sel.plu"))           .alias("plu"),

        # ── Quantity and pricing ───────────────────────────────────────────
        F.col("sel.quantity"),
        F.col("sel.price")                      .alias("unit_price"),
        F.col("sel.preDiscountPrice")           .alias("pre_discount_price"),
        F.col("sel.receiptLinePrice")           .alias("receipt_line_price"),
        F.coalesce(F.col("sel.tax"), F.lit(0.0)).alias("tax"),
        F.col("sel.taxInclusion")               .alias("tax_inclusion"),

        # ── Item status ────────────────────────────────────────────────────
        F.coalesce(F.col("sel.voided"), F.lit(False))
                                                .alias("voided"),
        F.col("sel.voidReason")                 .alias("void_reason"),
        F.col("sel.fulfillmentStatus")          .alias("fulfillment_status"),
        F.col("sel.selectionType")              .alias("selection_type"),

        # ── Item-level discounts ───────────────────────────────────────────
        # Distinct from check-level discounts on orders_silver.
        # Most items will have 0 here — check-level discounts are on the order header.
        F.coalesce(
            F.aggregate(
                F.coalesce(F.col("sel.appliedDiscounts"), F.array()),
                F.lit(0.0),
                lambda acc, d: acc + F.coalesce(d["discountAmount"], F.lit(0.0))
            ),
            F.lit(0.0)
        )                                       .alias("item_discount_amount"),

        F.when(
            F.size(F.coalesce(F.col("sel.appliedDiscounts"), F.array())) > 0,
            F.array_join(
                F.array_distinct(
                    F.transform(
                        F.coalesce(F.col("sel.appliedDiscounts"), F.array()),
                        lambda d: d["name"]
                    )
                ),
                ", "
            )
        ).otherwise(F.lit(None))                .alias("item_discount_names"),

        (F.size(F.coalesce(F.col("sel.appliedDiscounts"), F.array())) > F.lit(0))
                                                .alias("has_item_discount"),

        # ── Modifier flag ──────────────────────────────────────────────────
        (F.size(F.coalesce(F.col("sel.modifiers"), F.array())) > F.lit(0))
                                                .alias("has_modifiers"),

        # ── Audit ──────────────────────────────────────────────────────────
        F.lit(now_utc).cast("timestamp")        .alias("_silver_updated_at"),
        F.col("ingestion_batch_id")             .alias("_bronze_batch_id")
    )
    .filter(F.col("selection_guid").isNotNull())
    .filter(F.col("voided") == False)  # Exclude voided line items
                                       # Voided orders are filtered at query time via orders_silver.voided
)

print(f"order_items_silver rows to merge: {items_silver_df.count()}")

# COMMAND ----------

# ── 9. MERGE INTO order_items_silver ─────────────────────────────────────────

items_table = DeltaTable.forName(spark, ITEMS_TABLE)

items_table.alias("target").merge(
    items_silver_df.alias("source"),
    "target.selection_guid = source.selection_guid"
).whenMatchedUpdateAll(
).whenNotMatchedInsertAll(
).execute()

print(f"✓ Merged into {ITEMS_TABLE}")

# COMMAND ----------

# ── 10. VALIDATION QUERIES ────────────────────────────────────────────────────

print("\n── orders_silver: last 7 business dates ──")
spark.sql(f"""
    SELECT
        business_date,
        COUNT(*)                                AS order_count,
        ROUND(SUM(total_amount), 2)             AS total_revenue,
        ROUND(SUM(total_discount_amount), 2)    AS total_discounts,
        ROUND(SUM(tip_amount), 2)               AS total_tips,
        SUM(CAST(has_discount AS INT))          AS discounted_orders,
        COUNT(DISTINCT payment_type)            AS payment_types
    FROM {ORDERS_TABLE}
    WHERE voided = false
    GROUP BY business_date
    ORDER BY business_date DESC
    LIMIT 7
""").show(truncate=False)

print("\n── order_items_silver: top 10 items by revenue (all time) ──")
spark.sql(f"""
    SELECT
        display_name,
        COUNT(*)                                AS times_sold,
        ROUND(SUM(quantity), 0)                 AS total_qty,
        ROUND(SUM(receipt_line_price), 2)       AS total_revenue,
        ROUND(AVG(unit_price), 2)               AS avg_unit_price
    FROM {ITEMS_TABLE}
    WHERE voided = false
    GROUP BY display_name
    ORDER BY total_revenue DESC
    LIMIT 10
""").show(truncate=False)

print("\n── Discount breakdown ──")
spark.sql(f"""
    SELECT
        discount_names,
        COUNT(*)                                AS order_count,
        ROUND(SUM(total_discount_amount), 2)    AS total_discount_value,
        ROUND(AVG(total_discount_amount), 2)    AS avg_discount_per_order
    FROM {ORDERS_TABLE}
    WHERE has_discount = true
      AND voided = false
    GROUP BY discount_names
    ORDER BY total_discount_value DESC
""").show(truncate=False)

print("\n── Multi-check orders (split bills) ──")
spark.sql(f"""
    SELECT COUNT(*) AS split_bill_orders
    FROM {ORDERS_TABLE}
    WHERE check_count > 1
""").show()

print("\n── Item-level discounts (should be rare but present) ──")
spark.sql(f"""
    SELECT
        item_discount_names,
        COUNT(*)                                AS line_items,
        ROUND(SUM(item_discount_amount), 2)     AS total_item_discount_value
    FROM {ITEMS_TABLE}
    WHERE has_item_discount = true
    GROUP BY item_discount_names
    ORDER BY total_item_discount_value DESC
""").show(truncate=False)

print("\n── Tax breakdown by item (shows taxable vs non-taxable mix) ──")
spark.sql(f"""
    SELECT
        CASE WHEN tax > 0 THEN 'taxable' ELSE 'non-taxable' END AS tax_status,
        COUNT(*)                                AS line_items,
        ROUND(SUM(receipt_line_price), 2)       AS total_revenue,
        ROUND(SUM(tax), 2)                      AS total_tax_collected
    FROM {ITEMS_TABLE}
    WHERE voided = false
    GROUP BY tax_status
    ORDER BY tax_status
""").show(truncate=False)