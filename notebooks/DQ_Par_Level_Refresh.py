# Databricks notebook source
# MAGIC %md
# MAGIC # DQ — Par Level & Inventory Refresh
# MAGIC
# MAGIC Rebuilds `dq.par_level_suggestions` daily with fresh inventory and sales data.
# MAGIC
# MAGIC ## What this does
# MAGIC - Computes estimated `qty_on_hand` = last known inventory from bronze − sales since that event
# MAGIC - Computes 60-day sales velocity and last-sold date from silver
# MAGIC - Applies configured par levels from `reference.item_catalog`; fills suggested values
# MAGIC   (ceil(7 × daily_vel) / ceil(14 × daily_vel)) for items without configured levels
# MAGIC - Feeds the "Move It — Aged & Stale Inventory" and "Don't Run Out — Stockout Risk" dashboards
# MAGIC
# MAGIC ## Changelog
# MAGIC | Version | Date       | Description |
# MAGIC |---------|------------|-------------|
# MAGIC | v1      | 2026-05-24 | Initial daily refresh. Previous table was a static CTAS from 2026-04-13; stale for 6 weeks. |
# MAGIC | v2      | 2026-05-24 | Add "not tracked" detection: items with blank catalog inventory_quantity and no RECEIVE events in 90 days get qty_on_hand=NULL, inventory_status="not tracked". Prevents absurd negative quantities for items like Kneady Mama bread that stopped inventory tracking. |
# MAGIC | v3      | 2026-06-07 | Add _refreshed_at timestamp column for pipeline freshness monitoring on dashboards. |

# COMMAND ----------

CATALOG = "YOUR_CATALOG"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rebuild dq.par_level_suggestions

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.dq.par_level_suggestions AS
WITH

-- Most recent inventory position per item from bronze
last_inventory AS (
    SELECT
        item_version_id,
        quantity_on_hand AS last_known_qty,
        created_date     AS last_inventory_date
    FROM (
        SELECT
            item_version_id, quantity_on_hand, created_date,
            ROW_NUMBER() OVER (PARTITION BY item_version_id ORDER BY created_date DESC) AS rn
        FROM {CATALOG}.bronze.toast_inventory_history_raw
    ) WHERE rn = 1
),

-- Most recent RECEIVE event per item — used to detect "not tracked" items.
-- An item is considered not tracked if Toast marks it as unquantified
-- (inventory_quantity blank in item_catalog) AND it has had no RECEIVE event
-- in the last 90 days. This catches items like Kneady Mama bread that were
-- briefly tracked then removed from inventory management without a final count.
last_receive AS (
    SELECT item_version_id, MAX(created_date) AS last_receive_date
    FROM {CATALOG}.bronze.toast_inventory_history_raw
    WHERE adjustment_type = 'RECEIVE'
    GROUP BY item_version_id
),

-- Units sold since the last inventory event (deplete from last known qty)
sales_since_inv AS (
    SELECT
        s.item_guid,
        SUM(s.quantity) AS units_sold_since_last_inv
    FROM {CATALOG}.silver_sales.order_items_silver s
    INNER JOIN last_inventory li ON s.item_guid = li.item_version_id
    WHERE s.voided = false
      AND s.quantity > 0
      AND s.business_date >= CAST(li.last_inventory_date AS DATE)
    GROUP BY s.item_guid
),

-- 60-day velocity and last sale date within window
sales_60d AS (
    SELECT
        item_guid,
        COUNT(DISTINCT business_date)  AS selling_days_60d,
        ROUND(SUM(quantity) / 60.0, 3) AS avg_daily_units_60d,
        MAX(business_date)             AS last_sold_date
    FROM {CATALOG}.silver_sales.order_items_silver
    WHERE voided = false AND quantity > 0
      AND business_date >= DATE_SUB(CURRENT_DATE(), 60)
    GROUP BY item_guid
),

-- All-time last sale for items whose last sale is outside the 60-day window
all_time_last_sale AS (
    SELECT item_guid, MAX(business_date) AS last_sold_date_alltime
    FROM {CATALOG}.silver_sales.order_items_silver
    WHERE voided = false AND quantity > 0
    GROUP BY item_guid
),

computed AS (
    SELECT
        ic.item_id,
        ic.name,
        ic.barcode,
        ic.plu,
        ic.category_group,
        ic.category,
        ic.supplier,
        -- "Not tracked" items: Toast marks these with a blank inventory_quantity
        -- AND they have had no RECEIVE event in 90 days. Set qty to NULL so the
        -- dashboard shows them as not tracked rather than wildly negative.
        CASE
            WHEN TRIM(COALESCE(ic.inventory_quantity, '')) = ''
             AND (lr.last_receive_date IS NULL
                  OR DATEDIFF(CURRENT_DATE(), CAST(lr.last_receive_date AS DATE)) > 90)
            THEN NULL
            ELSE COALESCE(
                li.last_known_qty - COALESCE(ss.units_sold_since_last_inv, 0),
                CAST(NULLIF(TRIM(ic.inventory_quantity), '') AS DOUBLE)
            )
        END AS qty_on_hand,
        COALESCE(s60.avg_daily_units_60d, 0.0) AS avg_daily_units_60d,
        COALESCE(s60.selling_days_60d, 0)      AS selling_days_60d,
        COALESCE(s60.last_sold_date, atls.last_sold_date_alltime) AS last_sold_date,
        ic.receiving_units,
        ic.receiving_unit_quantities,
        -- Use configured par levels if set; otherwise compute from velocity
        CASE
            WHEN TRIM(COALESCE(ic.par_min, '')) != '' THEN ic.par_min
            WHEN COALESCE(s60.avg_daily_units_60d, 0) > 0
                THEN CAST(CEIL(7  * s60.avg_daily_units_60d) AS STRING)
            ELSE ''
        END AS par_min,
        CASE
            WHEN TRIM(COALESCE(ic.par_max, '')) != '' THEN ic.par_max
            WHEN COALESCE(s60.avg_daily_units_60d, 0) > 0
                THEN CAST(CEIL(14 * s60.avg_daily_units_60d) AS STRING)
            ELSE ''
        END AS par_max
    FROM {CATALOG}.reference.item_catalog ic
    LEFT JOIN last_inventory li       ON ic.item_id = li.item_version_id
    LEFT JOIN last_receive lr         ON ic.item_id = lr.item_version_id
    LEFT JOIN sales_since_inv ss      ON ic.item_id = ss.item_guid
    LEFT JOIN sales_60d s60           ON ic.item_id = s60.item_guid
    LEFT JOIN all_time_last_sale atls ON ic.item_id = atls.item_guid
    WHERE ic.is_discontinued = false
)

SELECT
    name,
    barcode,
    plu,
    category_group,
    category,
    supplier,
    qty_on_hand,
    CASE
        WHEN qty_on_hand IS NULL THEN 'not tracked'
        WHEN qty_on_hand > 0    THEN 'in stock'
        ELSE 'out of stock'
    END AS inventory_status,
    avg_daily_units_60d,
    selling_days_60d,
    last_sold_date,
    receiving_units,
    receiving_unit_quantities,
    par_min,
    par_max,
    CURRENT_TIMESTAMP() AS _refreshed_at
FROM computed
""")

print("✓ dq.par_level_suggestions rebuilt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation Summary

# COMMAND ----------

summary = spark.sql(f"""
SELECT
    COUNT(*)                                                            AS total_items,
    SUM(CASE WHEN inventory_status = 'out of stock' THEN 1 ELSE 0 END) AS out_of_stock,
    SUM(CASE WHEN inventory_status = 'in stock'     THEN 1 ELSE 0 END) AS in_stock,
    SUM(CASE WHEN inventory_status = 'not tracked'  THEN 1 ELSE 0 END) AS not_tracked,
    SUM(CASE WHEN par_min != ''                     THEN 1 ELSE 0 END) AS items_with_par,
    SUM(CASE WHEN avg_daily_units_60d > 0           THEN 1 ELSE 0 END) AS items_with_velocity,
    MAX(last_sold_date)                                                 AS latest_sale_date,
    MAX(_refreshed_at)                                                  AS refreshed_at
FROM {CATALOG}.dq.par_level_suggestions
""").collect()[0]

print(f"Total active items  : {summary['total_items']}")
print(f"  In stock          : {summary['in_stock']}")
print(f"  Out of stock      : {summary['out_of_stock']}")
print(f"  Not tracked       : {summary['not_tracked']}")
print(f"Items with par set  : {summary['items_with_par']}")
print(f"Items with velocity : {summary['items_with_velocity']}")
print(f"Latest sale date    : {summary['latest_sale_date']}")
print(f"Table refreshed at  : {summary['refreshed_at']}")

assert summary['total_items'] > 0, "FAIL: par_level_suggestions is empty"
assert summary['latest_sale_date'] is not None, "FAIL: no sales data found"
assert str(summary['latest_sale_date']) >= str(__import__('datetime').date.today() - __import__('datetime').timedelta(days=3)), \
    f"FAIL: latest sale date {summary['latest_sale_date']} is more than 3 days old — silver may not have refreshed"

print("\n✓ All checks passed")
