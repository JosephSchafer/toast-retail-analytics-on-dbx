# Databricks notebook source
# MAGIC %md
# MAGIC # DQ — Google Sheet Export Queries
# MAGIC
# MAGIC Reference notebook for all SQL queries used to populate the 3SP Analytics
# MAGIC data quality Google Sheet:
# MAGIC **[3SP DQ Dashboard](https://docs.google.com/spreadsheets/d/1NvTFhmACP_DFiB4sig3xEk8MEZo3tPCebL_XU4lSPj4)**
# MAGIC
# MAGIC Each section corresponds to one tab in the sheet. Run the query directly
# MAGIC to preview or validate the data. To refresh the sheet, re-run the
# MAGIC Claude Code / MCP session that writes via `mcp__google-sheets__update_cells`.
# MAGIC
# MAGIC | Sheet Tab | Source Table | Rows (approx) |
# MAGIC |---|---|---|
# MAGIC | Par Levels | `dq.par_level_suggestions` | ~735 |
# MAGIC | Discontinued | `dq.suggested_discontinued` + candidate query | 146 flagged + ~234 candidates |
# MAGIC | Missing Barcode | `dq.missing_barcode` | varies |
# MAGIC | Missing Cost | `dq.missing_cost` | varies |
# MAGIC | Missing Supplier | `dq.missing_supplier` | varies |
# MAGIC | Inverted Margin | `dq.inverted_margin` | varies |
# MAGIC | Category Drift | `dq.category_drift` | varies |
# MAGIC | Nomenclature Fix | `dq.nomenclature_fix` | varies |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Par Levels
# MAGIC
# MAGIC Sheet tab: **Par Levels** — columns A:O (15 columns)
# MAGIC Written to rows 2 onward; row 1 is the header.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     name,
# MAGIC     barcode,
# MAGIC     plu,
# MAGIC     category_group,
# MAGIC     category,
# MAGIC     supplier,
# MAGIC     qty_on_hand,
# MAGIC     inventory_status,
# MAGIC     avg_daily_units_60d,
# MAGIC     selling_days_60d,
# MAGIC     last_sold_date,
# MAGIC     receiving_units,
# MAGIC     receiving_unit_quantities,
# MAGIC     par_min,
# MAGIC     par_max
# MAGIC FROM 3sp_analytics_workspace.dq.par_level_suggestions
# MAGIC ORDER BY category_group, category, name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discontinued — Already Flagged
# MAGIC
# MAGIC Sheet tab: **Discontinued**, rows 2–147 (146 rows as of last export).
# MAGIC All rows have `is_discontinued = true`.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     name,
# MAGIC     barcode,
# MAGIC     plu,
# MAGIC     category_group,
# MAGIC     category,
# MAGIC     supplier,
# MAGIC     qty_on_hand,
# MAGIC     price,
# MAGIC     last_sold_date,
# MAGIC     days_since_last_sale,
# MAGIC     avg_daily_units_60d,
# MAGIC     is_discontinued,
# MAGIC     discontinued_reason
# MAGIC FROM 3sp_analytics_workspace.dq.suggested_discontinued
# MAGIC ORDER BY category_group, category, name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discontinued — Candidates ("Next to Discontinue")
# MAGIC
# MAGIC Sheet tab: **Discontinued**, rows 148 onward.
# MAGIC Items not yet flagged but meeting stale-inventory thresholds:
# MAGIC - **Zero/negative stock** with no sales for **45+ days**
# MAGIC - **Positive stock** with no sales for **90+ days**
# MAGIC
# MAGIC Excludes `category = 'Fine Art'` (consignment/gallery items managed separately).
# MAGIC All rows have `is_discontinued = false`.

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH last_sales AS (
# MAGIC     SELECT
# MAGIC         item_guid,
# MAGIC         MAX(business_date) AS last_sold_date
# MAGIC     FROM 3sp_analytics_workspace.silver_sales.order_items_silver
# MAGIC     WHERE voided = false
# MAGIC     GROUP BY item_guid
# MAGIC ),
# MAGIC sales_60d AS (
# MAGIC     SELECT
# MAGIC         item_guid,
# MAGIC         ROUND(SUM(quantity) / 60.0, 3) AS avg_daily_units_60d
# MAGIC     FROM 3sp_analytics_workspace.silver_sales.order_items_silver
# MAGIC     WHERE voided = false
# MAGIC       AND business_date >= DATEADD(DAY, -60, CURRENT_DATE())
# MAGIC     GROUP BY item_guid
# MAGIC ),
# MAGIC candidates AS (
# MAGIC     SELECT
# MAGIC         ic.name,
# MAGIC         ic.barcode,
# MAGIC         ic.plu,
# MAGIC         ic.category_group,
# MAGIC         ic.category,
# MAGIC         ic.supplier,
# MAGIC         CAST(NULLIF(ic.inventory_quantity, '') AS DOUBLE) AS qty_on_hand,
# MAGIC         ic.price,
# MAGIC         ls.last_sold_date,
# MAGIC         DATEDIFF(CURRENT_DATE(), ls.last_sold_date)        AS days_since_last_sale,
# MAGIC         COALESCE(s60.avg_daily_units_60d, 0)               AS avg_daily_units_60d,
# MAGIC         CASE
# MAGIC             WHEN CAST(NULLIF(ic.inventory_quantity, '') AS DOUBLE) <= 0
# MAGIC               AND (ls.last_sold_date IS NULL
# MAGIC                    OR DATEDIFF(CURRENT_DATE(), ls.last_sold_date) >= 45)
# MAGIC               THEN 'Candidate: zero stock, no sales '
# MAGIC                    || COALESCE(CAST(DATEDIFF(CURRENT_DATE(), ls.last_sold_date) AS STRING), '90+')
# MAGIC                    || ' days'
# MAGIC             WHEN CAST(NULLIF(ic.inventory_quantity, '') AS DOUBLE) > 0
# MAGIC               AND (ls.last_sold_date IS NULL
# MAGIC                    OR DATEDIFF(CURRENT_DATE(), ls.last_sold_date) >= 90)
# MAGIC               THEN 'Candidate: in stock, no sales '
# MAGIC                    || COALESCE(CAST(DATEDIFF(CURRENT_DATE(), ls.last_sold_date) AS STRING), '90+')
# MAGIC                    || ' days'
# MAGIC         END AS discontinued_reason
# MAGIC     FROM 3sp_analytics_workspace.reference.item_catalog ic
# MAGIC     LEFT JOIN last_sales ls  ON ic.item_id = ls.item_guid
# MAGIC     LEFT JOIN sales_60d s60  ON ic.item_id = s60.item_guid
# MAGIC     WHERE ic.is_discontinued = false
# MAGIC       AND ic.category_group IN ('Grocery', 'Alcohol', 'Housewares & Decor')
# MAGIC       AND ic.category != 'Fine Art'
# MAGIC )
# MAGIC SELECT
# MAGIC     name,
# MAGIC     barcode,
# MAGIC     plu,
# MAGIC     category_group,
# MAGIC     category,
# MAGIC     supplier,
# MAGIC     qty_on_hand,
# MAGIC     price,
# MAGIC     last_sold_date,
# MAGIC     days_since_last_sale,
# MAGIC     avg_daily_units_60d,
# MAGIC     false AS is_discontinued,
# MAGIC     discontinued_reason
# MAGIC FROM candidates
# MAGIC WHERE discontinued_reason IS NOT NULL
# MAGIC ORDER BY category_group, category, name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Missing Barcode
# MAGIC
# MAGIC Sheet tab: **Missing Barcode** — columns A:L (12 columns).
# MAGIC Active items without a UPC/EAN barcode, with sales velocity context.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     name,
# MAGIC     plu,
# MAGIC     category_group,
# MAGIC     category,
# MAGIC     brand,
# MAGIC     supplier,
# MAGIC     supplier_item_id,
# MAGIC     price,
# MAGIC     last_sold_date,
# MAGIC     avg_daily_units_90d,
# MAGIC     est_weekly_units,
# MAGIC     barcode
# MAGIC FROM 3sp_analytics_workspace.dq.missing_barcode
# MAGIC ORDER BY category_group, category, name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Missing Cost
# MAGIC
# MAGIC Sheet tab: **Missing Cost** — columns A:K (11 columns).
# MAGIC Active items with no unit cost on file — blocks margin calculations.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     name,
# MAGIC     barcode,
# MAGIC     plu,
# MAGIC     category_group,
# MAGIC     category,
# MAGIC     supplier,
# MAGIC     price,
# MAGIC     last_sold_date,
# MAGIC     avg_daily_units_60d,
# MAGIC     revenue_60d,
# MAGIC     cost
# MAGIC FROM 3sp_analytics_workspace.dq.missing_cost
# MAGIC ORDER BY category_group, category, name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Missing Supplier
# MAGIC
# MAGIC Sheet tab: **Missing Supplier** — columns A:M (13 columns).
# MAGIC Active items with no supplier assigned — blocks reordering workflows.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     name,
# MAGIC     barcode,
# MAGIC     plu,
# MAGIC     category_group,
# MAGIC     category,
# MAGIC     brand,
# MAGIC     qty_on_hand,
# MAGIC     inventory_status,
# MAGIC     price,
# MAGIC     last_sold_date,
# MAGIC     avg_daily_units_60d,
# MAGIC     supplier,
# MAGIC     supplier_item_id
# MAGIC FROM 3sp_analytics_workspace.dq.missing_supplier
# MAGIC ORDER BY category_group, category, name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inverted Margin
# MAGIC
# MAGIC Sheet tab: **Inverted Margin** — columns A:O (15 columns).
# MAGIC Items where cost ≥ price (selling at a loss). Includes estimated daily loss
# MAGIC and suggested corrected values.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     name,
# MAGIC     barcode,
# MAGIC     plu,
# MAGIC     category_group,
# MAGIC     category,
# MAGIC     supplier,
# MAGIC     price,
# MAGIC     cost,
# MAGIC     gross_margin,
# MAGIC     calculated_margin_pct,
# MAGIC     est_daily_loss_usd,
# MAGIC     revenue_60d,
# MAGIC     cost_corrected,
# MAGIC     price_corrected
# MAGIC FROM 3sp_analytics_workspace.dq.inverted_margin
# MAGIC ORDER BY est_daily_loss_usd DESC, category_group, category, name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Category Drift
# MAGIC
# MAGIC Sheet tab: **Category Drift** — columns A:H (8 columns).
# MAGIC Items where the Toast sales reporting category doesn't match the menu
# MAGIC hierarchy category — causes miscategorized revenue in Gold tables.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     name,
# MAGIC     barcode,
# MAGIC     plu,
# MAGIC     category_group,
# MAGIC     correct_category,
# MAGIC     current_sales_category,
# MAGIC     price,
# MAGIC     supplier
# MAGIC FROM 3sp_analytics_workspace.dq.category_drift
# MAGIC ORDER BY category_group, correct_category, name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Nomenclature Fix
# MAGIC
# MAGIC Sheet tab: **Nomenclature Fix** — columns A:J (10 columns).
# MAGIC Items with inconsistent or non-standard display names. `auto_fix_candidate = true`
# MAGIC means the suggested name can be applied without manual review.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     current_name,
# MAGIC     barcode,
# MAGIC     plu,
# MAGIC     brand,
# MAGIC     category_group,
# MAGIC     category,
# MAGIC     price,
# MAGIC     suggested_name,
# MAGIC     auto_fix_candidate,
# MAGIC     name
# MAGIC FROM 3sp_analytics_workspace.dq.nomenclature_fix
# MAGIC ORDER BY category_group, category, current_name
