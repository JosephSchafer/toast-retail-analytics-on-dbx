# Databricks notebook source
# MAGIC %md
# MAGIC # Platinum — Daily Sales Combined View
# MAGIC
# MAGIC Creates `platinum.daily_sales_combined` — a view that merges actuals and
# MAGIC the latest forecast into a single, duplication-free series for dashboards.
# MAGIC
# MAGIC ## Why this exists
# MAGIC
# MAGIC Actuals (`gold.daily_sales_summary`) and forecasts (`gold.daily_sales_forecast`)
# MAGIC are kept in separate tables to avoid double-counting. This view is the safe
# MAGIC interface for anything that needs both — Genie queries, dashboards, WoW comparisons.
# MAGIC
# MAGIC ## What it contains
# MAGIC
# MAGIC | Date range | Source | record_type |
# MAGIC |---|---|---|
# MAGIC | Any date with actual revenue > 0 | `gold.daily_sales_summary` | `actual` |
# MAGIC | Closed days (zero revenue) | `gold.daily_sales_summary` | `actual` |
# MAGIC | Future dates (after last actual) | `gold.daily_sales_forecast` (latest run only) | `forecast` |
# MAGIC
# MAGIC **Key guarantees:**
# MAGIC - No date appears twice — actual takes precedence over any forecast for the same date
# MAGIC - Only the most recent forecast run is included (MAX `forecast_created_at`)
# MAGIC - Past forecast rows (for dates that now have actuals) are excluded automatically
# MAGIC
# MAGIC ## Run this notebook when
# MAGIC - First time setup (creates the schema and view)
# MAGIC - After schema changes to either Gold table (re-creates the view)
# MAGIC
# MAGIC The view itself is always live — it does not need to be re-run daily.

# COMMAND ----------

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

from pyspark.sql import functions as F

# COMMAND ----------

# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────

CATALOG          = "3sp_analytics_workspace"
ACTUALS_TABLE    = f"{CATALOG}.gold.daily_sales_summary"
FORECAST_TABLE   = f"{CATALOG}.gold.daily_sales_forecast"
PLATINUM_SCHEMA  = f"{CATALOG}.platinum"
COMBINED_VIEW    = f"{PLATINUM_SCHEMA}.daily_sales_combined"

# COMMAND ----------

# ── 3. CREATE PLATINUM SCHEMA ─────────────────────────────────────────────────

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {PLATINUM_SCHEMA}")
print(f"✓ Schema ready: {PLATINUM_SCHEMA}")

# COMMAND ----------

# ── 4. CREATE OR REPLACE VIEW ─────────────────────────────────────────────────
# The view is defined once and re-evaluated on every query — no daily refresh needed.
#
# Logic:
#   1. Find the cutoff: the latest business_date in actuals with net_revenue > 0
#   2. Take ALL rows from actuals (including zero-revenue closed days)
#   3. Take forecast rows for dates STRICTLY AFTER the cutoff, from the LATEST run only
#   4. UNION both sets — no date can appear in both because actuals own every date up to cutoff
#
# The LATEST forecast run is identified by MAX(forecast_created_at) across the entire
# forecast table — this ensures we always show the most recent model output, not stale
# predictions from a previous run that may have used an older model version.

spark.sql(f"""
    CREATE OR REPLACE VIEW {COMBINED_VIEW}
    COMMENT 'Platinum: actuals + latest forecast in one duplication-free series. Use record_type to distinguish. Actuals from gold.daily_sales_summary; forecasts from gold.daily_sales_forecast (latest run only, for future dates only).'
    AS
    WITH last_actual AS (
        -- The last date we have real revenue for. Forecasts only appear after this.
        SELECT MAX(business_date) AS cutoff
        FROM {ACTUALS_TABLE}
        WHERE net_revenue > 0
    ),
    latest_forecast_run AS (
        -- Identify the most recent forecast batch
        SELECT MAX(forecast_created_at) AS latest_run
        FROM {FORECAST_TABLE}
    )

    -- ── Actuals: every day in gold.daily_sales_summary ──────────────────────
    SELECT
        a.business_date,
        'actual'                        AS record_type,
        a.order_count,
        a.gross_revenue,
        a.net_revenue,
        a.total_discounts,
        a.avg_ticket_size,
        a.item_count,
        a.day_of_week,
        a.day_name,
        a.is_weekend,
        a.is_bread_delivery_day,
        a.week_of_year,
        a.month,
        a.year,
        a.weather_high_f,
        a.weather_low_f,
        a.weather_feels_high_f,
        a.weather_feels_low_f,
        a.weather_category,
        a.weather_condition,
        a.weather_code,
        a.total_precip_in,
        a.total_snow_in,
        a.sunny_hours,
        a.avg_cloud_cover_pct,
        a.weather_data_type,
        -- Forecast-specific columns null for actuals
        CAST(NULL AS DOUBLE)            AS forecast_lower,
        CAST(NULL AS DOUBLE)            AS forecast_upper,
        CAST(NULL AS STRING)            AS forecast_model,
        CAST(NULL AS INTEGER)           AS forecast_model_version,
        CAST(NULL AS STRING)            AS forecast_run_id,
        CAST(NULL AS TIMESTAMP)         AS forecast_created_at,
        CAST(NULL AS INTEGER)           AS forecast_horizon_days,
        CAST(NULL AS BOOLEAN)           AS likely_future_closure
    FROM {ACTUALS_TABLE} a

    UNION ALL

    -- ── Forecasts: future dates only, latest run only ────────────────────────
    SELECT
        f.business_date,
        'forecast'                      AS record_type,
        f.order_count,
        CAST(NULL AS DOUBLE)            AS gross_revenue,
        f.net_revenue,
        CAST(NULL AS DOUBLE)            AS total_discounts,
        ROUND(f.net_revenue / NULLIF(f.order_count, 0), 2)
                                        AS avg_ticket_size,
        CAST(NULL AS INTEGER)           AS item_count,
        f.day_of_week,
        f.day_name,
        f.is_weekend,
        f.is_bread_delivery_day,
        f.week_of_year,
        f.month,
        f.year,
        f.weather_high_f,
        CAST(NULL AS DOUBLE)            AS weather_low_f,
        CAST(NULL AS DOUBLE)            AS weather_feels_high_f,
        CAST(NULL AS DOUBLE)            AS weather_feels_low_f,
        f.weather_category,
        f.weather_condition,
        CAST(NULL AS INTEGER)           AS weather_code,
        f.total_precip_in,
        CAST(NULL AS DOUBLE)            AS total_snow_in,
        CAST(NULL AS DOUBLE)            AS sunny_hours,
        CAST(NULL AS DOUBLE)            AS avg_cloud_cover_pct,
        CAST(NULL AS STRING)            AS weather_data_type,
        f.forecast_lower,
        f.forecast_upper,
        f.forecast_model,
        f.forecast_model_version,
        f.forecast_run_id,
        f.forecast_created_at,
        f.forecast_horizon_days,
        f.likely_future_closure
    FROM {FORECAST_TABLE} f
    CROSS JOIN last_actual
    CROSS JOIN latest_forecast_run
    WHERE f.business_date > last_actual.cutoff
      AND f.forecast_created_at = latest_forecast_run.latest_run
""")

print(f"✓ View created: {COMBINED_VIEW}")

# COMMAND ----------

# ── 5. APPLY UNITY CATALOG METADATA ───────────────────────────────────────────

spark.sql(f"""
    ALTER VIEW {COMBINED_VIEW}
    SET TAGS (
        'domain'  = 'retail',
        'layer'   = 'platinum',
        'pii'     = 'false',
        'refresh' = 'live_view',
        'dashboard' = 'primary'
    )
""")

print("✓ Tags applied")

# COMMAND ----------

# ── 6. VALIDATION ─────────────────────────────────────────────────────────────

print("── Row counts by record_type ──")
spark.sql(f"""
    SELECT
        record_type,
        COUNT(*)                        AS row_count,
        MIN(business_date)              AS earliest_date,
        MAX(business_date)              AS latest_date
    FROM {COMBINED_VIEW}
    GROUP BY record_type
    ORDER BY record_type
""").show(truncate=False)

print("\n── Duplication check — every date should appear exactly once ──")
dup_check = spark.sql(f"""
    SELECT
        business_date,
        COUNT(*) AS row_count
    FROM {COMBINED_VIEW}
    GROUP BY business_date
    HAVING COUNT(*) > 1
""")
dup_count = dup_check.count()
if dup_count > 0:
    print(f"  ✗ FAIL — {dup_count} dates appear more than once:")
    dup_check.show(10, truncate=False)
else:
    print(f"  ✓ PASS — no duplicate dates")

print("\n── Last 3 actuals + next 14 forecast days ──")
spark.sql(f"""
    SELECT
        business_date,
        day_name,
        record_type,
        order_count,
        ROUND(net_revenue, 2)           AS net_revenue,
        ROUND(forecast_lower, 2)        AS lower_bound,
        ROUND(forecast_upper, 2)        AS upper_bound,
        weather_high_f,
        weather_category,
        likely_future_closure
    FROM {COMBINED_VIEW}
    WHERE business_date BETWEEN CURRENT_DATE - INTERVAL 3 DAYS
                             AND CURRENT_DATE + INTERVAL 14 DAYS
    ORDER BY business_date
""").show(20, truncate=False)
