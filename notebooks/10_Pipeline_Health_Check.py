# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline Health Check
# MAGIC
# MAGIC Runs as the final task in the daily job after NB9. Validates that:
# MAGIC 1. Actuals are fresh (sales summary updated with yesterday's data)
# MAGIC 2. Forecast horizon is adequate (≥25 days forward)
# MAGIC 3. Feature table has no unexpected NULLs in key columns
# MAGIC 4. Platinum view returns both actuals and forecasts
# MAGIC 5. Rolling accuracy has not degraded past the 20% MAPE warning threshold
# MAGIC
# MAGIC **Critical checks raise RuntimeError** so the job status turns red and
# MAGIC the existing job failure notifications fire normally.
# MAGIC Accuracy warnings are printed but do not fail the job.

# COMMAND ----------

import datetime

CATALOG   = "3sp_analytics_workspace"
TODAY     = datetime.date.today()
YESTERDAY = TODAY - datetime.timedelta(days=1)

print(f"── Pipeline Health Check: {TODAY} ──\n")

# COMMAND ----------

# ── 1. ACTUALS FRESHNESS ──────────────────────────────────────────────────────
# gold.daily_sales_summary should have an actuals row for at least yesterday.
# If NB4 (Gold Sales) ran successfully, this will always be true.

result = spark.sql(f"""
    SELECT MAX(business_date) AS max_date
    FROM {CATALOG}.gold.daily_sales_summary
""").collect()[0]

max_actual = result["max_date"]

if max_actual is None or max_actual < YESTERDAY:
    raise RuntimeError(
        f"FAIL — daily_sales_summary is stale. "
        f"max_date={max_actual}, expected>={YESTERDAY}. "
        f"NB4 (Gold Sales Summary) likely did not run or the Toast ingest found no new orders."
    )
print(f"✓ Actuals fresh: daily_sales_summary through {max_actual}")

# COMMAND ----------

# ── 2. FORECAST HORIZON ───────────────────────────────────────────────────────
# gold.daily_sales_forecast should have rows written today covering ≥25 days forward.
# NB9 generates 30 days; ≥25 gives a buffer for feature data gaps in forecast_features.

result = spark.sql(f"""
    SELECT
        COUNT(DISTINCT business_date) AS fwd_rows,
        MAX(business_date)            AS max_fwd_date
    FROM {CATALOG}.gold.daily_sales_forecast
    WHERE business_date > CURRENT_DATE
      AND CAST(forecast_created_at AS DATE) = CURRENT_DATE
""").collect()[0]

min_horizon = TODAY + datetime.timedelta(days=25)

if result["max_fwd_date"] is None or result["max_fwd_date"] < min_horizon:
    raise RuntimeError(
        f"FAIL — forecast horizon too short. "
        f"max_fwd_date={result['max_fwd_date']}, expected>={min_horizon}. "
        f"NB9 (Forecast Generate) may not have run today, or forecast_features "
        f"is missing future weather rows (check NB2 weather ingest)."
    )
print(f"✓ Forecast horizon: {result['fwd_rows']} days through {result['max_fwd_date']}")

# COMMAND ----------

# ── 3. FEATURE TABLE INTEGRITY ────────────────────────────────────────────────
# Check for unexpected NULLs in the last 7 days of gold.forecast_features.
# Warnings only — NULLs in weather columns are filled with defaults in NB9,
# but they indicate the weather ingest (NB2) may have failed.

nulls = spark.sql(f"""
    SELECT
        COUNT(*)                                                              AS rows_checked,
        SUM(CASE WHEN weather_high_f          IS NULL THEN 1 ELSE 0 END)    AS null_weather,
        SUM(CASE WHEN is_bread_delivery_day   IS NULL THEN 1 ELSE 0 END)    AS null_bread,
        SUM(CASE WHEN weather_comfort_score   IS NULL THEN 1 ELSE 0 END)    AS null_comfort,
        SUM(CASE WHEN heat_stress             IS NULL THEN 1 ELSE 0 END)    AS null_heat
    FROM {CATALOG}.gold.forecast_features
    WHERE ds BETWEEN DATE_SUB(CURRENT_DATE, 7) AND CURRENT_DATE
""").collect()[0]

warnings = []
if nulls["null_weather"] and nulls["null_weather"] > 0:
    warnings.append(f"weather_high_f: {nulls['null_weather']} NULLs — weather ingest (NB2) may have failed")
if nulls["null_comfort"] and nulls["null_comfort"] > 0:
    warnings.append(f"weather_comfort_score: {nulls['null_comfort']} NULLs — NB6 feature compute incomplete")
if nulls["null_heat"] and nulls["null_heat"] > 0:
    warnings.append(f"heat_stress: {nulls['null_heat']} NULLs — NB6 feature compute incomplete")

if warnings:
    print(f"⚠ Feature integrity ({nulls['rows_checked']} rows checked):")
    for w in warnings:
        print(f"  - {w}")
else:
    print(f"✓ Feature integrity: {nulls['rows_checked']} recent rows, no NULLs in key columns")

# COMMAND ----------

# ── 4. PLATINUM VIEW SANITY CHECK ─────────────────────────────────────────────
# platinum.daily_sales_combined (VIEW) must return both ACTUAL and FORECAST rows.
# If it returns zero rows of either type, the upstream tables or the view DDL
# is broken — this should never happen after a successful pipeline run.

counts = spark.sql(f"""
    SELECT record_type, COUNT(*) AS cnt
    FROM {CATALOG}.platinum.daily_sales_combined
    GROUP BY record_type
""").collect()

type_map  = {r["record_type"]: r["cnt"] for r in counts}
n_actuals = type_map.get("actual",   0)
n_fcast   = type_map.get("forecast", 0)

if n_actuals == 0:
    raise RuntimeError(
        "FAIL — platinum.daily_sales_combined has no 'actual' rows. "
        "The VIEW or gold.daily_sales_summary may be broken."
    )
if n_fcast == 0:
    raise RuntimeError(
        "FAIL — platinum.daily_sales_combined has no 'forecast' rows. "
        "NB9 did not write forward rows, or the platinum VIEW is broken."
    )
print(f"✓ Platinum view: {n_actuals} actuals + {n_fcast} forecasts")

# COMMAND ----------

# ── 5. ROLLING ACCURACY RETROSPECTIVE ────────────────────────────────────────
# Compares historical forecast values to actuals by forecast horizon
# (1-day-out, 7-day-out, 14-day-out) over the last 30 days.
#
# Because NB9 appends one row per (business_date, forecast_created_at::date),
# we can look up what was predicted N days before each date occurred.
# MAPE > 20% triggers a retraining recommendation (warning, not failure).

accuracy_rows = spark.sql(f"""
    SELECT
        f.forecast_horizon_days                                          AS horizon_days,
        COUNT(*)                                                         AS days_compared,
        ROUND(AVG(ABS(f.net_revenue - s.net_revenue)
              / NULLIF(s.net_revenue, 0)) * 100, 1)                     AS mape_pct,
        ROUND(AVG(ABS(f.net_revenue - s.net_revenue)), 0)               AS mae_dollars
    FROM {CATALOG}.gold.daily_sales_forecast   f
    JOIN {CATALOG}.gold.daily_sales_summary    s ON s.business_date = f.business_date
    WHERE f.business_date BETWEEN DATE_SUB(CURRENT_DATE, 30)
                              AND DATE_SUB(CURRENT_DATE, 1)
      AND s.net_revenue > 0
      AND f.forecast_horizon_days IN (1, 7, 14)
    GROUP BY f.forecast_horizon_days
    ORDER BY f.forecast_horizon_days
""").collect()

accuracy_warning = False
if not accuracy_rows:
    print("  Accuracy retrospective: no matched forecast/actual pairs yet")
    print("  (Normal for first 14 days of deployment — will populate automatically)")
else:
    print(f"\n  Rolling accuracy (last 30 days vs actuals):")
    print(f"  {'Horizon':>10}  {'Days':>5}  {'MAPE':>7}  {'MAE':>10}")
    print(f"  {'-'*38}")
    for r in accuracy_rows:
        flag = " ⚠" if r["mape_pct"] and r["mape_pct"] > 20 else "  "
        print(f"  {str(r['horizon_days']) + 'd':>10}  {r['days_compared']:>5}  "
              f"{r['mape_pct']:>6.1f}%{flag}  ${r['mae_dollars']:>8,.0f}")
        if r["mape_pct"] and r["mape_pct"] > 20:
            accuracy_warning = True
    if accuracy_warning:
        print(f"\n  ⚠ MAPE above 20% — consider retraining NB7/NB8 (monthly cadence).")

# COMMAND ----------

# ── SUMMARY ───────────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"HEALTH CHECK PASSED — {TODAY}")
print(f"{'='*50}")
if warnings:
    print(f"  {len(warnings)} feature warning(s) above — check NB2 weather ingest.")
if accuracy_warning:
    print(f"  Accuracy degraded — run NB7 + NB8 to retrain Prophet models.")
if not warnings and not accuracy_warning:
    print(f"  All checks clean. Pipeline is healthy.")
