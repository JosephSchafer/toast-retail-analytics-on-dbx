# Databricks notebook source
# MAGIC %md
# MAGIC # Weekly Retrain Review — Auto-Promote & Seasonality Audit
# MAGIC
# MAGIC Runs after `7_Model_Prophet_Revenue` in the weekly retrain job.
# MAGIC
# MAGIC ## What this notebook does
# MAGIC
# MAGIC 1. **Selects the best candidate** from today's MLflow runs (lowest cv_mape)
# MAGIC 2. **Compares against current @production** — promotes only if cv_mape improves
# MAGIC    by more than the noise threshold (2%)
# MAGIC 3. **Triggers NB9 backfill** if a new model was promoted
# MAGIC 4. **Seasonality diagnostic** — checks whether the model is systematically off
# MAGIC    by DOW or by month, and whether the seasonal prior index needs recalibration
# MAGIC 5. **Writes a structured review row** to `gold.retrain_review_history` so every
# MAGIC    weekly run is auditable
# MAGIC
# MAGIC ## Promotion guardrail
# MAGIC
# MAGIC New model is promoted only if:
# MAGIC   `new_cv_mape < current_prod_cv_mape * PROMOTION_THRESHOLD`
# MAGIC
# MAGIC `PROMOTION_THRESHOLD = 1.02` — allows up to 2% regression before blocking.
# MAGIC This prevents flip-flopping when two models are statistically equivalent.
# MAGIC
# MAGIC ## Seasonality review
# MAGIC
# MAGIC During the first operating year (through Dec 2026), the model uses a prior from
# MAGIC the previous business at this location. As our own data accumulates, the prior
# MAGIC becomes less necessary and may start to constrain rather than guide. This notebook
# MAGIC flags three specific signals to watch:
# MAGIC
# MAGIC - **DOW bias**: any day-of-week with mean error > ±15% of its average actual
# MAGIC - **Monthly ramp divergence**: months where actual revenue is >20% above/below
# MAGIC   the seasonal prior index (signals the prior needs recalibration)
# MAGIC - **Trend acceleration**: 4-week trailing revenue growth rate vs 12-week rate
# MAGIC   (signals a structural shift the model may be missing)
# MAGIC
# MAGIC ## Change log
# MAGIC
# MAGIC | Version | Date | Author | Change |
# MAGIC |---|---|---|---|
# MAGIC | v1 | 2026-06-23 | JS | Initial build — auto-promote, seasonality diagnostic, review history |

# COMMAND ----------

# ── 1. INSTALL ────────────────────────────────────────────────────────────────

%pip install prophet mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ── 2. IMPORTS & CONFIG ───────────────────────────────────────────────────────

import mlflow
import pandas as pd
import numpy as np
import datetime
import json
import warnings
warnings.filterwarnings('ignore')

mlflow.set_registry_uri("databricks-uc")
client = mlflow.tracking.MlflowClient()

CATALOG    = "3sp_analytics_workspace"
MODEL_NAME = f"{CATALOG}.default.toast_revenue_prophet"

# Promote new model only if cv_mape is better than current prod by this factor.
# 1.02 = allow up to 2% regression (noise buffer). Anything worse keeps current prod.
PROMOTION_THRESHOLD = 1.02

# Flag DOW bias if mean error exceeds this fraction of the day's average actual revenue
DOW_BIAS_THRESHOLD = 0.15   # 15%

# Flag monthly divergence from prior index if actual is this far off the prior shape
MONTHLY_PRIOR_DIVERGENCE_THRESHOLD = 0.20  # 20%

# Trailing growth rate comparison window (weeks)
SHORT_WINDOW_WEEKS  = 4
LONG_WINDOW_WEEKS   = 12

print(f"MODEL_NAME: {MODEL_NAME}")
print(f"PROMOTION_THRESHOLD: {PROMOTION_THRESHOLD}")

# COMMAND ----------

# ── 3. FIND TODAY'S BEST CANDIDATE ────────────────────────────────────────────
# NB7 logged all variant runs to forecast_accuracy_history today.
# Pick the one with the lowest cv_mape.

_today = str(datetime.date.today())

candidates_df = spark.sql(f"""
    SELECT mlflow_run_id, notes AS variant_label,
           cv_mape, cv_mape_7d, cv_mape_14d,
           train_mape, train_r2, backtest_mape,
           training_data_through, training_rows
    FROM {CATALOG}.gold.forecast_accuracy_history
    WHERE training_date = DATE '{_today}'
      AND model_name = 'revenue'
      AND cv_mape IS NOT NULL
    ORDER BY cv_mape ASC
    LIMIT 1
""").toPandas()

if candidates_df.empty:
    print("No candidates found for today. NB7 may not have run yet or CV failed.")
    dbutils.notebook.exit("NO_CANDIDATES")

best = candidates_df.iloc[0]
best_run_id    = best["mlflow_run_id"]
best_cv_mape   = float(best["cv_mape"])
best_label     = best["variant_label"] or best_run_id[:12]
best_data_thru = str(best["training_data_through"])

print(f"Best candidate today: {best_label}")
print(f"  run_id:    {best_run_id}")
print(f"  cv_mape:   {best_cv_mape:.2f}%")
print(f"  data_thru: {best_data_thru}")

# COMMAND ----------

# ── 4. COMPARE AGAINST CURRENT @production ────────────────────────────────────

prod_cv_mape   = None
prod_version   = None
prod_run_id    = None

try:
    prod_mv      = client.get_model_version_by_alias(MODEL_NAME, "production")
    prod_version = prod_mv.version
    prod_run_id  = prod_mv.run_id

    # Pull cv_mape from accuracy history for the production run
    prod_metrics = spark.sql(f"""
        SELECT cv_mape FROM {CATALOG}.gold.forecast_accuracy_history
        WHERE mlflow_run_id = '{prod_run_id}'
          AND cv_mape IS NOT NULL
        ORDER BY logged_at DESC
        LIMIT 1
    """).toPandas()

    if not prod_metrics.empty:
        prod_cv_mape = float(prod_metrics.iloc[0]["cv_mape"])

    print(f"\nCurrent @production: v{prod_version}  run={prod_run_id[:12]}")
    print(f"  prod_cv_mape: {prod_cv_mape:.2f}%" if prod_cv_mape else "  prod_cv_mape: unknown (first run)")
except Exception as e:
    print(f"No current @production model (first promotion or error): {e}")

# Promotion decision
if prod_cv_mape is None:
    _should_promote = True
    _promotion_reason = "no_prior_production"
elif best_cv_mape < prod_cv_mape * PROMOTION_THRESHOLD:
    _should_promote = True
    _promotion_reason = f"cv_mape improved {prod_cv_mape:.2f}% → {best_cv_mape:.2f}%"
else:
    _should_promote = False
    _promotion_reason = f"cv_mape {best_cv_mape:.2f}% did not beat {prod_cv_mape:.2f}% * {PROMOTION_THRESHOLD} = {prod_cv_mape * PROMOTION_THRESHOLD:.2f}%"

print(f"\nPromotion decision: {'PROMOTE' if _should_promote else 'KEEP CURRENT'}")
print(f"  Reason: {_promotion_reason}")

# COMMAND ----------

# ── 5. PROMOTE (if warranted) ─────────────────────────────────────────────────

promoted_version = None

if _should_promote:
    # Deprecate old production
    if prod_version:
        try:
            client.set_model_version_tag(MODEL_NAME, prod_version, "status", "deprecated")
            print(f"  v{prod_version} → deprecated")
        except Exception:
            pass

    # Register the winning run
    mv = mlflow.register_model(f"runs:/{best_run_id}/prophet_revenue_model", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, mv.version, "status",        "production")
    client.set_model_version_tag(MODEL_NAME, mv.version, "version_tag",   best_label)
    client.set_model_version_tag(MODEL_NAME, mv.version, "training_date", _today)
    client.set_registered_model_alias(MODEL_NAME, "production", mv.version)

    promoted_version = mv.version
    print(f"  ✓ v{mv.version} → @production  ({best_label})")

    # Trigger NB9 backfill via the daily job's Generate_Forecasts task
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        nb9_run = w.jobs.run_now(
            job_id=601248779031413,
            notebook_params={"run_mode": "backfill"},
        )
        print(f"  ✓ NB9 backfill triggered (run_id: {nb9_run.run_id})")
    except Exception as e:
        print(f"  ⚠ NB9 trigger failed — trigger manually: {e}")
else:
    print(f"  Keeping @production v{prod_version} ({prod_cv_mape:.2f}% cv_mape)")

# COMMAND ----------

# ── 6. SEASONALITY DIAGNOSTIC ─────────────────────────────────────────────────
# Runs regardless of whether promotion happened.
# Flags three signals: DOW bias, monthly prior divergence, trend acceleration.

# ── 6a. DOW bias (last 28 days, horizon 1-14) ─────────────────────────────────

dow_df = spark.sql(f"""
    SELECT
        f.day_name,
        f.day_of_week,
        COUNT(*)                                                              AS n,
        ROUND(AVG(a.net_revenue), 0)                                         AS avg_actual,
        ROUND(AVG(f.net_revenue), 0)                                         AS avg_forecast,
        ROUND(AVG(a.net_revenue - f.net_revenue), 0)                         AS mean_error,
        ROUND(AVG(ABS(a.net_revenue - f.net_revenue) / a.net_revenue), 4)    AS mean_abs_pct_err,
        ROUND(AVG((a.net_revenue - f.net_revenue) / a.net_revenue), 4)       AS mean_pct_err
    FROM {CATALOG}.gold.daily_sales_forecast f
    JOIN {CATALOG}.gold.daily_sales_summary a ON f.business_date = a.business_date
    WHERE a.net_revenue > 300
      AND f.forecast_horizon_days BETWEEN 1 AND 14
      AND a.business_date >= date_sub(current_date(), 28)
    GROUP BY f.day_name, f.day_of_week
    ORDER BY f.day_of_week
""").toPandas()

print("\n── DOW Bias (last 28 days, horizon 1-14d) ──")
print(f"  {'Day':<12} {'n':>3}  {'avg_actual':>10}  {'avg_forecast':>12}  {'mean_err':>10}  {'mean_pct_err':>13}  flag")
dow_flags = []
for _, row in dow_df.iterrows():
    flag = ""
    if abs(row["mean_pct_err"]) > DOW_BIAS_THRESHOLD:
        direction = "UNDER" if row["mean_pct_err"] > 0 else "OVER"
        flag = f"⚠ {direction}-forecast {abs(row['mean_pct_err'])*100:.0f}%"
        dow_flags.append({"dow": row["day_name"], "mean_pct_err": row["mean_pct_err"], "flag": flag})
    print(f"  {row['day_name']:<12} {int(row['n']):>3}  ${row['avg_actual']:>9,.0f}  ${row['avg_forecast']:>11,.0f}  ${row['mean_error']:>9,.0f}  {row['mean_pct_err']*100:>+12.1f}%  {flag}")

if not dow_flags:
    print("  ✓ No DOW bias exceeding threshold")

# ── 6b. Monthly revenue vs prior index ────────────────────────────────────────

monthly_df = spark.sql(f"""
    SELECT
        YEAR(business_date)  AS yr,
        MONTH(business_date) AS mo,
        DATE_FORMAT(business_date, 'yyyy-MM') AS year_month,
        ROUND(SUM(net_revenue), 0)  AS actual_revenue,
        COUNT(*)                    AS days
    FROM {CATALOG}.gold.daily_sales_summary
    WHERE net_revenue > 100
      AND business_date >= '2026-01-01'
    GROUP BY 1, 2, 3
    HAVING COUNT(*) >= 20
    ORDER BY 1, 2
""").toPandas()

# Prior monthly linearity (same as NB7)
PRIOR_MONTHLY_LINEARITY = {
    1: 0.065, 2: 0.064, 3: 0.073, 4: 0.075,
    5: 0.108, 6: 0.122, 7: 0.132, 8: 0.126,
    9: 0.093, 10: 0.077, 11: 0.061, 12: 0.062,
}

monthly_flags = []
if not monthly_df.empty:
    # Normalize actuals to fractions of total (same scale as prior)
    total_rev = monthly_df["actual_revenue"].sum()
    monthly_df["actual_fraction"] = monthly_df["actual_revenue"] / total_rev
    monthly_df["prior_fraction"]  = monthly_df["mo"].map(PRIOR_MONTHLY_LINEARITY)

    print("\n── Monthly Revenue vs Prior Index ──")
    print(f"  {'Month':<10}  {'actual_frac':>12}  {'prior_frac':>11}  {'divergence':>11}  flag")
    for _, row in monthly_df.iterrows():
        divergence = (row["actual_fraction"] - row["prior_fraction"]) / row["prior_fraction"]
        flag = ""
        if abs(divergence) > MONTHLY_PRIOR_DIVERGENCE_THRESHOLD:
            direction = "ABOVE" if divergence > 0 else "BELOW"
            flag = f"⚠ {abs(divergence)*100:.0f}% {direction} prior"
            monthly_flags.append({"month": row["year_month"], "divergence": divergence, "flag": flag})
        print(f"  {row['year_month']:<10}  {row['actual_fraction']:>11.3f}  {row['prior_fraction']:>10.3f}  {divergence:>+10.1%}  {flag}")

    if not monthly_flags:
        print("  ✓ Monthly shape tracking prior within threshold")

# ── 6c. Trend acceleration ────────────────────────────────────────────────────

trend_df = spark.sql(f"""
    SELECT
        business_date,
        net_revenue,
        AVG(net_revenue) OVER (ORDER BY business_date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS ma_28d,
        AVG(net_revenue) OVER (ORDER BY business_date ROWS BETWEEN 83 PRECEDING AND CURRENT ROW) AS ma_84d
    FROM {CATALOG}.gold.daily_sales_summary
    WHERE net_revenue > 100
    ORDER BY business_date DESC
    LIMIT 1
""").toPandas()

trend_flags = []
if not trend_df.empty:
    ma_short = float(trend_df.iloc[0]["ma_28d"])
    ma_long  = float(trend_df.iloc[0]["ma_84d"])
    accel    = (ma_short - ma_long) / ma_long if ma_long > 0 else 0.0
    accel_flag = ""
    if abs(accel) > 0.10:
        direction = "ACCELERATING" if accel > 0 else "DECELERATING"
        accel_flag = f"⚠ Trend {direction} {abs(accel)*100:.0f}% above long-run avg — changepoint may lag"
        trend_flags.append({"accel": accel, "flag": accel_flag})

    print(f"\n── Trend Acceleration ──")
    print(f"  28-day avg:  ${ma_short:,.0f}/day")
    print(f"  84-day avg:  ${ma_long:,.0f}/day")
    print(f"  Acceleration: {accel:+.1%}  {accel_flag if accel_flag else '✓ within normal range'}")

# ── Summary ───────────────────────────────────────────────────────────────────

total_flags = len(dow_flags) + len(monthly_flags) + len(trend_flags)
print(f"\n── Seasonality Summary ──")
print(f"  DOW flags:           {len(dow_flags)}")
print(f"  Monthly prior flags: {len(monthly_flags)}")
print(f"  Trend flags:         {len(trend_flags)}")
if total_flags == 0:
    print("  ✓ No seasonality concerns flagged")
else:
    print(f"  ⚠ {total_flags} flag(s) — review the diagnostics above")
    if monthly_flags:
        print("\n  Monthly divergence suggests the prior store's seasonal shape")
        print("  may need recalibration. Consider:")
        print("    1. Are specific months consistently above/below the prior?")
        print("    2. If Jun-Aug are all 20%+ above prior, the prior underestimates summer peak.")
        print("    3. Update PRIOR_MONTHLY_LINEARITY in NB7 config if pattern is stable >2 months.")
    if trend_flags:
        print("\n  Trend acceleration: model may be underestimating the forward trajectory.")
        print("  Consider raising changepoint_prior_scale in the next retrain variant.")

# COMMAND ----------

# ── 7. WRITE REVIEW HISTORY ───────────────────────────────────────────────────

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.retrain_review_history (
        review_date             DATE      COMMENT 'Date this review ran',
        best_run_id             STRING    COMMENT 'MLflow run ID of best candidate',
        best_label              STRING    COMMENT 'Variant label of best candidate',
        best_cv_mape            DOUBLE    COMMENT 'CV MAPE of best candidate',
        best_data_through       DATE      COMMENT 'Training data cutoff of best candidate',
        prod_run_id             STRING    COMMENT 'MLflow run ID of prior @production',
        prod_cv_mape            DOUBLE    COMMENT 'CV MAPE of prior @production',
        promoted                BOOLEAN   COMMENT 'Whether a new model was promoted',
        promoted_version        INT       COMMENT 'Registry version promoted (null if no promotion)',
        promotion_reason        STRING    COMMENT 'Why promotion happened or did not',
        dow_flags               INT       COMMENT 'Number of DOW bias flags',
        monthly_prior_flags     INT       COMMENT 'Number of monthly prior divergence flags',
        trend_flags             INT       COMMENT 'Number of trend acceleration flags',
        flag_detail             STRING    COMMENT 'JSON array of flag details',
        notes                   STRING    COMMENT 'Free-text notes'
    )
    USING DELTA
    COMMENT 'One row per weekly retrain review. Tracks promotion decisions and seasonality health.'
""")

_flags_json = json.dumps({
    "dow":     dow_flags,
    "monthly": monthly_flags,
    "trend":   trend_flags,
}).replace("'", "\\'")

def _sql_str(v):
    if v is None:
        return "NULL"
    return f"'{str(v)}'"

def _sql_float(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NULL"
    return str(round(float(v), 4))

def _sql_int(v):
    if v is None:
        return "NULL"
    return str(int(v))

spark.sql(f"""
    INSERT INTO {CATALOG}.gold.retrain_review_history
    (review_date, best_run_id, best_label, best_cv_mape, best_data_through,
     prod_run_id, prod_cv_mape, promoted, promoted_version, promotion_reason,
     dow_flags, monthly_prior_flags, trend_flags, flag_detail, notes)
    VALUES (
        DATE '{_today}',
        {_sql_str(best_run_id)},
        {_sql_str(best_label)},
        {_sql_float(best_cv_mape)},
        DATE '{best_data_thru}',
        {_sql_str(prod_run_id)},
        {_sql_float(prod_cv_mape)},
        {str(_should_promote).upper()},
        {_sql_int(promoted_version)},
        {_sql_str(_promotion_reason)},
        {len(dow_flags)},
        {len(monthly_flags)},
        {len(trend_flags)},
        {_sql_str(_flags_json)},
        NULL
    )
""")

print(f"\n✓ Review row written to {CATALOG}.gold.retrain_review_history")

# COMMAND ----------

# ── 8. EXIT STATUS ────────────────────────────────────────────────────────────

_status = "PROMOTED" if _should_promote else "NO_CHANGE"
_summary = (
    f"{_status} | best={best_label} cv_mape={best_cv_mape:.1f}% | "
    f"flags: dow={len(dow_flags)} monthly={len(monthly_flags)} trend={len(trend_flags)}"
)
print(f"\n{_summary}")
dbutils.notebook.exit(_summary)
