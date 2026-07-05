# Databricks notebook source
# MAGIC %md
# MAGIC # Prophet Model — Order Count Forecast
# MAGIC
# MAGIC Trains a Facebook Prophet model to forecast daily order (ticket) count.
# MAGIC Mirrors the architecture of `7_Model_Prophet_Revenue` but targets
# MAGIC `y_orders` instead of `y_revenue`.
# MAGIC
# MAGIC ## Why a separate orders model?
# MAGIC
# MAGIC Order count and revenue are related but distinct signals:
# MAGIC - A high-revenue day could be few large tickets (catering) or many small ones (busy Saturday)
# MAGIC - Order count is a direct measure of foot traffic — useful for staffing decisions
# MAGIC - The two models may have different optimal parameters and different accuracy profiles
# MAGIC - Comparing revenue forecast vs orders forecast can flag catering days
# MAGIC   (revenue up, orders flat = large ticket day)
# MAGIC
# MAGIC ## Key modeling decisions
# MAGIC
# MAGIC - **Training start: Dec 1 2025** — same as revenue model
# MAGIC - **Linear growth, changepoint_prior_scale=0.01** — same conservative trend
# MAGIC - **Custom yearly seasonality** from prior store monthly linearity
# MAGIC - **Regressors:** bread delivery day, temperature — precipitation dropped
# MAGIC   (order count is less sensitive to rain than revenue; customers still come
# MAGIC   in, they just spend less)
# MAGIC
# MAGIC ## Dependencies
# MAGIC
# MAGIC **Reads from:**
# MAGIC - `3sp_analytics_workspace.gold.forecast_features`
# MAGIC - `3sp_analytics_workspace.reference.store_events`
# MAGIC
# MAGIC **Writes to:**
# MAGIC - MLflow experiment: `toast_prophet_orders`
# MAGIC - MLflow Model Registry: `toast_orders_prophet`
# MAGIC - `3sp_analytics_workspace.gold.daily_sales_summary` (forecast order_count)
# MAGIC
# MAGIC **Upstream:** `7_Model_Prophet_Revenue` — run revenue model first
# MAGIC **Downstream:** `11_Model_Evaluate_Register`
# MAGIC
# MAGIC ## Change log
# MAGIC
# MAGIC | Version | Date | Author | Change |
# MAGIC |---|---|---|---|
# MAGIC | v2 | 2026-04-19 | JS | Switched to flat growth; reduced seasonality_prior_scale grid 5/10/10→1/2/3 — same fix as revenue model v3 |
# MAGIC | v1 | 2026-03-28 | JS | Initial build |

# COMMAND ----------

# ── 1. INSTALL DEPENDENCIES ───────────────────────────────────────────────────

%pip install prophet mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ── 2. IMPORTS ────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import datetime
warnings.filterwarnings('ignore')

try:
    import mlflow
    import mlflow.prophet
    print("✓ MLflow imported")
except ModuleNotFoundError:
    raise RuntimeError("MLflow not found. Ensure cell 1 installed it and restartPython() ran.")

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from pyspark.sql import functions as F
from delta.tables import DeltaTable

print("✓ All libraries loaded")

# COMMAND ----------

# ── 3. CONFIGURATION ──────────────────────────────────────────────────────────

CATALOG        = "3sp_analytics_workspace"
FEATURES_TABLE = f"{CATALOG}.gold.forecast_features"
DAILY_TABLE    = f"{CATALOG}.gold.daily_sales_summary"

EXPERIMENT_NAME = "toast_prophet_orders"

# UC model registry — three-level name required (catalog.schema.model), matching NB7
# and NB9's loader. Fixed 2026-07-05: the one-level name rode on the legacy workspace
# registry being MLflow's default; a serverless env update flipped the default to UC
# and the weekly retrain job started failing on registration/search.
MODEL_NAME      = f"{CATALOG}.default.toast_orders_prophet"

# Set UC as the registry target before any register_model() calls (same as NB7).
import mlflow as _mlflow_init
_mlflow_init.set_registry_uri("databricks-uc")

FORECAST_DAYS = 30
TRAIN_START   = pd.Timestamp('2025-12-01')

# Prior store monthly linearity — same seasonal shape as revenue model
# Order count seasonality mirrors revenue seasonality at this location
PRIOR_MONTHLY_LINEARITY = {
    1:  0.065,  2:  0.064,  3:  0.073,  4:  0.075,
    5:  0.108,  6:  0.122,  7:  0.132,  8:  0.126,
    9:  0.093,  10: 0.077,  11: 0.061,  12: 0.062,
}

# Convert to 52 ISO-week weights for consistent display with revenue model
from calendar import monthrange as _monthrange
_REF_YEAR = 2025
_daily_lin_o = {}
for _m, _prop in PRIOR_MONTHLY_LINEARITY.items():
    _days = _monthrange(_REF_YEAR, _m)[1]
    for _d in range(1, _days + 1):
        _daily_lin_o[pd.Timestamp(_REF_YEAR, _m, _d)] = _prop / _days
_weekly_sums_o: dict = {}
for _dt, _v in _daily_lin_o.items():
    _w = int(_dt.isocalendar()[1])
    _weekly_sums_o[_w] = _weekly_sums_o.get(_w, 0.0) + _v
_wk_total_o = sum(_weekly_sums_o.values())
PRIOR_WEEKLY_LINEARITY = {w: _weekly_sums_o[w] / _wk_total_o for w in sorted(_weekly_sums_o)}

# Parameter grid — same structure as revenue model for comparability
# Revenue model winner was tight_trend_moderate_seasonal (cv_mape=21.7%)
# We start with that as the baseline and test a couple of variations
# seasonality_prior_scale reduced from 5/10/10 → 1/2/3 to avoid over-reliance
# on the prior store summer shape when we have no observed summer data yet.
PARAM_GRID = [
    {
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 1.0,
        "holidays_prior_scale":    10.0,
        "run_label":               "flat_soft_seasonal",
    },
    {
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 2.0,
        "holidays_prior_scale":    10.0,
        "run_label":               "flat_moderate_seasonal",
    },
    {
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 3.0,
        "holidays_prior_scale":    20.0,
        "run_label":               "flat_stronger_seasonal",
    },
]

# growth='flat': mirrors the revenue model fix. The store is at a stable
# plateau since Dec 2025; linear trend extrapolation caused 2-3x overforecasts.
PROPHET_FIXED = {
    "growth":             "flat",
    "seasonality_mode":   "multiplicative",
    "yearly_seasonality": False,
    "weekly_seasonality": True,
    "daily_seasonality":  False,
}

# COMMAND ----------

# ── 4. LOAD SPECIAL EVENTS FROM REFERENCE TABLE ───────────────────────────────

events_ref = spark.sql(f"""
    SELECT event_date, event_name, event_type, lower_window, upper_window
    FROM {CATALOG}.reference.store_events
    WHERE is_active = true
    ORDER BY event_date
""").toPandas()

events_ref['event_date'] = events_ref['event_date'].astype(str)

SPECIAL_EVENTS = {
    row['event_date']: (
        row['event_name'],
        row['event_type'],
        int(row['lower_window']),
        int(row['upper_window'])
    )
    for _, row in events_ref.iterrows()
    if row['event_type'] in ('PLANNED_EVENT', 'REVENUE_DISTORTION', 'ORGANIC_EVENT')
}

FUTURE_CLOSURES = {
    row['event_date']: row['event_name']
    for _, row in events_ref.iterrows()
    if row['event_type'] == 'FUTURE_CLOSURE'
}

print(f"✓ Loaded {len(events_ref)} events from reference.store_events")
for _, row in events_ref.iterrows():
    print(f"  {row['event_date']}  {row['event_name']:<35} {row['event_type']}")

# COMMAND ----------

# ── 5. LOAD AND PREPARE TRAINING DATA ─────────────────────────────────────────

features_df = spark.sql(f"""
    SELECT
        ds, y_revenue, y_orders,
        day_of_week, day_name, is_weekend,
        is_bread_delivery_day, is_holiday_window,
        store_closed, is_imputed,
        exclude_from_training, training_weight,
        weather_high_f, weather_low_f,
        total_precip_in, total_snow_in,
        weather_category
    FROM {FEATURES_TABLE}
    ORDER BY ds
""").toPandas()

features_df['ds'] = pd.to_datetime(features_df['ds'])

all_historical = features_df[features_df['y_orders'].notna()].copy()
future_df      = features_df[features_df['y_orders'].isna()].copy()

# Training: Dec 1 onward, not excluded, not imputed
train_df = all_historical[
    (all_historical['ds'] >= TRAIN_START) &
    (~all_historical['exclude_from_training']) &
    (all_historical['training_weight'] > 0.5)
].copy()

print(f"All historical rows:  {len(all_historical)}")
print(f"Training rows (Dec+): {len(train_df)}")
print(f"  Date range: {train_df['ds'].min().date()} → {train_df['ds'].max().date()}")
print(f"  Avg daily orders: {train_df['y_orders'].mean():.1f}")
print(f"  Max daily orders: {train_df['y_orders'].max():.0f}")
print(f"Future rows:          {len(future_df)}")

# COMMAND ----------

# ── 6. BUILD PROPHET HOLIDAYS DATAFRAME ───────────────────────────────────────

holiday_rows = []

for date_str, (name, etype, lower_w, upper_w) in SPECIAL_EVENTS.items():
    if etype == 'PLANNED_EVENT':
        holiday_rows.append({
            'holiday':      name,
            'ds':           pd.Timestamp(date_str),
            'lower_window': lower_w,
            'upper_window': upper_w,
        })

for date_str, name in FUTURE_CLOSURES.items():
    holiday_rows.append({
        'holiday':      f"Closed - {name}",
        'ds':           pd.Timestamp(date_str),
        'lower_window': 0,
        'upper_window': 0,
    })

for year in [2025, 2026]:
    for day in range(15, 32):
        try:
            holiday_rows.append({
                'holiday':      'Holiday Shopping Window',
                'ds':           pd.Timestamp(f"{year}-12-{day:02d}"),
                'lower_window': 0,
                'upper_window': 0,
            })
        except Exception:
            pass

holidays_df = pd.DataFrame(holiday_rows)
print(f"✓ Holidays: {len(holidays_df)} rows, {holidays_df['holiday'].nunique()} unique")

# COMMAND ----------

# ── 7. PRINT PRIOR SEASONALITY ────────────────────────────────────────────────

print("Prior seasonality (normalized, weekly):")
mean_weekly = np.mean(list(PRIOR_WEEKLY_LINEARITY.values()))
for week, lin in PRIOR_WEEKLY_LINEARITY.items():
    normalized  = (lin / mean_weekly) - 1.0
    bar         = "█" * int(abs(normalized) * 20)
    sign        = "+" if normalized >= 0 else "-"
    print(f"  Wk {week:02d}  {sign}{abs(normalized)*100:4.1f}%  {bar}")

# COMMAND ----------

# ── 8. CONFIGURE MLFLOW ───────────────────────────────────────────────────────

current_user = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{current_user}/{EXPERIMENT_NAME}")
print(f"✓ MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training — Parameter Grid Search
# MAGIC
# MAGIC Three configurations, same cross-validation approach as the revenue model.
# MAGIC The winner is selected by CV MAPE and registered to the Model Registry.

# COMMAND ----------

# ── 9. TRAIN ACROSS PARAMETER GRID ───────────────────────────────────────────

run_results = []

for params in PARAM_GRID:

    run_label   = params.pop("run_label")
    full_params = {**PROPHET_FIXED, **params}

    print(f"\n{'='*55}")
    print(f"Run: {run_label}")
    print(f"  changepoint_prior_scale: {params['changepoint_prior_scale']}")
    print(f"  seasonality_prior_scale: {params['seasonality_prior_scale']}")
    print(f"  holidays_prior_scale:    {params['holidays_prior_scale']}")

    with mlflow.start_run(run_name=f"prophet_orders_{run_label}") as run:

        run_id = run.info.run_id
        mlflow.log_params(full_params)
        mlflow.log_param("training_start",  str(TRAIN_START.date()))
        mlflow.log_param("training_rows",   len(train_df))
        mlflow.log_param("target",          "order_count")
        mlflow.log_param("seasonal_prior",  "prior_store_avg_2022_2023")
        mlflow.log_param("run_label",       run_label)
        mlflow.set_tag("training_date",           str(pd.Timestamp.today().date()))
        mlflow.log_param("training_data_through", str(train_df['ds'].max().date()))

        # ── Build model ───────────────────────────────────────────────────────
        model = Prophet(
            growth                  = full_params["growth"],
            changepoint_prior_scale = params["changepoint_prior_scale"],
            seasonality_prior_scale = params["seasonality_prior_scale"],
            holidays_prior_scale    = params["holidays_prior_scale"],
            seasonality_mode        = full_params["seasonality_mode"],
            yearly_seasonality      = full_params["yearly_seasonality"],
            weekly_seasonality      = full_params["weekly_seasonality"],
            daily_seasonality       = full_params["daily_seasonality"],
            holidays                = holidays_df,
        )

        model.add_seasonality(
            name          = "yearly_prior",
            period        = 365.25,
            fourier_order = 3,
            prior_scale   = params["seasonality_prior_scale"],
        )

        # Regressors for order count:
        # - bread delivery: strong effect on foot traffic
        # - temperature: negative correlation with orders (colder = more traffic)
        # - precipitation dropped vs revenue model — rain affects spend more than visits
        model.add_regressor("is_bread_delivery_day", mode="multiplicative")
        model.add_regressor("weather_high_f",         mode="additive")

        # ── Fit ───────────────────────────────────────────────────────────────
        fit_df = train_df[
            train_df["y_orders"].notna() &
            (train_df["training_weight"] > 0.5)
        ][["ds", "y_orders", "is_bread_delivery_day",
           "weather_high_f"]].rename(
            columns={"y_orders": "y"}
        ).dropna(subset=["ds", "y"])

        fit_df["weather_high_f"]       = fit_df["weather_high_f"].fillna(
            fit_df["weather_high_f"].median()
        )
        fit_df["is_bread_delivery_day"] = (
            fit_df["is_bread_delivery_day"].fillna(False).astype(int)
        )

        model.fit(fit_df[["ds", "y", "is_bread_delivery_day", "weather_high_f"]])

        # ── Forecast input ────────────────────────────────────────────────────
        hist_input = all_historical[["ds", "is_bread_delivery_day",
                                      "weather_high_f"]].copy()
        fwd_input  = future_df[["ds", "is_bread_delivery_day",
                                  "weather_high_f"]].copy()

        fwd_input["weather_high_f"] = fwd_input["weather_high_f"].fillna(
            train_df["weather_high_f"].median()
        )
        for df_part in [hist_input, fwd_input]:
            df_part["is_bread_delivery_day"] = (
                df_part["is_bread_delivery_day"].fillna(False).astype(int)
            )

        forecast_input = pd.concat(
            [hist_input, fwd_input], ignore_index=True
        ).drop_duplicates("ds")

        forecast = model.predict(forecast_input)

        # ── Accuracy metrics ──────────────────────────────────────────────────
        eval_df = forecast[["ds", "yhat"]].merge(
            train_df[["ds", "y_orders"]], on="ds", how="inner"
        )
        mae  = np.mean(np.abs(eval_df["yhat"] - eval_df["y_orders"]))
        nonz = eval_df[eval_df["y_orders"] > 0]
        mape = np.mean(np.abs(
            (nonz["yhat"] - nonz["y_orders"]) / nonz["y_orders"]
        )) * 100
        rmse = np.sqrt(np.mean((eval_df["yhat"] - eval_df["y_orders"]) ** 2))
        ss_res = np.sum((eval_df["y_orders"] - eval_df["yhat"]) ** 2)
        ss_tot = np.sum((eval_df["y_orders"] - eval_df["y_orders"].mean()) ** 2)
        r2   = 1 - (ss_res / ss_tot)

        mlflow.log_metric("mae",  round(mae,  2))
        mlflow.log_metric("mape", round(mape, 2))
        mlflow.log_metric("rmse", round(rmse, 2))
        mlflow.log_metric("r2",   round(r2,   4))

        # ── Cross-validation ──────────────────────────────────────────────────
        cv_mape = None
        try:
            df_cv = cross_validation(
                model, initial="60 days", period="30 days",
                horizon="14 days", parallel=None,
            )
            df_perf = performance_metrics(df_cv)
            cv_mae  = df_perf["mae"].mean()
            cv_mape = df_perf["mape"].mean() * 100
            cv_rmse = df_perf["rmse"].mean()
            mlflow.log_metric("cv_mae",  round(cv_mae,  2))
            mlflow.log_metric("cv_mape", round(cv_mape, 2))
            mlflow.log_metric("cv_rmse", round(cv_rmse, 2))
            print(f"  Training MAPE: {mape:.1f}%  |  CV MAPE: {cv_mape:.1f}%  |  R²: {r2:.3f}")
        except Exception as e:
            cv_mape = mape
            print(f"  Training MAPE: {mape:.1f}%  |  CV skipped: {e}")

        # ── Forecast chart ────────────────────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.fill_between(
            pd.to_datetime(forecast["ds"]),
            forecast["yhat_lower"].clip(0),
            forecast["yhat_upper"].clip(0),
            alpha=0.2, color="#2E4057", label="Confidence interval"
        )
        ax1.plot(pd.to_datetime(forecast["ds"]), forecast["yhat"].clip(0),
                 color="#2E4057", linewidth=2, label="Forecast")

        excluded = all_historical[all_historical["ds"] < TRAIN_START]
        included = all_historical[all_historical["ds"] >= TRAIN_START]
        ax1.scatter(excluded["ds"], excluded["y_orders"],
                    color="#8D99AE", s=20, alpha=0.5, label="Excluded (pre-Dec)")
        ax1.scatter(included["ds"], included["y_orders"],
                    color="#E84855", s=25, alpha=0.7, label="Training data (Dec+)")
        ax1.axvline(TRAIN_START, color="#06A77D", linewidth=2,
                    linestyle="--", alpha=0.7, label="Training start")

        ax1.set_title(
            f"Prophet Orders — {run_label}\n"
            f"Training MAPE: {mape:.1f}%  |  "
            f"CV MAPE: {cv_mape:.1f}%  |  R²: {r2:.3f}",
            fontsize=12, fontweight="bold"
        )
        ax1.set_ylabel("Daily Order Count (tickets)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha="right")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.set_ylim(-5, max(included["y_orders"].max() * 1.3, 120))
        fig1.tight_layout()

        chart_path = f"/tmp/prophet_orders_{run_label}.png"
        fig1.savefig(chart_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(chart_path)
        plt.close(fig1)

        mlflow.prophet.log_model(
            pr_model = model,
            name     = "prophet_orders_model",
        )

        run_results.append({
            "run_id":         run_id,
            "run_label":      run_label,
            "cv_mape":        cv_mape,
            "mape":           mape,
            "r2":             r2,
            "mae":            mae,
            "params":         params.copy(),
            "model":          model,
            "forecast":       forecast,
            "forecast_input": forecast_input,
        })

    params["run_label"] = run_label

# COMMAND ----------

# ── 10. SELECT AND REGISTER BEST MODEL ───────────────────────────────────────

print("\n" + "="*60)
print("TUNING RESULTS SUMMARY — ORDERS MODEL")
print("="*60)
print(f"  {'Run':<45} {'CV MAPE':>8} {'Train MAPE':>11} {'R²':>6}")
print("  " + "-"*60)

best = min(run_results, key=lambda x: x["cv_mape"])

for r in sorted(run_results, key=lambda x: x["cv_mape"]):
    marker = " ← BEST" if r["run_id"] == best["run_id"] else ""
    print(f"  {r['run_label']:<45} {r['cv_mape']:>7.1f}% "
          f"{r['mape']:>10.1f}% {r['r2']:>6.3f}{marker}")

print(f"\nBest: {best['run_label']}  (CV MAPE: {best['cv_mape']:.1f}%)")

model         = best["model"]
forecast      = best["forecast"]
forecast_input= best["forecast_input"]
run_id        = best["run_id"]
mape          = best["mape"]
r2            = best["r2"]
mae           = best["mae"]
cv_mape       = best["cv_mape"]

with mlflow.start_run(run_id=run_id):
    # Unity Catalog requires a model signature (input + output schema).
    from mlflow.models.signature import infer_signature
    _sig_input  = forecast_input[["ds", "is_bread_delivery_day", "weather_high_f"]].head(10)
    _sig_output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(10)
    signature   = infer_signature(_sig_input, _sig_output)
    mlflow.prophet.log_model(
        pr_model              = model,
        name                  = "prophet_orders_model",
        input_example         = _sig_input.head(5),
        registered_model_name = MODEL_NAME,
        signature             = signature,
    )
    mlflow.set_tag("best_run", "true")
    mlflow.set_tag("selection_criterion", "cv_mape")

    # Immediately promote the new version to @production so notebook 9 picks it up.
    # NOTE (fixed 2026-07-05): the UC model registry rejects search_model_versions'
    # order_by argument (MlflowException, killed the weekly retrain job) — fetch all
    # versions for this model and take the max client-side instead.
    _reg_client   = mlflow.tracking.MlflowClient()
    _new_versions = _reg_client.search_model_versions(f"name='{MODEL_NAME}'")
    if _new_versions:
        _new_ver = max(_new_versions, key=lambda v: int(v.version)).version
        _reg_client.set_registered_model_alias(MODEL_NAME, "production", _new_ver)
        print(f"  ✓ Promoted v{_new_ver} → @production alias")

print(f"\n✓ Best model registered as '{MODEL_NAME}'")
print(f"  View at: Experiments → {EXPERIMENT_NAME}")

# COMMAND ----------

# ── 11. VALIDATION ────────────────────────────────────────────────────────────
# Forecast rows are written by 9_Forecast_Generate.py — nothing to write here.

print("\n── Last 7 days actuals (orders) ──")
spark.sql(f"""
    SELECT
        business_date,
        day_name,
        order_count,
        ROUND(net_revenue, 2)           AS revenue,
        ROUND(net_revenue /
            NULLIF(order_count, 0), 2)  AS implied_avg_ticket,
        weather_high_f,
        weather_category,
        is_bread_delivery_day
    FROM {DAILY_TABLE}
    WHERE business_date BETWEEN
        CURRENT_DATE - INTERVAL 7 DAYS AND
        CURRENT_DATE
    ORDER BY business_date
""").show(10, truncate=False)

print(f"\n── Model summary ──")
print(f"  Training period:   {TRAIN_START.date()} → {train_df['ds'].max().date()}")
print(f"  Training rows:     {len(train_df)}")
print(f"  Avg daily orders:  {train_df['y_orders'].mean():.1f}")
print(f"  MAE:               {mae:.1f} orders/day")
print(f"  MAPE:              {mape:.1f}%")
print(f"  CV MAPE:           {cv_mape:.1f}%")
print(f"  R²:                {r2:.3f}")
print(f"  MLflow run:        {run_id}")
print(f"  Registered model:  {MODEL_NAME}")

# COMMAND ----------

# ── 12. LOG TO ACCURACY HISTORY ───────────────────────────────────────────────

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.gold.forecast_accuracy_history (
        logged_at              TIMESTAMP  COMMENT 'Row write timestamp',
        model_name             STRING     COMMENT 'revenue or orders',
        mlflow_run_id          STRING     COMMENT 'MLflow run ID',
        training_date          DATE       COMMENT 'Date of this retrain',
        training_rows          INT        COMMENT 'Observations in training set',
        training_data_through  DATE       COMMENT 'Last training date',
        train_mape             DOUBLE     COMMENT 'In-sample MAPE %',
        train_mae              DOUBLE     COMMENT 'In-sample MAE',
        train_r2               DOUBLE     COMMENT 'In-sample R²',
        cv_mape                DOUBLE     COMMENT 'Cross-validated MAPE %',
        cv_mae                 DOUBLE     COMMENT 'Cross-validated MAE',
        cv_mape_7d             DOUBLE     COMMENT 'CV MAPE at 7-day horizon %',
        cv_mape_14d            DOUBLE     COMMENT 'CV MAPE at 14-day horizon %',
        backtest_mape          DOUBLE     COMMENT 'Dec-Feb→Mar-Apr MAPE % (revenue only)',
        backtest_mae           DOUBLE     COMMENT 'Dec-Feb→Mar-Apr MAE (revenue only)',
        backtest_ci_coverage   DOUBLE     COMMENT 'CI coverage % in backtest (revenue only)',
        notes                  STRING     COMMENT 'Free-text notes'
    )
    USING DELTA
    COMMENT 'One row per model retrain. Tracks accuracy trends for retraining decisions.'
""")

_today_str   = str(pd.Timestamp.today().date())
_train_max   = str(train_df['ds'].max().date())
_cv_mape_str = str(round(cv_mape, 2)) if cv_mape is not None else "NULL"

spark.sql(f"""
    INSERT INTO {CATALOG}.gold.forecast_accuracy_history
    (logged_at, model_name, mlflow_run_id, training_date, training_rows,
     training_data_through, train_mape, train_mae, train_r2,
     cv_mape, cv_mae, cv_mape_7d, cv_mape_14d,
     backtest_mape, backtest_mae, backtest_ci_coverage, notes)
    VALUES (
        current_timestamp(),
        'orders',
        '{run_id}',
        DATE '{_today_str}',
        {len(train_df)},
        DATE '{_train_max}',
        {round(mape, 2)},
        {round(mae,  2)},
        {round(r2,   4)},
        {_cv_mape_str},
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        '{best["run_label"]}'
    )
""")

print(f"✓ Orders training run logged to {CATALOG}.gold.forecast_accuracy_history")
print(f"  train_mape={round(mape,2)}%  cv_mape={round(cv_mape,2) if cv_mape is not None else 'N/A'}%")