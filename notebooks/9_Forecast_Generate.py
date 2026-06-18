# Databricks notebook source
# MAGIC %md
# MAGIC # Forecast Generation
# MAGIC
# MAGIC Loads the production-aliased Prophet models from the MLflow Model Registry
# MAGIC and writes 30-day forward predictions to `gold.daily_sales_forecast`.
# MAGIC
# MAGIC ## What this notebook does
# MAGIC
# MAGIC 1. Loads `toast_revenue_prophet@production` and `toast_orders_prophet@production`
# MAGIC 2. Builds the forward feature input from `gold.forecast_features`
# MAGIC 3. Generates revenue and order count predictions for the next 30 days
# MAGIC 4. Writes forecast rows to `gold.daily_sales_forecast` (one row per business date)
# MAGIC 5. Labels known future closure dates as `likely_closed`
# MAGIC
# MAGIC ## Run modes
# MAGIC
# MAGIC | Mode | What it does |
# MAGIC |---|---|
# MAGIC | `scheduled` | Generates forecasts for next 30 days — run nightly after Gold sales summary |
# MAGIC | `backfill` | Regenerates all forecast rows from scratch — use after a model update |
# MAGIC
# MAGIC ## Architecture
# MAGIC
# MAGIC Forecasts live in their own table `gold.daily_sales_forecast` — separate from
# MAGIC actuals in `gold.daily_sales_summary`. The Platinum view `platinum.daily_sales_combined`
# MAGIC joins both tables for dashboards, using only the latest forecast run for future dates.
# MAGIC This prevents double-counting when dashboards query without a `record_type` filter.
# MAGIC
# MAGIC ## Schedule
# MAGIC
# MAGIC Add as a final task in `toast_sales_pipeline` job, dependent on `build_gold`.
# MAGIC Runs nightly after actuals are written so forecasts are always fresh.
# MAGIC
# MAGIC ## Dependencies
# MAGIC
# MAGIC **Reads from:**
# MAGIC - `3sp_analytics_workspace.gold.forecast_features`
# MAGIC - `3sp_analytics_workspace.reference.store_events`
# MAGIC - MLflow Model Registry: `toast_revenue_prophet@production`
# MAGIC - MLflow Model Registry: `toast_orders_prophet@production`
# MAGIC
# MAGIC **Writes to:**
# MAGIC - `3sp_analytics_workspace.gold.daily_sales_forecast`
# MAGIC
# MAGIC **Upstream:** `4_Gold_Sales_Summary` and both model notebooks must run first
# MAGIC **Downstream:** `platinum.daily_sales_combined` view reads from this table
# MAGIC
# MAGIC ## Change log
# MAGIC
# MAGIC | Version | Date | Author | Change |
# MAGIC |---|---|---|---|
# MAGIC | v1 | 2026-03-28 | JS | Initial build — wrote to daily_sales_summary |
# MAGIC | v3 | 2026-05-23 | JS | Split ne_seasonal_prior into ne_prior_weekday/weekend to match NB7 v7. Added exponentially-weighted rolling DOW blend (λ=0.75): 50/50 blend for days 1-7, 25/75 for days 8-14, Prophet-only for days 15-30. |
# MAGIC | v4 | 2026-06-18 | JS | Horizon-based bias correction (step 9c). Realized accuracy analysis (n=1,409 pairs) showed consistent over-forecasting of +$161–$362/day across all horizons even after DOW blend. Correction subtracted post-blend. |
| v2 | 2026-03-31 | JS | Retargeted to gold.daily_sales_forecast to prevent duplication |

# COMMAND ----------

# ── 1. INSTALL DEPENDENCIES ───────────────────────────────────────────────────

%pip install prophet mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ── 2. IMPORTS ────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import mlflow
    import mlflow.prophet
    print("✓ MLflow imported")
except ModuleNotFoundError:
    raise RuntimeError("MLflow not found.")

from prophet import Prophet
from pyspark.sql import functions as F
from delta.tables import DeltaTable

print("✓ All libraries loaded")

# COMMAND ----------

# ── 3. WIDGET ─────────────────────────────────────────────────────────────────

dbutils.widgets.dropdown(
    name         = "run_mode",
    defaultValue = "scheduled",
    choices      = ["scheduled", "backfill"],
    label        = "Run Mode"
)

# COMMAND ----------

# ── 4. CONFIGURATION ──────────────────────────────────────────────────────────

RUN_MODE = dbutils.widgets.get("run_mode")

CATALOG          = "3sp_analytics_workspace"
FEATURES_TABLE   = f"{CATALOG}.gold.forecast_features"
FORECAST_TABLE   = f"{CATALOG}.gold.daily_sales_forecast"
ACTUALS_TABLE    = f"{CATALOG}.gold.daily_sales_summary"
WEATHER_TABLE    = f"{CATALOG}.bronze.weather_hourly"

REVENUE_MODEL_URI = "models:/toast_revenue_prophet@production"
ORDERS_MODEL_URI  = "models:/toast_orders_prophet@production"

FORECAST_DAYS = 30

NOW_UTC      = datetime.datetime.now(datetime.timezone.utc)
TODAY        = NOW_UTC.date()
YESTERDAY    = (NOW_UTC - datetime.timedelta(days=1)).date()
FORECAST_END = TODAY + datetime.timedelta(days=FORECAST_DAYS)

print(f"Run mode:     {RUN_MODE}")
print(f"Today:        {TODAY}")
print(f"Forecast end: {FORECAST_END}")

# COMMAND ----------

# ── 5. TABLE SETUP ────────────────────────────────────────────────────────────

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FORECAST_TABLE} (
        business_date           DATE        NOT NULL    COMMENT 'The forecast target date',

        -- Predicted sales metrics
        net_revenue             DOUBLE                  COMMENT 'Predicted net revenue for the day',
        order_count             INTEGER                 COMMENT 'Predicted number of orders for the day',
        forecast_lower          DOUBLE                  COMMENT 'Lower confidence bound from revenue model',
        forecast_upper          DOUBLE                  COMMENT 'Upper confidence bound from revenue model',

        -- Day-of-week features
        day_of_week             INTEGER                 COMMENT 'Day of week: 1=Sunday, 2=Monday ... 7=Saturday',
        day_name                STRING                  COMMENT 'Full day name (Monday, Tuesday, etc.)',
        is_weekend              BOOLEAN                 COMMENT 'True for Saturday and Sunday',
        is_bread_delivery_day   BOOLEAN                 COMMENT 'True on Tuesdays and Fridays',
        week_of_year            INTEGER                 COMMENT 'ISO week number',
        month                   INTEGER                 COMMENT 'Month number (1-12)',
        year                    INTEGER                 COMMENT 'Calendar year',

        -- Weather at time of forecast
        weather_high_f          DOUBLE                  COMMENT 'Forecast high temperature in Fahrenheit',
        weather_category        STRING                  COMMENT 'Forecast weather category: Clear, Cloudy, Rainy, Snowy, Stormy, Foggy',
        weather_condition       STRING                  COMMENT 'Forecast weather condition',
        total_precip_in         DOUBLE                  COMMENT 'Forecast precipitation in inches',

        -- Closure flag
        likely_future_closure   BOOLEAN                 COMMENT 'True if this date is flagged as a likely store closure in reference.store_events',

        -- Model provenance — one row per forecast run per date
        forecast_model          STRING                  COMMENT 'Model name and alias used to generate this forecast (e.g. toast_revenue_prophet@production v3)',
        forecast_model_version  INTEGER                 COMMENT 'Model Registry version number',
        forecast_run_id         STRING                  COMMENT 'MLflow run ID of the model that generated this forecast',
        forecast_created_at     TIMESTAMP               COMMENT 'When this forecast row was generated — use MAX() to get the latest run',
        forecast_horizon_days   INTEGER                 COMMENT 'Days between forecast creation and the predicted date — used for accuracy-by-horizon analysis',

        -- Audit
        _forecast_updated_at    TIMESTAMP               COMMENT 'When this row was last written',
        _batch_id               STRING                  COMMENT 'UUID for this forecast run'
    )
    USING DELTA
    PARTITIONED BY (year, month)
    COMMENT 'Gold: daily revenue and order count forecasts from Prophet models. One row per business date per forecast run. Use MAX(forecast_created_at) to get the latest forecast. Join to gold.daily_sales_summary via platinum.daily_sales_combined.'
    TBLPROPERTIES ('quality' = 'gold', 'delta.enableChangeDataFeed' = 'true')
""")

print(f"✓ {FORECAST_TABLE} ready")

# COMMAND ----------

# ── 6. LOAD PRODUCTION MODELS ─────────────────────────────────────────────────

print("Loading production models from MLflow registry...")

client = mlflow.tracking.MlflowClient()

revenue_model = mlflow.prophet.load_model(REVENUE_MODEL_URI)
revenue_mv    = client.get_model_version_by_alias("toast_revenue_prophet", "production")
revenue_run   = client.get_run(revenue_mv.run_id)
REVENUE_MODEL_VERSION = revenue_mv.version
REVENUE_MODEL_RUN_ID  = revenue_mv.run_id
REVENUE_MODEL_CV_MAPE = revenue_run.data.metrics.get("cv_mape", None)

print(f"✓ Revenue model loaded: {REVENUE_MODEL_URI}")
print(f"  Version:  {REVENUE_MODEL_VERSION}")
print(f"  Run ID:   {REVENUE_MODEL_RUN_ID}")
print(f"  CV MAPE:  {REVENUE_MODEL_CV_MAPE:.1f}%" if REVENUE_MODEL_CV_MAPE else "  CV MAPE: N/A")

orders_model = mlflow.prophet.load_model(ORDERS_MODEL_URI)
orders_mv    = client.get_model_version_by_alias("toast_orders_prophet", "production")
orders_run   = client.get_run(orders_mv.run_id)
ORDERS_MODEL_VERSION  = orders_mv.version
ORDERS_MODEL_RUN_ID   = orders_mv.run_id
ORDERS_MODEL_CV_MAPE  = orders_run.data.metrics.get("cv_mape", None)

print(f"✓ Orders model loaded:  {ORDERS_MODEL_URI}")
print(f"  Version:  {ORDERS_MODEL_VERSION}")
print(f"  Run ID:   {ORDERS_MODEL_RUN_ID}")
print(f"  CV MAPE:  {ORDERS_MODEL_CV_MAPE:.1f}%" if ORDERS_MODEL_CV_MAPE else "  CV MAPE: N/A")

import uuid
FORECAST_CREATED_AT = NOW_UTC
BATCH_ID = str(uuid.uuid4())

# COMMAND ----------

# ── 7. LOAD SPECIAL EVENTS ────────────────────────────────────────────────────

events_ref = spark.sql(f"""
    SELECT event_date, event_name, event_type, lower_window, upper_window
    FROM {CATALOG}.reference.store_events
    WHERE is_active = true
    ORDER BY event_date
""").toPandas()

events_ref['event_date'] = events_ref['event_date'].astype(str)

FUTURE_CLOSURES = {
    row['event_date']: row['event_name']
    for _, row in events_ref.iterrows()
    if row['event_type'] == 'FUTURE_CLOSURE'
}

print(f"✓ Loaded {len(events_ref)} events")
print(f"  Future closures: {list(FUTURE_CLOSURES.keys())}")

# COMMAND ----------

# ── 8. BUILD FORECAST INPUT ───────────────────────────────────────────────────

future_features = spark.sql(f"""
    SELECT
        ds,
        is_bread_delivery_day,
        weather_high_f,
        weather_feels_high_f,
        weather_comfort_score,
        heat_stress,
        total_precip_in,
        likely_future_closure
    FROM {FEATURES_TABLE}
    WHERE y_revenue IS NULL
      AND ds <= '{FORECAST_END}'
    ORDER BY ds
""").toPandas()

future_features['ds'] = pd.to_datetime(future_features['ds'])
future_features['weather_high_f']        = future_features['weather_high_f'].fillna(55.0)
future_features['weather_feels_high_f']  = future_features['weather_feels_high_f'].fillna(55.0)
future_features['weather_comfort_score'] = future_features['weather_comfort_score'].fillna(0.7)
future_features['heat_stress']           = future_features['heat_stress'].fillna(0.0)
future_features['total_precip_in']       = future_features['total_precip_in'].fillna(0.0)
future_features['is_bread_delivery_day'] = (
    future_features['is_bread_delivery_day'].fillna(False).astype(int)
)

# Prior store weekly seasonality index — required by the revenue model regressor.
# Same source data as notebook 7 (PRIOR_MONTHLY_LINEARITY), converted to 52 ISO-week
# weights by distributing each month's proportion evenly across its days then
# summing per week. Must match notebook 7 exactly.
_PRIOR_MONTHLY_LINEARITY = {
    1: 0.065, 2: 0.064, 3: 0.073, 4: 0.075, 5: 0.108, 6: 0.122,
    7: 0.132, 8: 0.126, 9: 0.093, 10: 0.077, 11: 0.061, 12: 0.062,
}
from calendar import monthrange as _monthrange
_REF_YEAR = 2025
_daily_lin = {}
for _m, _prop in _PRIOR_MONTHLY_LINEARITY.items():
    _days = _monthrange(_REF_YEAR, _m)[1]
    for _d in range(1, _days + 1):
        _daily_lin[pd.Timestamp(_REF_YEAR, _m, _d)] = _prop / _days
_weekly_sums: dict = {}
for _dt, _v in _daily_lin.items():
    _w = int(_dt.isocalendar()[1])
    _weekly_sums[_w] = _weekly_sums.get(_w, 0.0) + _v
_wk_total = sum(_weekly_sums.values())
_PRIOR_WEEKLY_LINEARITY = {w: _weekly_sums[w] / _wk_total for w in sorted(_weekly_sums)}
_prior_mean_weekly = np.mean(list(_PRIOR_WEEKLY_LINEARITY.values()))

def _prior_seasonal_index(d):
    d = pd.Timestamp(d)
    iso_week = int(d.isocalendar()[1])
    days_to_thu = 4 - d.isoweekday()
    thu_cur = d + pd.Timedelta(days=days_to_thu)
    if d < thu_cur:
        w_prev   = iso_week - 1 if iso_week > 1 else 52
        thu_prev = thu_cur - pd.Timedelta(weeks=1)
        v1 = (_PRIOR_WEEKLY_LINEARITY.get(w_prev,    _prior_mean_weekly) / _prior_mean_weekly) - 1.0
        v2 = (_PRIOR_WEEKLY_LINEARITY[iso_week] / _prior_mean_weekly) - 1.0
        t  = (d - thu_prev).days / 7
    else:
        w_next = iso_week + 1 if iso_week < 52 else 1
        v1 = (_PRIOR_WEEKLY_LINEARITY[iso_week] / _prior_mean_weekly) - 1.0
        v2 = (_PRIOR_WEEKLY_LINEARITY.get(w_next,    _prior_mean_weekly) / _prior_mean_weekly) - 1.0
        t  = (d - thu_cur).days / 7
    return v1 + (v2 - v1) * t

future_features['ne_seasonal_prior'] = future_features['ds'].apply(_prior_seasonal_index)

# Summer raw index (May-Sep) reaches +0.22 to +0.50 above the annual mean.
# Cap limits how far we extrapolate beyond the training window (all negative/near-zero).
# Raised to 0.20 to let the tourism/snowbird summer pattern through while still
# preventing the full +50% extrapolation (model hasn't observed a summer yet).
_SEASONAL_PRIOR_CAP = 0.20
future_features['ne_seasonal_prior'] = future_features['ne_seasonal_prior'].clip(upper=_SEASONAL_PRIOR_CAP)

# Split into weekend/weekday components — must match NB7 v7 regressor definitions exactly.
# The revenue model was trained with these two regressors; passing the unified ne_seasonal_prior
# would cause a schema mismatch at predict() time.
_is_wknd_fwd = (future_features['ds'].dt.dayofweek >= 5).astype(int)  # 5=Sat, 6=Sun
future_features['ne_prior_weekend'] = future_features['ne_seasonal_prior'] * _is_wknd_fwd
future_features['ne_prior_weekday'] = future_features['ne_seasonal_prior'] * (1 - _is_wknd_fwd)

print(f"✓ Forward feature rows: {len(future_features)}")
print(f"  Date range: {future_features['ds'].min().date()} → {future_features['ds'].max().date()}")
print(f"  Seasonal prior range: {future_features['ne_seasonal_prior'].min():.3f} → {future_features['ne_seasonal_prior'].max():.3f} (cap={_SEASONAL_PRIOR_CAP})")
print(f"  Future closures in window: {future_features['likely_future_closure'].sum()}")

# COMMAND ----------

# ── 9. GENERATE REVENUE FORECAST ──────────────────────────────────────────────

revenue_input    = future_features[['ds', 'ne_prior_weekday', 'ne_prior_weekend',
                                     'is_bread_delivery_day',
                                     'weather_high_f', 'total_precip_in']].copy()
revenue_forecast = revenue_model.predict(revenue_input)
revenue_preds    = revenue_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
revenue_preds['yhat']       = revenue_preds['yhat'].clip(0).round(2)
revenue_preds['yhat_lower'] = revenue_preds['yhat_lower'].clip(0).round(2)
revenue_preds['yhat_upper'] = revenue_preds['yhat_upper'].clip(0).round(2)

print(f"✓ Revenue forecast generated: {len(revenue_preds)} days")
print(f"\n  Sample forecast (next 7 days):")
print(f"  {'Date':<12} {'Day':<10} {'Revenue':>10} {'Lower':>10} {'Upper':>10}")
print(f"  {'-'*52}")
for _, row in revenue_preds.head(7).iterrows():
    day_name = row['ds'].strftime('%A')
    print(f"  {str(row['ds'].date()):<12} {day_name:<10} "
          f"${row['yhat']:>9,.0f} ${row['yhat_lower']:>9,.0f} ${row['yhat_upper']:>9,.0f}")

# COMMAND ----------

# ── 9b. ROLLING DOW BLEND ──────────────────────────────────────────────────────
# Blends the Prophet output with an exponentially-weighted rolling day-of-week mean.
# Short-horizon forecasts (days 1-7) anchor 50% to recent actuals per DOW; this
# self-corrects the Prophet baseline to the current revenue level without retraining.
# Days 8-14 taper to 25% rolling / 75% Prophet. Days 15-30 are Prophet-only.
#
# Lambda=0.75: weight decay per occurrence. Last week = 1.0, 2 weeks ago = 0.75, etc.
# Using last 6 occurrences per DOW (6 weeks of history for each day of week).

_BLEND_LAMBDA = 0.75
_DOW_LOOKBACK_WEEKS = 6

_actuals_dow = spark.sql(f"""
    SELECT
        business_date,
        dayofweek(business_date) AS dow_num,
        net_revenue
    FROM {ACTUALS_TABLE}
    WHERE net_revenue > 0
      AND business_date >= date_sub(current_date(), {_DOW_LOOKBACK_WEEKS * 7})
    ORDER BY business_date
""").toPandas()

_rolling_dow = {}
for _dow in range(1, 8):
    _rows = _actuals_dow[_actuals_dow['dow_num'] == _dow].sort_values('business_date')
    if len(_rows) == 0:
        continue
    _n = len(_rows)
    _w = np.array([_BLEND_LAMBDA ** (_n - 1 - i) for i in range(_n)])
    _w /= _w.sum()
    _rolling_dow[_dow] = float(np.dot(_rows['net_revenue'].values, _w))

_DOW_NAMES = {1: 'Sun', 2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}
print(f"Rolling DOW means (exp-weighted λ={_BLEND_LAMBDA}, last {_DOW_LOOKBACK_WEEKS} weeks):")
for _d, _v in sorted(_rolling_dow.items()):
    print(f"  {_DOW_NAMES[_d]}: ${_v:,.0f}")

# Spark dayofweek: 1=Sun, 2=Mon, ..., 7=Sat
# pandas dayofweek: 0=Mon, ..., 6=Sun  →  map to Spark convention
_pd_to_spark = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 1}
revenue_preds['_dow_num'] = revenue_preds['ds'].dt.dayofweek.map(_pd_to_spark)
revenue_preds['_horizon'] = (revenue_preds['ds'].dt.date.apply(
    lambda d: (d - TODAY).days
))

def _apply_blend(row):
    horizon  = row['_horizon']
    dow_mean = _rolling_dow.get(row['_dow_num'], row['yhat'])
    w_roll   = 0.5 if horizon <= 7 else (0.25 if horizon <= 14 else 0.0)
    if w_roll == 0.0:
        return row['yhat'], row['yhat_lower'], row['yhat_upper']
    blended = w_roll * dow_mean + (1.0 - w_roll) * row['yhat']
    scale   = blended / row['yhat'] if row['yhat'] > 0 else 1.0
    return blended, row['yhat_lower'] * scale, row['yhat_upper'] * scale

_blended = revenue_preds.apply(
    lambda r: pd.Series(_apply_blend(r), index=['yhat', 'yhat_lower', 'yhat_upper']),
    axis=1
)
revenue_preds[['yhat', 'yhat_lower', 'yhat_upper']] = _blended
revenue_preds['yhat']       = revenue_preds['yhat'].clip(0).round(2)
revenue_preds['yhat_lower'] = revenue_preds['yhat_lower'].clip(0).round(2)
revenue_preds['yhat_upper'] = revenue_preds['yhat_upper'].clip(0).round(2)
revenue_preds.drop(columns=['_dow_num', '_horizon'], inplace=True)

print(f"\n  Blended forecast (next 7 days — 50% rolling DOW + 50% Prophet):")
print(f"  {'Date':<12} {'Day':<10} {'Blended':>10} {'Lower':>10} {'Upper':>10}")
print(f"  {'-'*52}")
for _, row in revenue_preds.head(7).iterrows():
    print(f"  {str(row['ds'].date()):<12} {row['ds'].strftime('%A'):<10} "
          f"${row['yhat']:>9,.0f} ${row['yhat_lower']:>9,.0f} ${row['yhat_upper']:>9,.0f}")

# COMMAND ----------

# ── 9c. BIAS CORRECTION ───────────────────────────────────────────────────────
# Realized accuracy analysis (n=1,409 forecast-vs-actual pairs from gold.daily_sales_forecast
# vs gold.daily_sales_summary) showed consistent over-forecasting at every horizon,
# even after the rolling DOW blend in step 9b. The amounts below are the RESIDUAL bias
# remaining after the blend — subtract them to center the forecast distribution.
#
# Horizon  | Observed bias | n
# 1–3 days |        +$161  | 204
# 4–7 days |        +$207  | 247
# 8–14 days|        +$246  | 358
# 15–21 days|       +$292  | 294
# 22–30 days|       +$362  | 306

_HORIZON_BIAS = [
    (3,  160.0),
    (7,  200.0),
    (14, 245.0),
    (21, 290.0),
    (30, 360.0),
]

def _get_bias(horizon_days: int) -> float:
    for cutoff, bias in _HORIZON_BIAS:
        if horizon_days <= cutoff:
            return bias
    return _HORIZON_BIAS[-1][1]

revenue_preds['_horizon_tmp'] = revenue_preds['ds'].dt.date.apply(lambda d: (d - TODAY).days)
revenue_preds['_bias_tmp']    = revenue_preds['_horizon_tmp'].apply(_get_bias)

revenue_preds['yhat']       = (revenue_preds['yhat']       - revenue_preds['_bias_tmp']).clip(0).round(2)
revenue_preds['yhat_lower'] = (revenue_preds['yhat_lower'] - revenue_preds['_bias_tmp']).clip(0).round(2)
revenue_preds['yhat_upper'] = (revenue_preds['yhat_upper'] - revenue_preds['_bias_tmp']).clip(0).round(2)
revenue_preds.drop(columns=['_horizon_tmp', '_bias_tmp'], inplace=True)

print(f"✓ Horizon-based bias correction applied ($160 at 1–3d → $360 at 22–30d)")
print(f"  Bias-corrected forecast (next 7 days):")
print(f"  {'Date':<12} {'Day':<10} {'Revenue':>10} {'Lower':>10} {'Upper':>10}")
print(f"  {'-'*52}")
for _, row in revenue_preds.head(7).iterrows():
    print(f"  {str(row['ds'].date()):<12} {row['ds'].strftime('%A'):<10} "
          f"${row['yhat']:>9,.0f} ${row['yhat_lower']:>9,.0f} ${row['yhat_upper']:>9,.0f}")

# COMMAND ----------

# ── 10. GENERATE ORDERS FORECAST ──────────────────────────────────────────────

orders_input    = future_features[['ds', 'is_bread_delivery_day', 'weather_high_f']].copy()
orders_forecast = orders_model.predict(orders_input)
orders_preds    = orders_forecast[['ds', 'yhat']].copy()
orders_preds['yhat'] = orders_preds['yhat'].clip(0).round(0).astype(int)

print(f"✓ Orders forecast generated: {len(orders_preds)} days")

# COMMAND ----------

# ── 11. COMBINE AND LABEL CLOSURE DATES ───────────────────────────────────────

combined = revenue_preds.merge(
    orders_preds.rename(columns={'yhat': 'order_count_pred'}),
    on='ds', how='left'
)
combined = combined.merge(
    future_features[['ds', 'likely_future_closure', 'weather_high_f', 'total_precip_in']],
    on='ds', how='left'
)
combined['implied_avg_ticket'] = (
    combined['yhat'] / combined['order_count_pred'].replace(0, np.nan)
).round(2)

print(f"\n  Combined forecast summary:")
print(f"  {'Date':<12} {'Day':<10} {'Revenue':>10} {'Orders':>7} "
      f"{'Avg Ticket':>11} {'Closure?':>9}")
print(f"  {'-'*62}")
for _, row in combined.head(14).iterrows():
    day_name = row['ds'].strftime('%A')
    closure  = "CLOSED" if row.get('likely_future_closure') else ""
    print(f"  {str(row['ds'].date()):<12} {day_name:<10} "
          f"${row['yhat']:>9,.0f} {row['order_count_pred']:>7} "
          f"${row['implied_avg_ticket']:>10,.2f} {closure:>9}")

# COMMAND ----------

# ── 12. WRITE TO gold.daily_sales_forecast ────────────────────────────────────
# One row per (business_date). Each nightly run updates the forecast for the
# upcoming 30 days. Older forecast rows are preserved for accuracy tracking —
# forecast_created_at records when each prediction was made, enabling
# accuracy-by-horizon analysis (1-day-out vs 7-day-out vs 14-day-out).

combined_spark = spark.createDataFrame(
    combined[['ds', 'yhat', 'yhat_lower', 'yhat_upper',
              'order_count_pred', 'likely_future_closure',
              'weather_high_f', 'total_precip_in']].rename(columns={
        'ds':                    'business_date',
        'yhat':                  'net_revenue',
        'yhat_lower':            'forecast_lower',
        'yhat_upper':            'forecast_upper',
        'order_count_pred':      'order_count',
        'likely_future_closure': 'likely_future_closure',
        'weather_high_f':        'weather_high_f',
        'total_precip_in':       'total_precip_in',
    })
).withColumn("business_date", F.col("business_date").cast("date")) \
 .withColumn("order_count",   F.col("order_count").cast("integer")) \
 .withColumn("day_of_week",   F.dayofweek(F.col("business_date"))) \
 .withColumn("day_name",      F.date_format(F.col("business_date"), "EEEE")) \
 .withColumn("is_weekend",    F.dayofweek(F.col("business_date")).isin(1, 7)) \
 .withColumn("is_bread_delivery_day", F.dayofweek(F.col("business_date")).isin(3, 6)) \
 .withColumn("week_of_year",  F.weekofyear(F.col("business_date"))) \
 .withColumn("month",         F.month(F.col("business_date"))) \
 .withColumn("year",          F.year(F.col("business_date"))) \
 .withColumn("weather_category",   F.lit(None).cast("string")) \
 .withColumn("weather_condition",  F.lit(None).cast("string")) \
 .withColumn("forecast_model",        F.lit(f"toast_revenue_prophet@production v{REVENUE_MODEL_VERSION}")) \
 .withColumn("forecast_model_version",F.lit(int(REVENUE_MODEL_VERSION))) \
 .withColumn("forecast_run_id",       F.lit(REVENUE_MODEL_RUN_ID)) \
 .withColumn("forecast_created_at",   F.lit(str(FORECAST_CREATED_AT)).cast("timestamp")) \
 .withColumn("forecast_horizon_days", F.datediff(
     F.col("business_date"),
     F.lit(str(TODAY)).cast("date")
 )) \
 .withColumn("_forecast_updated_at",  F.current_timestamp()) \
 .withColumn("_batch_id",             F.lit(BATCH_ID))

# Enrich with weather forecast from Open-Meteo (best available for future dates)
weather_fwd = spark.sql(f"""
    SELECT
        date,
        MAX_BY(weather_category, weather_code)  AS weather_category,
        MAX_BY(weather_condition, weather_code) AS weather_condition
    FROM {WEATHER_TABLE}
    WHERE date > '{YESTERDAY}'
      AND date <= '{FORECAST_END}'
    GROUP BY date
""")

forecast_final = combined_spark \
    .drop("weather_category", "weather_condition") \
    .join(weather_fwd, combined_spark.business_date == weather_fwd.date, how="left") \
    .drop("date")

# MERGE on (business_date, forecast_created_at::date) — one row per target date per
# forecast run. Each nightly run inserts 30 new rows rather than overwriting.
# This preserves the full forecast history needed for accuracy-by-age analysis
# in platinum.forecast_accuracy. ~11k rows/year — negligible storage.
DeltaTable.forName(spark, FORECAST_TABLE).alias("t").merge(
    forecast_final.alias("s"),
    "t.business_date = s.business_date AND CAST(t.forecast_created_at AS DATE) = CAST(s.forecast_created_at AS DATE)"
).whenMatchedUpdateAll(
).whenNotMatchedInsertAll(
).execute()

print(f"✓ Wrote {len(combined)} forecast rows to {FORECAST_TABLE}")

# COMMAND ----------

# ── 13. VALIDATION ────────────────────────────────────────────────────────────

print("\n── Next 14 days forecast ──")
spark.sql(f"""
    SELECT
        f.business_date,
        f.day_name,
        f.order_count,
        ROUND(f.net_revenue, 2)             AS revenue,
        ROUND(f.forecast_lower, 2)          AS lower_bound,
        ROUND(f.forecast_upper, 2)          AS upper_bound,
        ROUND(f.net_revenue /
            NULLIF(f.order_count, 0), 2)    AS avg_ticket,
        f.weather_high_f,
        f.weather_category,
        f.likely_future_closure,
        f.forecast_model
    FROM {FORECAST_TABLE} f
    WHERE f.business_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL 14 DAYS
    ORDER BY f.business_date
""").show(15, truncate=False)

print("\n── Forecast accuracy: predictions vs actuals ──")
spark.sql(f"""
    SELECT
        f.business_date,
        f.day_name,
        f.forecast_horizon_days             AS horizon_days,
        ROUND(f.net_revenue, 0)             AS predicted,
        ROUND(a.net_revenue, 0)             AS actual,
        ROUND(ABS(f.net_revenue
            - a.net_revenue), 0)            AS abs_error,
        ROUND(ABS(f.net_revenue
            - a.net_revenue)
            / NULLIF(a.net_revenue, 0)
            * 100, 1)                       AS pct_error,
        f.forecast_model,
        f.forecast_created_at::DATE         AS forecast_date
    FROM {FORECAST_TABLE} f
    JOIN {ACTUALS_TABLE} a
        ON f.business_date = a.business_date
        AND a.net_revenue > 0
    WHERE f.forecast_created_at IS NOT NULL
    ORDER BY f.business_date DESC, f.forecast_horizon_days
    LIMIT 20
""").show(truncate=False)

print("\n── Avg ticket sanity check ──")
spark.sql(f"""
    SELECT
        business_date,
        day_name,
        order_count,
        ROUND(net_revenue, 0)               AS revenue,
        ROUND(net_revenue /
            NULLIF(order_count, 0), 2)      AS implied_avg_ticket,
        CASE WHEN net_revenue /
            NULLIF(order_count, 0) > 150
            OR net_revenue /
            NULLIF(order_count, 0) < 15
            THEN '⚠ CHECK' ELSE '✓ OK' END  AS ticket_check
    FROM {FORECAST_TABLE}
    WHERE business_date > CURRENT_DATE
    ORDER BY business_date
    LIMIT 14
""").show(truncate=False)
