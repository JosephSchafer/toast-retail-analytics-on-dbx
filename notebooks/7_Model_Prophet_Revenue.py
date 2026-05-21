# Databricks notebook source
# MAGIC %md
# MAGIC # Prophet Model — Revenue Forecast
# MAGIC
# MAGIC Trains a Facebook Prophet model to forecast daily net revenue.
# MAGIC Logs the experiment to MLflow and registers the best model.
# MAGIC
# MAGIC ## What is Prophet?
# MAGIC
# MAGIC Prophet is a forecasting library built by Meta's data science team.
# MAGIC It works by decomposing historical data into three components:
# MAGIC
# MAGIC - **Trend** — the long-term direction of revenue over time
# MAGIC - **Seasonality** — repeating weekly and yearly patterns
# MAGIC - **Events/Holidays** — specific dates that behave differently
# MAGIC
# MAGIC It then adds those components back together to make predictions,
# MAGIC with upper and lower confidence bounds on every forecast date.
# MAGIC
# MAGIC ## Key modeling decisions
# MAGIC
# MAGIC - **Training start: Dec 1 2025** — excludes September-October ramp-up
# MAGIC   period when the store had no liquor license and no advertising.
# MAGIC   Those weeks are not representative of steady-state operations.
# MAGIC
# MAGIC - **Logistic growth with $12,000/day cap** — foot traffic at this
# MAGIC   location is constrained by Cohasset's population and store location.
# MAGIC   Cap set at ~25% above the prior store's peak daily revenue, reflecting
# MAGIC   a better product mix and higher average ticket size.
# MAGIC
# MAGIC - **Custom yearly seasonality from prior store data** — with only
# MAGIC   4 months of our own data we cannot fit a reliable yearly curve.
# MAGIC   We encode the avg 2022-2023 monthly linearity from the prior store
# MAGIC   at this location as a soft prior. Our own data overrides it as months
# MAGIC   accumulate. By next September this prior will be largely superseded.
# MAGIC
# MAGIC - **Weekly seasonality from our own data** — we have enough history
# MAGIC   to fit the weekly pattern reliably. Friday/Saturday peaks are clear.
# MAGIC
# MAGIC ## Dependencies
# MAGIC
# MAGIC **Reads from:**
# MAGIC - `3sp_analytics_workspace.gold.forecast_features`
# MAGIC - `3sp_analytics_workspace.bronze.weather_hourly`
# MAGIC
# MAGIC **Writes to:**
# MAGIC - MLflow experiment: `toast_prophet_revenue`
# MAGIC - MLflow Model Registry: `toast_revenue_prophet`
# MAGIC - `3sp_analytics_workspace.gold.daily_sales_summary` (forecast rows)
# MAGIC
# MAGIC **Upstream:** `6_Feature_Engineering` must run first
# MAGIC **Downstream:** `11_Model_Evaluate_Register` compares this against LightGBM
# MAGIC
# MAGIC ## Change log
# MAGIC
# MAGIC | Version | Date | Author | Change |
# MAGIC |---|---|---|---|
# MAGIC | v6 | 2026-04-29 | JS | ne_seasonal_prior mode: multiplicative→additive. Multiplicative compounded the weekly peak with summer index (May Fri $3k+); additive adds a fixed $/day seasonal bonus instead. prior_scale restored to 5.0 — regularization doesn't help when data is strongly informative. |
| v5 | 2026-04-28 | JS | Smooth seasonal prior: interpolate between month midpoints (15th) instead of step-function per month. Eliminates hard Apr→May forecast jump. |
| v4 | 2026-04-19 | JS | Summer seasonality: switched from Fourier add_seasonality to ne_seasonal_prior multiplicative regressor. Added Dec-Feb→Mar-Apr backtest. |
# MAGIC | v3 | 2026-04-19 | JS | Switched to flat growth — logistic S-curve projected 2-3x actuals for summer. seasonality_prior_scale 5→2, yearly prior_scale 3→1 |
# MAGIC | v2 | 2026-03-27 | JS | Logistic growth cap $12k, custom seasonal prior from prior store, training start Dec 1 |
# MAGIC | v1 | 2026-03-27 | JS | Initial build — linear trend, no prior, full history |

# COMMAND ----------

# ── 1. INSTALL DEPENDENCIES ───────────────────────────────────────────────────
# restartPython() is required after pip install on Databricks serverless —
# the new packages are not available until the Python kernel restarts.

%pip install prophet mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ── 2. IMPORTS ────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import datetime
import json
warnings.filterwarnings('ignore')

try:
    import mlflow
    import mlflow.prophet
    from mlflow.models.signature import infer_signature
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
WEATHER_TABLE  = f"{CATALOG}.bronze.weather_hourly"

EXPERIMENT_NAME = "toast_prophet_revenue"
MODEL_NAME      = "toast_revenue_prophet"

FORECAST_DAYS   = 30

# Training start: Dec 1 2025
# Excludes Sep-Oct ramp-up (no liquor license, no advertising)
TRAIN_START = pd.Timestamp('2025-12-01')

# Daily revenue cap for logistic growth
# Prior store peak ~$9,300/day. We set cap at ~25% above that.
# Foot traffic is location-constrained — cap reflects higher ticket size,
# not more customers than the location can physically support.
REVENUE_CAP = 12000.0

# ── Prior store seasonal shape (avg 2022-2023 monthly linearity) ──────────────
# Source: actual sales data from the prior business at this location.
# These represent the fraction of annual revenue in each month, averaged
# across two stable pre-ownership-change years.
# Used as a soft yearly seasonality prior — our data overrides it month
# by month as we accumulate a full year of observations.
PRIOR_MONTHLY_LINEARITY = {
    1:  0.065,   # January
    2:  0.064,   # February
    3:  0.073,   # March
    4:  0.075,   # April
    5:  0.108,   # May       ↑ tourism season starts
    6:  0.122,   # June      ↑ peak summer tourism
    7:  0.132,   # July      ← peak (tourism + snowbird return)
    8:  0.126,   # August    ↑ late summer
    9:  0.093,   # September ↑ shoulder season
    10: 0.077,   # October
    11: 0.061,   # November
    12: 0.062,   # December
}

# Distribute monthly proportions evenly across days, then sum by ISO week and normalize.
# This produces 52 weekly weights whose shape mirrors the monthly prior but at finer
# resolution — no hard month-boundary steps.
from calendar import monthrange as _monthrange
_REF_YEAR = 2025
_daily_lin = {}
for _m, _prop in PRIOR_MONTHLY_LINEARITY.items():
    _days = _monthrange(_REF_YEAR, _m)[1]
    for _d in range(1, _days + 1):
        _daily_lin[pd.Timestamp(_REF_YEAR, _m, _d)] = _prop / _days
_weekly_sums: dict = {}
for _dt, _v in _daily_lin.items():
    _w = int(_dt.isocalendar()[1])
    _weekly_sums[_w] = _weekly_sums.get(_w, 0.0) + _v
_wk_total = sum(_weekly_sums.values())
PRIOR_WEEKLY_LINEARITY = {w: _weekly_sums[w] / _wk_total for w in sorted(_weekly_sums)}

# ── Prophet hyperparameters ────────────────────────────────────────────────────
# growth='flat': the store has been at a stable plateau since Dec 2025.
# With only 5 months of data, logistic growth was interpreting the natural
# winter→spring seasonal uptick as ongoing exponential S-curve growth,
# then projecting 2-3x actual revenue for summer. Flat growth makes the
# trend a constant baseline; weekly seasonality and the yearly prior
# handle the rest.
PROPHET_PARAMS = {
    "growth":                   "flat",       # plateau since Dec — no trend component
    "changepoint_prior_scale":  0.05,         # kept for compatibility (unused with flat growth)
    "seasonality_prior_scale":  2.0,          # reduced from 5.0 — weekly pattern is reliable
    "holidays_prior_scale":     10.0,         # events can have large effects
    "seasonality_mode":         "multiplicative",
    "yearly_seasonality":       False,        # replaced by ne_seasonal_prior regressor
    "weekly_seasonality":       True,         # reliable from our own data
    "daily_seasonality":        False,
}

# ── Special events ────────────────────────────────────────────────────────────
SPECIAL_EVENTS = {
    '2025-12-13': ('Town Stroll',         'PLANNED_EVENT',      -1, 0),
    '2026-03-02': ('Wine Tasting',        'REVENUE_DISTORTION',  0, 0),
    '2025-12-11': ('Dec event / unknown', 'ORGANIC_EVENT',       0, 0),
}

FUTURE_CLOSURES = {
    '2026-11-26': 'Thanksgiving',
    '2026-12-25': 'Christmas Day',
}

# COMMAND ----------

# ── 4. LOAD AND PREPARE TRAINING DATA ─────────────────────────────────────────

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

# Historical rows: have a revenue value
# Future rows: revenue is null — model will predict these
all_historical = features_df[features_df['y_revenue'].notna()].copy()
future_df      = features_df[features_df['y_revenue'].isna()].copy()

# Training rows: from Dec 1 onward, not excluded, not imputed closure days
# The Dec 1 cutoff removes the unrepresentative ramp-up period
train_df = all_historical[
    (all_historical['ds'] >= TRAIN_START) &
    (~all_historical['exclude_from_training']) &
    (all_historical['training_weight'] > 0.5)
].copy()

print(f"All historical rows:  {len(all_historical)}")
print(f"Training rows (Dec+): {len(train_df)}")
print(f"  Date range: {train_df['ds'].min().date()} → {train_df['ds'].max().date()}")
print(f"Future rows:          {len(future_df)}")
print(f"\nNote: {len(all_historical) - len(train_df)} pre-Dec rows excluded from training")
print(f"(Sep-Oct ramp-up without liquor license or advertising)")

# COMMAND ----------

# ── 5. ADD LOGISTIC GROWTH CAP ────────────────────────────────────────────────
# Prophet's logistic growth requires a 'cap' column on every row.
# We also add a 'floor' of 0 — revenue cannot go negative.
# The cap represents the maximum plausible daily revenue given
# the physical constraints of this location.

train_df['cap']   = REVENUE_CAP
train_df['floor'] = 0.0

future_df['cap']   = REVENUE_CAP
future_df['floor'] = 0.0

print(f"✓ Logistic growth cap set: ${REVENUE_CAP:,.0f}/day")
print(f"  Rationale: ~25% above prior store's peak daily revenue (~$9,300)")
print(f"  Reflects higher ticket size but same location foot traffic constraints")

# COMMAND ----------

# ── 6. BUILD PRIOR SEASONAL REGRESSOR FROM PRIOR STORE DATA ───────────────────
# With only 5 months of our own data, Prophet cannot fit a reliable yearly curve
# via Fourier seasonality — 8 free Fourier coefficients (order=4) extrapolate
# wildly into unobserved months (tested: prior_scale=3 → May $4k+, May $8k+ at 6).
#
# Better approach: treat the prior store's monthly pattern as a REGRESSOR.
#   - Pre-compute one scalar per date: (monthly_linearity / mean_linearity) - 1
#   - Add as a multiplicative regressor with a moderate prior_scale
#   - Model learns ONE coefficient from Dec-Apr data: how strongly this store
#     tracks the prior store's seasonal pattern
#   - That coefficient naturally extrapolates summer without Fourier freedom
#
# Example: if the learned coefficient is 0.5 and July prior index = +0.38,
# the multiplicative contribution is +19% above baseline for July → reasonable.

_mean_weekly = np.mean(list(PRIOR_WEEKLY_LINEARITY.values()))

def prior_seasonal_index(ds):
    # Linear interpolation between ISO-week Thursday midpoints (7-day spans).
    # Finer than monthly midpoints — no visible steps at month boundaries.
    if not isinstance(ds, pd.Timestamp):
        ds = pd.Timestamp(ds)
    iso_week = int(ds.isocalendar()[1])
    days_to_thu = 4 - ds.isoweekday()          # Thursday = isoweekday 4
    thu_cur = ds + pd.Timedelta(days=days_to_thu)
    if ds < thu_cur:                            # before this week's Thursday
        w_prev   = iso_week - 1 if iso_week > 1 else 52
        thu_prev = thu_cur - pd.Timedelta(weeks=1)
        v1 = (PRIOR_WEEKLY_LINEARITY.get(w_prev,   _mean_weekly) / _mean_weekly) - 1.0
        v2 = (PRIOR_WEEKLY_LINEARITY[iso_week] / _mean_weekly) - 1.0
        t  = (ds - thu_prev).days / 7
    else:                                       # Thursday through Saturday
        w_next   = iso_week + 1 if iso_week < 52 else 1
        v1 = (PRIOR_WEEKLY_LINEARITY[iso_week] / _mean_weekly) - 1.0
        v2 = (PRIOR_WEEKLY_LINEARITY.get(w_next,   _mean_weekly) / _mean_weekly) - 1.0
        t  = (ds - thu_cur).days / 7
    return v1 + (v2 - v1) * t

def prior_seasonality(ds_series):
    return pd.Series([prior_seasonal_index(d) for d in pd.to_datetime(ds_series)])

# Attach to all dataframes
all_historical['ne_seasonal_prior'] = prior_seasonality(all_historical['ds']).values
future_df['ne_seasonal_prior']      = prior_seasonality(future_df['ds']).values
train_df['ne_seasonal_prior']       = prior_seasonality(train_df['ds']).values

print("Prior seasonal index (smooth interpolation — 0.0 = annual average):")
print("  Weekly midpoint values (Thursday of each ISO week):")
for week, lin in PRIOR_WEEKLY_LINEARITY.items():
    val  = (lin / _mean_weekly) - 1.0
    bar  = "█" * int(abs(val) * 30)
    sign = "+" if val >= 0 else "-"
    print(f"  Wk {week:02d}  {sign}{abs(val)*100:4.1f}%  {bar}")
print()
print("  Boundary check (Apr 30 → May 1 should be near-identical):")
for ds_str in ['2026-04-28', '2026-04-30', '2026-05-01', '2026-05-03']:
    v = prior_seasonal_index(pd.Timestamp(ds_str))
    print(f"  {ds_str}  {v:+.4f}")

# COMMAND ----------

# ── 7. BUILD PROPHET HOLIDAYS DATAFRAME ───────────────────────────────────────

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
print(f"✓ Holidays: {len(holidays_df)} rows, {holidays_df['holiday'].nunique()} unique events")

# COMMAND ----------

# ── 8. CONFIGURE MLFLOW ───────────────────────────────────────────────────────

current_user = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{current_user}/{EXPERIMENT_NAME}")
print(f"✓ MLflow experiment: {EXPERIMENT_NAME}")
print(f"  View at: Databricks sidebar → Experiments → {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training the Model
# MAGIC
# MAGIC Prophet will fit:
# MAGIC 1. A **logistic trend** bounded by the $12,000/day cap
# MAGIC 2. The **weekly pattern** from our own Dec-March data
# MAGIC 3. A **custom yearly seasonality** from the prior store's seasonal shape
# MAGIC 4. **Holiday/event effects** for Town Stroll and the December window
# MAGIC 5. **Regressors** for bread delivery days, temperature, and precipitation
# MAGIC
# MAGIC After this runs, the forecast chart should show a realistic summer ramp-up
# MAGIC rather than the runaway linear extrapolation from the first attempt.

# COMMAND ----------

# ── 9. BACKTEST: Train Dec-Feb → Evaluate Mar-Apr ────────────────────────────
# With 7 months of actuals, we can do a genuine holdout test.
# Train on Dec 2025 - Feb 28 2026 (winter only).
# Evaluate on Mar 1 - Apr 18 2026 (spring actuals we already have).
# This quantifies how well the yearly_prior extrapolates the seasonal lift
# BEFORE we trust it for the unseen summer months.
#
# Target: MAPE < 25%, CI coverage > 60%.
# If the seasonal prior correctly encodes "spring is busier than winter",
# Mar-Apr predictions should track actuals reasonably well.

BACKTEST_CUTOFF = pd.Timestamp('2026-03-01')

bt_train = train_df[train_df['ds'] < BACKTEST_CUTOFF].copy()
bt_test  = train_df[train_df['ds'] >= BACKTEST_CUTOFF].copy()

print(f"── Backtest: Dec-Feb train → Mar-Apr evaluation ──")
print(f"  Train: {bt_train['ds'].min().date()} → {bt_train['ds'].max().date()} ({len(bt_train)} rows)")
print(f"  Test:  {bt_test['ds'].min().date()} → {bt_test['ds'].max().date()} ({len(bt_test)} rows)")

_bt_model = Prophet(
    growth                  = PROPHET_PARAMS["growth"],
    changepoint_prior_scale = PROPHET_PARAMS["changepoint_prior_scale"],
    seasonality_prior_scale = PROPHET_PARAMS["seasonality_prior_scale"],
    holidays_prior_scale    = PROPHET_PARAMS["holidays_prior_scale"],
    seasonality_mode        = PROPHET_PARAMS["seasonality_mode"],
    yearly_seasonality      = False,
    weekly_seasonality      = True,
    daily_seasonality       = False,
    holidays                = holidays_df,
)
_bt_model.add_regressor('ne_seasonal_prior',     mode='additive', prior_scale=5.0)
_bt_model.add_regressor('is_bread_delivery_day', mode='multiplicative')
_bt_model.add_regressor('weather_high_f',         mode='additive')
_bt_model.add_regressor('total_precip_in',        mode='additive')

# Add prior seasonal index to backtest sets
bt_train['ne_seasonal_prior'] = prior_seasonality(bt_train['ds']).values
bt_test['ne_seasonal_prior']  = prior_seasonality(bt_test['ds']).values

_bt_fit = bt_train[['ds', 'y_revenue', 'cap', 'floor',
                     'ne_seasonal_prior',
                     'is_bread_delivery_day', 'weather_high_f', 'total_precip_in']].rename(
    columns={'y_revenue': 'y'}
).dropna(subset=['ds', 'y'])
for _col in ['weather_high_f', 'total_precip_in']:
    _bt_fit[_col] = _bt_fit[_col].fillna(_bt_fit[_col].median())
_bt_fit['is_bread_delivery_day'] = _bt_fit['is_bread_delivery_day'].fillna(False).astype(int)

_bt_model.fit(_bt_fit[['ds', 'y', 'cap', 'floor',
                        'ne_seasonal_prior',
                        'is_bread_delivery_day', 'weather_high_f', 'total_precip_in']])

_bt_pred_in = bt_test[['ds', 'ne_seasonal_prior',
                        'is_bread_delivery_day', 'weather_high_f', 'total_precip_in']].copy()
for _col in ['weather_high_f', 'total_precip_in']:
    _bt_pred_in[_col] = _bt_pred_in[_col].fillna(bt_train[_col].median())
_bt_pred_in['is_bread_delivery_day'] = _bt_pred_in['is_bread_delivery_day'].fillna(False).astype(int)
_bt_pred_in['cap']   = REVENUE_CAP
_bt_pred_in['floor'] = 0.0

_bt_fc = _bt_model.predict(_bt_pred_in)

_bt_eval = _bt_fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
    bt_test[['ds', 'y_revenue']].rename(columns={'y_revenue': 'actual'}),
    on='ds', how='inner'
)
_bt_eval = _bt_eval[_bt_eval['actual'] > 0].copy()

_bt_mae  = np.mean(np.abs(_bt_eval['yhat'] - _bt_eval['actual']))
_bt_mape = np.mean(np.abs((_bt_eval['yhat'] - _bt_eval['actual']) / _bt_eval['actual'])) * 100
_bt_ci_cover = (
    (_bt_eval['actual'] >= _bt_eval['yhat_lower']) &
    (_bt_eval['actual'] <= _bt_eval['yhat_upper'])
).mean() * 100

_pass = _bt_mape < 25 and _bt_ci_cover >= 55

print(f"\n── Backtest results ──")
print(f"  MAE:         ${_bt_mae:,.0f}/day")
print(f"  MAPE:         {_bt_mape:.1f}%")
print(f"  CI Coverage:  {_bt_ci_cover:.0f}% of actuals within 80% confidence interval")
print(f"\n  {'✓ PASS' if _pass else '⚠ REVIEW'} — {'Seasonal prior extrapolates spring lift well.' if _pass else 'Check prior_scale or seasonality mode.'}")

print(f"\n  {'Date':<12} {'Day':<4} {'Actual':>8} {'Predicted':>10} {'Error%':>8} {'In CI':>6}")
print(f"  {'-'*52}")
for _, _row in _bt_eval.sort_values('ds').iterrows():
    _err = (_row['yhat'] - _row['actual']) / _row['actual'] * 100
    _ci  = "✓" if _row['yhat_lower'] <= _row['actual'] <= _row['yhat_upper'] else "✗"
    _dn  = pd.Timestamp(_row['ds']).strftime("%a")
    print(f"  {str(_row['ds'].date()):<12} {_dn:<4} ${_row['actual']:>6,.0f}  ${_row['yhat']:>7,.0f}  {_err:>+6.1f}%  {_ci:>4}")

# COMMAND ----------

# ── 10. TRAIN PROPHET MODEL ───────────────────────────────────────────────────

with mlflow.start_run(run_name="prophet_revenue_v6_additive_seasonal_prior") as run:

    run_id = run.info.run_id
    print(f"MLflow run started: {run_id}")

    # Log all configuration
    mlflow.log_params(PROPHET_PARAMS)
    mlflow.log_param("training_start",   str(TRAIN_START.date()))
    mlflow.log_param("training_rows",    len(train_df))
    mlflow.log_param("revenue_cap",      REVENUE_CAP)
    mlflow.log_param("forecast_days",    FORECAST_DAYS)
    mlflow.log_param("holiday_count",    len(holidays_df))
    mlflow.log_param("target",           "net_revenue")
    mlflow.log_param("seasonal_prior",   "prior_store_avg_2022_2023")
    mlflow.log_metric("backtest_mape",        round(_bt_mape,     2))
    mlflow.log_metric("backtest_mae",         round(_bt_mae,      2))
    mlflow.log_metric("backtest_ci_coverage", round(_bt_ci_cover, 2))
    mlflow.set_tag("training_date",           str(pd.Timestamp.today().date()))
    mlflow.log_param("training_data_through", str(train_df['ds'].max().date()))

    # ── Instantiate Prophet ───────────────────────────────────────────────────
    model = Prophet(
        growth                  = PROPHET_PARAMS["growth"],
        changepoint_prior_scale = PROPHET_PARAMS["changepoint_prior_scale"],
        seasonality_prior_scale = PROPHET_PARAMS["seasonality_prior_scale"],
        holidays_prior_scale    = PROPHET_PARAMS["holidays_prior_scale"],
        seasonality_mode        = PROPHET_PARAMS["seasonality_mode"],
        yearly_seasonality      = PROPHET_PARAMS["yearly_seasonality"],
        weekly_seasonality      = PROPHET_PARAMS["weekly_seasonality"],
        daily_seasonality       = PROPHET_PARAMS["daily_seasonality"],
        holidays                = holidays_df,
    )

    # ne_seasonal_prior: the prior store's monthly seasonal index used as a regressor.
    # mode='additive': the coefficient is learned in $/day units, not as a fraction of
    # trend. This prevents multiplicative compounding where a high-revenue Friday gets
    # BOTH the Friday weekly boost AND a 21% summer percentage multiplier, which was
    # projecting May Fridays at $3,000+ while April Fridays were running $2,000-2,500.
    # In additive mode the seasonal effect is a flat $/day bonus that doesn't scale with
    # the day's base level — slow Mondays stay slow, busy Saturdays get the same bonus.
    model.add_regressor('ne_seasonal_prior',      mode='additive', prior_scale=5.0)
    model.add_regressor('is_bread_delivery_day',  mode='multiplicative')
    model.add_regressor('weather_high_f',          mode='additive')
    model.add_regressor('total_precip_in',         mode='additive')

    # ── Prepare fit dataframe ─────────────────────────────────────────────────
    fit_df = train_df[['ds', 'y_revenue', 'cap', 'floor',
                        'ne_seasonal_prior',
                        'is_bread_delivery_day',
                        'weather_high_f',
                        'total_precip_in']].rename(
        columns={'y_revenue': 'y'}
    ).dropna(subset=['ds', 'y'])

    for col in ['weather_high_f', 'total_precip_in']:
        fit_df[col] = fit_df[col].fillna(fit_df[col].median())
    fit_df['is_bread_delivery_day'] = fit_df['is_bread_delivery_day'].fillna(False).astype(int)

    print("Fitting Prophet model...")
    model.fit(fit_df[['ds', 'y', 'cap', 'floor',
                       'ne_seasonal_prior',
                       'is_bread_delivery_day',
                       'weather_high_f', 'total_precip_in']])
    print("✓ Model fitted")

    # ── Build forecast input ──────────────────────────────────────────────────
    # Combine ALL historical dates (not just training) plus future dates
    # so the chart shows how the model fits the full history
    hist_input = all_historical[['ds', 'ne_seasonal_prior',
                                  'is_bread_delivery_day',
                                  'weather_high_f', 'total_precip_in']].copy()
    fwd_input  = future_df[['ds', 'ne_seasonal_prior',
                              'is_bread_delivery_day',
                              'weather_high_f', 'total_precip_in']].copy()

    for col in ['weather_high_f', 'total_precip_in']:
        fwd_input[col] = fwd_input[col].fillna(train_df[col].median())
    for df_part in [hist_input, fwd_input]:
        df_part['is_bread_delivery_day'] = df_part['is_bread_delivery_day'].fillna(False).astype(int)

    forecast_input = pd.concat([hist_input, fwd_input], ignore_index=True).drop_duplicates('ds')
    forecast_input['cap']   = REVENUE_CAP
    forecast_input['floor'] = 0.0

    forecast = model.predict(forecast_input)

    # ── Accuracy metrics (on training period only) ────────────────────────────
    eval_df = forecast[['ds', 'yhat']].merge(
        train_df[['ds', 'y_revenue']],
        on='ds', how='inner'
    )
    mae  = np.mean(np.abs(eval_df['yhat'] - eval_df['y_revenue']))
    nonz = eval_df[eval_df['y_revenue'] > 0]
    mape = np.mean(np.abs((nonz['yhat'] - nonz['y_revenue']) / nonz['y_revenue'])) * 100
    rmse = np.sqrt(np.mean((eval_df['yhat'] - eval_df['y_revenue']) ** 2))
    ss_res = np.sum((eval_df['y_revenue'] - eval_df['yhat']) ** 2)
    ss_tot = np.sum((eval_df['y_revenue'] - eval_df['y_revenue'].mean()) ** 2)
    r2   = 1 - (ss_res / ss_tot)

    mlflow.log_metric("mae",  round(mae,  2))
    mlflow.log_metric("mape", round(mape, 2))
    mlflow.log_metric("rmse", round(rmse, 2))
    mlflow.log_metric("r2",   round(r2,   4))

    print(f"\nModel accuracy (training period Dec 1 onward):")
    print(f"  MAE:  ${mae:,.2f}  (average dollar error per day)")
    print(f"  MAPE: {mape:.1f}%  (average % error)")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²:   {r2:.3f}   (1.0 = perfect)")

    # ── Forecast chart ────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.fill_between(pd.to_datetime(forecast['ds']),
                     forecast['yhat_lower'].clip(0),
                     forecast['yhat_upper'].clip(0, REVENUE_CAP),
                     alpha=0.2, color='#2E4057', label='Confidence interval')
    ax1.plot(pd.to_datetime(forecast['ds']), forecast['yhat'].clip(0),
             color='#2E4057', linewidth=2, label='Forecast')

    # Plot actual dots — color-coded by training inclusion
    excluded = all_historical[all_historical['ds'] < TRAIN_START]
    included = all_historical[all_historical['ds'] >= TRAIN_START]
    ax1.scatter(excluded['ds'], excluded['y_revenue'],
                color='#8D99AE', s=20, alpha=0.5, label='Excluded (pre-Dec ramp-up)')
    ax1.scatter(included['ds'], included['y_revenue'],
                color='#E84855', s=25, alpha=0.7, label='Training data (Dec+)')

    # Revenue cap line
    ax1.axhline(REVENUE_CAP, color='#F4A261', linewidth=1.5, linestyle=':',
                label=f'Revenue cap (${REVENUE_CAP:,.0f})')

    # Mark training start
    ax1.axvline(TRAIN_START, color='#06A77D', linewidth=2, linestyle='--',
                label='Training start (Dec 1)', alpha=0.7)

    ax1.set_title('Prophet Revenue Forecast — Logistic Growth with Prior Seasonality',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Net Revenue ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha='right')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim(-500, REVENUE_CAP * 1.15)
    fig1.tight_layout()
    fig1.savefig('/tmp/prophet_revenue_forecast_v2.png', dpi=150, bbox_inches='tight')
    mlflow.log_artifact('/tmp/prophet_revenue_forecast_v2.png')
    plt.show()

    # ── Component chart ───────────────────────────────────────────────────────
    fig2 = model.plot_components(forecast, figsize=(14, 12))
    fig2.suptitle('Prophet Components — Trend, Seasonality, Events',
                  fontsize=13, fontweight='bold', y=1.01)
    fig2.savefig('/tmp/prophet_revenue_components_v2.png', dpi=150, bbox_inches='tight')
    mlflow.log_artifact('/tmp/prophet_revenue_components_v2.png')
    plt.show()

    # ── Log model ─────────────────────────────────────────────────────────────
    # Unity Catalog requires a model signature (input + output schema).
    signature = infer_signature(
        forecast_input[['ds', 'cap', 'floor', 'ne_seasonal_prior',
                         'is_bread_delivery_day',
                         'weather_high_f', 'total_precip_in']],
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
    )
    mlflow.prophet.log_model(
        pr_model              = model,
        artifact_path         = "prophet_revenue_model",
        registered_model_name = MODEL_NAME,
        signature             = signature,
    )

    # Immediately promote the new version to @production so notebook 9 picks it up
    _reg_client  = mlflow.tracking.MlflowClient()
    _new_versions = _reg_client.search_model_versions(
        f"name='{MODEL_NAME}'", order_by=["version_number DESC"], max_results=1
    )
    if _new_versions:
        _new_ver = _new_versions[0].version
        _reg_client.set_registered_model_alias(MODEL_NAME, "production", _new_ver)
        print(f"  ✓ Promoted v{_new_ver} → @production alias")

    print(f"\n✓ Model logged and registered")
    print(f"  Run ID:     {run_id}")
    print(f"  Model name: {MODEL_NAME}")
    print(f"  View at:    Experiments → {EXPERIMENT_NAME}")

# COMMAND ----------

# ── 10. CROSS-VALIDATION ──────────────────────────────────────────────────────
# With ~4 months of training data (Dec-Mar), we have limited room for
# cross-validation folds. We try with conservative settings.

cv_mae = cv_mape = cv_rmse = None
cv_mape_7d = cv_mape_14d = None
print("Running cross-validation...")
with mlflow.start_run(run_id=run_id):
    try:
        df_cv = cross_validation(
            model,
            initial  = '60 days',
            period   = '30 days',
            horizon  = '14 days',
            parallel = None,
        )
        df_perf = performance_metrics(df_cv)
        cv_mae  = df_perf['mae'].mean()
        cv_mape = df_perf['mape'].mean() * 100
        cv_rmse = df_perf['rmse'].mean()

        mlflow.log_metric("cv_mae",  round(cv_mae,  2))
        mlflow.log_metric("cv_mape", round(cv_mape, 2))
        mlflow.log_metric("cv_rmse", round(cv_rmse, 2))

        print(f"\nCross-validation results (honest out-of-sample accuracy):")
        print(f"  CV MAE:   ${cv_mae:,.2f}")
        print(f"  CV MAPE:  {cv_mape:.1f}%")
        print(f"  CV RMSE:  ${cv_rmse:,.2f}")
        print(f"\nNote: CV metrics are higher than training metrics — that is correct.")
        print(f"They reflect real forecast error on data the model had not seen.")

        # Accuracy by horizon chart
        fig3, ax = plt.subplots(figsize=(10, 4))
        horizon_mape = df_cv.copy()
        horizon_mape = horizon_mape[horizon_mape['y'] > 0]
        horizon_mape['abs_pct_err'] = np.abs(
            (horizon_mape['yhat'] - horizon_mape['y']) / horizon_mape['y']
        ) * 100
        horizon_mape['days'] = horizon_mape['horizon'].dt.days
        hm = horizon_mape.groupby('days')['abs_pct_err'].mean().reset_index()

        ax.plot(hm['days'], hm['abs_pct_err'],
                color='#2E4057', linewidth=2.5, marker='o', markersize=5)
        ax.fill_between(hm['days'], hm['abs_pct_err'], alpha=0.1, color='#2E4057')
        ax.axhline(cv_mape, color='#E84855', linestyle='--', linewidth=1.5,
                   label=f'Average MAPE: {cv_mape:.1f}%')
        ax.set_xlabel('Days into forecast horizon')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('Forecast Accuracy by Horizon\n(how accuracy changes as we predict further out)',
                     fontsize=12, fontweight='bold')
        ax.legend()
        fig3.savefig('/tmp/prophet_cv_horizon_v2.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('/tmp/prophet_cv_horizon_v2.png')
        plt.show()

        for _h in [7, 14]:
            _hm_row = hm[hm['days'] == _h]
            if not _hm_row.empty:
                _v = round(float(_hm_row['abs_pct_err'].values[0]), 2)
                mlflow.log_metric(f"cv_mape_{_h}d", _v)
                if _h == 7:  cv_mape_7d  = _v
                if _h == 14: cv_mape_14d = _v

    except Exception as e:
        print(f"Cross-validation skipped: {e}")
        print("This can happen with limited training data. Model is still valid.")

# COMMAND ----------

# ── 11. VALIDATION ────────────────────────────────────────────────────────────
# Forecast rows are written by 9_Forecast_Generate.py — nothing to write here.

print("\n── Last 7 days actuals ──")
spark.sql(f"""
    SELECT
        business_date,
        day_name,
        ROUND(net_revenue, 2)           AS revenue,
        order_count,
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
print(f"  Revenue cap:       ${REVENUE_CAP:,.0f}/day")
print(f"  MAE:               ${mae:,.2f}")
print(f"  MAPE:              {mape:.1f}%")
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

_today_str    = str(pd.Timestamp.today().date())
_train_max    = str(train_df['ds'].max().date())
_cv_mape_str  = str(round(cv_mape,     2)) if cv_mape     is not None else "NULL"
_cv_mae_str   = str(round(cv_mae,      2)) if cv_mae      is not None else "NULL"
_cv_7d_str    = str(cv_mape_7d)            if cv_mape_7d  is not None else "NULL"
_cv_14d_str   = str(cv_mape_14d)           if cv_mape_14d is not None else "NULL"

spark.sql(f"""
    INSERT INTO {CATALOG}.gold.forecast_accuracy_history
    (logged_at, model_name, mlflow_run_id, training_date, training_rows,
     training_data_through, train_mape, train_mae, train_r2,
     cv_mape, cv_mae, cv_mape_7d, cv_mape_14d,
     backtest_mape, backtest_mae, backtest_ci_coverage, notes)
    VALUES (
        current_timestamp(),
        'revenue',
        '{run_id}',
        DATE '{_today_str}',
        {len(train_df)},
        DATE '{_train_max}',
        {round(mape, 2)},
        {round(mae,  2)},
        {round(r2,   4)},
        {_cv_mape_str},
        {_cv_mae_str},
        {_cv_7d_str},
        {_cv_14d_str},
        {round(_bt_mape,    2)},
        {round(_bt_mae,     2)},
        {round(_bt_ci_cover, 2)},
        NULL
    )
""")

print(f"✓ Revenue training run logged to {CATALOG}.gold.forecast_accuracy_history")
print(f"  train_mape={round(mape,2)}%  "
      f"cv_mape={round(cv_mape,2) if cv_mape is not None else 'N/A'}%  "
      f"backtest_mape={round(_bt_mape,2)}%")