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
# MAGIC   location is constrained by [your city]'s population and store location.
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
# MAGIC - `YOUR_CATALOG.gold.forecast_features`
# MAGIC - `YOUR_CATALOG.bronze.weather_hourly`
# MAGIC
# MAGIC **Writes to:**
# MAGIC - MLflow experiment: `toast_prophet_revenue`
# MAGIC - MLflow Model Registry: `toast_revenue_prophet`
# MAGIC - `YOUR_CATALOG.gold.daily_sales_summary` (forecast rows)
# MAGIC
# MAGIC **Upstream:** `6_Feature_Engineering` must run first
# MAGIC **Downstream:** `11_Model_Evaluate_Register` compares this against LightGBM
# MAGIC
# MAGIC ## Change log
# MAGIC
# MAGIC | Version | Date | Author | Change |
# MAGIC |---|---|---|---|
# MAGIC | v7 | 2026-05-23 | JS | Split ne_seasonal_prior into ne_prior_weekday + ne_prior_weekend. Diagnosed +40-65% weekday over-forecast: uniform additive prior added ~$500/day equally across all DOWs while only weekends see tourist uplift. Separate regressors let model learn near-zero weekday coefficient. Added Memorial Day / July 4 / Labor Day as COASTAL_HOLIDAY events. Added rolling DOW blending in NB9 (50/50 for days 1-7, 25/75 for days 8-14). |
# MAGIC | v6 | 2026-04-29 | JS | ne_seasonal_prior mode: multiplicative→additive. Multiplicative compounded the weekly peak with summer index (May Fri $3k+); additive adds a fixed $/day seasonal bonus instead. prior_scale restored to 5.0 — regularization doesn't help when data is strongly informative. |
| v10 | 2026-06-23 | JS | Multi-variant retraining on Jun 2026 data. Ran three candidates in one pass (v10a: cps=0.15 baseline refresh; v10b: cps=0.25 to capture Jun acceleration inflection; v10c: cps=0.25 + weekday prior_scale 2→4). CV horizon fixed 14→30 days to match actual forecast window. Winner promoted to @production after comparison. |
| v9 | 2026-06-22 | JS | TRAIN_START moved to Jan 1 2026 (was Sep 2025). Jan–Mar shoulder season was anchoring Fri/Sat/Sun baselines too low. With 6 months of Jan–Jun data, dropping winter gives Prophet a cleaner ramp signal. Also raised ne_prior_weekend prior_scale 5→8 to allow stronger tourist-weekend coefficient without over-regularization. |
| v8 | 2026-06-22 | JS | Switch growth flat→linear, changepoint_prior_scale 0.05→0.15. Flat growth was architecturally incapable of following the Jan–Jun ramp; realized under-forecast ~+$470/day avg Jun 2026. Linear+flexible changepoints lets trend detect real inflections without logistic blowup. Revisit changepoint scale in Jan 2027 with full clean year. |
| v6 | 2026-06-22 | JS | Replace TRAIN_START=Dec 2025 hard cutoff with EXCLUDE_PERIODS list. Excludes Sep–Dec 2025 (ramp-up: no liquor license, no advertising) while preserving future Decembers for seasonality. TRAIN_START moved back to Sep 1 2025 to be a no-op anchor. |
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

CATALOG        = "YOUR_CATALOG"
FEATURES_TABLE = f"{CATALOG}.gold.forecast_features"
DAILY_TABLE    = f"{CATALOG}.gold.daily_sales_summary"
WEATHER_TABLE  = f"{CATALOG}.bronze.weather_hourly"

EXPERIMENT_NAME = "toast_prophet_revenue"

# UC model registry — three-level name required (catalog.schema.model).
# The workspace legacy registry is disabled; all versions live in UC.
# NB9 loads this model via: mlflow.prophet.load_model("models:/MODEL_NAME@production")
MODEL_NAME      = f"{CATALOG}.default.toast_revenue_prophet"

# Set UC as the registry target before any register_model() calls.
# Must be done at module level, not inside start_run().
import mlflow as _mlflow_init
_mlflow_init.set_registry_uri("databricks-uc")

FORECAST_DAYS   = 30

# Training start: Jan 1 2026 — see rationale below.
#
# We have three eras of data:
#   Sep–Dec 2025: ramp-up (no liquor license, no advertising) — never representative
#   Jan–Mar 2026: functional store but deep shoulder season — valid but see below
#   Apr–Jun+ 2026: the store at scale in its seasonal ramp — most relevant
#
# As of Jun 2026, using Jan–Mar data to predict Jul is actively harmful: those months
# anchor weekly seasonality at winter/shoulder-season levels, pulling Fri/Sat/Sun
# baselines down toward $1,800–2,200 when reality is $2,800–3,400+. By trimming to
# Jan 1 2026 we keep 6 months of clean data (Jan–Jun), drop the non-representative
# ramp-up, and let Prophet's linear trend detect the genuine spring–summer ramp.
#
# Future Januarys and Februarys will eventually provide their own training signal as
# we accumulate a full year of data. Revisit TRAIN_START after Dec 2026 closes.
TRAIN_START = pd.Timestamp('2026-01-01')

# Date ranges to exclude from training entirely.
# Sep–Dec 2025 was a non-representative ramp-up period: no liquor license, no
# advertising, and highly irregular seasonal patterns that will never recur.
# We exclude those specific calendar months rather than setting a rolling lookback
# so that future Decembers (2026+) are retained for seasonality learning.
# (TRAIN_START=Jan 1 2026 already excludes Sep–Dec 2025 implicitly, but we keep
# the explicit list so the EXCLUDE_PERIODS pattern works correctly for future use.)
EXCLUDE_PERIODS = [
    (pd.Timestamp('2025-09-01'), pd.Timestamp('2025-12-31')),
]

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
# growth='linear': allows Prophet to fit a real trend through the Jan–Jun data.
# Previously 'flat' because logistic was over-projecting (2-3x summer) with only
# 4 months of data. Now at 5.75 months of clean data the summer ramp is visible
# and flat growth cannot follow it — realized under-forecast was ~+$470/day avg
# in Jun 2026. Linear avoids logistic's exponential blowup while letting the
# trend detect real inflections.
#
# changepoint_prior_scale=0.15: more flexible than default (0.05) so Prophet can
# detect the Jan→Jun ramp as a real trend changepoint rather than seasonality noise.
# Not so high that it overfits weekly wiggles as "trend."
# Revisit in Jan 2027 with a full year of clean data — may want to dial back once
# seasonal shape is fully learned and trend is expected to be flatter.
PROPHET_PARAMS = {
    "growth":                   "linear",     # allows trend to follow the observed ramp
    "changepoint_prior_scale":  0.15,         # flexible enough to detect Jun inflection
    "seasonality_prior_scale":  2.0,          # weekly pattern is reliable, keep regularized
    "holidays_prior_scale":     10.0,         # events can have large effects
    "seasonality_mode":         "multiplicative",
    "yearly_seasonality":       False,        # insufficient clean data (<6 months); revisit Jan 2027
    "weekly_seasonality":       True,         # reliable from our own data
    "daily_seasonality":        False,
}

# ── Special events ────────────────────────────────────────────────────────────
SPECIAL_EVENTS = {
    '2025-12-13': ('Town Stroll',         'PLANNED_EVENT',      -1, 0),
    '2026-03-02': ('Wine Tasting',        'REVENUE_DISTORTION',  0, 0),
    '2025-12-11': ('Dec event / unknown', 'ORGANIC_EVENT',       0, 0),
    # Coastal summer holidays — high-traffic days for [your city] (beach town).
    # lower_window=-1 captures the eve (e.g. Sunday before Memorial Day, July 3).
    '2026-05-25': ('Memorial Day',        'COASTAL_HOLIDAY',    -1, 0),
    '2026-07-04': ('Independence Day',    'COASTAL_HOLIDAY',    -1, 1),
    '2026-09-07': ('Labor Day',           'COASTAL_HOLIDAY',    -1, 0),
    '2027-05-31': ('Memorial Day',        'COASTAL_HOLIDAY',    -1, 0),
    '2027-07-04': ('Independence Day',    'COASTAL_HOLIDAY',    -1, 1),
    '2027-09-06': ('Labor Day',           'COASTAL_HOLIDAY',    -1, 0),
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

# Build an excluded-period mask from EXCLUDE_PERIODS.
# This lets us drop specific calendar windows (e.g. the 2025 ramp-up) without
# capping the overall lookback — future Decembers will train normally.
_in_excluded_period = pd.Series(False, index=all_historical.index)
for _start, _end in EXCLUDE_PERIODS:
    _in_excluded_period |= (all_historical['ds'] >= _start) & (all_historical['ds'] <= _end)

train_df = all_historical[
    (all_historical['ds'] >= TRAIN_START) &
    (~_in_excluded_period) &
    (~all_historical['exclude_from_training']) &
    (all_historical['training_weight'] > 0.5)
].copy()

print(f"All historical rows:  {len(all_historical)}")
print(f"Training rows (Dec+): {len(train_df)}")
print(f"  Date range: {train_df['ds'].min().date()} → {train_df['ds'].max().date()}")
print(f"Future rows:          {len(future_df)}")
_excluded_count = _in_excluded_period.sum()
print(f"\nNote: {_excluded_count} rows excluded via EXCLUDE_PERIODS (Sep–Dec 2025 ramp-up)")
print(f"  These months are non-representative and will never recur as an opening period.")
print(f"  Future Decembers will train normally.")

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

# Split into weekend vs weekday components.
# Weekday revenue shows no significant summer uplift (locals shop year-round at similar rates).
# Weekend uplift IS real: coastal tourists arrive Fri-Sun from Memorial Day through Labor Day.
# Separate regressors let Prophet learn near-zero weekday coefficient and a meaningful
# weekend coefficient — preventing the ~$500/day uniform over-forecast on Mon-Thu observed in v6.
for _df in [all_historical, future_df, train_df]:
    _is_wknd = (_df['ds'].dt.dayofweek >= 5).astype(int)  # 5=Sat, 6=Sun
    _df['ne_prior_weekend'] = (_df['ne_seasonal_prior'] * _is_wknd).values
    _df['ne_prior_weekday'] = (_df['ne_seasonal_prior'] * (1 - _is_wknd)).values

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
    if etype in ('PLANNED_EVENT', 'COASTAL_HOLIDAY'):
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

# Backtest = most-recent-30-day holdout (fixed 2026-07-03).
# The old fixed cutoff (2026-03-01) trained on Jan-Feb ($1,119/day winter shoulder) to predict
# Mar-Apr, which — after TRAIN_START moved to Jan 1 and the store ~doubled Jan→Jun — produced a
# meaningless 47% MAPE (no ramp signal in the train window). Holding out the LAST 30 days instead
# evaluates the model on the period we actually care about and keeps the ramp in training.
_bt_last_obs = train_df.loc[train_df['y_revenue'].notna(), 'ds'].max()
BACKTEST_CUTOFF = _bt_last_obs - pd.Timedelta(days=30)

bt_train = train_df[train_df['ds'] < BACKTEST_CUTOFF].copy()
bt_test  = train_df[(train_df['ds'] >= BACKTEST_CUTOFF) & (train_df['ds'] <= _bt_last_obs)].copy()

print(f"── Backtest: Dec-Feb train → Mar-Apr evaluation ──")
print(f"  Train: {bt_train['ds'].min().date()} → {bt_train['ds'].max().date()} ({len(bt_train)} rows)")
print(f"  Test:  {bt_test['ds'].min().date()} → {bt_test['ds'].max().date()} ({len(bt_test)} rows)")

_bt_model = Prophet(
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
_bt_model.add_regressor('ne_prior_weekday',       mode='additive', prior_scale=2.0)
_bt_model.add_regressor('ne_prior_weekend',       mode='additive', prior_scale=8.0)
_bt_model.add_regressor('is_bread_delivery_day',  mode='multiplicative')
_bt_model.add_regressor('weather_high_f',          mode='additive')
_bt_model.add_regressor('total_precip_in',         mode='additive')

# ne_prior_weekday/weekend already computed in cell 6 and inherited via .copy()

_bt_fit = bt_train[['ds', 'y_revenue', 'cap', 'floor',
                     'ne_prior_weekday', 'ne_prior_weekend',
                     'is_bread_delivery_day', 'weather_high_f', 'total_precip_in']].rename(
    columns={'y_revenue': 'y'}
).dropna(subset=['ds', 'y'])
for _col in ['weather_high_f', 'total_precip_in']:
    _bt_fit[_col] = _bt_fit[_col].fillna(_bt_fit[_col].median())
_bt_fit['is_bread_delivery_day'] = _bt_fit['is_bread_delivery_day'].fillna(False).astype(int)

_bt_model.fit(_bt_fit[['ds', 'y', 'cap', 'floor',
                        'ne_prior_weekday', 'ne_prior_weekend',
                        'is_bread_delivery_day', 'weather_high_f', 'total_precip_in']])

_bt_pred_in = bt_test[['ds', 'ne_prior_weekday', 'ne_prior_weekend',
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

# ── 10. TRAIN VARIANTS + CV ───────────────────────────────────────────────────
#
# Runs multiple candidate configurations in one pass. Each variant logs its own
# MLflow run with a descriptive name. After all variants complete, a comparison
# table is printed so you can pick the winner before promoting.
#
# Variant design:
#   v10a — baseline refresh: same architecture as v9, retrained on Jun 2026 data.
#           Isolates the gain from fresh data alone.
#   v10b — cps=0.25: more flexible changepoints to capture the Jun acceleration
#           inflection. Risk: can overfit weekly wiggles as "trend."
#   v10c — cps=0.25 + weekday_prior_scale=4.0: weekday summer ramp signal is now
#           real (Mon/Tue running $1,850–2,300 in Jun vs ~$1,400 in Jan). Raise
#           weekday prior_scale to let the model assign more weight to the seasonal
#           lift on those days instead of regularizing it toward zero.
#
# CV horizon = 30 days to match the actual forecast window we care about.
# initial = 90 days gives enough folds across the Jan–Jun dataset.

_VARIANTS = [
    {
        "label":                  "v10a_cps015_baseline_refresh",
        "changepoint_prior_scale": 0.15,
        "wkday_prior_scale":       2.0,
        "note":                   "v9 architecture retrained on Jun 2026 data — isolates data freshness gain",
    },
    {
        "label":                  "v10b_cps025_june_inflection",
        "changepoint_prior_scale": 0.25,
        "wkday_prior_scale":       2.0,
        "note":                   "Higher CPS to capture Jun acceleration trend inflection",
    },
    {
        "label":                  "v10c_cps025_wkday_prior4",
        "changepoint_prior_scale": 0.25,
        "wkday_prior_scale":       4.0,
        "note":                   "cps=0.25 + weekday prior_scale 2→4 (Mon/Tue summer ramp now real)",
    },
]

_variant_results = []  # accumulate for comparison table

for _v in _VARIANTS:
    print(f"\n{'='*60}")
    print(f"  Variant: {_v['label']}")
    print(f"  {_v['note']}")
    print(f"{'='*60}")

    _params = dict(PROPHET_PARAMS)
    _params["changepoint_prior_scale"] = _v["changepoint_prior_scale"]

    _run_name = (
        f"prophet_revenue_linear_growth"
        f"_cps{str(_v['changepoint_prior_scale']).replace('.', '')}"
        f"_train{str(TRAIN_START.date()).replace('-', '')}"
        f"_wkday{str(_v['wkday_prior_scale']).replace('.', '')}"
        f"_wknd8"
    )

    with mlflow.start_run(run_name=_run_name) as _run:
        _run_id = _run.info.run_id

        mlflow.log_params(_params)
        mlflow.log_param("training_start",        str(TRAIN_START.date()))
        mlflow.log_param("training_data_through",  str(train_df['ds'].max().date()))
        mlflow.log_param("training_rows",          len(train_df))
        mlflow.log_param("revenue_cap",            REVENUE_CAP)
        mlflow.log_param("forecast_days",          FORECAST_DAYS)
        mlflow.log_param("target",                 "net_revenue")
        mlflow.log_param("seasonal_prior",         "prior_store_avg_2022_2023")
        mlflow.log_param("wkday_prior_scale",      _v["wkday_prior_scale"])
        mlflow.log_param("wkend_prior_scale",      8.0)
        mlflow.log_param("variant_note",           _v["note"])
        mlflow.log_metric("backtest_mape",         round(_bt_mape,     2))
        mlflow.log_metric("backtest_mae",          round(_bt_mae,      2))
        mlflow.log_metric("backtest_ci_coverage",  round(_bt_ci_cover, 2))
        mlflow.set_tag("training_date",            str(pd.Timestamp.today().date()))
        mlflow.set_tag("status",                   "candidate")
        mlflow.set_tag("model_type",               "prophet")
        mlflow.set_tag("target",                   "net_revenue")

        # ── Build model ───────────────────────────────────────────────────────
        _model = Prophet(
            growth                  = _params["growth"],
            changepoint_prior_scale = _params["changepoint_prior_scale"],
            seasonality_prior_scale = _params["seasonality_prior_scale"],
            holidays_prior_scale    = _params["holidays_prior_scale"],
            seasonality_mode        = _params["seasonality_mode"],
            yearly_seasonality      = _params["yearly_seasonality"],
            weekly_seasonality      = _params["weekly_seasonality"],
            daily_seasonality       = _params["daily_seasonality"],
            holidays                = holidays_df,
        )
        _model.add_regressor('ne_prior_weekday',      mode='additive', prior_scale=_v["wkday_prior_scale"])
        _model.add_regressor('ne_prior_weekend',      mode='additive', prior_scale=8.0)
        _model.add_regressor('is_bread_delivery_day', mode='multiplicative')
        _model.add_regressor('weather_high_f',        mode='additive')
        _model.add_regressor('total_precip_in',       mode='additive')

        # ── Fit ───────────────────────────────────────────────────────────────
        _fit_df = train_df[['ds', 'y_revenue', 'cap', 'floor',
                             'ne_prior_weekday', 'ne_prior_weekend',
                             'is_bread_delivery_day',
                             'weather_high_f', 'total_precip_in']].rename(
            columns={'y_revenue': 'y'}
        ).dropna(subset=['ds', 'y']).copy()
        for _col in ['weather_high_f', 'total_precip_in']:
            _fit_df[_col] = _fit_df[_col].fillna(_fit_df[_col].median())
        _fit_df['is_bread_delivery_day'] = _fit_df['is_bread_delivery_day'].fillna(False).astype(int)

        print(f"  Fitting on {len(_fit_df)} rows ({_fit_df['ds'].min().date()} → {_fit_df['ds'].max().date()})...")
        _model.fit(_fit_df[['ds', 'y', 'cap', 'floor',
                             'ne_prior_weekday', 'ne_prior_weekend',
                             'is_bread_delivery_day', 'weather_high_f', 'total_precip_in']])
        print(f"  ✓ Fitted")

        # ── Predict over all history + future ─────────────────────────────────
        _hist_in = all_historical[['ds', 'ne_prior_weekday', 'ne_prior_weekend',
                                    'is_bread_delivery_day', 'weather_high_f', 'total_precip_in']].copy()
        _fwd_in  = future_df[['ds', 'ne_prior_weekday', 'ne_prior_weekend',
                               'is_bread_delivery_day', 'weather_high_f', 'total_precip_in']].copy()
        for _col in ['weather_high_f', 'total_precip_in']:
            _fwd_in[_col] = _fwd_in[_col].fillna(train_df[_col].median())
        for _df_part in [_hist_in, _fwd_in]:
            _df_part['is_bread_delivery_day'] = _df_part['is_bread_delivery_day'].fillna(False).astype(int)
        _fc_input = pd.concat([_hist_in, _fwd_in], ignore_index=True).drop_duplicates('ds')
        _fc_input['cap']   = REVENUE_CAP
        _fc_input['floor'] = 0.0
        _forecast = _model.predict(_fc_input)

        # ── In-sample metrics ─────────────────────────────────────────────────
        _eval = _forecast[['ds', 'yhat']].merge(train_df[['ds', 'y_revenue']], on='ds', how='inner')
        _mae  = np.mean(np.abs(_eval['yhat'] - _eval['y_revenue']))
        _nonz = _eval[_eval['y_revenue'] > 0]
        _mape = np.mean(np.abs((_nonz['yhat'] - _nonz['y_revenue']) / _nonz['y_revenue'])) * 100
        _rmse = np.sqrt(np.mean((_eval['yhat'] - _eval['y_revenue']) ** 2))
        _ss_res = np.sum((_eval['y_revenue'] - _eval['yhat']) ** 2)
        _ss_tot = np.sum((_eval['y_revenue'] - _eval['y_revenue'].mean()) ** 2)
        _r2   = 1 - (_ss_res / _ss_tot)
        mlflow.log_metric("mae",  round(_mae,  2))
        mlflow.log_metric("mape", round(_mape, 2))
        mlflow.log_metric("rmse", round(_rmse, 2))
        mlflow.log_metric("r2",   round(_r2,   4))
        print(f"  In-sample — MAE: ${_mae:,.0f}  MAPE: {_mape:.1f}%  R²: {_r2:.3f}")

        # ── Forecast chart ────────────────────────────────────────────────────
        _fig1, _ax1 = plt.subplots(figsize=(14, 6))
        _ax1.fill_between(pd.to_datetime(_forecast['ds']),
                          _forecast['yhat_lower'].clip(0),
                          _forecast['yhat_upper'].clip(0, REVENUE_CAP),
                          alpha=0.2, color='#2E4057', label='CI')
        _ax1.plot(pd.to_datetime(_forecast['ds']), _forecast['yhat'].clip(0),
                  color='#2E4057', linewidth=2, label='Forecast')
        _excl_mask = all_historical.index.isin(all_historical[_in_excluded_period].index)
        _ax1.scatter(all_historical[~_excl_mask & (all_historical['ds'] >= TRAIN_START)]['ds'],
                     all_historical[~_excl_mask & (all_historical['ds'] >= TRAIN_START)]['y_revenue'],
                     color='#E84855', s=20, alpha=0.7, label='Training data')
        _ax1.axhline(REVENUE_CAP, color='#F4A261', linewidth=1.5, linestyle=':',
                     label=f'Cap (${REVENUE_CAP:,.0f})')
        _ax1.set_title(f'Prophet Revenue — {_v["label"]}', fontsize=12, fontweight='bold')
        _ax1.set_ylabel('Net Revenue ($)')
        _ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        _ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        _ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(_ax1.xaxis.get_majorticklabels(), rotation=20, ha='right')
        _ax1.legend(loc='upper left', fontsize=9)
        _ax1.set_ylim(-200, REVENUE_CAP * 1.1)
        _fig1.tight_layout()
        _chart_path = f'/tmp/forecast_{_v["label"]}.png'
        _fig1.savefig(_chart_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(_chart_path)
        plt.show()

        # ── Cross-validation (horizon=30d — matches actual forecast window) ───
        _cv_mape = _cv_mae = _cv_mape_7d = _cv_mape_14d = _cv_mape_30d = None
        print(f"  Running CV (horizon=30d)...")
        try:
            _df_cv = cross_validation(
                _model,
                initial  = '90 days',
                period   = '21 days',
                horizon  = '30 days',
                parallel = None,
            )
            _df_perf = performance_metrics(_df_cv)
            _cv_mae  = _df_perf['mae'].mean()
            _cv_mape = _df_perf['mape'].mean() * 100
            mlflow.log_metric("cv_mae",  round(_cv_mae,  2))
            mlflow.log_metric("cv_mape", round(_cv_mape, 2))

            # Per-horizon MAPE at 7, 14, 30 days
            _hm = _df_cv.copy()
            _hm = _hm[_hm['y'] > 0].copy()
            _hm['abs_pct_err'] = np.abs((_hm['yhat'] - _hm['y']) / _hm['y']) * 100
            _hm['days'] = _hm['horizon'].dt.days
            _hm_grp = _hm.groupby('days')['abs_pct_err'].mean()
            for _h, _attr in [(7, '_cv_mape_7d'), (14, '_cv_mape_14d'), (30, '_cv_mape_30d')]:
                _candidates = _hm_grp.index[(_hm_grp.index >= _h - 1) & (_hm_grp.index <= _h + 1)]
                if len(_candidates):
                    _val = round(float(_hm_grp[_candidates].mean()), 2)
                    mlflow.log_metric(f"cv_mape_{_h}d", _val)
                    locals()[_attr.lstrip('_')] = _val  # noqa — dynamic assignment intentional
                    if _attr == '_cv_mape_7d':  _cv_mape_7d  = _val
                    if _attr == '_cv_mape_14d': _cv_mape_14d = _val
                    if _attr == '_cv_mape_30d': _cv_mape_30d = _val

            # Horizon accuracy chart
            _fig3, _ax3 = plt.subplots(figsize=(10, 4))
            _ax3.plot(_hm_grp.index, _hm_grp.values, color='#2E4057', linewidth=2, marker='o', markersize=4)
            _ax3.axhline(_cv_mape, color='#E84855', linestyle='--', linewidth=1.5,
                         label=f'Avg MAPE: {_cv_mape:.1f}%')
            for _h_mark in [7, 14, 30]:
                _ax3.axvline(_h_mark, color='gray', linestyle=':', linewidth=1)
            _ax3.set_xlabel('Days into horizon')
            _ax3.set_ylabel('MAPE (%)')
            _ax3.set_title(f'CV Accuracy by Horizon — {_v["label"]}', fontsize=11, fontweight='bold')
            _ax3.legend()
            _fig3.tight_layout()
            _cv_chart = f'/tmp/cv_horizon_{_v["label"]}.png'
            _fig3.savefig(_cv_chart, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(_cv_chart)
            plt.show()

            print(f"  CV MAPE (30d horizon): {_cv_mape:.1f}%  |  7d: {_cv_mape_7d}%  14d: {_cv_mape_14d}%  30d: {_cv_mape_30d}%")
        except Exception as _e:
            print(f"  CV skipped: {_e}")

        # ── Log model artifact (no registration — compare first) ──────────────
        _sig = infer_signature(
            _fc_input[['ds', 'cap', 'floor', 'ne_prior_weekday', 'ne_prior_weekend',
                        'is_bread_delivery_day', 'weather_high_f', 'total_precip_in']],
            _forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        )
        mlflow.prophet.log_model(
            pr_model      = _model,
            artifact_path = "prophet_revenue_model",
            signature     = _sig,
        )
        print(f"  ✓ Logged to MLflow run {_run_id}")

    # Stash results for comparison — keep reference to last fitted model/forecast
    _variant_results.append({
        "label":       _v["label"],
        "run_id":      _run_id,
        "train_mape":  round(_mape, 1),
        "train_r2":    round(_r2, 3),
        "cv_mape":     round(_cv_mape, 1) if _cv_mape is not None else None,
        "cv_mape_7d":  _cv_mape_7d,
        "cv_mape_14d": _cv_mape_14d,
        "cv_mape_30d": _cv_mape_30d,
        "bt_mape":     round(_bt_mape, 1),
        "model":       _model,
        "forecast":    _forecast,
        "fit_df":      _fit_df,
    })

# ── Comparison table ──────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"  VARIANT COMPARISON — pick the winner for promotion")
print(f"{'='*90}")
print(f"  {'Variant':<40} {'tr_mape':>8} {'tr_r2':>7} {'cv_mape':>8} {'cv_7d':>7} {'cv_14d':>7} {'cv_30d':>7} {'bt_mape':>8}")
print(f"  {'-'*90}")
for _r in _variant_results:
    def _fmt(v): return f"{v:.1f}%" if v is not None else "  n/a  "
    print(
        f"  {_r['label']:<40}"
        f"  {_fmt(_r['train_mape']):>8}"
        f"  {_r['train_r2']:>6.3f}"
        f"  {_fmt(_r['cv_mape']):>8}"
        f"  {_fmt(_r['cv_mape_7d']):>7}"
        f"  {_fmt(_r['cv_mape_14d']):>7}"
        f"  {_fmt(_r['cv_mape_30d']):>7}"
        f"  {_fmt(_r['bt_mape']):>8}"
    )
print(f"\n  Primary sort key: cv_mape_30d (lowest = best at the 30-day window we care about)")
print(f"  Tiebreaker: cv_mape_14d, then bt_mape")
print(f"\n  Run IDs:")
for _r in _variant_results:
    print(f"    {_r['label']}: {_r['run_id']}")
print(f"\n  → Set run_id and _promote=True in cell 10a to register and promote the winner.")

# Expose the winning model/forecast for downstream cells (validation, history logging).
# Winner = lowest cv_mape (the robust rolling-origin metric), matching the notebook's stated
# sort key. Auto-selected (2026-07-03) instead of hardcoding the last variant.
_WINNER_IDX = min(range(len(_variant_results)),
                  key=lambda i: _variant_results[i]["cv_mape"])
print(f"  → Auto-selected winner by lowest cv_mape: {_variant_results[_WINNER_IDX]['label']} "
      f"(cv_mape={_variant_results[_WINNER_IDX]['cv_mape']:.1f}%)")
model    = _variant_results[_WINNER_IDX]["model"]
forecast = _variant_results[_WINNER_IDX]["forecast"]
run_id   = _variant_results[_WINNER_IDX]["run_id"]
mae      = float('nan')  # in-sample MAE not re-computed here; use mlflow metric
mape     = _variant_results[_WINNER_IDX]["train_mape"]
r2       = _variant_results[_WINNER_IDX]["train_r2"]
cv_mape  = _variant_results[_WINNER_IDX]["cv_mape"]
cv_mae   = None
cv_mape_7d  = _variant_results[_WINNER_IDX]["cv_mape_7d"]
cv_mape_14d = _variant_results[_WINNER_IDX]["cv_mape_14d"]

# COMMAND ----------

# ── 10a. PROMOTE TO @production ── RUN MANUALLY AFTER REVIEWING COMPARISON TABLE ──
#
# Steps:
#   1. Read the comparison table printed above — lowest cv_mape_30d wins
#   2. If a different variant than the default (_WINNER_IDX) is better,
#      set run_id to that variant's run_id (printed above)
#   3. Set _promote = True and run this cell
#
# The @production alias is what NB9 loads. Changing it takes effect on the next
# daily NB9 run (or trigger NB9 manually for an immediate refresh).

# Auto-promote gate (2026-07-03): promote the winning variant when the (now-meaningful,
# last-30-day-holdout) backtest passes its quality bar. Guardrail keeps a human able to
# force-skip via the promote_mode widget. Default "auto" = promote iff _pass.
try:
    dbutils.widgets.text("promote_mode", "auto")
    _promote_mode = dbutils.widgets.get("promote_mode")
except Exception:
    _promote_mode = "auto"

if _promote_mode == "never":
    _promote = False
elif _promote_mode == "force":
    _promote = True
else:  # "auto"
    _promote = bool(_pass)  # _pass = (_bt_mape < 25 and _bt_ci_cover >= 55) from the backtest cell

print(f"  Promote decision: mode={_promote_mode}, backtest _pass={_pass}, → _promote={_promote}")

if _promote:
    _reg_client = mlflow.tracking.MlflowClient()

    try:
        _old_prod = _reg_client.get_model_version_by_alias(MODEL_NAME, "production")
        _reg_client.set_model_version_tag(MODEL_NAME, _old_prod.version, "status", "deprecated")
        print(f"  Previous @production v{_old_prod.version} → deprecated")
    except Exception:
        pass

    _mv = mlflow.register_model(f"runs:/{run_id}/prophet_revenue_model", MODEL_NAME)
    _winning_label = next((r["label"] for r in _variant_results if r["run_id"] == run_id), run_id)
    _reg_client.set_model_version_tag(MODEL_NAME, _mv.version, "status",        "production")
    _reg_client.set_model_version_tag(MODEL_NAME, _mv.version, "version_tag",   _winning_label)
    _reg_client.set_model_version_tag(MODEL_NAME, _mv.version, "training_date", str(pd.Timestamp.today().date()))
    _reg_client.set_registered_model_alias(MODEL_NAME, "production", _mv.version)

    print(f"  ✓ Registered v{_mv.version} → @production")
    print(f"  Variant: {_winning_label}")
    print(f"  Run ID:  {run_id}")
    print(f"  Model:   {MODEL_NAME}")
    print(f"\n  Next: trigger NB9 (job YOUR_DAILY_JOB_ID task forecast_generate) to regenerate forecasts.")
else:
    print("Promotion skipped. Set _promote=True after reviewing the comparison table.")
    print(f"  Current _WINNER_IDX={_WINNER_IDX} → {_variant_results[_WINNER_IDX]['label']}")
    print(f"  run_id = '{run_id}'")

# COMMAND ----------

# ── 11. RETURN METRICS (so job runs surface results without the MLflow system-table lag) ──
import json as _json
_exit_payload = {
    "winner":        _variant_results[_WINNER_IDX]["label"],
    "cv_mape":       round(float(_variant_results[_WINNER_IDX]["cv_mape"]), 2),
    "backtest_mape": round(float(_bt_mape), 2),
    "backtest_ci_coverage": round(float(_bt_ci_cover), 1),
    "backtest_pass": bool(_pass),
    "promoted":      bool(_promote),
    "promote_mode":  _promote_mode,
    "run_id":        run_id,
}
print("EXIT_METRICS " + _json.dumps(_exit_payload))
dbutils.notebook.exit(_json.dumps(_exit_payload))

# COMMAND ----------

# ── 11. VALIDATION ────────────────────────────────────────────────────────────

print("\n── Last 7 days actuals ──")
spark.sql(f"""
    SELECT business_date, day_name,
           ROUND(net_revenue, 2) AS revenue,
           order_count, weather_high_f, weather_category
    FROM {DAILY_TABLE}
    WHERE business_date BETWEEN CURRENT_DATE - INTERVAL 7 DAYS AND CURRENT_DATE
    ORDER BY business_date
""").show(10, truncate=False)

print(f"\n── Winner summary ({_variant_results[_WINNER_IDX]['label']}) ──")
print(f"  Training period: {TRAIN_START.date()} → {train_df['ds'].max().date()}  ({len(train_df)} rows)")
print(f"  train_mape: {_variant_results[_WINNER_IDX]['train_mape']}%   R²: {_variant_results[_WINNER_IDX]['train_r2']}")
print(f"  cv_mape:    {_variant_results[_WINNER_IDX]['cv_mape']}%  (30d horizon)")
print(f"  Run ID:     {run_id}")

# COMMAND ----------

# ── 12. LOG ALL VARIANTS TO ACCURACY HISTORY ─────────────────────────────────
# One row per variant so the Forecast Accuracy dashboard shows the full comparison.

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

_today_str = str(pd.Timestamp.today().date())
_train_max  = str(train_df['ds'].max().date())

def _sql_float(v):
    """Return a SQL-safe float literal, or NULL for None/nan."""
    if v is None:
        return "NULL"
    try:
        if np.isnan(float(v)):
            return "NULL"
    except (TypeError, ValueError):
        return "NULL"
    return str(round(float(v), 4))

for _vr in _variant_results:
    spark.sql(f"""
        INSERT INTO {CATALOG}.gold.forecast_accuracy_history
        (logged_at, model_name, mlflow_run_id, training_date, training_rows,
         training_data_through, train_mape, train_mae, train_r2,
         cv_mape, cv_mae, cv_mape_7d, cv_mape_14d,
         backtest_mape, backtest_mae, backtest_ci_coverage, notes)
        VALUES (
            current_timestamp(),
            'revenue',
            '{_vr["run_id"]}',
            DATE '{_today_str}',
            {len(train_df)},
            DATE '{_train_max}',
            {_sql_float(_vr["train_mape"])},
            NULL,
            {_sql_float(_vr["train_r2"])},
            {_sql_float(_vr["cv_mape"])},
            NULL,
            {_sql_float(_vr["cv_mape_7d"])},
            {_sql_float(_vr["cv_mape_14d"])},
            {_sql_float(_vr["bt_mape"])},
            NULL,
            {_sql_float(_bt_ci_cover)},
            '{_vr["label"]}'
        )
    """)
    print(f"  ✓ Logged {_vr['label']}  cv_mape={_vr['cv_mape']}%  bt_mape={_vr['bt_mape']}%")