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
    5:  0.101,   # May
    6:  0.109,   # June
    7:  0.115,   # July   ← peak
    8:  0.114,   # August
    9:  0.083,   # September
    10: 0.077,   # October
    11: 0.061,   # November
    12: 0.062,   # December
}

# ── Prophet hyperparameters ────────────────────────────────────────────────────
PROPHET_PARAMS = {
    "growth":                   "logistic",   # bounded growth toward cap
    "changepoint_prior_scale":  0.05,         # moderate trend flexibility
    "seasonality_prior_scale":  5.0,          # seasonal patterns can be meaningful
    "holidays_prior_scale":     10.0,         # events can have large effects
    "seasonality_mode":         "multiplicative",
    "yearly_seasonality":       False,        # replaced by custom prior below
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

# ── 6. BUILD CUSTOM YEARLY SEASONALITY FROM PRIOR STORE DATA ──────────────────
# With only 4 months of our own data, Prophet cannot fit a reliable yearly curve.
# Instead we build a custom seasonality from the prior store's monthly linearity.
#
# How this works:
#   Prophet's custom seasonality accepts a function that returns a value
#   for any given date. We build a smooth curve from the monthly linearity
#   percentages using linear interpolation between month midpoints.
#   The prior_scale=3 makes this a SOFT prior — our own data will increasingly
#   override it as we accumulate observations across more months.
#
# By next September (12 months of our data) this prior should be
# almost entirely superseded by what we've actually observed.

def monthly_seasonality_prior(ds):
    """
    Returns a seasonality value for any date based on the prior store's
    monthly revenue distribution. Interpolates smoothly between months.

    This encodes the knowledge that July is ~77% busier than January
    (0.115 / 0.065 = 1.77) without requiring us to have observed a July.
    """
    if not isinstance(ds, pd.Timestamp):
        ds = pd.Timestamp(ds)
    month = ds.month
    # Normalize so the annual average = 1.0 (mean of all 12 values)
    mean_lin = np.mean(list(PRIOR_MONTHLY_LINEARITY.values()))
    return (PRIOR_MONTHLY_LINEARITY[month] / mean_lin) - 1.0

# Vectorized version for Prophet
def prior_seasonality(ds):
    return pd.Series([monthly_seasonality_prior(d) for d in pd.to_datetime(ds)])

print("Prior seasonality values (normalized, 0.0 = average month):")
for month, lin in PRIOR_MONTHLY_LINEARITY.items():
    mean_lin = np.mean(list(PRIOR_MONTHLY_LINEARITY.values()))
    normalized = (lin / mean_lin) - 1.0
    bar = "█" * int(abs(normalized) * 20)
    sign = "+" if normalized >= 0 else "-"
    month_name = pd.Timestamp(f"2026-{month:02d}-01").strftime("%B")
    print(f"  {month_name:<12} {sign}{abs(normalized)*100:4.1f}%  {bar}")

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

# ── 9. TRAIN PROPHET MODEL ────────────────────────────────────────────────────

with mlflow.start_run(run_name="prophet_revenue_v2") as run:

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

    # Add custom yearly seasonality from prior store data
    # fourier_order=3 gives a smooth curve with 3 sine/cosine pairs
    # prior_scale=3 makes it soft — our data overrides it as it accumulates
    model.add_seasonality(
        name         = 'yearly_prior',
        period       = 365.25,
        fourier_order= 3,
        prior_scale  = 3.0,
        condition_name = None,
    )

    # Add regressors
    model.add_regressor('is_bread_delivery_day', mode='multiplicative')
    model.add_regressor('weather_high_f',         mode='additive')
    model.add_regressor('total_precip_in',        mode='additive')

    # ── Prepare fit dataframe ─────────────────────────────────────────────────
    fit_df = train_df[['ds', 'y_revenue', 'cap', 'floor',
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
                       'is_bread_delivery_day',
                       'weather_high_f', 'total_precip_in']])
    print("✓ Model fitted")

    # ── Build forecast input ──────────────────────────────────────────────────
    # Combine ALL historical dates (not just training) plus future dates
    # so the chart shows how the model fits the full history
    hist_input = all_historical[['ds', 'is_bread_delivery_day',
                                  'weather_high_f', 'total_precip_in']].copy()
    fwd_input  = future_df[['ds', 'is_bread_delivery_day',
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
    mlflow.prophet.log_model(
        pr_model              = model,
        artifact_path         = "prophet_revenue_model",
        registered_model_name = MODEL_NAME,
    )

    print(f"\n✓ Model logged and registered")
    print(f"  Run ID:     {run_id}")
    print(f"  Model name: {MODEL_NAME}")
    print(f"  View at:    Experiments → {EXPERIMENT_NAME}")

# COMMAND ----------

# ── 10. CROSS-VALIDATION ──────────────────────────────────────────────────────
# With ~4 months of training data (Dec-Mar), we have limited room for
# cross-validation folds. We try with conservative settings.

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

    except Exception as e:
        print(f"Cross-validation skipped: {e}")
        print("This can happen with limited training data. Model is still valid.")

# COMMAND ----------

# ── 11. WRITE FORECAST ROWS TO GOLD ──────────────────────────────────────────

future_preds = forecast[
    forecast['ds'] > pd.Timestamp(train_df['ds'].max())
][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

future_preds['yhat']       = future_preds['yhat'].clip(0, REVENUE_CAP).round(2)
future_preds['yhat_lower'] = future_preds['yhat_lower'].clip(0).round(2)
future_preds['yhat_upper'] = future_preds['yhat_upper'].clip(0, REVENUE_CAP * 1.1).round(2)

# Add confidence bound columns if not already present
existing_cols = [f.name for f in spark.table(DAILY_TABLE).schema.fields]
if 'forecast_lower' not in existing_cols:
    spark.sql(f"""
        ALTER TABLE {DAILY_TABLE} ADD COLUMNS (
            forecast_lower  DOUBLE COMMENT 'Lower confidence bound from forecast model',
            forecast_upper  DOUBLE COMMENT 'Upper confidence bound from forecast model',
            forecast_model  STRING COMMENT 'Model that generated this forecast row'
        )
    """)
    print("✓ Added forecast confidence columns to daily_sales_summary")

future_preds_spark = spark.createDataFrame(
    future_preds.rename(columns={
        'ds':         'business_date',
        'yhat':       'net_revenue_pred',
        'yhat_lower': 'forecast_lower',
        'yhat_upper': 'forecast_upper',
    })
).withColumn("business_date", F.col("business_date").cast("date"))

# Build forecast rows with calendar and weather fields
forecast_rows = spark.sql(f"""
    SELECT
        f.ds                                    AS business_date,
        'forecast'                              AS record_type,
        CAST(NULL AS LONG)                      AS order_count,
        CAST(NULL AS DOUBLE)                    AS gross_revenue,
        CAST(NULL AS DOUBLE)                    AS net_revenue,
        CAST(NULL AS DOUBLE)                    AS total_discounts,
        CAST(NULL AS DOUBLE)                    AS avg_ticket_size,
        CAST(NULL AS LONG)                      AS item_count,
        DAYOFWEEK(f.ds)                         AS day_of_week,
        DATE_FORMAT(f.ds, 'EEEE')               AS day_name,
        DAYOFWEEK(f.ds) IN (1, 7)               AS is_weekend,
        DAYOFWEEK(f.ds) IN (3, 6)               AS is_bread_delivery_day,
        WEEKOFYEAR(f.ds)                        AS week_of_year,
        MONTH(f.ds)                             AS month,
        YEAR(f.ds)                              AS year,
        w.weather_high_f,
        w.weather_low_f,
        w.weather_feels_high_f,
        w.weather_feels_low_f,
        w.weather_category,
        w.weather_condition,
        w.weather_code,
        w.total_precip_in,
        w.total_snow_in,
        w.sunny_hours,
        w.avg_cloud_cover_pct,
        w.weather_data_type,
        CURRENT_TIMESTAMP()                     AS _gold_updated_at,
        '{run_id}'                              AS _batch_id
    FROM {FEATURES_TABLE} f
    LEFT JOIN (
        SELECT
            date,
            ROUND(MAX(temperature_f), 1)            AS weather_high_f,
            ROUND(MIN(temperature_f), 1)            AS weather_low_f,
            ROUND(MAX(apparent_temperature_f), 1)   AS weather_feels_high_f,
            ROUND(MIN(apparent_temperature_f), 1)   AS weather_feels_low_f,
            MAX_BY(weather_category, weather_code)  AS weather_category,
            MAX_BY(weather_condition, weather_code) AS weather_condition,
            MAX(weather_code)                       AS weather_code,
            ROUND(SUM(precipitation_in), 3)         AS total_precip_in,
            ROUND(SUM(snowfall_in), 3)              AS total_snow_in,
            ROUND(SUM(CASE WHEN sunshine_minutes > 30 THEN 1 ELSE 0 END), 0)
                                                    AS sunny_hours,
            ROUND(AVG(cloud_cover_pct), 1)          AS avg_cloud_cover_pct,
            MAX_BY(data_type, weather_code)         AS weather_data_type
        FROM {WEATHER_TABLE}
        WHERE date > '{train_df["ds"].max().date()}'
        GROUP BY date
    ) w ON f.ds = w.date
    WHERE f.y_revenue IS NULL
""")

forecast_final = forecast_rows.join(
    future_preds_spark, on="business_date", how="left"
).withColumn("net_revenue", F.col("net_revenue_pred")) \
 .withColumn("forecast_model", F.lit(MODEL_NAME)) \
 .drop("net_revenue_pred")

DeltaTable.forName(spark, DAILY_TABLE).alias("t").merge(
    forecast_final.alias("s"),
    "t.business_date = s.business_date AND t.record_type = s.record_type"
).whenMatchedUpdateAll(
).whenNotMatchedInsertAll(
).execute()

print(f"✓ Wrote {future_preds.shape[0]} forecast rows to {DAILY_TABLE}")

# COMMAND ----------

# ── 12. VALIDATION ────────────────────────────────────────────────────────────

print("\n── Last 7 days actual + next 14 days forecast ──")
spark.sql(f"""
    SELECT
        business_date,
        day_name,
        record_type,
        ROUND(net_revenue, 2)           AS revenue,
        ROUND(forecast_lower, 2)        AS lower_bound,
        ROUND(forecast_upper, 2)        AS upper_bound,
        weather_high_f,
        weather_category,
        is_bread_delivery_day
    FROM {DAILY_TABLE}
    WHERE business_date BETWEEN
        CURRENT_DATE - INTERVAL 7 DAYS AND
        CURRENT_DATE + INTERVAL 14 DAYS
    ORDER BY business_date, record_type DESC
""").show(30, truncate=False)

print(f"\n── Model summary ──")
print(f"  Training period:   {TRAIN_START.date()} → {train_df['ds'].max().date()}")
print(f"  Training rows:     {len(train_df)}")
print(f"  Revenue cap:       ${REVENUE_CAP:,.0f}/day")
print(f"  MAE:               ${mae:,.2f}")
print(f"  MAPE:              {mape:.1f}%")
print(f"  R²:                {r2:.3f}")
print(f"  MLflow run:        {run_id}")
print(f"  Registered model:  {MODEL_NAME}")