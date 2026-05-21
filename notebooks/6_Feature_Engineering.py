# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering — Sales Forecast
# MAGIC
# MAGIC ## What is this notebook?
# MAGIC
# MAGIC A forecast model is only as good as the information you give it.
# MAGIC **Feature engineering** is the process of transforming raw data into
# MAGIC the specific signals — called **features** — that the model will use
# MAGIC to make predictions.
# MAGIC
# MAGIC Think of it like preparing ingredients before cooking. The raw data
# MAGIC (daily revenue, weather readings, dates) is the grocery bag. This
# MAGIC notebook chops, measures, and organizes everything into the exact
# MAGIC form the model needs.
# MAGIC
# MAGIC **What this notebook produces:**
# MAGIC `3sp_analytics_workspace.gold.forecast_features`
# MAGIC — one row per day, with every feature the model will train on.
# MAGIC Both the Prophet and LightGBM models read from this table.
# MAGIC
# MAGIC ## Feature categories we're building
# MAGIC
# MAGIC | Category | Examples | Why it matters |
# MAGIC |---|---|---|
# MAGIC | Calendar | Day of week, month, week of year | Weekly and seasonal patterns |
# MAGIC | Special days | Bread delivery, holidays, events | Known demand drivers |
# MAGIC | Weather | Temperature, precipitation, snow | Affects foot traffic |
# MAGIC | Lag features | Revenue 7 days ago, 14 days ago | Recent momentum |
# MAGIC | Rolling averages | 4-week average for this day-of-week | Baseline expectation |
# MAGIC | Outlier flags | Imputed day, catering day | Model confidence weights |
# MAGIC
# MAGIC ## TODO: Future features (not yet implemented)
# MAGIC
# MAGIC - **Cohasset school calendar** — no-school days and half days likely
# MAGIC   drive meaningful foot traffic increases. Source: Cohasset Public Schools
# MAGIC   annual PDF calendar. Implementation: manual CSV → Unity Catalog reference
# MAGIC   table → join here. Add `is_no_school_day` and `is_half_day` boolean flags.
# MAGIC   Validate with t-test (same method as bread delivery analysis in EDA).
# MAGIC
# MAGIC - **Local events calendar** — civic events, school sports, concerts.
# MAGIC   Harder to source systematically. Consider a manual log maintained by
# MAGIC   store staff as events are noticed.

# COMMAND ----------

# ── WIDGET ────────────────────────────────────────────────────────────────────

dbutils.widgets.dropdown(
    name="run_mode",
    defaultValue="incremental",
    choices=["incremental", "full_rebuild"],
    label="Run Mode"
)

# COMMAND ----------

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import datetime
import uuid
from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql import Window

# COMMAND ----------

# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────

RUN_MODE = dbutils.widgets.get("run_mode")

CATALOG       = "3sp_analytics_workspace"
GOLD_SCHEMA   = f"{CATALOG}.gold"
BRONZE_SCHEMA = f"{CATALOG}.bronze"

DAILY_SUMMARY   = f"{GOLD_SCHEMA}.daily_sales_summary"
WEATHER_HOURLY  = f"{BRONZE_SCHEMA}.weather_hourly"
FEATURES_TABLE  = f"{GOLD_SCHEMA}.forecast_features"

BATCH_ID = str(uuid.uuid4())
NOW_UTC  = datetime.datetime.now(datetime.timezone.utc)

# ── Load special events from reference table ──────────────────────────────────
# Single source of truth — add new events to reference.store_events notebook.
# All model notebooks read from here so changes only need to happen once.

events_ref = spark.sql(f"""
    SELECT event_date, event_name, event_type, lower_window, upper_window
    FROM {CATALOG}.reference.store_events
    WHERE is_active = true
    ORDER BY event_date
""").toPandas()

events_ref['event_date'] = events_ref['event_date'].astype(str)

SPECIAL_EVENTS = {
    row['event_date']: (row['event_name'], row['event_type'])
    for _, row in events_ref.iterrows()
    if row['event_type'] in ('PLANNED_EVENT', 'REVENUE_DISTORTION', 'ORGANIC_EVENT')
}

FUTURE_CLOSURES = {
    row['event_date']: row['event_name']
    for _, row in events_ref.iterrows()
    if row['event_type'] == 'FUTURE_CLOSURE'
}

print(f"✓ Loaded {len(events_ref)} events from reference.store_events")
print(f"  Training events: {len(SPECIAL_EVENTS)}")
print(f"  Future closures: {len(FUTURE_CLOSURES)}")
for _, row in events_ref.iterrows():
    status = "active" if row.get('is_active', True) else "inactive"
    print(f"  {row['event_date']}  {row['event_name']:<35} {row['event_type']}")

# Catering detection threshold — single ticket above this
# likely represents a catering/event purchase, not retail
CATERING_THRESHOLD = 500

# Holiday shopping window — high demand but real signal, NOT excluded
# Model should learn the December demand curve from this
HOLIDAY_WINDOW_MONTHS = [12]
HOLIDAY_WINDOW_DAYS   = list(range(15, 32))   # Dec 15-31

# ── EDA-validated feature flags ───────────────────────────────────────────────
# These thresholds come directly from the EDA analysis.
# Change them here if the EDA is re-run with updated data.

SIGNIFICANT_LAGS     = [1, 6, 7, 8, 14]   # from autocorrelation analysis
WEATHER_FEATURES     = ['weather_high_f', 'weather_low_f', 'total_precip_in']
SNOW_CLOSURE_THRESH  = 3.0   # inches prior day — likely closure risk

# COMMAND ----------

# ── 3. SETUP OUTPUT TABLE ─────────────────────────────────────────────────────

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FEATURES_TABLE} (

        -- ── Key ────────────────────────────────────────────────────────────
        ds                      DATE        NOT NULL    COMMENT 'Date — named ds for Prophet compatibility (Prophet requires this column name)',

        -- ── Targets (what we are forecasting) ─────────────────────────────
        -- Both targets are available for both models.
        -- Prophet trains one model per target.
        -- LightGBM can predict both in one pass.
        y_revenue               DOUBLE                  COMMENT 'Net revenue — primary forecast target. Imputed for closure days.',
        y_orders                DOUBLE                  COMMENT 'Order count — secondary forecast target. Imputed for closure days.',

        -- ── Calendar features ──────────────────────────────────────────────
        day_of_week             INTEGER                 COMMENT '1=Sunday through 7=Saturday (Spark convention)',
        day_name                STRING                  COMMENT 'Full day name: Monday, Tuesday, etc.',
        is_weekend              BOOLEAN                 COMMENT 'True for Saturday and Sunday',
        week_of_year            INTEGER                 COMMENT 'ISO week number (1-53)',
        month                   INTEGER                 COMMENT 'Month number (1-12)',
        year                    INTEGER                 COMMENT 'Calendar year',
        day_of_year             INTEGER                 COMMENT 'Day of year (1-365/366) — captures smooth seasonality curve',
        days_since_start        INTEGER                 COMMENT 'Days since first observation — captures long-term trend',

        -- ── Cyclical encoding ──────────────────────────────────────────────
        -- Raw day-of-week (1-7) treats Monday and Sunday as far apart,
        -- but they are both near the weekend in behavior. Cyclical encoding
        -- wraps the scale so day 7 connects back to day 1 smoothly.
        -- LightGBM handles this automatically but it helps linear models.
        dow_sin                 DOUBLE                  COMMENT 'Sine encoding of day-of-week — captures cyclical weekly pattern',
        dow_cos                 DOUBLE                  COMMENT 'Cosine encoding of day-of-week — paired with dow_sin',
        month_sin               DOUBLE                  COMMENT 'Sine encoding of month — captures cyclical yearly pattern',
        month_cos               DOUBLE                  COMMENT 'Cosine encoding of month — paired with month_sin',

        -- ── Special day flags ──────────────────────────────────────────────
        is_bread_delivery_day   BOOLEAN                 COMMENT 'True on Tuesdays and Fridays — Kneady Mama delivery. EDA confirmed +39% revenue lift.',
        is_friday               BOOLEAN                 COMMENT 'Explicit Friday flag — Friday behaves like weekend (2.65x Monday multiplier)',
        is_holiday_window       BOOLEAN                 COMMENT 'True Dec 15-31 — holiday shopping season. High demand is real signal, not outlier.',
        is_planned_event        BOOLEAN                 COMMENT 'True on known recurring special events (e.g. Town Stroll). Model learns pattern.',
        event_name              STRING                  COMMENT 'Name of the special event if applicable, null otherwise.',

        -- ── Closure and data quality flags ────────────────────────────────
        store_closed            BOOLEAN                 COMMENT 'True if actual revenue was $0 — store was closed for any reason.',
        is_imputed              BOOLEAN                 COMMENT 'True if y_revenue is an imputed estimate rather than observed revenue.',
        is_catering_day         BOOLEAN                 COMMENT 'True if the day likely included catering revenue above the $500 threshold.',
        exclude_from_training   BOOLEAN                 COMMENT 'True for REVENUE_DISTORTION and ORGANIC_EVENT days — do not train on these.',
        training_weight         DOUBLE                  COMMENT 'Suggested sample weight for model training: 1.0=normal, 0.5=imputed, 0.0=exclude.',
        likely_snow_closure_risk BOOLEAN                COMMENT 'True if prior day had >3 inches of snow — elevated risk of closure.',
        likely_future_closure   BOOLEAN                 COMMENT 'True on known future holiday closures (Thanksgiving, Christmas). Forecast labeled closed.',

        -- ── Weather features ───────────────────────────────────────────────
        -- EDA confirmed negative temperature correlation (colder = higher revenue
        -- for a coastal New England grocery/cafe). Precipitation negative.
        -- Sunshine hours and cloud cover near zero correlation — excluded.
        weather_high_f          DOUBLE                  COMMENT 'Daily high temperature in Fahrenheit. Negative correlation with revenue (r=-0.25).',
        weather_low_f           DOUBLE                  COMMENT 'Daily low temperature in Fahrenheit. Negative correlation with revenue (r=-0.29).',
        weather_feels_high_f    DOUBLE                  COMMENT 'Daily high feels-like temperature. Often stronger behavioral driver than actual temp.',
        weather_comfort_score   DOUBLE                  COMMENT 'Piecewise-linear comfort score (0-1), peaks at 62F feels-like. Cold min 40F, hot max 85F.',
        heat_stress             DOUBLE                  COMMENT 'Linear heat penalty above 82F feels-like. Zero below threshold.',
        total_precip_in         DOUBLE                  COMMENT 'Total precipitation in inches. Negative correlation with revenue (r=-0.12).',
        total_snow_in           DOUBLE                  COMMENT 'Total snowfall in inches. Strong suppressor — closure risk at high values.',
        weather_category        STRING                  COMMENT 'Predominant weather category: Clear, Cloudy, Rainy, Snowy, Stormy, Foggy.',
        is_precipitation_day    BOOLEAN                 COMMENT 'True if any measurable rain or snow fell.',
        is_snow_day             BOOLEAN                 COMMENT 'True if snowfall > 0.1 inches.',
        weather_data_type       STRING                  COMMENT 'archive, archive_pending, or forecast — source of weather data for this row.',

        -- ── Lag features ───────────────────────────────────────────────────
        -- Autocorrelation analysis identified lags 1, 6, 7, 8, 14 as significant.
        -- Lag 7 is the strongest — same day last week is the best single predictor.
        -- Used by LightGBM only (Prophet handles temporal patterns internally).
        lag_1_revenue           DOUBLE                  COMMENT 'Net revenue 1 day ago.',
        lag_6_revenue           DOUBLE                  COMMENT 'Net revenue 6 days ago.',
        lag_7_revenue           DOUBLE                  COMMENT 'Net revenue 7 days ago (same day last week). Strongest lag predictor.',
        lag_8_revenue           DOUBLE                  COMMENT 'Net revenue 8 days ago.',
        lag_14_revenue          DOUBLE                  COMMENT 'Net revenue 14 days ago (same day 2 weeks ago).',
        lag_7_orders            DOUBLE                  COMMENT 'Order count 7 days ago.',
        lag_14_orders           DOUBLE                  COMMENT 'Order count 14 days ago.',

        -- ── Rolling averages ───────────────────────────────────────────────
        -- Rolling averages provide a "recent baseline" expectation.
        -- We compute them for the same day-of-week to avoid mixing e.g.
        -- Monday averages into a Saturday prediction.
        rolling_4w_same_dow_revenue  DOUBLE             COMMENT 'Average revenue for this day-of-week over the prior 4 weeks. Best baseline predictor.',
        rolling_4w_same_dow_orders   DOUBLE             COMMENT 'Average order count for this day-of-week over the prior 4 weeks.',
        rolling_2w_revenue      DOUBLE                  COMMENT '14-day trailing average revenue (all days). Captures recent momentum.',
        rolling_4w_revenue      DOUBLE                  COMMENT '28-day trailing average revenue (all days). Captures medium-term trend.',

        -- ── Audit ──────────────────────────────────────────────────────────
        _feature_computed_at    TIMESTAMP               COMMENT 'When this feature row was computed.',
        _batch_id               STRING                  COMMENT 'Batch ID for this feature engineering run.'
    )
    USING DELTA
    COMMENT 'Gold: forecast feature table. One row per day. Consumed by Prophet and LightGBM forecast models. Rebuilt nightly after Gold sales summary.'
    TBLPROPERTIES ('quality' = 'gold', 'delta.enableChangeDataFeed' = 'true')
""")

print(f"✓ {FEATURES_TABLE} ready")

# COMMAND ----------

# ── 4. DETERMINE DATE RANGE ───────────────────────────────────────────────────
# For features we need to go a bit further back than the training window
# because lag features look back up to 14 days. We fetch 30 extra days
# of history to ensure all lag values can be computed for the earliest
# training rows.

# Include 30 days of future weather forecast for forward prediction rows
# Must be >= FORECAST_DAYS in 9_Forecast_Generate.py (currently 30)
FUTURE_DAYS   = 30
LAG_BUFFER    = 30   # extra days before training start for lag computation

yesterday = (NOW_UTC - datetime.timedelta(days=1)).date()
future_end = (NOW_UTC + datetime.timedelta(days=FUTURE_DAYS)).date()

if RUN_MODE == "incremental":
    try:
        last_feature_date = spark.sql(f"""
            SELECT MAX(ds) AS d FROM {FEATURES_TABLE}
            WHERE NOT exclude_from_training
        """).collect()[0]["d"]
    except Exception:
        last_feature_date = None

    if last_feature_date is None:
        history_start = spark.sql(f"""
            SELECT MIN(business_date) AS d FROM {DAILY_SUMMARY}
        """).collect()[0]["d"] - datetime.timedelta(days=LAG_BUFFER)
    else:
        # Reprocess last 14 days to refresh any lag features that may have changed
        history_start = last_feature_date - datetime.timedelta(days=14)

    print(f"Incremental mode: {history_start} → {future_end}")
else:
    history_start = spark.sql(f"""
        SELECT MIN(business_date) AS d FROM {DAILY_SUMMARY}
    """).collect()[0]["d"] - datetime.timedelta(days=LAG_BUFFER)
    print(f"Full rebuild: {history_start} → {future_end}")

# COMMAND ----------

# ── 5. LOAD SOURCE DATA ───────────────────────────────────────────────────────

# Load actual sales history (includes imputed closure days)
sales_df = spark.sql(f"""
    SELECT
        business_date           AS ds,
        order_count,
        net_revenue,
        avg_ticket_size,
        day_of_week,
        day_name,
        is_weekend,
        is_bread_delivery_day,
        week_of_year,
        month,
        year,
        weather_high_f,
        weather_low_f,
        weather_feels_high_f,
        total_precip_in,
        total_snow_in,
        weather_category,
        weather_data_type
    FROM {DAILY_SUMMARY}
    WHERE business_date >= '{history_start}'
    ORDER BY business_date
""").toPandas()

# Load forward weather forecast rows from the weather table
# These provide weather features for future prediction rows
future_weather_df = spark.sql(f"""
    SELECT
        date                                        AS ds,
        ROUND(MAX(temperature_f), 1)                AS weather_high_f,
        ROUND(MIN(temperature_f), 1)                AS weather_low_f,
        ROUND(MAX(apparent_temperature_f), 1)       AS weather_feels_high_f,
        ROUND(SUM(precipitation_in), 3)             AS total_precip_in,
        ROUND(SUM(snowfall_in), 3)                  AS total_snow_in,
        MAX_BY(weather_category, weather_code)      AS weather_category,
        MAX_BY(data_type, weather_code)             AS weather_data_type
    FROM {WEATHER_HOURLY}
    WHERE date > '{yesterday}'
      AND date <= '{future_end}'
    GROUP BY date
    ORDER BY date
""").toPandas()

sales_df['ds']       = pd.to_datetime(sales_df['ds'])
future_weather_df['ds'] = pd.to_datetime(future_weather_df['ds'])

print(f"Loaded {len(sales_df)} historical days")
print(f"Loaded {len(future_weather_df)} future weather days")

# COMMAND ----------

# ── 6. BUILD FULL DATE SPINE ──────────────────────────────────────────────────
# Generate every date from history_start to future_end.
# Historical dates get actual sales data.
# Future dates get only calendar and weather features — no sales targets yet.

all_dates = pd.date_range(start=history_start, end=future_end, freq='D')
spine_df  = pd.DataFrame({'ds': all_dates})

# Merge sales history
df = spine_df.merge(sales_df, on='ds', how='left')

# Merge future weather (fills weather columns for forward dates)
df = df.merge(future_weather_df,
              on='ds', how='left',
              suffixes=('', '_future'))

# For future dates, use the future weather values where historical are missing
for col in ['weather_high_f', 'weather_low_f', 'weather_feels_high_f',
            'total_precip_in', 'total_snow_in', 'weather_category', 'weather_data_type']:
    future_col = col + '_future'
    if future_col in df.columns:
        df[col] = df[col].fillna(df[future_col])
        df.drop(columns=[future_col], inplace=True)

print(f"Full spine: {len(df)} days ({history_start} → {future_end})")

# COMMAND ----------

# ── 7. CALENDAR FEATURES ──────────────────────────────────────────────────────

df['day_of_week']  = df['ds'].dt.dayofweek + 1   # 1=Monday ... 7=Sunday
df['day_name']     = df['ds'].dt.day_name()
df['is_weekend']   = df['day_of_week'].isin([6, 7])
df['week_of_year'] = df['ds'].dt.isocalendar().week.astype(int)
df['month']        = df['ds'].dt.month
df['year']         = df['ds'].dt.year
df['day_of_year']  = df['ds'].dt.dayofyear
df['is_friday']    = df['day_of_week'] == 5

# Days since first observation — captures long-term growth trend
first_date = df['ds'].min()
df['days_since_start'] = (df['ds'] - first_date).dt.days

# ── Cyclical encoding ─────────────────────────────────────────────────────────
# Converts day-of-week (1-7) and month (1-12) into sine/cosine pairs.
# This tells the model that Sunday (7) is behaviorally close to Monday (1)
# and that December (12) is close to January (1) on the seasonal cycle.
df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# ── Bread delivery day ────────────────────────────────────────────────────────
# EDA confirmed +39% revenue lift on Tuesdays and Fridays.
# day_of_week: 2=Tuesday, 5=Friday (Monday=1 convention)
df['is_bread_delivery_day'] = df['day_of_week'].isin([2, 5])

# ── Holiday window ────────────────────────────────────────────────────────────
df['is_holiday_window'] = (
    df['month'].isin(HOLIDAY_WINDOW_MONTHS) &
    df['day_of_year'].between(
        pd.Timestamp(f"{df['year'].iloc[0]}-12-15").dayofyear,
        365
    )
)
# Simpler approach that works across years
df['is_holiday_window'] = (
    (df['month'] == 12) & (df['ds'].dt.day >= 15)
)

print("✓ Calendar features computed")

# COMMAND ----------

# ── 8. SPECIAL EVENT FLAGS ────────────────────────────────────────────────────

event_dates      = set(SPECIAL_EVENTS.keys())
event_labels_map = {d: info[0] for d, info in SPECIAL_EVENTS.items()}
event_types_map  = {d: info[1] for d, info in SPECIAL_EVENTS.items()}

df['ds_str']          = df['ds'].dt.strftime('%Y-%m-%d')
df['is_planned_event']= df['ds_str'].isin(
    [d for d, (_, t) in SPECIAL_EVENTS.items() if t == 'PLANNED_EVENT']
)
df['event_name']      = df['ds_str'].map(event_labels_map)
df['event_type']      = df['ds_str'].map(event_types_map)

# Exclude revenue distortions and organic events from training
EXCLUDE_TYPES = {'REVENUE_DISTORTION', 'ORGANIC_EVENT'}
df['exclude_from_training'] = df['event_type'].isin(EXCLUDE_TYPES)

print(f"✓ Special event flags: {df['is_planned_event'].sum()} planned events")
print(f"  Excluded from training: {df['exclude_from_training'].sum()} days")

# COMMAND ----------

# ── 9. CLOSURE AND DATA QUALITY FLAGS ─────────────────────────────────────────

# Store closed = zero revenue on a historical day
df['store_closed'] = (df['net_revenue'] == 0) & df['net_revenue'].notna()

# Future dates have no revenue yet — not the same as closed
df['store_closed'] = df['store_closed'].fillna(False)

# Impute zero-revenue days with day-of-week × month median from open days
df['y_revenue'] = df['net_revenue'].copy()
df['y_orders']  = df['order_count'].copy()
df['is_imputed'] = False

open_mask = (~df['store_closed']) & df['net_revenue'].notna()
open_days = df[open_mask]

for idx, row in df[df['store_closed']].iterrows():
    similar = open_days[
        (open_days['day_of_week'] == row['day_of_week']) &
        (open_days['month']       == row['month'])
    ]
    if len(similar) == 0:
        similar = open_days[open_days['day_of_week'] == row['day_of_week']]
    if len(similar) > 0:
        df.at[idx, 'y_revenue'] = similar['net_revenue'].median()
        df.at[idx, 'y_orders']  = similar['order_count'].median()
        df.at[idx, 'is_imputed'] = True

# Future dates: y_revenue and y_orders are null (to be filled by the model)
# This is correct — we are not predicting the past

# Catering day detection (daily avg ticket above threshold)
df['is_catering_day'] = (
    df['avg_ticket_size'].notna() &
    (df['avg_ticket_size'] > CATERING_THRESHOLD)
)

# Snow closure risk — prior day snowfall above threshold
df['prior_day_snow']         = df['total_snow_in'].shift(1)
df['likely_snow_closure_risk'] = df['prior_day_snow'] > SNOW_CLOSURE_THRESH

# Future closure labeling
future_closure_dates = set(FUTURE_CLOSURES.keys())
df['likely_future_closure'] = df['ds_str'].isin(future_closure_dates)

# ── Training weights ──────────────────────────────────────────────────────────
# 1.0 = normal observed day — full confidence
# 0.5 = imputed closure day — real demand pattern but estimated value
# 0.0 = excluded day — do not train on this row
df['training_weight'] = 1.0
df.loc[df['is_imputed'],             'training_weight'] = 0.5
df.loc[df['exclude_from_training'],  'training_weight'] = 0.0
df.loc[df['store_closed'] & ~df['is_imputed'], 'training_weight'] = 0.0

print(f"✓ Closure flags: {df['store_closed'].sum()} closed, {df['is_imputed'].sum()} imputed")
print(f"  Training weight distribution:")
print(f"    1.0 (full):     {(df['training_weight'] == 1.0).sum()} days")
print(f"    0.5 (imputed):  {(df['training_weight'] == 0.5).sum()} days")
print(f"    0.0 (excluded): {(df['training_weight'] == 0.0).sum()} days")

# COMMAND ----------

# ── 10. WEATHER FEATURES ──────────────────────────────────────────────────────
# EDA findings:
#   - Temperature (high and low): negative correlation — colder = more revenue
#     for a coastal New England grocery/café. Counter-intuitive but real.
#   - Precipitation: negative correlation — rain suppresses foot traffic
#   - Sunshine/cloud cover: near-zero correlation — excluded
#   - Snowfall: strong closure risk at high values (handled in closure flags)

df['is_precipitation_day'] = df['total_precip_in'].fillna(0) > 0.01
df['is_snow_day']          = df['total_snow_in'].fillna(0) > 0.1

# Fill missing weather with day-of-month seasonal averages
# (better than leaving nulls which break tree models)
for col in ['weather_high_f', 'weather_low_f', 'weather_feels_high_f',
            'total_precip_in', 'total_snow_in']:
    if df[col].isna().any():
        monthly_avg = df.groupby('month')[col].transform('median')
        df[col] = df[col].fillna(monthly_avg)

df['weather_category'] = df['weather_category'].fillna('Unknown')
df['weather_data_type']= df['weather_data_type'].fillna('unknown')

print("✓ Weather features computed")

# COMMAND ----------

# ── 10b. NON-LINEAR TEMPERATURE FEATURES ──────────────────────────────────────
# Linear weather_high_f misses the U-shaped temperature response:
# cold extremes (closure risk) and hot extremes (beach day / town empties) both
# suppress revenue, while a comfortable 55-65°F window drives peak foot traffic.
# Uses weather_feels_high_f (already shown to be a stronger behavioral driver
# than actual temp for this coastal store).
#
# weather_comfort_score: piecewise-linear 0→1→0 peaked at ~62°F.
#   Evaluate as a candidate regressor when retraining with summer data.
# heat_stress: linear penalty above 82°F feels-like.
#   Separates "too hot to be outside" from the comfort curve.

COMFORT_PEAK_F   = 62.0
COMFORT_COLD_MIN = 40.0
COMFORT_HOT_MAX  = 85.0

df['weather_comfort_score'] = np.where(
    df['weather_feels_high_f'] <= COMFORT_COLD_MIN, 0.0,
    np.where(
        df['weather_feels_high_f'] <= COMFORT_PEAK_F,
        (df['weather_feels_high_f'] - COMFORT_COLD_MIN) / (COMFORT_PEAK_F - COMFORT_COLD_MIN),
        np.where(
            df['weather_feels_high_f'] <= COMFORT_HOT_MAX,
            (COMFORT_HOT_MAX - df['weather_feels_high_f']) / (COMFORT_HOT_MAX - COMFORT_PEAK_F),
            0.0
        )
    )
).clip(0.0, 1.0)

df['heat_stress'] = (df['weather_feels_high_f'] - 82.0).clip(lower=0.0)

print(f"✓ Non-linear temperature features computed "
      f"(comfort_score: {df['weather_comfort_score'].min():.2f}–{df['weather_comfort_score'].max():.2f}, "
      f"heat_stress: {df['heat_stress'].min():.1f}–{df['heat_stress'].max():.1f}°F above threshold)")

# COMMAND ----------

# ── 11. LAG FEATURES ──────────────────────────────────────────────────────────
# Lag features use past values of the target as predictors.
# We use y_revenue (imputed) so closure days don't create artificial zeros
# in the lag window — a zero from a closed day would incorrectly signal
# "last week was terrible" to the model.
#
# IMPORTANT: lag features are only valid for HISTORICAL dates.
# Future prediction rows will have null lag values — that is correct.
# LightGBM handles null lag features gracefully.

# Sort by date to ensure shift() computes correctly
df = df.sort_values('ds').reset_index(drop=True)

for lag in SIGNIFICANT_LAGS:
    df[f'lag_{lag}_revenue'] = df['y_revenue'].shift(lag)

df['lag_7_orders']  = df['y_orders'].shift(7)
df['lag_14_orders'] = df['y_orders'].shift(14)

print(f"✓ Lag features computed: {[f'lag_{l}_revenue' for l in SIGNIFICANT_LAGS]}")

# COMMAND ----------

# ── 12. ROLLING AVERAGE FEATURES ──────────────────────────────────────────────
# Rolling averages provide a "recent baseline" that captures current momentum.
#
# same_dow (same day of week) rolling average is the most powerful baseline:
# "What was average revenue on the last 4 Tuesdays?" tells you more about
# this Tuesday than a general 28-day average would.

# General rolling averages (all days)
df['rolling_2w_revenue'] = df['y_revenue'].shift(1).rolling(window=14,  min_periods=7).mean()
df['rolling_4w_revenue'] = df['y_revenue'].shift(1).rolling(window=28,  min_periods=14).mean()

# Same day-of-week rolling average
# For each row, look back at the last 4 occurrences of the same day of week
# (equivalent to looking back ~4 weeks at the same weekday)
dow_rolling = {}
for dow in range(1, 8):
    mask = df['day_of_week'] == dow
    dow_df = df[mask].copy().sort_values('ds')
    dow_df['rolling_4w_same_dow_revenue'] = (
        dow_df['y_revenue'].shift(1).rolling(window=4, min_periods=2).mean()
    )
    dow_df['rolling_4w_same_dow_orders'] = (
        dow_df['y_orders'].shift(1).rolling(window=4, min_periods=2).mean()
    )
    dow_rolling[dow] = dow_df[['ds', 'rolling_4w_same_dow_revenue',
                                'rolling_4w_same_dow_orders']]

dow_rolling_df = pd.concat(dow_rolling.values()).sort_values('ds')
df = df.merge(dow_rolling_df, on='ds', how='left')

print("✓ Rolling average features computed")

# COMMAND ----------

# ── 13. AUDIT COLUMNS ─────────────────────────────────────────────────────────

df['_feature_computed_at'] = NOW_UTC
df['_batch_id']            = BATCH_ID

# COMMAND ----------

# ── 14. FINAL COLUMN SELECTION AND TYPE CLEANUP ───────────────────────────────

final_cols = [
    'ds',
    'y_revenue', 'y_orders',
    'day_of_week', 'day_name', 'is_weekend', 'week_of_year',
    'month', 'year', 'day_of_year', 'days_since_start',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_bread_delivery_day', 'is_friday', 'is_holiday_window',
    'is_planned_event', 'event_name',
    'store_closed', 'is_imputed', 'is_catering_day',
    'exclude_from_training', 'training_weight',
    'likely_snow_closure_risk', 'likely_future_closure',
    'weather_high_f', 'weather_low_f', 'weather_feels_high_f',
    'weather_comfort_score', 'heat_stress',
    'total_precip_in', 'total_snow_in', 'weather_category',
    'is_precipitation_day', 'is_snow_day', 'weather_data_type',
    'lag_1_revenue', 'lag_6_revenue', 'lag_7_revenue',
    'lag_8_revenue', 'lag_14_revenue',
    'lag_7_orders', 'lag_14_orders',
    'rolling_4w_same_dow_revenue', 'rolling_4w_same_dow_orders',
    'rolling_2w_revenue', 'rolling_4w_revenue',
    '_feature_computed_at', '_batch_id',
]

df_final = df[final_cols].copy()

# Trim to dates that will actually be written (exclude lag buffer before training start)
actual_start = spark.sql(f"""
    SELECT MIN(business_date) FROM {DAILY_SUMMARY}
""").collect()[0][0]

df_final = df_final[df_final['ds'] >= pd.Timestamp(actual_start)]

print(f"Final feature rows: {len(df_final)}")
print(f"  Historical (with targets): {df_final['y_revenue'].notna().sum()}")
print(f"  Future (targets null):     {df_final['y_revenue'].isna().sum()}")

# COMMAND ----------

# ── 15. WRITE TO DELTA ────────────────────────────────────────────────────────

spark_df = spark.createDataFrame(df_final) \
    .withColumn("ds", F.col("ds").cast("date")) \
    .withColumn("_feature_computed_at", F.col("_feature_computed_at").cast("timestamp"))

DeltaTable.forName(spark, FEATURES_TABLE).alias("t").merge(
    spark_df.alias("s"),
    "t.ds = s.ds"
).whenMatchedUpdateAll(
).whenNotMatchedInsertAll(
).execute()

print(f"✓ Merged into {FEATURES_TABLE}")

# COMMAND ----------

# ── 16. UNITY CATALOG METADATA ────────────────────────────────────────────────

spark.sql(f"""
    COMMENT ON TABLE {FEATURES_TABLE} IS
    'Gold: forecast feature table. One row per day covering full history plus 16-day forward window.
Consumed by Prophet and LightGBM forecast models — both read from this single table.
Features include calendar signals, weather, lag values, rolling averages, and event flags.
y_revenue and y_orders are imputed for closure days and null for future dates.
training_weight column controls how each row is weighted during model training:
1.0 = normal observed day, 0.5 = imputed closure day, 0.0 = excluded distortion/event.'
""")

spark.sql(f"""
    ALTER TABLE {FEATURES_TABLE} SET TAGS (
        'domain'  = 'retail',
        'layer'   = 'gold',
        'purpose' = 'ml_features',
        'models'  = 'prophet_lightgbm',
        'refresh' = 'daily',
        'pii'     = 'false'
    )
""")

print(f"✓ Metadata applied to {FEATURES_TABLE}")

# COMMAND ----------

# ── 17. VALIDATION ────────────────────────────────────────────────────────────

print("\n── Feature table overview ──")
spark.sql(f"""
    SELECT
        COUNT(*)                            AS total_rows,
        MIN(ds)                             AS earliest_date,
        MAX(ds)                             AS latest_date,
        SUM(CAST(y_revenue IS NOT NULL AS INT)) AS historical_rows,
        SUM(CAST(y_revenue IS NULL AS INT))     AS future_rows,
        SUM(CAST(store_closed AS INT))          AS closed_days,
        SUM(CAST(is_imputed AS INT))            AS imputed_days,
        SUM(CAST(exclude_from_training AS INT)) AS excluded_days,
        ROUND(AVG(training_weight), 3)          AS avg_training_weight
    FROM {FEATURES_TABLE}
""").show(truncate=False)

print("\n── Training weight distribution ──")
spark.sql(f"""
    SELECT
        training_weight,
        COUNT(*)    AS days,
        MIN(ds)     AS example_from,
        MAX(ds)     AS example_to
    FROM {FEATURES_TABLE}
    GROUP BY training_weight
    ORDER BY training_weight DESC
""").show(truncate=False)

print("\n── Forward forecast window (future rows with weather) ──")
spark.sql(f"""
    SELECT
        ds,
        DATE_FORMAT(ds, 'EEEE')         AS day_name,
        is_bread_delivery_day,
        is_holiday_window,
        likely_future_closure,
        ROUND(weather_high_f, 1)        AS high_f,
        ROUND(weather_low_f, 1)         AS low_f,
        weather_category,
        ROUND(total_precip_in, 2)       AS precip_in,
        likely_snow_closure_risk
    FROM {FEATURES_TABLE}
    WHERE y_revenue IS NULL
    ORDER BY ds
""").show(20, truncate=False)

print("\n── Lag feature null check (first training rows should have nulls) ──")
spark.sql(f"""
    SELECT
        ds,
        lag_1_revenue,
        lag_7_revenue,
        lag_14_revenue,
        rolling_4w_same_dow_revenue
    FROM {FEATURES_TABLE}
    WHERE y_revenue IS NOT NULL
    ORDER BY ds
    LIMIT 20
""").show(truncate=False)