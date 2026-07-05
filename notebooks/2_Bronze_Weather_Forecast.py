# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze — Weather Forecast Ingestion (Open-Meteo Forecast API)
# MAGIC
# MAGIC Fetches hourly weather forecast data for [your city], MA and merges it into
# MAGIC `YOUR_CATALOG.bronze.weather_hourly` — the same table used by
# MAGIC the historical archive notebook.
# MAGIC
# MAGIC **How this fits the pipeline:**
# MAGIC
# MAGIC | Notebook | Source | Coverage | Runs |
# MAGIC |---|---|---|---|
# MAGIC | `1_Bronze_Weather_Ingestion` | Open-Meteo Archive | YOUR_TOAST_GOLIVE_DATE → ~5 days ago | Daily 01:00 AM |
# MAGIC | `2_Bronze_Weather_Forecast` (this one) | Open-Meteo Forecast | ~3 days ago → +16 days | Daily 01:15 AM |
# MAGIC
# MAGIC The two notebooks share the same table and MERGE on `time`. Where they
# MAGIC overlap, the archive notebook's values eventually overwrite forecast values
# MAGIC as the archive catches up — reanalysis is more accurate than the forecast
# MAGIC was at the time. The `data_type` column records which source each row
# MAGIC came from (`archive` or `forecast`) for full lineage.
# MAGIC
# MAGIC **Why Open-Meteo forecast instead of NWS:**
# MAGIC Identical field names, units, and structure to the archive API — zero
# MAGIC training-serving skew for the ML sales forecast model. Open-Meteo forecast
# MAGIC accuracy at 1-7 days is comparable to NWS for temperature and precipitation.

# COMMAND ----------

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

import requests
import pandas as pd
import datetime
import uuid
from delta.tables import DeltaTable
from pyspark.sql import functions as F

# COMMAND ----------

# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────

# [your city] village coordinates — must match archive notebook exactly
LATITUDE  = YOUR_LATITUDE
LONGITUDE = YOUR_LONGITUDE
TIMEZONE  = "America/New_York"

# Open-Meteo forecast endpoint
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Identical fields to the archive notebook — zero schema divergence
HOURLY_PARAMS = ",".join([
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
    "sunshine_duration",
    "cloud_cover",
])

# How many days back to include from the forecast API
# Open-Meteo forecast includes up to 3 days of recent actuals before today.
# This bridges the ~5 day archive lag window.
PAST_DAYS    = 3
FORECAST_DAYS = 16   # Open-Meteo max; covers well beyond the 7-day sales forecast need

CATALOG    = "YOUR_CATALOG"
SCHEMA     = "bronze"
TABLE      = "weather_hourly"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

# COMMAND ----------

# ── 3. CALL OPEN-METEO FORECAST API ──────────────────────────────────────────

api_url = (
    f"{FORECAST_URL}"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    f"&hourly={HOURLY_PARAMS}"
    f"&timezone={TIMEZONE.replace('/', '%2F')}"
    f"&past_days={PAST_DAYS}"
    f"&forecast_days={FORECAST_DAYS}"
)

print(f"Calling: {api_url}")

response = requests.get(api_url, timeout=60)
response.raise_for_status()

data = response.json()

if "hourly" not in data:
    raise RuntimeError(
        f"Open-Meteo response missing 'hourly' key. Full response: {data}"
    )

total_hours = len(data["hourly"]["time"])
print(f"✓ API returned {total_hours} hourly records")
print(f"  Coverage: {data['hourly']['time'][0]} → {data['hourly']['time'][-1]}")

# COMMAND ----------

# ── 4. TRANSFORM ──────────────────────────────────────────────────────────────
# Identical transform logic to the archive notebook to ensure the two
# sources produce structurally identical rows in the shared table.

WEATHER_CODE_MAP = {
    0:  ("Clear sky",             "Clear"),
    1:  ("Mainly clear",          "Clear"),
    2:  ("Partly cloudy",         "Cloudy"),
    3:  ("Overcast",              "Cloudy"),
    45: ("Fog",                   "Foggy"),
    48: ("Rime fog",              "Foggy"),
    51: ("Light drizzle",         "Rainy"),
    53: ("Moderate drizzle",      "Rainy"),
    55: ("Dense drizzle",         "Rainy"),
    61: ("Slight rain",           "Rainy"),
    63: ("Moderate rain",         "Rainy"),
    65: ("Heavy rain",            "Rainy"),
    71: ("Slight snow",           "Snowy"),
    73: ("Moderate snow",         "Snowy"),
    75: ("Heavy snow",            "Snowy"),
    77: ("Snow grains",           "Snowy"),
    80: ("Slight rain showers",   "Rainy"),
    81: ("Moderate rain showers", "Rainy"),
    82: ("Heavy rain showers",    "Rainy"),
    85: ("Slight snow showers",   "Snowy"),
    86: ("Heavy snow showers",    "Snowy"),
    95: ("Thunderstorm",          "Stormy"),
    96: ("Thunderstorm w/ hail",  "Stormy"),
    99: ("Thunderstorm w/ hail",  "Stormy"),
}

df = pd.DataFrame(data["hourly"])

df = df.rename(columns={
    "temperature_2m":       "temperature_c",
    "apparent_temperature": "apparent_temperature_c",
    "precipitation":        "precipitation_mm",
    "snowfall":             "snowfall_cm",
    "wind_speed_10m":       "wind_speed_kmh",
    "sunshine_duration":    "sunshine_seconds",
    "cloud_cover":          "cloud_cover_pct",
})

# Unit conversions — identical to archive notebook
df["temperature_f"]          = (df["temperature_c"] * 9 / 5 + 32).round(1)
df["apparent_temperature_f"] = (df["apparent_temperature_c"] * 9 / 5 + 32).round(1)
df["precipitation_in"]       = (df["precipitation_mm"] / 25.4).round(3)
df["snowfall_in"]            = (df["snowfall_cm"] / 2.54).round(2)
df["wind_speed_mph"]         = (df["wind_speed_kmh"] * 0.621371).round(1)
df["sunshine_minutes"]       = (df["sunshine_seconds"] / 60).round(1)
df["is_sunny_hour"]          = df["sunshine_minutes"] > 30

df["weather_condition"] = df["weather_code"].map(
    lambda c: WEATHER_CODE_MAP.get(c, (f"Unknown ({c})", "Unknown"))[0]
)
df["weather_category"] = df["weather_code"].map(
    lambda c: WEATHER_CODE_MAP.get(c, (f"Unknown ({c})", "Unknown"))[1]
)

df["time"] = pd.to_datetime(df["time"])
df["date"] = df["time"].dt.date

# data_type distinguishes forecast rows from archive rows in the shared table
# Past rows will eventually be overwritten by archive reanalysis values —
# data_type='archive' rows are considered more accurate than 'forecast' rows
# for the same timestamp.
today = datetime.datetime.now(datetime.timezone.utc).date()
df["data_type"] = df["date"].apply(
    lambda d: "archive_pending" if d <= today else "forecast"
)
# archive_pending = within the ~5 day lag window; will be overwritten by archive notebook
# forecast        = future dates; will remain as forecast until the date passes + 5 days

batch_id = str(uuid.uuid4())
now_utc  = datetime.datetime.now(datetime.timezone.utc)
df["_ingested_at"]        = now_utc
df["_ingestion_batch_id"] = batch_id

df = df[[
    "time", "date",
    "temperature_f", "apparent_temperature_f",
    "temperature_c", "apparent_temperature_c",
    "precipitation_in", "precipitation_mm",
    "snowfall_in", "snowfall_cm",
    "weather_code", "weather_condition", "weather_category",
    "wind_speed_mph", "wind_speed_kmh",
    "sunshine_minutes", "sunshine_seconds", "is_sunny_hour",
    "cloud_cover_pct",
    "data_type",
    "_ingested_at", "_ingestion_batch_id",
]]

now_local = datetime.datetime.now()
future_rows = (df["date"] > today).sum()
past_rows   = (df["date"] <= today).sum()
print(f"Rows covering past/gap ({PAST_DAYS}d): {past_rows}")
print(f"Rows covering future ({FORECAST_DAYS - PAST_DAYS}d): {future_rows}")

# COMMAND ----------

# ── 5. ADD data_type COLUMN TO TABLE IF NOT EXISTS ────────────────────────────
# The archive notebook predates this column. Add it safely if missing.

existing_cols = [f.name for f in spark.table(FULL_TABLE).schema.fields]

if "data_type" not in existing_cols:
    spark.sql(f"""
        ALTER TABLE {FULL_TABLE}
        ADD COLUMN data_type STRING
        COMMENT 'Source of this row: archive = Open-Meteo reanalysis (final), archive_pending = within lag window (will be overwritten), forecast = future prediction'
    """)
    # Backfill existing rows as archive since they came from the archive notebook
    spark.sql(f"UPDATE {FULL_TABLE} SET data_type = 'archive' WHERE data_type IS NULL")
    print("✓ Added data_type column to existing table and backfilled as archive")
else:
    print("✓ data_type column already present")

# COMMAND ----------

# ── 6. MERGE INTO SHARED TABLE ────────────────────────────────────────────────
# Merge on time (hourly timestamp).
# For past/gap rows (archive_pending): insert if missing, but do NOT overwrite
#   existing rows that are already marked 'archive' — reanalysis wins over forecast.
# For future rows: always upsert — forecasts update daily with better models.

spark_df = (
    spark.createDataFrame(df)
    .withColumn("time",          F.col("time").cast("timestamp"))
    .withColumn("date",          F.col("date").cast("date"))
    .withColumn("cloud_cover_pct", F.col("cloud_cover_pct").cast("integer"))
)

weather_table = DeltaTable.forName(spark, FULL_TABLE)

weather_table.alias("target").merge(
    spark_df.alias("source"),
    "target.time = source.time"
).whenMatchedUpdate(
    # Only overwrite existing rows if they are NOT finalized archive data
    condition="target.data_type != 'archive'",
    set={col: f"source.{col}" for col in df.columns}
).whenNotMatchedInsertAll(
).execute()

print(f"✓ Merged into {FULL_TABLE}")

# COMMAND ----------

# ── 7. VALIDATION ─────────────────────────────────────────────────────────────

print("\n── Row counts by data_type ──")
spark.sql(f"""
    SELECT
        data_type,
        COUNT(*)            AS hours,
        MIN(date)           AS from_date,
        MAX(date)           AS to_date
    FROM {FULL_TABLE}
    GROUP BY data_type
    ORDER BY from_date
""").show(truncate=False)

print("\n── 7-day forward forecast summary ──")
spark.sql(f"""
    SELECT
        date,
        DAYOFWEEK(date)                         AS day_of_week,
        DATE_FORMAT(date, 'EEEE')               AS day_name,
        ROUND(MIN(temperature_f), 1)            AS low_f,
        ROUND(MAX(temperature_f), 1)            AS high_f,
        ROUND(MIN(apparent_temperature_f), 1)   AS feels_low_f,
        ROUND(MAX(apparent_temperature_f), 1)   AS feels_high_f,
        ROUND(SUM(precipitation_in), 2)         AS total_precip_in,
        ROUND(SUM(snowfall_in), 2)              AS total_snow_in,
        ROUND(SUM(sunshine_minutes) / 60, 1)   AS sunny_hours,
        ROUND(AVG(cloud_cover_pct), 0)          AS avg_cloud_pct,
        MAX(weather_category)                   AS predominant_category
    FROM {FULL_TABLE}
    WHERE date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL 6 DAYS
    GROUP BY date
    ORDER BY date
""").show(truncate=False)

print("\n── Gap coverage check (last 7 days — should have no nulls) ──")
spark.sql(f"""
    SELECT
        date,
        data_type,
        COUNT(*)    AS hours,
        SUM(CASE WHEN temperature_f IS NULL THEN 1 ELSE 0 END) AS null_temp_hours
    FROM {FULL_TABLE}
    WHERE date BETWEEN CURRENT_DATE - INTERVAL 7 DAYS AND CURRENT_DATE
    GROUP BY date, data_type
    ORDER BY date
""").show(truncate=False)