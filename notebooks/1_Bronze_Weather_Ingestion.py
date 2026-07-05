# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze — Weather Ingestion (Open-Meteo Historical Archive)
# MAGIC
# MAGIC Ingests hourly historical weather data for [your city], MA into
# MAGIC `YOUR_CATALOG.bronze.weather_hourly`.
# MAGIC
# MAGIC **Data source:** Open-Meteo Historical Archive API (free, no API key required)
# MAGIC https://open-meteo.com/en/docs/historical-weather-api
# MAGIC
# MAGIC **Note on data sources:**
# MAGIC - Historical weather (this notebook): Open-Meteo archive — reliable, hourly, going back years
# MAGIC - 7-day forecast (separate notebook): NWS / weather.gov API
# MAGIC - Both feed the same Gold weather table used by the sales forecast model
# MAGIC
# MAGIC ## Schedule
# MAGIC Run daily at 04:00 AM ET — after the Bronze sales ingestion at 2:59am.
# MAGIC Open-Meteo archive data lags ~5 days behind real-time, so the daily
# MAGIC run keeps the table current within that window automatically.
# MAGIC
# MAGIC ## Incremental logic
# MAGIC On each run the notebook finds the latest timestamp already in the table
# MAGIC and fetches only new data from that date forward. Safe to rerun — all
# MAGIC writes use MERGE INTO on `time` (hourly timestamp, UTC-offset).

# COMMAND ----------

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

import requests
import pandas as pd
import datetime
from delta.tables import DeltaTable
from pyspark.sql import functions as F

# COMMAND ----------

# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────

# [your city] village coordinates
LATITUDE  = YOUR_LATITUDE
LONGITUDE = YOUR_LONGITUDE
TIMEZONE  = "America/New_York"

# Open-Meteo archive endpoint
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Fields to request — chosen to support sales forecast model features:
#   temperature_2m        : actual air temp (°C) — converted to °F in transform
#   apparent_temperature  : feels-like temp — stronger behavioral driver than actual temp
#   precipitation         : rainfall (mm) — converted to inches in transform
#   snowfall              : snow depth (cm) — separate from rain, strong sales suppressor
#   weather_code          : WMO condition code — decoded to human-readable string
#   wind_speed_10m        : wind at 10m (km/h) — affects foot traffic on exposed coastal site
#   sunshine_duration     : seconds of sunshine per hour — proxy for "nice day" effect
#   cloud_cover           : total cloud cover (%) — complements sunshine_duration
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

# Destination — aligned with main pipeline catalog
CATALOG      = "YOUR_CATALOG"
SCHEMA       = "bronze"
TABLE        = "weather_hourly"
FULL_TABLE   = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Historical backfill start — aligned with Toast go-live
BACKFILL_START = "YOUR_TOAST_GOLIVE_DATE"

# Open-Meteo archive lags ~5 days. Don't request beyond this to avoid
# empty responses that could corrupt the watermark.
MAX_END_DATE = (
    datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5)
).strftime("%Y-%m-%d")

# COMMAND ----------

# ── 3. SETUP TABLE ────────────────────────────────────────────────────────────

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FULL_TABLE} (
        time                    TIMESTAMP   NOT NULL    COMMENT 'Hourly timestamp in America/New_York timezone',
        date                    DATE        NOT NULL    COMMENT 'Calendar date — derived from time for easy daily joins',

        temperature_f           DOUBLE                  COMMENT 'Air temperature in Fahrenheit at 2m height',
        apparent_temperature_f  DOUBLE                  COMMENT 'Feels-like temperature in Fahrenheit — stronger behavioral driver than actual temp',
        temperature_c           DOUBLE                  COMMENT 'Air temperature in Celsius (source value from API)',
        apparent_temperature_c  DOUBLE                  COMMENT 'Feels-like temperature in Celsius (source value from API)',

        precipitation_in        DOUBLE                  COMMENT 'Precipitation in inches',
        precipitation_mm        DOUBLE                  COMMENT 'Precipitation in millimeters (source value from API)',
        snowfall_in             DOUBLE                  COMMENT 'Snowfall in inches',
        snowfall_cm             DOUBLE                  COMMENT 'Snowfall in centimeters (source value from API)',

        weather_code            INTEGER                 COMMENT 'WMO weather condition code (raw)',
        weather_condition       STRING                  COMMENT 'Human-readable weather condition decoded from weather_code',
        weather_category        STRING                  COMMENT 'Simplified category for modeling: Clear, Cloudy, Rainy, Snowy, Stormy, Foggy',

        wind_speed_mph          DOUBLE                  COMMENT 'Wind speed in mph at 10m height',
        wind_speed_kmh          DOUBLE                  COMMENT 'Wind speed in km/h (source value from API)',

        sunshine_minutes        DOUBLE                  COMMENT 'Minutes of sunshine in the hour (converted from seconds)',
        sunshine_seconds        DOUBLE                  COMMENT 'Seconds of sunshine in the hour (source value from API)',
        is_sunny_hour           BOOLEAN                 COMMENT 'True if sunshine_minutes > 30 — more than half the hour was sunny',

        cloud_cover_pct         INTEGER                 COMMENT 'Total cloud cover as a percentage (0-100)',

        _ingested_at            TIMESTAMP               COMMENT 'UTC timestamp when this row was written by the pipeline',
        _ingestion_batch_id     STRING                  COMMENT 'UUID for the ingestion run'
    )
    USING DELTA
    PARTITIONED BY (date)
    COMMENT 'Bronze: hourly historical weather for [your city] MA from Open-Meteo archive. Covers Toast go-live (YOUR_TOAST_GOLIVE_DATE) onwards.'
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'quality' = 'bronze'
    )
""")

print(f"✓ {FULL_TABLE} ready")

# COMMAND ----------

# ── 4. DETERMINE DATE RANGE ───────────────────────────────────────────────────
# Find the latest date already in the table and fetch from there forward.
# Uses timezone-aware datetime throughout to avoid deprecation warnings.

try:
    latest_row = spark.sql(f"""
        SELECT MAX(date) AS latest_date FROM {FULL_TABLE}
    """).collect()[0]["latest_date"]
except Exception:
    latest_row = None

if latest_row is None:
    # No data yet — full backfill from go-live date
    start_date = BACKFILL_START
    print(f"No existing data — backfilling from {start_date}")
else:
    # Resume from the day after the last complete day in the table
    # We re-fetch the latest date too in case the previous run was partial
    start_date = (latest_row - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Incremental run — fetching from {start_date} (re-fetches last day for safety)")

end_date = MAX_END_DATE

if start_date >= end_date:
    print(f"Table is current (latest: {latest_row}, archive lag end: {end_date}). Nothing to fetch.")
    dbutils.notebook.exit("UP_TO_DATE")

print(f"Fetching: {start_date} → {end_date}")

# COMMAND ----------

# ── 5. CALL OPEN-METEO API ────────────────────────────────────────────────────

api_url = (
    f"{ARCHIVE_URL}"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    f"&start_date={start_date}"
    f"&end_date={end_date}"
    f"&hourly={HOURLY_PARAMS}"
    f"&timezone={TIMEZONE.replace('/', '%2F')}"
)

print(f"Calling: {api_url}")

response = requests.get(api_url, timeout=60)
response.raise_for_status()

data = response.json()

if "hourly" not in data:
    raise RuntimeError(
        f"Open-Meteo response missing 'hourly' key. Full response: {data}"
    )

print(f"✓ API returned {len(data['hourly']['time'])} hourly records")

# COMMAND ----------

# ── 6. TRANSFORM ──────────────────────────────────────────────────────────────

# WMO weather code → human-readable condition
WEATHER_CODE_MAP = {
    0:  ("Clear sky",            "Clear"),
    1:  ("Mainly clear",         "Clear"),
    2:  ("Partly cloudy",        "Cloudy"),
    3:  ("Overcast",             "Cloudy"),
    45: ("Fog",                  "Foggy"),
    48: ("Rime fog",             "Foggy"),
    51: ("Light drizzle",        "Rainy"),
    53: ("Moderate drizzle",     "Rainy"),
    55: ("Dense drizzle",        "Rainy"),
    61: ("Slight rain",          "Rainy"),
    63: ("Moderate rain",        "Rainy"),
    65: ("Heavy rain",           "Rainy"),
    71: ("Slight snow",          "Snowy"),
    73: ("Moderate snow",        "Snowy"),
    75: ("Heavy snow",           "Snowy"),
    77: ("Snow grains",          "Snowy"),
    80: ("Slight rain showers",  "Rainy"),
    81: ("Moderate rain showers","Rainy"),
    82: ("Heavy rain showers",   "Rainy"),
    85: ("Slight snow showers",  "Snowy"),
    86: ("Heavy snow showers",   "Snowy"),
    95: ("Thunderstorm",         "Stormy"),
    96: ("Thunderstorm w/ hail", "Stormy"),
    99: ("Thunderstorm w/ hail", "Stormy"),
}

df = pd.DataFrame(data["hourly"])

# Rename source columns to make intent clear before transforming
df = df.rename(columns={
    "temperature_2m":       "temperature_c",
    "apparent_temperature": "apparent_temperature_c",
    "precipitation":        "precipitation_mm",
    "snowfall":             "snowfall_cm",
    "wind_speed_10m":       "wind_speed_kmh",
    "sunshine_duration":    "sunshine_seconds",
    "cloud_cover":          "cloud_cover_pct",
})

# Unit conversions
df["temperature_f"]         = (df["temperature_c"] * 9 / 5 + 32).round(1)
df["apparent_temperature_f"]= (df["apparent_temperature_c"] * 9 / 5 + 32).round(1)
df["precipitation_in"]      = (df["precipitation_mm"] / 25.4).round(3)
df["snowfall_in"]           = (df["snowfall_cm"] / 2.54).round(2)
df["wind_speed_mph"]        = (df["wind_speed_kmh"] * 0.621371).round(1)
df["sunshine_minutes"]      = (df["sunshine_seconds"] / 60).round(1)
df["is_sunny_hour"]         = df["sunshine_minutes"] > 30

# Decode weather codes
df["weather_condition"] = df["weather_code"].map(
    lambda c: WEATHER_CODE_MAP.get(c, (f"Unknown ({c})", "Unknown"))[0]
)
df["weather_category"] = df["weather_code"].map(
    lambda c: WEATHER_CODE_MAP.get(c, (f"Unknown ({c})", "Unknown"))[1]
)

# Parse timestamp and extract date
df["time"] = pd.to_datetime(df["time"])
df["date"] = df["time"].dt.date

# Audit columns
batch_id = str(__import__("uuid").uuid4())
now_utc  = datetime.datetime.now(datetime.timezone.utc)
df["_ingested_at"]        = now_utc
df["_ingestion_batch_id"] = batch_id

# Final column order matching table DDL
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
    "_ingested_at", "_ingestion_batch_id",
]]

print(f"Transformed {len(df)} rows covering {df['date'].min()} → {df['date'].max()}")

# COMMAND ----------

# ── 7. MERGE INTO BRONZE ──────────────────────────────────────────────────────

spark_df = spark.createDataFrame(df) \
    .withColumn("time",   F.col("time").cast("timestamp")) \
    .withColumn("date",   F.col("date").cast("date")) \
    .withColumn("cloud_cover_pct", F.col("cloud_cover_pct").cast("integer"))

weather_table = DeltaTable.forName(spark, FULL_TABLE)

weather_table.alias("target").merge(
    spark_df.alias("source"),
    "target.time = source.time"
).whenMatchedUpdateAll(
).whenNotMatchedInsertAll(
).execute()

print(f"✓ Merged into {FULL_TABLE}")

# COMMAND ----------

# ── 8. APPLY UNITY CATALOG METADATA ──────────────────────────────────────────
# Applied on every run so metadata stays current after schema changes.

spark.sql(f"""
    COMMENT ON TABLE {FULL_TABLE} IS
    'Bronze: hourly historical weather observations for [your city] MA (lat YOUR_LATITUDE, lon YOUR_LONGITUDE).
Source: Open-Meteo Historical Archive API (free, no key required). Covers YOUR_TOAST_GOLIVE_DATE onwards.
Updated daily — Open-Meteo archive lags ~5 days behind real-time.
Used as input to the sales forecast model and the Gold weather summary table.
Join to gold.daily_sales_summary on date for weather-correlated sales analysis.'
""")

spark.sql(f"""
    ALTER TABLE {FULL_TABLE} SET TAGS (
        'domain'   = 'weather',
        'source'   = 'open_meteo_archive',
        'location' = 'cohasset_ma',
        'pii'      = 'false',
        'layer'    = 'bronze',
        'refresh'  = 'daily'
    )
""")

column_comments = {
    "time":                   "Hourly timestamp in America/New_York timezone. Primary key — one row per hour.",
    "date":                   "Calendar date derived from time. Use this for daily joins to sales data.",
    "temperature_f":          "Actual air temperature in Fahrenheit at 2 meters height.",
    "apparent_temperature_f": "Feels-like temperature in Fahrenheit. Stronger predictor of customer behavior than actual temp.",
    "temperature_c":          "Actual air temperature in Celsius. Source value from Open-Meteo.",
    "apparent_temperature_c": "Feels-like temperature in Celsius. Source value from Open-Meteo.",
    "precipitation_in":       "Total precipitation (rain) in inches for the hour.",
    "precipitation_mm":       "Total precipitation in millimeters. Source value from Open-Meteo.",
    "snowfall_in":            "Snowfall in inches for the hour. Distinct from rain — strong foot traffic suppressor.",
    "snowfall_cm":            "Snowfall in centimeters. Source value from Open-Meteo.",
    "weather_code":           "WMO weather condition code (raw integer). See weather_condition for decoded label.",
    "weather_condition":      "Human-readable weather description decoded from weather_code (e.g. Moderate rain, Partly cloudy).",
    "weather_category":       "Simplified weather category for modeling: Clear, Cloudy, Rainy, Snowy, Stormy, Foggy.",
    "wind_speed_mph":         "Wind speed in miles per hour at 10 meters height.",
    "wind_speed_kmh":         "Wind speed in km/h. Source value from Open-Meteo.",
    "sunshine_minutes":       "Minutes of sunshine in the hour (0-60). Converted from sunshine_seconds.",
    "sunshine_seconds":       "Seconds of sunshine in the hour. Source value from Open-Meteo.",
    "is_sunny_hour":          "True if more than 30 minutes of the hour had direct sunshine.",
    "cloud_cover_pct":        "Total cloud cover as a percentage (0 = clear sky, 100 = fully overcast).",
    "_ingested_at":           "UTC timestamp when this row was written to the Bronze table.",
    "_ingestion_batch_id":    "UUID for the ingestion run. Links all rows written in the same execution.",
}

for col_name, comment in column_comments.items():
    safe_comment = comment.replace("'", "''")
    spark.sql(f"""
        ALTER TABLE {FULL_TABLE}
        ALTER COLUMN `{col_name}`
        COMMENT '{safe_comment}'
    """)

column_tags = {
    "time":                   {"semantic": "primary_key",  "semantic_type": "timestamp"},
    "date":                   {"semantic": "foreign_key",  "semantic_type": "date"},
    "temperature_f":          {"semantic": "metric",       "ml_feature": "true"},
    "apparent_temperature_f": {"semantic": "metric",       "ml_feature": "true"},
    "precipitation_in":       {"semantic": "metric",       "ml_feature": "true"},
    "snowfall_in":            {"semantic": "metric",       "ml_feature": "true"},
    "weather_category":       {"semantic": "dimension",    "ml_feature": "true"},
    "wind_speed_mph":         {"semantic": "metric",       "ml_feature": "true"},
    "sunshine_minutes":       {"semantic": "metric",       "ml_feature": "true"},
    "cloud_cover_pct":        {"semantic": "metric",       "ml_feature": "true"},
    "is_sunny_hour":          {"semantic": "flag",         "ml_feature": "true"},
    "_ingested_at":           {"semantic": "audit"},
    "_ingestion_batch_id":    {"semantic": "audit"},
}

for col_name, tags in column_tags.items():
    tag_pairs = ", ".join(f"'{k}' = '{v}'" for k, v in tags.items())
    spark.sql(f"""
        ALTER TABLE {FULL_TABLE}
        ALTER COLUMN `{col_name}`
        SET TAGS ({tag_pairs})
    """)

print(f"✓ Metadata applied to {FULL_TABLE}")

# COMMAND ----------

# ── 9. VALIDATION ─────────────────────────────────────────────────────────────

print("\n── Coverage and row counts ──")
spark.sql(f"""
    SELECT
        MIN(date)           AS earliest_date,
        MAX(date)           AS latest_date,
        COUNT(*)            AS total_hours,
        COUNT(DISTINCT date) AS total_days,
        ROUND(COUNT(*) / COUNT(DISTINCT date), 1) AS avg_hours_per_day
    FROM {FULL_TABLE}
""").show(truncate=False)

print("\n── Weather category distribution ──")
spark.sql(f"""
    SELECT
        weather_category,
        COUNT(*)                        AS hours,
        ROUND(COUNT(*) * 100.0
            / SUM(COUNT(*)) OVER (), 1) AS pct_of_hours
    FROM {FULL_TABLE}
    GROUP BY weather_category
    ORDER BY hours DESC
""").show(truncate=False)

print("\n── Recent 3 days sample ──")
spark.sql(f"""
    SELECT
        date,
        ROUND(AVG(temperature_f), 1)            AS avg_temp_f,
        ROUND(AVG(apparent_temperature_f), 1)   AS avg_feels_like_f,
        ROUND(SUM(precipitation_in), 2)         AS total_precip_in,
        ROUND(SUM(snowfall_in), 2)              AS total_snow_in,
        ROUND(AVG(cloud_cover_pct), 0)          AS avg_cloud_pct,
        ROUND(SUM(sunshine_minutes) / 60, 1)   AS sunny_hours,
        MAX(weather_category)                   AS predominant_category
    FROM {FULL_TABLE}
    WHERE date >= CURRENT_DATE - INTERVAL 3 DAYS
    GROUP BY date
    ORDER BY date DESC
""").show(truncate=False)