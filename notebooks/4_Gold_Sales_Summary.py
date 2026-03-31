# Databricks notebook source
# MAGIC %md
# MAGIC # Gold — Sales Summary Tables
# MAGIC
# MAGIC Builds three Gold tables from Silver sales data, joined to weather and
# MAGIC the reference item catalog. **Actuals only** — forecasts live in
# MAGIC `gold.daily_sales_forecast` (written by `9_Forecast_Generate`).
# MAGIC The Platinum view `platinum.daily_sales_combined` joins both for dashboards.
# MAGIC
# MAGIC | Table | Grain | Powers |
# MAGIC |---|---|---|
# MAGIC | `gold.hourly_sales_summary` | Hour × day | Today's hourly revenue + traffic strip |
# MAGIC | `gold.daily_sales_summary` | Day (actuals only) | 7-day back revenue + weather |
# MAGIC | `gold.daily_sales_by_category` | Day × category | Category revenue breakdown chart |
# MAGIC
# MAGIC ## Run modes
# MAGIC | Mode | What it does |
# MAGIC |---|---|
# MAGIC | `incremental` | Processes only dates not yet in Gold — run on daily schedule |
# MAGIC | `full_refresh` | Rebuilds all Gold tables from scratch — run after schema changes |
# MAGIC
# MAGIC ## Schedule
# MAGIC Run daily at 02:00 AM ET — after Bronze sales (midnight) and weather (01:00 AM).

# COMMAND ----------

# ── WIDGET ────────────────────────────────────────────────────────────────────

dbutils.widgets.dropdown(
    name="run_mode",
    defaultValue="incremental",
    choices=["incremental", "full_refresh"],
    label="Run Mode"
)

# COMMAND ----------

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────

import datetime
import uuid
from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql import Window

# COMMAND ----------

# ── 2. CONFIGURATION ──────────────────────────────────────────────────────────

RUN_MODE = dbutils.widgets.get("run_mode")

CATALOG  = "3sp_analytics_workspace"

# Source tables
ORDERS_SILVER   = f"{CATALOG}.silver_sales.orders_silver"
ITEMS_SILVER    = f"{CATALOG}.silver_sales.order_items_silver"
ITEM_CATALOG    = f"{CATALOG}.reference.item_catalog"
WEATHER_HOURLY  = f"{CATALOG}.bronze.weather_hourly"

# Gold targets
GOLD_SCHEMA             = f"{CATALOG}.gold"
HOURLY_TABLE            = f"{GOLD_SCHEMA}.hourly_sales_summary"
DAILY_TABLE             = f"{GOLD_SCHEMA}.daily_sales_summary"
CATEGORY_TABLE          = f"{GOLD_SCHEMA}.daily_sales_by_category"

# Store hours in LOCAL TIME (America/New_York).
# opened_date in Silver is UTC — we convert to ET before extracting the hour.
# Data shows first orders ~10am ET, last ~6pm ET.
# Spine covers 9am-7pm ET to include potential edge cases on both ends.
STORE_OPEN_HOUR  = 9    # 9 AM ET
STORE_CLOSE_HOUR = 19   # 7 PM ET
LOCAL_TZ         = "America/New_York"

NOW_UTC   = datetime.datetime.now(datetime.timezone.utc)
BATCH_ID  = str(uuid.uuid4())
NOW_TS    = NOW_UTC.isoformat()

# COMMAND ----------

# ── 3. SETUP SCHEMA AND TABLES ────────────────────────────────────────────────

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD_SCHEMA}")

# ── hourly_sales_summary ──────────────────────────────────────────────────────
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {HOURLY_TABLE} (
        business_date           DATE        NOT NULL    COMMENT 'The business date for this hour',
        hour_of_day             INTEGER     NOT NULL    COMMENT 'Hour of day in local time (0-23)',
        hour_label              STRING                  COMMENT 'Display label for the hour (e.g. 9 AM, 2 PM)',

        -- Sales metrics
        order_count             INTEGER                 COMMENT 'Number of orders (tickets) opened in this hour',
        gross_revenue           DOUBLE                  COMMENT 'Sum of order total_amount before discounts',
        net_revenue             DOUBLE                  COMMENT 'Sum of order total_amount after discounts',
        total_discounts         DOUBLE                  COMMENT 'Sum of all discounts applied in this hour',
        avg_ticket_size         DOUBLE                  COMMENT 'Average net revenue per order in this hour',
        item_count              INTEGER                 COMMENT 'Total line items sold in this hour',

        -- Weather for this hour
        temperature_f           DOUBLE                  COMMENT 'Actual or forecast temperature in Fahrenheit for this hour',
        apparent_temperature_f  DOUBLE                  COMMENT 'Feels-like temperature in Fahrenheit for this hour',
        weather_condition       STRING                  COMMENT 'Human-readable weather condition for this hour',
        weather_category        STRING                  COMMENT 'Simplified weather category: Clear, Cloudy, Rainy, Snowy, Stormy, Foggy',
        weather_code            INTEGER                 COMMENT 'WMO weather code for icon mapping in the dashboard',
        precipitation_in        DOUBLE                  COMMENT 'Precipitation in inches for this hour',
        weather_data_type       STRING                  COMMENT 'Whether weather data is archive or forecast for this hour',

        -- Flags
        is_store_hours          BOOLEAN                 COMMENT 'True if this hour falls within the standard 9am-7pm store window',
        has_transactions        BOOLEAN                 COMMENT 'True if at least one order occurred in this hour',

        -- Audit
        _gold_updated_at        TIMESTAMP               COMMENT 'When this row was last written to Gold',
        _batch_id               STRING                  COMMENT 'Batch ID for this Gold run'
    )
    USING DELTA
    PARTITIONED BY (business_date)
    COMMENT 'Gold: hourly sales summary for Cohasset store. One row per hour per day covering store hours (9am-7pm) plus any hours with actual transactions outside that window. Powers the hourly revenue and traffic strip on the dashboard.'
    TBLPROPERTIES ('quality' = 'gold', 'delta.enableChangeDataFeed' = 'true')
""")

# ── daily_sales_summary ───────────────────────────────────────────────────────
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {DAILY_TABLE} (
        business_date           DATE        NOT NULL    COMMENT 'The business date',

        -- Sales metrics (actuals from Silver only)
        order_count             INTEGER                 COMMENT 'Number of orders (tickets) for the day',
        gross_revenue           DOUBLE                  COMMENT 'Total revenue before discounts',
        net_revenue             DOUBLE                  COMMENT 'Total revenue after discounts, excluding tips',
        total_discounts         DOUBLE                  COMMENT 'Total discount value applied',
        avg_ticket_size         DOUBLE                  COMMENT 'Average net revenue per order',
        item_count              INTEGER                 COMMENT 'Total line items sold',

        -- Day-of-week features (useful for dashboard annotations and model features)
        day_of_week             INTEGER                 COMMENT 'Day of week: 1=Sunday, 2=Monday ... 7=Saturday',
        day_name                STRING                  COMMENT 'Full day name (Monday, Tuesday, etc.)',
        is_weekend              BOOLEAN                 COMMENT 'True for Saturday and Sunday',
        is_bread_delivery_day   BOOLEAN                 COMMENT 'True on Tuesdays and Fridays — Kneady Mama bread delivery days, associated with higher foot traffic',
        week_of_year            INTEGER                 COMMENT 'ISO week number — useful for week-on-week comparison',
        month                   INTEGER                 COMMENT 'Month number (1-12)',
        year                    INTEGER                 COMMENT 'Calendar year',

        -- Weather summary for the day
        weather_high_f          DOUBLE                  COMMENT 'Daily high temperature in Fahrenheit',
        weather_low_f           DOUBLE                  COMMENT 'Daily low temperature in Fahrenheit',
        weather_feels_high_f    DOUBLE                  COMMENT 'Daily high feels-like temperature in Fahrenheit',
        weather_feels_low_f     DOUBLE                  COMMENT 'Daily low feels-like temperature in Fahrenheit',
        weather_category        STRING                  COMMENT 'Predominant weather category for the day: Clear, Cloudy, Rainy, Snowy, Stormy, Foggy',
        weather_condition       STRING                  COMMENT 'Most descriptive weather condition observed during the day',
        weather_code            INTEGER                 COMMENT 'WMO weather code for the predominant condition — use this for icon mapping in the dashboard',
        total_precip_in         DOUBLE                  COMMENT 'Total precipitation in inches for the day',
        total_snow_in           DOUBLE                  COMMENT 'Total snowfall in inches for the day',
        sunny_hours             DOUBLE                  COMMENT 'Number of hours with more than 30 minutes of direct sunshine',
        avg_cloud_cover_pct     DOUBLE                  COMMENT 'Average cloud cover percentage across all daylight hours',
        weather_data_type       STRING                  COMMENT 'archive, archive_pending, or forecast — indicates source of weather data',

        -- Audit
        _gold_updated_at        TIMESTAMP               COMMENT 'When this row was last written to Gold',
        _batch_id               STRING                  COMMENT 'Batch ID for this Gold run'
    )
    USING DELTA
    PARTITIONED BY (year, month)
    COMMENT 'Gold: daily sales actuals with weather. One row per business day. Actuals only — forecasts live in gold.daily_sales_forecast. Join both via platinum.daily_sales_combined.'
    TBLPROPERTIES ('quality' = 'gold', 'delta.enableChangeDataFeed' = 'true')
""")

# ── daily_sales_by_category ───────────────────────────────────────────────────
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {CATEGORY_TABLE} (
        business_date           DATE        NOT NULL    COMMENT 'The business date',
        category_group          STRING      NOT NULL    COMMENT 'Top-level category group: Grocery, Alcohol, Cafe Menu, Housewares and Decor, etc.',
        category                STRING      NOT NULL    COMMENT 'Mid-level menu category: Candy, Beer, Sandwiches, etc. Primary reporting dimension.',

        -- Sales metrics
        order_count             INTEGER                 COMMENT 'Number of distinct orders containing at least one item in this category',
        item_count              INTEGER                 COMMENT 'Total quantity of items sold in this category',
        gross_revenue           DOUBLE                  COMMENT 'Sum of receipt_line_price for all items in this category',
        avg_item_price          DOUBLE                  COMMENT 'Average unit price for items in this category on this day',

        -- Tax flag (useful for tax reporting)
        has_taxable_items       BOOLEAN                 COMMENT 'True if any items in this category on this day had tax > 0',

        -- Audit
        _gold_updated_at        TIMESTAMP               COMMENT 'When this row was last written to Gold',
        _batch_id               STRING                  COMMENT 'Batch ID for this Gold run'
    )
    USING DELTA
    PARTITIONED BY (business_date)
    COMMENT 'Gold: daily sales broken down by category. One row per date + category_group + category. Actuals only — joins Silver order items to reference item catalog for category names. Powers the category revenue breakdown chart.'
    TBLPROPERTIES ('quality' = 'gold', 'delta.enableChangeDataFeed' = 'true')
""")

print(f"✓ {HOURLY_TABLE}")
print(f"✓ {DAILY_TABLE}")
print(f"✓ {CATEGORY_TABLE}")

# COMMAND ----------

# ── 4. DETERMINE DATES TO PROCESS ─────────────────────────────────────────────

yesterday = (NOW_UTC - datetime.timedelta(days=1)).date()

if RUN_MODE == "incremental":
    try:
        last_gold_date = spark.sql(f"""
            SELECT MAX(business_date) AS d
            FROM {DAILY_TABLE}
        """).collect()[0]["d"]
    except Exception:
        last_gold_date = None

    if last_gold_date is None:
        # First run — process everything in Silver
        process_from = spark.sql(f"""
            SELECT MIN(business_date) AS d FROM {ORDERS_SILVER}
        """).collect()[0]["d"]
    else:
        # Reprocess last 2 days in case of late-arriving orders
        process_from = last_gold_date - datetime.timedelta(days=2)

    print(f"Incremental mode — processing {process_from} → {yesterday}")
else:
    process_from = spark.sql(f"""
        SELECT MIN(business_date) AS d FROM {ORDERS_SILVER}
    """).collect()[0]["d"]
    print(f"Full refresh — processing {process_from} → {yesterday}")

date_filter = f"business_date BETWEEN '{process_from}' AND '{yesterday}'"

# COMMAND ----------

# ── 5. BUILD WEATHER DAILY AGGREGATES ─────────────────────────────────────────
# Pre-aggregate weather to daily grain for joining to both Gold tables.
# Predominant weather category = the category with the most hours in the day.
# Predominant weather code = the code from the single worst/most notable hour
# (highest code value, as WMO codes generally increase in severity).

weather_daily = spark.sql(f"""
    WITH hourly AS (
        SELECT
            date,
            temperature_f,
            apparent_temperature_f,
            precipitation_in,
            snowfall_in,
            sunshine_minutes,
            cloud_cover_pct,
            weather_category,
            weather_condition,
            weather_code,
            data_type,
            COUNT(*) OVER (PARTITION BY date, weather_category) AS category_hours
        FROM {WEATHER_HOURLY}
        WHERE date BETWEEN '{process_from}' AND '{yesterday}'
    )
    SELECT
        date,
        ROUND(MAX(temperature_f), 1)                AS weather_high_f,
        ROUND(MIN(temperature_f), 1)                AS weather_low_f,
        ROUND(MAX(apparent_temperature_f), 1)       AS weather_feels_high_f,
        ROUND(MIN(apparent_temperature_f), 1)       AS weather_feels_low_f,
        ROUND(SUM(precipitation_in), 3)             AS total_precip_in,
        ROUND(SUM(snowfall_in), 3)                  AS total_snow_in,
        ROUND(SUM(CASE WHEN sunshine_minutes > 30 THEN 1 ELSE 0 END), 0)
                                                    AS sunny_hours,
        ROUND(AVG(cloud_cover_pct), 1)              AS avg_cloud_cover_pct,
        MAX_BY(weather_category, category_hours)    AS weather_category,
        MAX_BY(weather_condition, weather_code)     AS weather_condition,
        MAX(weather_code)                           AS weather_code,
        MAX_BY(data_type, weather_code)             AS weather_data_type
    FROM hourly
    GROUP BY date
""")

weather_daily.createOrReplaceTempView("weather_daily_agg")
print(f"✓ Weather daily aggregates: {weather_daily.count()} days")

# COMMAND ----------

# ── 6. BUILD daily_sales_summary (actuals) ────────────────────────────────────
# We use a date spine approach so that closed days (no orders in Silver)
# appear as explicit zero-revenue rows in Gold rather than being absent.
# This is critical for the feature engineering notebook to detect closures.
#
# Approach:
#   1. Generate a spine of every date in the processing window
#   2. LEFT JOIN Silver orders onto the spine — closed days get nulls
#   3. COALESCE nulls to zero for metric columns
#   4. Mark rows with zero revenue as store_closed = true

# Step 1: build the spine using Spark's SEQUENCE function
date_spine = spark.sql(f"""
    SELECT EXPLODE(SEQUENCE(
        DATE '{process_from}',
        DATE '{yesterday}',
        INTERVAL 1 DAY
    )) AS business_date
""")

# Step 2: aggregate Silver orders and items SEPARATELY then join.
# Never sum revenue (o.total_amount) in a query joined to items_silver —
# the join multiplies order rows by item count, inflating revenue.
# Revenue lives on the orders table. Item count lives on the items table.

orders_agg = spark.sql(f"""
    SELECT
        business_date,
        COUNT(DISTINCT order_guid)              AS order_count,
        ROUND(SUM(gross_amount), 2)                       AS gross_revenue,
        ROUND(SUM(total_amount - tip_amount), 2)          AS net_revenue,
        ROUND(SUM(total_discount_amount), 2)              AS total_discounts,
        ROUND(
            SUM(total_amount - tip_amount) / NULLIF(COUNT(DISTINCT order_guid), 0),
        2)                                                AS avg_ticket_size
    FROM {ORDERS_SILVER}
    WHERE {date_filter}
      AND voided = false
    GROUP BY business_date
""")

items_agg = spark.sql(f"""
    SELECT
        business_date,
        COUNT(*) AS item_count
    FROM {ITEMS_SILVER}
    WHERE {date_filter}
      AND voided = false
    GROUP BY business_date
""")

silver_agg = orders_agg.join(items_agg, on="business_date", how="left") \
    .fillna(0, subset=["item_count"])

# Step 3: join spine to aggregated Silver — closed days will have null metrics
spine_with_sales = date_spine.join(silver_agg, on="business_date", how="left")

# Step 4: join weather and fill nulls to produce final Gold rows
spine_with_sales.createOrReplaceTempView("spine_with_sales")

daily_actuals = spark.sql(f"""
    SELECT
        s.business_date,

        -- Sales metrics — zero for closed days, actual values for open days
        COALESCE(s.order_count, 0)              AS order_count,
        COALESCE(s.gross_revenue, 0.0)          AS gross_revenue,
        COALESCE(s.net_revenue, 0.0)            AS net_revenue,
        COALESCE(s.total_discounts, 0.0)        AS total_discounts,
        COALESCE(s.avg_ticket_size, 0.0)        AS avg_ticket_size,
        COALESCE(s.item_count, 0)               AS item_count,

        -- Day-of-week features (derived from date, always available)
        DAYOFWEEK(s.business_date)              AS day_of_week,
        DATE_FORMAT(s.business_date, 'EEEE')    AS day_name,
        DAYOFWEEK(s.business_date) IN (1, 7)    AS is_weekend,
        DAYOFWEEK(s.business_date) IN (3, 6)    AS is_bread_delivery_day,
        WEEKOFYEAR(s.business_date)             AS week_of_year,
        MONTH(s.business_date)                  AS month,
        YEAR(s.business_date)                   AS year,

        -- Weather (null for days without weather data)
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

        -- Audit
        CURRENT_TIMESTAMP()                     AS _gold_updated_at,
        '{BATCH_ID}'                            AS _batch_id

    FROM spine_with_sales s
    LEFT JOIN weather_daily_agg w
        ON s.business_date = w.date
""")

# Report closed days found
open_count   = daily_actuals.filter(F.col("net_revenue") > 0).count()
closed_count = daily_actuals.filter(F.col("net_revenue") == 0).count()
print(f"daily_sales_summary rows to merge: {daily_actuals.count()}")
print(f"  Open days:   {open_count}")
print(f"  Closed days: {closed_count} (written as zero-revenue rows)")

DAILY_UPDATE_COLS = [
    "order_count", "gross_revenue", "net_revenue",
    "total_discounts", "avg_ticket_size", "item_count",
    "day_of_week", "day_name", "is_weekend", "is_bread_delivery_day",
    "week_of_year", "month", "year",
    "weather_high_f", "weather_low_f", "weather_feels_high_f",
    "weather_feels_low_f", "weather_category", "weather_condition",
    "weather_code", "total_precip_in", "total_snow_in",
    "sunny_hours", "avg_cloud_cover_pct", "weather_data_type",
    "_gold_updated_at", "_batch_id",
]

DeltaTable.forName(spark, DAILY_TABLE).alias("t").merge(
    daily_actuals.alias("s"),
    "t.business_date = s.business_date"
).whenMatchedUpdate(
    set={col: f"s.{col}" for col in DAILY_UPDATE_COLS}
).whenNotMatchedInsertAll(
).execute()

print(f"✓ Merged into {DAILY_TABLE}")

# COMMAND ----------

# ── 7. BUILD hourly_sales_summary ─────────────────────────────────────────────
# Spine: for each day in scope, generate every hour from STORE_OPEN_HOUR to
# STORE_CLOSE_HOUR, then UNION with any transaction hours outside that window.
# This ensures empty hours still appear as zero-row placeholders on the chart.

# Get the distinct dates we're processing
dates_in_scope = spark.sql(f"""
    SELECT DISTINCT business_date
    FROM {ORDERS_SILVER}
    WHERE {date_filter} AND voided = false
""")

# Generate the store-hours spine for each date
hour_spine = spark.sql(f"""
    WITH dates AS (
        SELECT DISTINCT business_date
        FROM {ORDERS_SILVER}
        WHERE {date_filter} AND voided = false
    ),
    hours AS (
        SELECT EXPLODE(SEQUENCE({STORE_OPEN_HOUR}, {STORE_CLOSE_HOUR})) AS hour_of_day
    )
    SELECT d.business_date, h.hour_of_day
    FROM dates d CROSS JOIN hours h
""")

# Aggregate actual transactions to hour grain.
# opened_date is stored as UTC — convert to America/New_York before
# extracting the hour so the hourly chart aligns with local store time.
# Revenue aggregated from orders only (no items join) to avoid multiplication.
hourly_actuals = spark.sql(f"""
    SELECT
        o.business_date,
        HOUR(CONVERT_TIMEZONE('UTC', 'America/New_York', o.opened_date))
                                                AS hour_of_day,
        COUNT(DISTINCT o.order_guid)            AS order_count,
        ROUND(SUM(o.gross_amount), 2)                          AS gross_revenue,
        ROUND(SUM(o.total_amount - o.tip_amount), 2)           AS net_revenue,
        ROUND(SUM(o.total_discount_amount), 2)                 AS total_discounts,
        ROUND(
            SUM(o.total_amount - o.tip_amount) / NULLIF(COUNT(DISTINCT o.order_guid), 0),
        2)                                                     AS avg_ticket_size
    FROM {ORDERS_SILVER} o
    WHERE o.{date_filter}
      AND o.voided = false
      AND o.opened_date IS NOT NULL
    GROUP BY
        o.business_date,
        HOUR(CONVERT_TIMEZONE('UTC', 'America/New_York', o.opened_date))
""")

# Item count aggregated separately to avoid join multiplication
hourly_items = spark.sql(f"""
    SELECT
        i.business_date,
        HOUR(CONVERT_TIMEZONE('UTC', 'America/New_York', o.opened_date))
                                                AS hour_of_day,
        COUNT(*)                                AS item_count
    FROM {ITEMS_SILVER} i
    JOIN {ORDERS_SILVER} o ON i.order_guid = o.order_guid
    WHERE i.{date_filter}
      AND i.voided = false
      AND o.opened_date IS NOT NULL
    GROUP BY
        i.business_date,
        HOUR(CONVERT_TIMEZONE('UTC', 'America/New_York', o.opened_date))
""")

hourly_actuals = hourly_actuals.join(
    hourly_items, on=["business_date", "hour_of_day"], how="left"
).fillna(0, subset=["item_count"])

# Merge spine with actuals — spine hours get zeros where no transactions occurred
hourly_with_spine = (
    hour_spine
    .join(hourly_actuals,
          on=["business_date", "hour_of_day"],
          how="full_outer")
    .select(
        F.coalesce(
            hour_spine["business_date"],
            hourly_actuals["business_date"]
        ).alias("business_date"),
        F.coalesce(
            hour_spine["hour_of_day"],
            hourly_actuals["hour_of_day"]
        ).alias("hour_of_day"),
        F.coalesce(hourly_actuals["order_count"],  F.lit(0)).alias("order_count"),
        F.coalesce(hourly_actuals["gross_revenue"], F.lit(0.0)).alias("gross_revenue"),
        F.coalesce(hourly_actuals["net_revenue"],   F.lit(0.0)).alias("net_revenue"),
        F.coalesce(hourly_actuals["total_discounts"],F.lit(0.0)).alias("total_discounts"),
        F.coalesce(hourly_actuals["avg_ticket_size"],F.lit(0.0)).alias("avg_ticket_size"),
        F.coalesce(hourly_actuals["item_count"],    F.lit(0)).alias("item_count"),
    )
)

# Join weather for the hour
hourly_weather = spark.sql(f"""
    SELECT
        date                                    AS business_date,
        hour_of_day,
        temperature_f,
        apparent_temperature_f,
        weather_condition,
        weather_category,
        weather_code,
        precipitation_in,
        data_type                               AS weather_data_type
    FROM (
        SELECT *,
            -- weather time is already in America/New_York (ingested with that timezone)
            -- so HOUR(time) gives local hour directly — matches the ET hours in hourly_actuals
            HOUR(time) AS hour_of_day,
            ROW_NUMBER() OVER (
                PARTITION BY date, HOUR(time)
                ORDER BY time
            ) AS rn
        FROM {WEATHER_HOURLY}
        WHERE date BETWEEN '{process_from}' AND '{yesterday}'
    )
    WHERE rn = 1
""")

hourly_final = (
    hourly_with_spine
    .join(
        hourly_weather,
        on=(
            (hourly_with_spine["business_date"] == hourly_weather["business_date"]) &
            (hourly_with_spine["hour_of_day"]   == hourly_weather["hour_of_day"])
        ),
        how="left"
    )
    .select(
        hourly_with_spine["business_date"],
        hourly_with_spine["hour_of_day"],
        "order_count", "gross_revenue", "net_revenue",
        "total_discounts", "avg_ticket_size", "item_count",
        "temperature_f", "apparent_temperature_f",
        "weather_condition", "weather_category", "weather_code",
        "precipitation_in", "weather_data_type",
    )
    .withColumn(
        "hour_label",
        F.when(F.col("hour_of_day") == 0,  F.lit("12 AM"))
         .when(F.col("hour_of_day") < 12,  F.concat(F.col("hour_of_day").cast("string"), F.lit(" AM")))
         .when(F.col("hour_of_day") == 12, F.lit("12 PM"))
         .otherwise(F.concat((F.col("hour_of_day") - 12).cast("string"), F.lit(" PM")))
    )
    .withColumn(
        "is_store_hours",
        (F.col("hour_of_day") >= STORE_OPEN_HOUR) &
        (F.col("hour_of_day") <= STORE_CLOSE_HOUR)
    )
    .withColumn("has_transactions", F.col("order_count") > F.lit(0))
    .withColumn("_gold_updated_at", F.lit(NOW_TS).cast("timestamp"))
    .withColumn("_batch_id", F.lit(BATCH_ID))
    .select(
        "business_date", "hour_of_day", "hour_label",
        "order_count", "gross_revenue", "net_revenue",
        "total_discounts", "avg_ticket_size", "item_count",
        "temperature_f", "apparent_temperature_f",
        "weather_condition", "weather_category", "weather_code",
        "precipitation_in", "weather_data_type",
        "is_store_hours", "has_transactions",
        "_gold_updated_at", "_batch_id"
    )
)

print(f"hourly_sales_summary rows to merge: {hourly_final.count()}")

HOURLY_UPDATE_COLS = [
    "hour_label", "order_count", "gross_revenue", "net_revenue",
    "total_discounts", "avg_ticket_size", "item_count",
    "temperature_f", "apparent_temperature_f",
    "weather_condition", "weather_category", "weather_code",
    "precipitation_in", "weather_data_type",
    "is_store_hours", "has_transactions",
    "_gold_updated_at", "_batch_id",
]

HOURLY_INSERT_SET = {
    col: f"s.{col}"
    for col in ["business_date", "hour_of_day"] + HOURLY_UPDATE_COLS
}

DeltaTable.forName(spark, HOURLY_TABLE).alias("t").merge(
    hourly_final.alias("s"),
    "t.business_date = s.business_date AND t.hour_of_day = s.hour_of_day"
).whenMatchedUpdate(
    set={col: f"s.{col}" for col in HOURLY_UPDATE_COLS}
).whenNotMatchedInsert(
    values=HOURLY_INSERT_SET
).execute()

print(f"✓ Merged into {HOURLY_TABLE}")

# COMMAND ----------

# ── 8. BUILD daily_sales_by_category ──────────────────────────────────────────

category_actuals = spark.sql(f"""
    SELECT
        i.business_date,
        COALESCE(c.category_group, 'Unknown')   AS category_group,
        COALESCE(c.category, 'Unknown')         AS category,

        COUNT(DISTINCT i.order_guid)            AS order_count,
        ROUND(SUM(i.quantity), 0)               AS item_count,
        ROUND(SUM(i.receipt_line_price), 2)     AS gross_revenue,
        ROUND(AVG(i.unit_price), 2)             AS avg_item_price,
        MAX(CAST(i.tax > 0 AS INT)) = 1         AS has_taxable_items,

        CURRENT_TIMESTAMP()                     AS _gold_updated_at,
        '{BATCH_ID}'                            AS _batch_id

    FROM {ITEMS_SILVER} i
    LEFT JOIN {ITEM_CATALOG} c
        ON i.item_guid = c.item_id
    WHERE i.{date_filter}
      AND i.voided = false
    GROUP BY
        i.business_date,
        COALESCE(c.category_group, 'Unknown'),
        COALESCE(c.category, 'Unknown')
""")

print(f"daily_sales_by_category rows to merge: {category_actuals.count()}")

CATEGORY_UPDATE_COLS = [
    "order_count", "item_count", "gross_revenue",
    "avg_item_price", "has_taxable_items",
    "_gold_updated_at", "_batch_id",
]

CATEGORY_INSERT_SET = {
    col: f"s.{col}"
    for col in ["business_date", "category_group", "category"] + CATEGORY_UPDATE_COLS
}

DeltaTable.forName(spark, CATEGORY_TABLE).alias("t").merge(
    category_actuals.alias("s"),
    """t.business_date = s.business_date
       AND t.category_group = s.category_group
       AND t.category = s.category"""
).whenMatchedUpdate(
    set={col: f"s.{col}" for col in CATEGORY_UPDATE_COLS}
).whenNotMatchedInsert(
    values=CATEGORY_INSERT_SET
).execute()

print(f"✓ Merged into {CATEGORY_TABLE}")

# COMMAND ----------

# ── 9. UNITY CATALOG METADATA ─────────────────────────────────────────────────

for table, description, tags in [
    (HOURLY_TABLE,
     "Gold: hourly sales summary for the Cohasset store. One row per hour per business day. Covers store hours (9am-7pm) plus any hours with transactions outside that window. Joined to hourly weather from Open-Meteo. Powers the hourly revenue and ticket count strip on the dashboard.",
     {'domain':'retail','layer':'gold','refresh':'daily','pii':'false','dashboard':'hourly_strip'}),
    (DAILY_TABLE,
     "Gold: daily sales actuals with weather. One row per business date. Actuals only — forecasts live in gold.daily_sales_forecast. Join both via platinum.daily_sales_combined for dashboards.",
     {'domain':'retail','layer':'gold','refresh':'daily','pii':'false','dashboard':'7day_view'}),
    (CATEGORY_TABLE,
     "Gold: daily sales broken down by category group and category. One row per date + category. Actuals only. Joins Silver order items to the reference item catalog for category names. Powers the category revenue breakdown chart on the dashboard.",
     {'domain':'retail','layer':'gold','refresh':'daily','pii':'false','dashboard':'category_chart'}),
]:
    spark.sql(f"COMMENT ON TABLE {table} IS '{description}'")
    tag_sql = ", ".join(f"'{k}' = '{v}'" for k, v in tags.items())
    spark.sql(f"ALTER TABLE {table} SET TAGS ({tag_sql})")

print("✓ Metadata applied to all three Gold tables")

# COMMAND ----------

# ── 10. VALIDATION — ASSERT GOLD MATCHES SILVER ───────────────────────────────
# This cell compares Gold daily_sales_summary against Silver orders directly.
# It will raise an exception and halt the notebook if any assertion fails,
# preventing bad data from silently flowing downstream to the feature
# engineering and forecast model notebooks.
#
# Checks performed:
#   1. Row count — Gold has the expected number of actual rows
#   2. Revenue reconciliation — Gold revenue matches Silver within tolerance
#   3. Order count reconciliation — Gold order count matches Silver exactly
#   4. No inflated revenue — avg ticket size within plausible range
#   5. No future actual rows — Gold actuals don't exceed yesterday
#   6. Closed days present — zero-revenue days exist as explicit rows

import datetime

TOLERANCE_PCT  = 0.01   # allow 1% revenue variance for rounding differences
MAX_AVG_TICKET = 500.0  # flag any day where avg ticket exceeds this threshold
                        # (likely indicates the join multiplication bug)

validation_passed = True
failures = []
warnings_list = []

print("=" * 60)
print("GOLD TABLE VALIDATION")
print("=" * 60)

# ── Pull Silver ground truth for the processing window ───────────────────────
silver_truth = spark.sql(f"""
    SELECT
        business_date,
        COUNT(DISTINCT order_guid)               AS silver_orders,
        ROUND(SUM(total_amount - tip_amount), 2) AS silver_revenue
    FROM {ORDERS_SILVER}
    WHERE {date_filter}
      AND voided = false
    GROUP BY business_date
""").toPandas()

# ── Pull Gold actuals for the same window ─────────────────────────────────────
gold_actuals = spark.sql(f"""
    SELECT
        business_date,
        order_count                     AS gold_orders,
        ROUND(net_revenue, 2)           AS gold_revenue,
        ROUND(avg_ticket_size, 2)       AS gold_avg_ticket
    FROM {DAILY_TABLE}
    WHERE business_date BETWEEN '{process_from}' AND '{yesterday}'
""").toPandas()

gold_actuals['business_date']  = gold_actuals['business_date'].astype(str)
silver_truth['business_date']  = silver_truth['business_date'].astype(str)

# ── CHECK 1: Row counts ───────────────────────────────────────────────────────
# Gold should have one actual row per day in the window (including closed days)
import pandas as pd
expected_days = (
    pd.date_range(start=str(process_from), end=str(yesterday), freq='D')
    .shape[0]
)
gold_day_count = len(gold_actuals)

print(f"\n[1] Row count")
print(f"    Expected days in window: {expected_days}")
print(f"    Gold actual rows:        {gold_day_count}")

if gold_day_count != expected_days:
    msg = f"Row count mismatch: expected {expected_days} days, got {gold_day_count}"
    failures.append(msg)
    print(f"    ✗ FAIL — {msg}")
else:
    print(f"    ✓ PASS")

# ── CHECK 2: Revenue reconciliation ──────────────────────────────────────────
# Compare Gold revenue against Silver for each open day.
# Closed days (zero revenue) are excluded — Silver has no rows for them.
open_gold = gold_actuals[gold_actuals['gold_revenue'] > 0].copy()
merged = open_gold.merge(silver_truth, on='business_date', how='inner')

revenue_mismatches = []
for _, row in merged.iterrows():
    if row['silver_revenue'] == 0:
        continue
    variance_pct = abs(row['gold_revenue'] - row['silver_revenue']) / row['silver_revenue']
    if variance_pct > TOLERANCE_PCT:
        revenue_mismatches.append({
            'date':           row['business_date'],
            'gold_revenue':   row['gold_revenue'],
            'silver_revenue': row['silver_revenue'],
            'variance_pct':   round(variance_pct * 100, 1),
        })

print(f"\n[2] Revenue reconciliation (tolerance: {TOLERANCE_PCT*100:.0f}%)")
print(f"    Days checked: {len(merged)}")
if revenue_mismatches:
    validation_passed = False
    for m in revenue_mismatches[:5]:   # show first 5
        failures.append(
            f"Revenue mismatch on {m['date']}: "
            f"Gold=${m['gold_revenue']:,.2f} vs Silver=${m['silver_revenue']:,.2f} "
            f"({m['variance_pct']}% variance)"
        )
        print(f"    ✗ FAIL — {failures[-1]}")
    if len(revenue_mismatches) > 5:
        print(f"    ... and {len(revenue_mismatches) - 5} more mismatches")
else:
    total_gold_rev    = merged['gold_revenue'].sum()
    total_silver_rev  = merged['silver_revenue'].sum()
    overall_variance  = abs(total_gold_rev - total_silver_rev) / total_silver_rev * 100
    print(f"    Gold total:   ${total_gold_rev:,.2f}")
    print(f"    Silver total: ${total_silver_rev:,.2f}")
    print(f"    Variance:     {overall_variance:.3f}%")
    print(f"    ✓ PASS")

# ── CHECK 3: Order count reconciliation ──────────────────────────────────────
order_mismatches = []
for _, row in merged.iterrows():
    if row['gold_orders'] != row['silver_orders']:
        order_mismatches.append({
            'date':          row['business_date'],
            'gold_orders':   row['gold_orders'],
            'silver_orders': row['silver_orders'],
        })

print(f"\n[3] Order count reconciliation")
if order_mismatches:
    validation_passed = False
    for m in order_mismatches[:5]:
        failures.append(
            f"Order count mismatch on {m['date']}: "
            f"Gold={m['gold_orders']} vs Silver={m['silver_orders']}"
        )
        print(f"    ✗ FAIL — {failures[-1]}")
else:
    print(f"    ✓ PASS — all {len(merged)} days match exactly")

# ── CHECK 4: No inflated average ticket sizes ─────────────────────────────────
# The join multiplication bug caused avg ticket > $500 on most days.
# Flag any open day with avg ticket above the threshold.
inflated = gold_actuals[
    (gold_actuals['gold_revenue'] > 0) &
    (gold_actuals['gold_avg_ticket'] > MAX_AVG_TICKET)
]

print(f"\n[4] Average ticket size sanity check (max: ${MAX_AVG_TICKET:.0f})")
if len(inflated) > 0:
    validation_passed = False
    for _, row in inflated.iterrows():
        failures.append(
            f"Inflated avg ticket on {row['business_date']}: "
            f"${row['gold_avg_ticket']:,.2f} — possible join multiplication bug"
        )
        print(f"    ✗ FAIL — {failures[-1]}")
else:
    max_ticket = gold_actuals[gold_actuals['gold_revenue'] > 0]['gold_avg_ticket'].max()
    print(f"    Max avg ticket in window: ${max_ticket:,.2f}")
    print(f"    ✓ PASS")

# ── CHECK 5: No future actual rows ───────────────────────────────────────────
future_actuals = spark.sql(f"""
    SELECT COUNT(*) AS cnt
    FROM {DAILY_TABLE}
    WHERE business_date > '{yesterday}'
""").collect()[0]['cnt']

print(f"\n[5] No future actual rows")
if future_actuals > 0:
    validation_passed = False
    failures.append(f"Found {future_actuals} actual rows with business_date > {yesterday}")
    print(f"    ✗ FAIL — {failures[-1]}")
else:
    print(f"    ✓ PASS")

# ── CHECK 6: Closed days present as explicit zero rows ────────────────────────
# Silver has no rows for closed days. Gold should have explicit zero rows.
silver_dates  = set(silver_truth['business_date'].tolist())
gold_dates    = set(gold_actuals['business_date'].tolist())
all_dates     = set(
    d.strftime('%Y-%m-%d')
    for d in pd.date_range(start=str(process_from), end=str(yesterday), freq='D')
)
missing_from_gold = all_dates - gold_dates
closed_in_gold    = set(
    gold_actuals[gold_actuals['gold_revenue'] == 0]['business_date'].tolist()
)
expected_closed   = all_dates - silver_dates

print(f"\n[6] Closed days present as explicit zero rows")
print(f"    Days with no Silver data (expected closed): {len(expected_closed)}")
print(f"    Gold zero-revenue rows:                     {len(closed_in_gold)}")

if missing_from_gold:
    validation_passed = False
    failures.append(
        f"{len(missing_from_gold)} dates missing from Gold entirely: "
        f"{sorted(missing_from_gold)[:3]}{'...' if len(missing_from_gold) > 3 else ''}"
    )
    print(f"    ✗ FAIL — {failures[-1]}")
else:
    print(f"    ✓ PASS — all dates present")

if expected_closed:
    print(f"\n    Closed days in this window:")
    for d in sorted(expected_closed):
        print(f"      {d}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
if validation_passed:
    print("✓ ALL VALIDATION CHECKS PASSED")
    print(f"  {len(gold_actuals)} Gold rows validated against Silver")
    print(f"  Revenue and order counts reconcile within tolerance")
else:
    print(f"✗ VALIDATION FAILED — {len(failures)} issue(s) found:")
    for i, f in enumerate(failures, 1):
        print(f"  {i}. {f}")
print("=" * 60)

# Raise exception to halt notebook and fail the job if validation fails.
# This prevents bad data from flowing to feature engineering and the model.
if not validation_passed:
    raise ValueError(
        f"Gold table validation failed with {len(failures)} issue(s). "
        f"See output above for details. Fix the upstream issue and re-run."
    )

# ── DISPLAY TABLES (informational, after assertions pass) ─────────────────────
print("\n── daily_sales_summary: last 7 days ──")
spark.sql(f"""
    SELECT
        business_date,
        day_name,
        order_count,
        ROUND(net_revenue, 2)           AS net_revenue,
        ROUND(avg_ticket_size, 2)       AS avg_ticket,
        weather_high_f,
        weather_category,
        is_bread_delivery_day
    FROM {DAILY_TABLE}
    WHERE business_date >= CURRENT_DATE - INTERVAL 7 DAYS
    ORDER BY business_date DESC
""").show(truncate=False)

print("\n── hourly_sales_summary: yesterday ──")
spark.sql(f"""
    SELECT
        hour_label,
        order_count,
        ROUND(net_revenue, 2)           AS net_revenue,
        ROUND(avg_ticket_size, 2)       AS avg_ticket,
        ROUND(temperature_f, 1)         AS temp_f,
        weather_category,
        is_store_hours,
        has_transactions
    FROM {HOURLY_TABLE}
    WHERE business_date = CURRENT_DATE - INTERVAL 1 DAY
    ORDER BY hour_of_day
""").show(24, truncate=False)

print("\n── Week-on-week revenue comparison ──")
spark.sql(f"""
    SELECT
        business_date,
        day_name,
        ROUND(net_revenue, 2)           AS this_week_revenue,
        ROUND(LAG(net_revenue, 7) OVER (ORDER BY business_date), 2)
                                        AS prior_week_revenue,
        ROUND(
            (net_revenue - LAG(net_revenue, 7) OVER (ORDER BY business_date))
            / NULLIF(LAG(net_revenue, 7) OVER (ORDER BY business_date), 0) * 100
        , 1)                            AS wow_pct_change
    FROM {DAILY_TABLE}
    WHERE business_date >= CURRENT_DATE - INTERVAL 14 DAYS
    ORDER BY business_date DESC
    LIMIT 14
""").show(truncate=False)
