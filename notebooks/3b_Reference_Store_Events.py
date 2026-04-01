# Databricks notebook source
# MAGIC %md
# MAGIC # Reference — Store Events
# MAGIC
# MAGIC Creates and maintains the `reference.store_events` table.
# MAGIC This is the single source of truth for all special events that
# MAGIC affect the forecast model. All model notebooks read from here
# MAGIC rather than maintaining their own hardcoded event lists.
# MAGIC
# MAGIC ## Event types
# MAGIC
# MAGIC | Type | Treatment | Example |
# MAGIC |---|---|---|
# MAGIC | `PLANNED_EVENT` | Keep in training, Prophet holiday flag | Town Stroll |
# MAGIC | `REVENUE_DISTORTION` | Exclude from training — revenue inflated by catering | Wine Tasting |
# MAGIC | `ORGANIC_EVENT` | Exclude from training — unpredictable one-off | School group |
# MAGIC | `FUTURE_CLOSURE` | Label forecast rows as closed — do not predict revenue | Christmas |
# MAGIC
# MAGIC ## How to add a new event
# MAGIC
# MAGIC Add a row to the `events` list in section 3 and re-run this notebook.
# MAGIC All model notebooks will pick up the change on their next run.
# MAGIC
# MAGIC ## Dependencies
# MAGIC
# MAGIC **Writes to:**
# MAGIC - `3sp_analytics_workspace.reference.store_events`
# MAGIC
# MAGIC **Downstream:** `6_Feature_Engineering`, `7_Model_Prophet_Revenue`,
# MAGIC `8_Model_Prophet_Orders` all read from this table.
# MAGIC
# MAGIC ## Change log
# MAGIC
# MAGIC | Version | Date | Author | Change |
# MAGIC |---|---|---|---|
# MAGIC | v1 | 2026-03-27 | JS | Initial build — seeded with known events |

# COMMAND ----------

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────

CATALOG   = "3sp_analytics_workspace"
SCHEMA    = "reference"
TABLE     = f"{CATALOG}.{SCHEMA}.store_events"

# COMMAND ----------

# ── 2. CREATE TABLE ───────────────────────────────────────────────────────────

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        event_date          DATE        NOT NULL    COMMENT 'Date of the event',
        event_name          STRING      NOT NULL    COMMENT 'Human-readable name for the event',
        event_type          STRING      NOT NULL    COMMENT 'PLANNED_EVENT, REVENUE_DISTORTION, ORGANIC_EVENT, or FUTURE_CLOSURE',
        lower_window        INTEGER                 COMMENT 'Days before event_date where the effect begins (Prophet holiday param). 0 = event day only, -1 = also day before.',
        upper_window        INTEGER                 COMMENT 'Days after event_date where the effect continues (Prophet holiday param). 0 = event day only, 1 = also day after.',
        notes               STRING                  COMMENT 'Optional context about the event — what happened, why it matters for the model.',
        is_active           BOOLEAN                 COMMENT 'Set to false to disable an event without deleting it.',
        created_at          TIMESTAMP               COMMENT 'When this row was added.',
        updated_at          TIMESTAMP               COMMENT 'When this row was last modified.'
    )
    USING DELTA
    COMMENT 'Reference: store events that affect the forecast model. Single source of truth for all special dates. Add new events here — all model notebooks read from this table.'
    TBLPROPERTIES (
        'quality'  = 'reference',
        'domain'   = 'retail',
        'purpose'  = 'ml_features'
    )
""")

print(f"✓ {TABLE} ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3 — Seed Events
# MAGIC
# MAGIC Add new events here. Re-run this cell to upsert them into the table.
# MAGIC Existing rows are updated if the date already exists.

# COMMAND ----------

# ── 3. SEED EVENTS ────────────────────────────────────────────────────────────
# Add new events to this list. Re-run the notebook to upsert them.
#
# lower_window / upper_window:
#   These tell Prophet how many days around the event date to apply
#   the holiday effect. For a one-day event: lower=0, upper=0.
#   For an event with build-up (e.g. Town Stroll — people may shop the
#   day before too): lower=-1, upper=0.

from pyspark.sql import functions as F

events = [
    # ── Historical events ─────────────────────────────────────────────────────
    {
        "event_date":   "2025-12-13",
        "event_name":   "Cohasset Village Town Stroll",
        "event_type":   "PLANNED_EVENT",
        "lower_window": -1,     # effect starts day before (pre-event shopping)
        "upper_window": 0,
        "notes":        "Annual holiday event in Cohasset village. High foot traffic, many small tickets. Real demand signal — keep in training. Prophet holiday flag applied.",
        "is_active":    True,
    },
    {
        "event_date":   "2025-12-11",
        "event_name":   "December event / unknown",
        "event_type":   "ORGANIC_EVENT",
        "lower_window": 0,
        "upper_window": 0,
        "notes":        "Anomalous revenue spike in December. Cause unknown — possible civic or school group visit. Excluded from training as unpredictable one-off.",
        "is_active":    True,
    },
    {
        "event_date":   "2026-03-03",
        "event_name":   "Databricks Wine Tasting Event",
        "event_type":   "REVENUE_DISTORTION",
        "lower_window": 0,
        "upper_window": 0,
        "notes":        "Single large catering/event ticket inflated daily revenue artificially. Not representative of normal retail demand. Excluded from model training.",
        "is_active":    True,
    },

    # ── Future closures ───────────────────────────────────────────────────────
    {
        "event_date":   "2026-11-26",
        "event_name":   "Thanksgiving 2026",
        "event_type":   "FUTURE_CLOSURE",
        "lower_window": 0,
        "upper_window": 0,
        "notes":        "Store closed. Forecast rows labeled likely_closed. Revenue prediction still generated to support stay-open decision analysis.",
        "is_active":    True,
    },
    {
        "event_date":   "2026-12-25",
        "event_name":   "Christmas Day 2026",
        "event_type":   "FUTURE_CLOSURE",
        "lower_window": 0,
        "upper_window": 0,
        "notes":        "Store closed. Forecast rows labeled likely_closed. Revenue prediction still generated to support stay-open decision analysis.",
        "is_active":    True,
    },
]

# Convert to Spark DataFrame and upsert
from delta.tables import DeltaTable
import datetime

now = datetime.datetime.now(datetime.timezone.utc)

events_df = spark.createDataFrame([
    {
        **e,
        "event_date": e["event_date"],
        "created_at": now,
        "updated_at": now,
    }
    for e in events
]).withColumn("event_date", F.col("event_date").cast("date")) \
  .withColumn("created_at", F.col("created_at").cast("timestamp")) \
  .withColumn("updated_at", F.col("updated_at").cast("timestamp"))

DeltaTable.forName(spark, TABLE).alias("t").merge(
    events_df.alias("s"),
    "t.event_date = s.event_date AND t.event_name = s.event_name"
).whenMatchedUpdate(set={
    "event_type":   "s.event_type",
    "lower_window": "s.lower_window",
    "upper_window": "s.upper_window",
    "notes":        "s.notes",
    "is_active":    "s.is_active",
    "updated_at":   "s.updated_at",
}).whenNotMatchedInsertAll(
).execute()

print(f"✓ Upserted {len(events)} events into {TABLE}")

# COMMAND ----------

# ── 4. VALIDATION ─────────────────────────────────────────────────────────────

print("\nCurrent store events:")
spark.sql(f"""
    SELECT
        event_date,
        event_name,
        event_type,
        lower_window,
        upper_window,
        is_active,
        notes
    FROM {TABLE}
    ORDER BY event_date
""").show(truncate=False)

print("\nEvents by type:")
spark.sql(f"""
    SELECT
        event_type,
        COUNT(*) AS count,
        SUM(CAST(is_active AS INT)) AS active_count
    FROM {TABLE}
    GROUP BY event_type
    ORDER BY event_type
""").show(truncate=False)