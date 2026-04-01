# Databricks notebook source
# MAGIC %md
# MAGIC # Store Events Manager
# MAGIC
# MAGIC Add, edit, or deactivate events in `reference.store_events`.
# MAGIC
# MAGIC ## How to use
# MAGIC
# MAGIC 1. Set the widgets at the top of the notebook
# MAGIC 2. Run all cells
# MAGIC 3. The event will be upserted — if the date + name already exists it
# MAGIC    will be updated, otherwise a new row is inserted
# MAGIC
# MAGIC ## Event types
# MAGIC
# MAGIC | Type | Effect on model |
# MAGIC |---|---|
# MAGIC | `PLANNED_EVENT` | Keep in training, Prophet holiday flag (e.g. Town Stroll) |
# MAGIC | `REVENUE_DISTORTION` | Exclude from training — catering inflated revenue |
# MAGIC | `ORGANIC_EVENT` | Exclude from training — unpredictable one-off |
# MAGIC | `FUTURE_CLOSURE` | Forecast row labeled closed, revenue still predicted |
# MAGIC
# MAGIC ## Change log
# MAGIC
# MAGIC | Version | Date | Author | Change |
# MAGIC |---|---|---|---|
# MAGIC | v1 | 2026-03-27 | JS | Initial build |

# COMMAND ----------

# ── 1. WIDGETS ────────────────────────────────────────────────────────────────

dbutils.widgets.dropdown(
    name         = "action",
    defaultValue = "view",
    choices      = ["view", "upsert", "deactivate", "reactivate"],
    label        = "Action"
)
dbutils.widgets.text(
    name         = "event_date",
    defaultValue = "",
    label        = "Event Date (YYYY-MM-DD)"
)
dbutils.widgets.text(
    name         = "event_name",
    defaultValue = "",
    label        = "Event Name"
)
dbutils.widgets.dropdown(
    name         = "event_type",
    defaultValue = "PLANNED_EVENT",
    choices      = ["PLANNED_EVENT", "REVENUE_DISTORTION", "ORGANIC_EVENT", "FUTURE_CLOSURE"],
    label        = "Event Type"
)
dbutils.widgets.text(
    name         = "lower_window",
    defaultValue = "0",
    label        = "Lower Window (days before, e.g. -1)"
)
dbutils.widgets.text(
    name         = "upper_window",
    defaultValue = "0",
    label        = "Upper Window (days after, e.g. 1)"
)
dbutils.widgets.text(
    name         = "notes",
    defaultValue = "",
    label        = "Notes (optional)"
)

# COMMAND ----------

# ── 2. IMPORTS AND CONFIG ─────────────────────────────────────────────────────

import datetime
from delta.tables import DeltaTable
from pyspark.sql import functions as F

CATALOG = "3sp_analytics_workspace"
TABLE   = f"{CATALOG}.reference.store_events"

ACTION       = dbutils.widgets.get("action").strip()
EVENT_DATE   = dbutils.widgets.get("event_date").strip()
EVENT_NAME   = dbutils.widgets.get("event_name").strip()
EVENT_TYPE   = dbutils.widgets.get("event_type").strip()
LOWER_WINDOW = dbutils.widgets.get("lower_window").strip()
UPPER_WINDOW = dbutils.widgets.get("upper_window").strip()
NOTES        = dbutils.widgets.get("notes").strip()

print(f"Action:      {ACTION}")
if ACTION != "view":
    print(f"Event date:  {EVENT_DATE}")
    print(f"Event name:  {EVENT_NAME}")
    print(f"Event type:  {EVENT_TYPE}")

# COMMAND ----------

# ── 3. VALIDATE INPUTS ────────────────────────────────────────────────────────

if ACTION != "view":

    errors = []

    # Date validation
    if not EVENT_DATE:
        errors.append("Event date is required")
    else:
        try:
            parsed_date = datetime.date.fromisoformat(EVENT_DATE)
        except ValueError:
            errors.append(f"Event date '{EVENT_DATE}' is not a valid date. Use YYYY-MM-DD format.")

    # Name validation
    if ACTION in ("upsert",) and not EVENT_NAME:
        errors.append("Event name is required for upsert")

    # Window validation
    try:
        lower_w = int(LOWER_WINDOW)
        if lower_w > 0:
            errors.append(f"Lower window should be 0 or negative (got {lower_w}). Use 0 for event day only, -1 to also include the day before.")
    except ValueError:
        errors.append(f"Lower window '{LOWER_WINDOW}' must be an integer")

    try:
        upper_w = int(UPPER_WINDOW)
        if upper_w < 0:
            errors.append(f"Upper window should be 0 or positive (got {upper_w}). Use 0 for event day only, 1 to also include the day after.")
    except ValueError:
        errors.append(f"Upper window '{UPPER_WINDOW}' must be an integer")

    if errors:
        for e in errors:
            print(f"✗ {e}")
        raise ValueError(f"Input validation failed with {len(errors)} error(s). Fix the widgets and re-run.")
    else:
        print("✓ Inputs valid")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Action

# COMMAND ----------

# ── 4. EXECUTE ACTION ─────────────────────────────────────────────────────────

now = datetime.datetime.now(datetime.timezone.utc)

if ACTION == "view":
    # ── VIEW: show all events ─────────────────────────────────────────────────
    print("Current store events:\n")
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
    """).show(50, truncate=False)

elif ACTION == "upsert":
    # ── UPSERT: add new event or update existing ──────────────────────────────
    new_row = spark.createDataFrame([{
        "event_date":   EVENT_DATE,
        "event_name":   EVENT_NAME,
        "event_type":   EVENT_TYPE,
        "lower_window": int(LOWER_WINDOW),
        "upper_window": int(UPPER_WINDOW),
        "notes":        NOTES,
        "is_active":    True,
        "created_at":   now,
        "updated_at":   now,
    }]).withColumn("event_date",  F.col("event_date").cast("date")) \
       .withColumn("created_at", F.col("created_at").cast("timestamp")) \
       .withColumn("updated_at", F.col("updated_at").cast("timestamp"))

    DeltaTable.forName(spark, TABLE).alias("t").merge(
        new_row.alias("s"),
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

    print(f"✓ Upserted: {EVENT_DATE} — {EVENT_NAME} ({EVENT_TYPE})")

elif ACTION == "deactivate":
    # ── DEACTIVATE: soft delete — keeps the row but excludes from model ───────
    if not EVENT_DATE:
        raise ValueError("Event date required for deactivate")

    # Find matching active rows
    matches = spark.sql(f"""
        SELECT event_date, event_name, event_type
        FROM {TABLE}
        WHERE event_date = '{EVENT_DATE}'
          AND is_active = true
          AND ('{EVENT_NAME}' = '' OR event_name = '{EVENT_NAME}')
    """).collect()

    if not matches:
        print(f"⚠ No active events found for {EVENT_DATE}"
              + (f" with name '{EVENT_NAME}'" if EVENT_NAME else ""))
    else:
        name_filter = f"AND event_name = '{EVENT_NAME}'" if EVENT_NAME else ""
        spark.sql(f"""
            UPDATE {TABLE}
            SET is_active = false,
                updated_at = CURRENT_TIMESTAMP()
            WHERE event_date = '{EVENT_DATE}'
              AND is_active = true
              {name_filter}
        """)
        print(f"✓ Deactivated {len(matches)} event(s) on {EVENT_DATE}:")
        for m in matches:
            print(f"  {m['event_date']}  {m['event_name']}  ({m['event_type']})")

elif ACTION == "reactivate":
    # ── REACTIVATE: restore a previously deactivated event ───────────────────
    if not EVENT_DATE:
        raise ValueError("Event date required for reactivate")

    matches = spark.sql(f"""
        SELECT event_date, event_name, event_type
        FROM {TABLE}
        WHERE event_date = '{EVENT_DATE}'
          AND is_active = false
          AND ('{EVENT_NAME}' = '' OR event_name = '{EVENT_NAME}')
    """).collect()

    if not matches:
        print(f"⚠ No inactive events found for {EVENT_DATE}"
              + (f" with name '{EVENT_NAME}'" if EVENT_NAME else ""))
    else:
        name_filter = f"AND event_name = '{EVENT_NAME}'" if EVENT_NAME else ""
        spark.sql(f"""
            UPDATE {TABLE}
            SET is_active = true,
                updated_at = CURRENT_TIMESTAMP()
            WHERE event_date = '{EVENT_DATE}'
              AND is_active = false
              {name_filter}
        """)
        print(f"✓ Reactivated {len(matches)} event(s) on {EVENT_DATE}:")
        for m in matches:
            print(f"  {m['event_date']}  {m['event_name']}  ({m['event_type']})")

# COMMAND ----------

# ── 5. SHOW CURRENT STATE ─────────────────────────────────────────────────────

if ACTION != "view":
    print("\nUpdated event table:")
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
    """).show(50, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick reference — common operations
# MAGIC
# MAGIC **Add a new Town Stroll for next December:**
# MAGIC - Action: `upsert`
# MAGIC - Event Date: `2026-12-12`
# MAGIC - Event Name: `Cohasset Village Town Stroll`
# MAGIC - Event Type: `PLANNED_EVENT`
# MAGIC - Lower Window: `-1`
# MAGIC - Notes: `Annual holiday stroll`
# MAGIC
# MAGIC **Log a catering event that just happened:**
# MAGIC - Action: `upsert`
# MAGIC - Event Date: `2026-05-15`
# MAGIC - Event Name: `Catering order — graduation party`
# MAGIC - Event Type: `REVENUE_DISTORTION`
# MAGIC - Notes: `Single large catering ticket, ~$800`
# MAGIC
# MAGIC **Deactivate an event that was entered incorrectly:**
# MAGIC - Action: `deactivate`
# MAGIC - Event Date: `2026-03-03`
# MAGIC - Event Name: `Databricks Wine Tasting Event` (leave blank to deactivate all events on that date)
# MAGIC
# MAGIC **After any change:** re-run `6_Feature_Engineering` and the model
# MAGIC notebooks to incorporate the update into the next forecast.