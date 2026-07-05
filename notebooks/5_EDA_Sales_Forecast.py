# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Data Analysis — Sales Forecast
# MAGIC
# MAGIC ## What is this notebook?
# MAGIC
# MAGIC Before building any forecast model, good data scientists spend time
# MAGIC *looking* at their data carefully. This process is called **Exploratory
# MAGIC Data Analysis** (EDA). Think of it like a chef tasting and smelling
# MAGIC ingredients before deciding how to cook them.
# MAGIC
# MAGIC This notebook answers the questions that shape our forecast model:
# MAGIC
# MAGIC - How does revenue behave across the week? Across the year?
# MAGIC - How much does weather actually affect sales?
# MAGIC - Which days are genuine outliers that could mislead our model?
# MAGIC - Does bread delivery day actually drive more traffic?
# MAGIC - How strong is the seasonal curve?
# MAGIC
# MAGIC **You don't need to understand the code** — focus on the charts and the
# MAGIC plain-English commentary beneath each one. Every chart has a "What this
# MAGIC means" section written for a non-data-scientist reader.
# MAGIC
# MAGIC **Data source:** `YOUR_CATALOG.gold.daily_sales_summary`
# MAGIC — 164 days of actual sales data from the store, enriched with weather.

# COMMAND ----------

# ── 1. SETUP ──────────────────────────────────────────────────────────────────
# Load the libraries we need. Think of these as specialized tools:
#   pandas    = a spreadsheet engine for Python
#   matplotlib = a drawing engine for charts
#   seaborn   = a prettier wrapper around matplotlib
#   scipy     = statistics functions (correlations, significance tests)
#   numpy     = math functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Visual style ───────────────────────────────────────────────────────────────
# Set a clean, professional look for all charts.
# Using a style that renders well in Databricks notebooks.
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.color':        '#cccccc',
    'font.family':       'sans-serif',
    'font.size':         11,
    'axes.titlesize':    14,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# Brand colors for the store's charts
COLOR_PRIMARY   = '#2E4057'   # deep navy
COLOR_ACCENT    = '#E84855'   # warm red
COLOR_WEATHER   = '#3A86FF'   # sky blue
COLOR_POSITIVE  = '#06A77D'   # teal green
COLOR_WARNING   = '#F4A261'   # amber
COLOR_NEUTRAL   = '#8D99AE'   # gray

print("✓ Libraries loaded and chart style set")

# COMMAND ----------

# ── 2. LOAD DATA ──────────────────────────────────────────────────────────────
# Pull the Gold daily sales summary into a pandas DataFrame.
# We filter to actual rows only (not forecast rows, which don't exist yet).

df = spark.sql("""
    SELECT
        business_date,
        day_name,
        day_of_week,
        is_weekend,
        is_bread_delivery_day,
        week_of_year,
        month,
        year,
        order_count,
        net_revenue,
        avg_ticket_size,
        total_discounts,
        weather_high_f,
        weather_low_f,
        weather_feels_high_f,
        weather_category,
        weather_code,
        total_precip_in,
        total_snow_in,
        sunny_hours,
        avg_cloud_cover_pct
    FROM YOUR_CATALOG.gold.daily_sales_summary
    WHERE record_type = 'actual'
    ORDER BY business_date
""").toPandas()

# Convert date column to a proper datetime type so charts render correctly
df['business_date'] = pd.to_datetime(df['business_date'])

# ── Load special events from reference table ──────────────────────────────────
events_ref = spark.sql("""
    SELECT event_date, event_name, event_type, lower_window, upper_window
    FROM YOUR_CATALOG.reference.store_events
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

EVENT_LABELS = {date: info[0] for date, info in SPECIAL_EVENTS.items()}
EVENT_TYPES  = {date: info[1] for date, info in SPECIAL_EVENTS.items()}

print(f"✓ Loaded {len(events_ref)} events from reference.store_events")

# Catering detection threshold — any single day where avg ticket exceeds
# this value likely had a catering or event purchase inflating the total
CATERING_THRESHOLD = 500

# FUTURE_CLOSURES already loaded from reference table above

# ── Zero revenue day handling ─────────────────────────────────────────────────
# The cleanest possible rule: zero revenue = store was closed.
# The reason doesn't matter — snow, holiday, family emergency, power outage.
# These days are NOT demand signal. They are missing data.
#
# Strategy: IMPUTE rather than exclude.
# We fill zero-revenue days with the median revenue for the same day-of-week
# and month combination from surrounding open days. This preserves the
# seasonality and day-of-week patterns while filling the gap honestly.
# Imputed rows are flagged so the model can weight them lower during training.

df['store_closed']   = df['net_revenue'] == 0
closed_count         = df['store_closed'].sum()

# Identify prior-day snow as likely cause where applicable
# (joining to snow data — a zero revenue day after heavy snow is detectable)
df['prior_day_snow'] = df['total_snow_in'].shift(1)
df['likely_snow_closure'] = (df['store_closed']) & (df['prior_day_snow'] > 3)

print(f"Zero revenue (closed) days found: {closed_count}")
print(f"  Of which likely snow closures: {df['likely_snow_closure'].sum()}")
print()
print("Closed days:")
for _, row in df[df['store_closed']].iterrows():
    snow_note = f" (prior day snow: {row['prior_day_snow']:.1f}in)" if row['prior_day_snow'] > 0 else ""
    print(f"  {row['business_date'].date()}  {row['day_name']}{snow_note}")

# ── Impute zero-revenue days ───────────────────────────────────────────────────
# For each closed day, estimate what revenue would have been if open.
# Method: median of the same day-of-week in the same month, from open days only.
# This is conservative and interpretable — no black-box statistics.

df['net_revenue_imputed'] = df['net_revenue'].copy()
df['order_count_imputed'] = df['order_count'].copy()
df['is_imputed']          = False

open_days = df[~df['store_closed']]

for idx, row in df[df['store_closed']].iterrows():
    # Find open days with the same day-of-week and same month
    similar = open_days[
        (open_days['day_of_week'] == row['day_of_week']) &
        (open_days['month']       == row['month'])
    ]
    if len(similar) == 0:
        # Fall back to same day-of-week across all months if no same-month match
        similar = open_days[open_days['day_of_week'] == row['day_of_week']]

    if len(similar) > 0:
        df.at[idx, 'net_revenue_imputed'] = similar['net_revenue'].median()
        df.at[idx, 'order_count_imputed'] = similar['order_count'].median()
        df.at[idx, 'is_imputed']          = True

print(f"\nImputed {df['is_imputed'].sum()} closed days")
print(f"Imputed revenue values:")
for _, row in df[df['is_imputed']].iterrows():
    print(f"  {row['business_date'].date()}  {row['day_name']:<12} "
          f"actual: $0  →  imputed: ${row['net_revenue_imputed']:,.0f}")

# ── Flag special event days ───────────────────────────────────────────────────
df['is_special_event'] = df['business_date'].dt.strftime('%Y-%m-%d').isin(EVENT_LABELS.keys())
df['event_label']      = df['business_date'].dt.strftime('%Y-%m-%d').map(EVENT_LABELS)
df['event_type']       = df['business_date'].dt.strftime('%Y-%m-%d').map(EVENT_TYPES)

# Flag seasonal context periods (high demand but real signal — NOT excluded from training)
df['is_holiday_window'] = (
    (df['business_date'].dt.month == 12) &
    (df['business_date'].dt.day >= 15)
)

# ── Dataset summary ───────────────────────────────────────────────────────────
print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"  Date range:     {df['business_date'].min().date()} → {df['business_date'].max().date()}")
print(f"  Days of data:   {len(df)}")
print(f"  Open days:      {(~df['store_closed']).sum()}")
print(f"  Closed days:    {df['store_closed'].sum()} (imputed for training)")
print(f"  Total orders:   {df['order_count'].sum():,.0f}")
print(f"  Total revenue:  ${df['net_revenue'].sum():,.2f}")
print(f"  Avg daily rev (open days): ${df[~df['store_closed']]['net_revenue'].mean():,.2f}")
print(f"  Avg ticket:     ${df[~df['store_closed']]['avg_ticket_size'].mean():,.2f}")
print(f"  Special events: {df['is_special_event'].sum()}")
print(f"  Missing weather:{df['weather_high_f'].isna().sum()} days")
print("=" * 55)

# ── df_clean: the analysis-ready dataset ─────────────────────────────────────
# Exclude only events that DISTORT the revenue signal:
#   REVENUE_DISTORTION — artificial inflation from catering tickets
#   ORGANIC_EVENT      — unpredictable one-offs that can't be learned
#
# KEEP in training:
#   PLANNED_EVENT   — Town Stroll etc. (real signal, recurring, model should learn)
#   SEASONAL_SPIKE  — Christmas wine rush etc. (real demand, not a distortion)
#   is_holiday_window days — real high-demand period the model must learn

EXCLUDE_EVENT_TYPES = {'REVENUE_DISTORTION', 'ORGANIC_EVENT'}
df['exclude_from_training'] = df['event_type'].isin(EXCLUDE_EVENT_TYPES)

df_clean = df[~df['exclude_from_training']].copy()
df_clean['net_revenue'] = df_clean['net_revenue_imputed']
df_clean['order_count'] = df_clean['order_count_imputed']

# For charts where we specifically want open-day patterns only
df_open = df[~df['store_closed'] & ~df['exclude_from_training']].copy()

print(f"Training dataset: {len(df_clean)} days")
print(f"  Excluded (distortion/organic): {df['exclude_from_training'].sum()} days")
print(f"  Holiday window days kept:      {df_clean['is_holiday_window'].sum()} days")
print(f"  Planned events kept:           {(df_clean['event_type'] == 'PLANNED_EVENT').sum()} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 0 — Closure Days & Imputation
# MAGIC
# MAGIC Before any analysis, we need to understand the zero-revenue days in
# MAGIC the dataset. These days represent times the store was closed — for
# MAGIC whatever reason. Including them as-is would teach the forecast model
# MAGIC that certain days of the year have zero demand, which is not true.
# MAGIC
# MAGIC Instead, we **impute** them: we fill in an estimated "what would revenue
# MAGIC have been if we were open?" value based on similar open days. This
# MAGIC preserves the seasonal and day-of-week patterns without artificial gaps.
# MAGIC
# MAGIC Importantly, this also powers the **"should we stay open?"** decision:
# MAGIC the model will generate a revenue prediction for future Thanksgiving Days
# MAGIC and Christmas Days so you can weigh that against operating costs.

# COMMAND ----------

# ── CHART 0: Closure days visualization ──────────────────────────────────────

if df["store_closed"].sum() > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    open_days_df   = df[~df["store_closed"]]
    closed_days_df = df[df["store_closed"]].copy()

    ax1.plot(df["business_date"], df["net_revenue_imputed"],
             color=COLOR_NEUTRAL, linewidth=0.8, alpha=0.5, zorder=1)
    ax1.scatter(open_days_df["business_date"], open_days_df["net_revenue"],
                color=COLOR_PRIMARY, s=15, alpha=0.5, zorder=2, label="Open day")
    ax1.scatter(closed_days_df["business_date"], [0] * len(closed_days_df),
                color=COLOR_ACCENT, s=80, zorder=4, marker="X",
                label="Closed day (actual: $0)", edgecolors="white", linewidth=0.5)
    ax1.scatter(closed_days_df["business_date"], closed_days_df["net_revenue_imputed"],
                color=COLOR_POSITIVE, s=60, zorder=3, marker="D",
                label="Imputed value", edgecolors="white", linewidth=0.5)

    for _, row in closed_days_df.iterrows():
        ax1.plot([row["business_date"], row["business_date"]],
                 [0, row["net_revenue_imputed"]],
                 color=COLOR_POSITIVE, linewidth=1.5, linestyle=":", alpha=0.7, zorder=3)

    ax1.set_title("Revenue Timeline — Closed Days & Imputed Values", pad=12)
    ax1.set_ylabel("Net Revenue ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha="right")
    ax1.legend(fontsize=9, loc="upper left")

    ax2 = axes[1]
    labels = [f"{row['business_date'].strftime('%b %d')}\n({row['day_name'][:3]})"
              for _, row in closed_days_df.iterrows()]
    x = np.arange(len(closed_days_df))

    bars = ax2.bar(x, closed_days_df["net_revenue_imputed"].values,
                   0.5, color=COLOR_POSITIVE, alpha=0.8,
                   label="Imputed (estimated open-day revenue)", edgecolor="white")

    for i, (_, row) in enumerate(closed_days_df.iterrows()):
        ax2.text(i, row["net_revenue_imputed"] + 50,
                 f"${row['net_revenue_imputed']:,.0f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold",
                 color=COLOR_POSITIVE)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("Imputed Revenue for Each Closure Day", pad=12)
    ax2.set_ylabel("Net Revenue ($)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("/tmp/chart0_closures.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Closure chart saved. {len(closed_days_df)} closed days visualized.")
else:
    print("No closed days found in the dataset.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this means
# MAGIC
# MAGIC The **X marks** show actual zero-revenue days — the store was closed.
# MAGIC The **diamond marks** show what we estimate revenue would have been
# MAGIC if the store had been open, based on the median of similar days
# MAGIC (same day of week, same month).
# MAGIC
# MAGIC These estimates power the forward "should we stay open?" decision.
# MAGIC The model will generate a predicted revenue number for known future
# MAGIC holidays so you can compare it against operating costs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1 — The Full Revenue Picture
# MAGIC
# MAGIC The first thing we want to see is the complete revenue history laid out
# MAGIC on a timeline. This is the raw material our forecast model will learn from.
# MAGIC We're looking for:
# MAGIC - **Overall trend** — is revenue growing, flat, or declining over time?
# MAGIC - **Seasonality** — are there obvious high and low periods?
# MAGIC - **Outliers** — any days that look dramatically different from their neighbors?
# MAGIC - **Volatility** — how much does day-to-day revenue bounce around?

# COMMAND ----------

# ── CHART 1: Full revenue timeline ────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2.5, 1]})

# ── Top panel: daily revenue ──────────────────────────────────────────────────
ax1 = axes[0]

# Main revenue line
ax1.plot(df['business_date'], df['net_revenue'],
         color=COLOR_PRIMARY, linewidth=1.2, alpha=0.7, zorder=2)

# Shaded area under the line for visual weight
ax1.fill_between(df['business_date'], df['net_revenue'],
                 alpha=0.1, color=COLOR_PRIMARY, zorder=1)

# 7-day rolling average — smooths out the day-to-day noise so we can
# see the underlying trend more clearly
rolling_7 = df['net_revenue'].rolling(window=7, center=True).mean()
ax1.plot(df['business_date'], rolling_7,
         color=COLOR_ACCENT, linewidth=2.5, label='7-day rolling average', zorder=3)

# Mark special events with vertical lines and labels
# Color-coded by event type so the chart tells the whole story at a glance
EVENT_TYPE_COLORS = {
    'PLANNED_EVENT':      COLOR_WARNING,
    'REVENUE_DISTORTION': COLOR_ACCENT,
    'ORGANIC_EVENT':      COLOR_NEUTRAL,
}
for date_str, (label, etype) in SPECIAL_EVENTS.items():
    date = pd.to_datetime(date_str)
    if date >= df['business_date'].min() and date <= df['business_date'].max():
        event_rev = df[df['business_date'] == date]['net_revenue'].values
        if len(event_rev) > 0:
            ecolor = EVENT_TYPE_COLORS.get(etype, COLOR_WARNING)
            ax1.axvline(date, color=ecolor, linewidth=2,
                       linestyle='--', alpha=0.8, zorder=4)
            ax1.annotate(f"{label}\n({etype.replace('_',' ').lower()})",
                        xy=(date, event_rev[0]),
                        xytext=(10, 15),
                        textcoords='offset points',
                        fontsize=8.5,
                        color=ecolor,
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=ecolor, lw=1.5))

# Weekend shading — light gray bands on Saturdays and Sundays
for _, row in df[df['is_weekend']].iterrows():
    ax1.axvspan(row['business_date'] - pd.Timedelta(hours=12),
                row['business_date'] + pd.Timedelta(hours=12),
                alpha=0.06, color=COLOR_PRIMARY, zorder=0)

ax1.set_title('Daily Net Revenue — Full History', pad=15)
ax1.set_ylabel('Net Revenue ($)')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax1.legend(loc='upper left', framealpha=0.9)

# Add a subtle note about the weekend shading
ax1.text(0.99, 0.02, 'Light bands = weekends',
         transform=ax1.transAxes, fontsize=8,
         color=COLOR_NEUTRAL, ha='right', va='bottom')

# ── Bottom panel: order count ─────────────────────────────────────────────────
ax2 = axes[1]
ax2.bar(df['business_date'], df['order_count'],
        color=COLOR_PRIMARY, alpha=0.5, width=0.8, zorder=2)
rolling_orders = df['order_count'].rolling(window=7, center=True).mean()
ax2.plot(df['business_date'], rolling_orders,
         color=COLOR_ACCENT, linewidth=2, zorder=3)
ax2.set_ylabel('Orders (tickets)')
ax2.set_xlabel('Date')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax2.set_title('Daily Order Count', pad=10)

plt.tight_layout(h_pad=3)
plt.savefig('/tmp/chart1_revenue_timeline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this means
# MAGIC
# MAGIC This is the store's full revenue story since opening day. A few things
# MAGIC to notice:
# MAGIC
# MAGIC - The **red line** is a 7-day smoothed average — it filters out the
# MAGIC   noisy day-to-day swings and shows the underlying momentum.
# MAGIC - **Light gray bands** mark weekends — notice how the bars tend to be
# MAGIC   taller on those days.
# MAGIC - The **orange dashed lines** mark our two known special events.
# MAGIC - The **bottom panel** shows order count (number of customer transactions)
# MAGIC   — this tells a slightly different story than revenue because a high-revenue
# MAGIC   day could be driven by fewer large purchases or many small ones.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2 — Day of Week Patterns
# MAGIC
# MAGIC One of the strongest signals in retail data is the **day-of-week effect**.
# MAGIC Most stores see dramatically different traffic patterns on different days.
# MAGIC We need to quantify exactly how strong this effect is before building
# MAGIC the forecast model — it will be one of the most important features.

# COMMAND ----------

# ── CHART 2: Day-of-week revenue distribution ─────────────────────────────────

# Day ordering: Monday through Sunday
DAY_ORDER    = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_COLORS   = [COLOR_NEUTRAL] * 5 + [COLOR_ACCENT, COLOR_ACCENT]  # weekdays gray, weekends red

# Exclude special event days from this analysis so they don't skew the averages
df_clean = df[~df['is_special_event']].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Left: box plot of revenue by day ─────────────────────────────────────────
ax1 = axes[0]

day_data = [df_clean[df_clean['day_name'] == day]['net_revenue'].values
            for day in DAY_ORDER]

bp = ax1.boxplot(day_data, patch_artist=True, notch=False,
                 medianprops=dict(color='white', linewidth=2.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5),
                 flierprops=dict(marker='o', markersize=4, alpha=0.5))

for patch, color in zip(bp['boxes'], DAY_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax1.set_xticklabels([d[:3] for d in DAY_ORDER])
ax1.set_title('Revenue Distribution by Day of Week\n(special events excluded)', pad=12)
ax1.set_ylabel('Net Revenue ($)')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Annotate with median values above each box
for i, day in enumerate(DAY_ORDER):
    med = df_clean[df_clean['day_name'] == day]['net_revenue'].median()
    ax1.text(i + 1, med + 200, f'${med:,.0f}',
             ha='center', va='bottom', fontsize=8.5, fontweight='bold',
             color=DAY_COLORS[i])

# ── Right: average orders by day ──────────────────────────────────────────────
ax2 = axes[1]

avg_orders = [df_clean[df_clean['day_name'] == day]['order_count'].mean()
              for day in DAY_ORDER]

bars = ax2.bar([d[:3] for d in DAY_ORDER], avg_orders,
               color=DAY_COLORS, alpha=0.8, edgecolor='white', linewidth=0.5)

# Add value labels on top of bars
for bar, val in zip(bars, avg_orders):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{val:.0f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_title('Average Order Count by Day of Week\n(special events excluded)', pad=12)
ax2.set_ylabel('Average Orders')

# Add a legend explaining the colors
weekend_patch = mpatches.Patch(color=COLOR_ACCENT, alpha=0.8, label='Weekend')
weekday_patch = mpatches.Patch(color=COLOR_NEUTRAL, alpha=0.8, label='Weekday')
ax2.legend(handles=[weekend_patch, weekday_patch], loc='upper left')

plt.tight_layout()
plt.savefig('/tmp/chart2_day_of_week.png', dpi=150, bbox_inches='tight')
plt.show()

# Print the numbers for easy reference
print("\nMedian revenue and average orders by day:")
print(f"{'Day':<12} {'Median Revenue':>15} {'Avg Orders':>12} {'vs Monday':>10}")
print("-" * 52)
monday_rev = df_clean[df_clean['day_name'] == 'Monday']['net_revenue'].median()
for day in DAY_ORDER:
    med_rev  = df_clean[df_clean['day_name'] == day]['net_revenue'].median()
    avg_ord  = df_clean[df_clean['day_name'] == day]['order_count'].mean()
    vs_mon   = (med_rev / monday_rev - 1) * 100
    marker   = " ◀ bread delivery" if day in ['Tuesday', 'Friday'] else ""
    print(f"{day:<12} ${med_rev:>13,.0f} {avg_ord:>11.1f} {vs_mon:>+9.0f}%{marker}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this means
# MAGIC
# MAGIC The **box plot** on the left shows the spread of revenue for each day.
# MAGIC The line in the middle of each box is the median (the "typical" day).
# MAGIC The box covers the middle 50% of days. Dots outside the whiskers are
# MAGIC unusually high or low days.
# MAGIC
# MAGIC The **bar chart** on the right shows average customer count by day.
# MAGIC
# MAGIC This analysis directly answers: *"How much more should we expect to
# MAGIC sell on a Saturday vs a Monday?"* That multiplier becomes a core feature
# MAGIC in our forecast model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3 — Seasonality
# MAGIC
# MAGIC You described expecting roughly **2x the monthly revenue in summer vs winter**
# MAGIC with shoulder months at about 1/12 of yearly sales. With only 7 months of
# MAGIC data we can't see a full year yet, but we can see the early shape of the
# MAGIC curve and validate whether the data is tracking toward your expectation.

# COMMAND ----------

# ── CHART 3: Monthly and weekly revenue trends ────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# ── Top: monthly revenue ──────────────────────────────────────────────────────
ax1 = axes[0]

monthly = df_clean.groupby(df_clean['business_date'].dt.to_period('M')).agg(
    total_revenue=('net_revenue', 'sum'),
    total_orders=('order_count', 'sum'),
    days=('net_revenue', 'count')
).reset_index()
monthly['month_str']    = monthly['business_date'].astype(str)
monthly['daily_avg_rev'] = monthly['total_revenue'] / monthly['days']

bars = ax1.bar(monthly['month_str'], monthly['total_revenue'],
               color=COLOR_PRIMARY, alpha=0.7, edgecolor='white', linewidth=0.5)

# Annotate bars with total and daily average
for bar, (_, row) in zip(bars, monthly.iterrows()):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 200,
             f'${row["total_revenue"]:,.0f}\n(${row["daily_avg_rev"]:,.0f}/day)',
             ha='center', va='bottom', fontsize=8.5, color=COLOR_PRIMARY)

ax1.set_title('Monthly Total Revenue\n(special events excluded)', pad=12)
ax1.set_ylabel('Total Revenue ($)')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha='right')

# ── Bottom: weekly rolling revenue with trend line ────────────────────────────
ax2 = axes[1]

weekly = df_clean.resample('W', on='business_date')['net_revenue'].sum().reset_index()
ax2.bar(weekly['business_date'], weekly['net_revenue'],
        color=COLOR_PRIMARY, alpha=0.4, width=5, label='Weekly revenue')

# Add a trend line using linear regression
x_numeric = (weekly['business_date'] - weekly['business_date'].min()).dt.days
slope, intercept, r, p, _ = stats.linregress(x_numeric, weekly['net_revenue'])
trend_line = slope * x_numeric + intercept
ax2.plot(weekly['business_date'], trend_line,
         color=COLOR_ACCENT, linewidth=2.5, linestyle='--',
         label=f'Trend ({"+" if slope > 0 else ""}{slope:.1f}/day, R²={r**2:.2f})')

ax2.set_title('Weekly Revenue with Trend Line', pad=12)
ax2.set_ylabel('Weekly Revenue ($)')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha='right')
ax2.legend(loc='upper left')

# Annotate what R² means in plain English
r2_text = "Strong trend" if r**2 > 0.5 else "Moderate trend" if r**2 > 0.25 else "Weak trend"
ax2.text(0.99, 0.05,
         f'R² = {r**2:.2f} ({r2_text})\nSlope = ${slope*7:+.0f}/week',
         transform=ax2.transAxes, fontsize=9,
         ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

plt.tight_layout(h_pad=3)
plt.savefig('/tmp/chart3_seasonality.png', dpi=150, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this means
# MAGIC
# MAGIC The monthly bars show total revenue for each month we have data.
# MAGIC The number in parentheses is the daily average — this corrects for the
# MAGIC fact that some months have more days than others, or that we only have
# MAGIC partial data for the first and last months.
# MAGIC
# MAGIC The **trend line** in the weekly chart shows the overall direction of
# MAGIC the business. The **R² value** (R-squared) measures how well a straight
# MAGIC line fits the data — 1.0 would be a perfect straight line, 0.0 would be
# MAGIC random noise. The **slope** tells you how much weekly revenue is changing
# MAGIC per week on average.
# MAGIC
# MAGIC Since we only have data from September, we're seeing the tail end of
# MAGIC fall moving into winter. The summer comparison won't be visible until
# MAGIC we have a full year of data — but after 12 months, this chart will show
# MAGIC the full seasonal arc you described.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4 — The Bread Delivery Effect
# MAGIC
# MAGIC Tuesdays and Fridays are Kneady Mama bread delivery days. The hypothesis
# MAGIC is that customers come in specifically for fresh bread, and while they're
# MAGIC there they buy other things too — driving higher overall traffic.
# MAGIC Let's test whether this is statistically real or just a feeling.

# COMMAND ----------

# ── CHART 4: Bread delivery day analysis ─────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Split the data three ways: bread delivery days, same days of week without delivery,
# and all other weekdays
tue_fri_bread = df_clean[df_clean['is_bread_delivery_day']]
tue_fri_all   = df_clean[df_clean['day_name'].isin(['Tuesday', 'Friday'])]
other_weekdays = df_clean[df_clean['day_name'].isin(['Monday', 'Wednesday', 'Thursday'])]

# ── Left: revenue comparison ──────────────────────────────────────────────────
ax1 = axes[0]
groups    = ['Mon/Wed/Thu\n(non-delivery)', 'Tue/Fri\n(bread delivery)']
rev_means = [other_weekdays['net_revenue'].mean(), tue_fri_bread['net_revenue'].mean()]
rev_stds  = [other_weekdays['net_revenue'].std(), tue_fri_bread['net_revenue'].std()]

bars = ax1.bar(groups, rev_means, color=[COLOR_NEUTRAL, COLOR_POSITIVE],
               alpha=0.8, edgecolor='white', width=0.5)
ax1.errorbar(groups, rev_means, yerr=rev_stds,
             fmt='none', color='black', capsize=6, linewidth=1.5)

for bar, val in zip(bars, rev_means):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 50,
             f'${val:,.0f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

# Run a t-test to check if the difference is statistically significant
t_stat, p_val = stats.ttest_ind(tue_fri_bread['net_revenue'],
                                 other_weekdays['net_revenue'])
significance = "Statistically significant ✓" if p_val < 0.05 else f"Not significant (p={p_val:.2f})"
ax1.set_title(f'Average Daily Revenue\n{significance}', pad=12)
ax1.set_ylabel('Average Net Revenue ($)')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# ── Middle: order count comparison ───────────────────────────────────────────
ax2 = axes[1]
ord_means = [other_weekdays['order_count'].mean(), tue_fri_bread['order_count'].mean()]
ord_stds  = [other_weekdays['order_count'].std(),  tue_fri_bread['order_count'].std()]

bars2 = ax2.bar(groups, ord_means, color=[COLOR_NEUTRAL, COLOR_POSITIVE],
                alpha=0.8, edgecolor='white', width=0.5)
ax2.errorbar(groups, ord_means, yerr=ord_stds,
             fmt='none', color='black', capsize=6, linewidth=1.5)

for bar, val in zip(bars2, ord_means):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

t_stat2, p_val2 = stats.ttest_ind(tue_fri_bread['order_count'],
                                   other_weekdays['order_count'])
significance2 = "Statistically significant ✓" if p_val2 < 0.05 else f"Not significant (p={p_val2:.2f})"
ax2.set_title(f'Average Order Count\n{significance2}', pad=12)
ax2.set_ylabel('Average Orders')

# ── Right: Tuesday vs Friday breakdown ───────────────────────────────────────
ax3 = axes[2]
tue_rev = df_clean[df_clean['day_name'] == 'Tuesday']['net_revenue']
fri_rev = df_clean[df_clean['day_name'] == 'Friday']['net_revenue']

bp = ax3.boxplot([tue_rev, fri_rev], patch_artist=True,
                 medianprops=dict(color='white', linewidth=2.5))
bp['boxes'][0].set_facecolor(COLOR_POSITIVE)
bp['boxes'][0].set_alpha(0.75)
bp['boxes'][1].set_facecolor(COLOR_ACCENT)
bp['boxes'][1].set_alpha(0.75)

ax3.set_xticklabels(['Tuesday', 'Friday'])
ax3.set_title('Tuesday vs Friday Revenue\n(both are bread delivery days)', pad=12)
ax3.set_ylabel('Net Revenue ($)')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Annotate medians
for i, (data, day) in enumerate([(tue_rev, 'Tue'), (fri_rev, 'Fri')]):
    med = data.median()
    ax3.text(i + 1, med + 100, f'Median\n${med:,.0f}',
             ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.suptitle('Kneady Mama Bread Delivery Day Effect', fontsize=14,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/tmp/chart4_bread_delivery.png', dpi=150, bbox_inches='tight')
plt.show()

# Print lift calculation
lift_rev = (tue_fri_bread['net_revenue'].mean() / other_weekdays['net_revenue'].mean() - 1) * 100
lift_ord = (tue_fri_bread['order_count'].mean() / other_weekdays['order_count'].mean() - 1) * 100
print(f"\nBread delivery day lift:")
print(f"  Revenue:      +{lift_rev:.1f}% vs other weekdays")
print(f"  Order count:  +{lift_ord:.1f}% vs other weekdays")
print(f"  p-value (rev): {p_val:.3f} ({'significant' if p_val < 0.05 else 'not significant'})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this means
# MAGIC
# MAGIC The **error bars** on the bar charts show the typical range of variation —
# MAGIC if the bars are similar heights but the error bars overlap a lot, the
# MAGIC difference might just be random noise.
# MAGIC
# MAGIC The **p-value** is a statistical measure of confidence. A p-value below
# MAGIC 0.05 means we're at least 95% confident the difference is real and not
# MAGIC due to chance. This is the standard threshold used in science.
# MAGIC
# MAGIC The Tuesday vs Friday breakdown is interesting because even though both
# MAGIC are bread delivery days, Friday is also going into the weekend — so it's
# MAGIC hard to separate the bread effect from the natural Friday traffic increase.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5 — Weather Effects
# MAGIC
# MAGIC Does weather actually drive sales? Intuitively it seems like it should —
# MAGIC a beautiful sunny day might bring more foot traffic than a cold rainy one.
# MAGIC But intuition can be wrong. Let's look at the data.

# COMMAND ----------

# ── CHART 5: Weather correlation analysis ────────────────────────────────────

# Filter to days with weather data for this section
df_weather = df_clean[df_clean['weather_high_f'].notna()].copy()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Helper to add a regression line and correlation annotation
def add_regression(ax, x, y, color=COLOR_ACCENT):
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return
    slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_line, slope * x_line + intercept,
            color=color, linewidth=2, linestyle='--', alpha=0.8)
    sig = "p<0.05 ✓" if p < 0.05 else f"p={p:.2f}"
    ax.text(0.05, 0.95, f'r = {r:.2f}\n{sig}',
            transform=ax.transAxes, fontsize=9,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# ── Revenue vs high temperature ───────────────────────────────────────────────
ax = axes[0, 0]
scatter = ax.scatter(df_weather['weather_high_f'], df_weather['net_revenue'],
                     c=df_weather['is_weekend'].map({True: COLOR_ACCENT, False: COLOR_PRIMARY}),
                     alpha=0.6, s=40, edgecolors='none')
add_regression(ax, df_weather['weather_high_f'], df_weather['net_revenue'])
ax.set_xlabel('Daily High Temperature (°F)')
ax.set_ylabel('Net Revenue ($)')
ax.set_title('Revenue vs Temperature')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
weekend_dot = plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=COLOR_ACCENT, markersize=8, label='Weekend')
weekday_dot = plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=COLOR_PRIMARY, markersize=8, label='Weekday')
ax.legend(handles=[weekend_dot, weekday_dot], fontsize=8)

# ── Orders vs high temperature ────────────────────────────────────────────────
ax = axes[0, 1]
ax.scatter(df_weather['weather_high_f'], df_weather['order_count'],
           c=df_weather['is_weekend'].map({True: COLOR_ACCENT, False: COLOR_PRIMARY}),
           alpha=0.6, s=40, edgecolors='none')
add_regression(ax, df_weather['weather_high_f'], df_weather['order_count'])
ax.set_xlabel('Daily High Temperature (°F)')
ax.set_ylabel('Order Count')
ax.set_title('Orders vs Temperature')

# ── Revenue vs precipitation ──────────────────────────────────────────────────
ax = axes[0, 2]
ax.scatter(df_weather['total_precip_in'], df_weather['net_revenue'],
           c=df_weather['is_weekend'].map({True: COLOR_ACCENT, False: COLOR_PRIMARY}),
           alpha=0.6, s=40, edgecolors='none')
add_regression(ax, df_weather['total_precip_in'], df_weather['net_revenue'], COLOR_WEATHER)
ax.set_xlabel('Total Precipitation (inches)')
ax.set_ylabel('Net Revenue ($)')
ax.set_title('Revenue vs Precipitation')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# ── Revenue by weather category ───────────────────────────────────────────────
ax = axes[1, 0]
cat_order  = ['Clear', 'Cloudy', 'Rainy', 'Snowy', 'Stormy', 'Foggy']
cat_colors = [COLOR_WARNING, COLOR_NEUTRAL, COLOR_WEATHER, '#ADE8F4', COLOR_ACCENT, '#6C757D']
cat_data   = []
cat_labels = []
cat_clrs   = []
for cat, clr in zip(cat_order, cat_colors):
    data = df_weather[df_weather['weather_category'] == cat]['net_revenue']
    if len(data) > 0:
        cat_data.append(data.values)
        cat_labels.append(f'{cat}\n(n={len(data)})')
        cat_clrs.append(clr)

bp = ax.boxplot(cat_data, patch_artist=True,
                medianprops=dict(color='white', linewidth=2))
for patch, clr in zip(bp['boxes'], cat_clrs):
    patch.set_facecolor(clr)
    patch.set_alpha(0.75)
ax.set_xticklabels(cat_labels, fontsize=8)
ax.set_title('Revenue by Weather Category')
ax.set_ylabel('Net Revenue ($)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# ── Revenue vs sunshine hours ─────────────────────────────────────────────────
ax = axes[1, 1]
ax.scatter(df_weather['sunny_hours'], df_weather['net_revenue'],
           c=df_weather['is_weekend'].map({True: COLOR_ACCENT, False: COLOR_PRIMARY}),
           alpha=0.6, s=40, edgecolors='none')
add_regression(ax, df_weather['sunny_hours'], df_weather['net_revenue'], COLOR_WARNING)
ax.set_xlabel('Sunny Hours in Day')
ax.set_ylabel('Net Revenue ($)')
ax.set_title('Revenue vs Sunshine')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# ── Correlation heatmap ───────────────────────────────────────────────────────
ax = axes[1, 2]
weather_cols = ['net_revenue', 'order_count', 'weather_high_f',
                'weather_low_f', 'total_precip_in', 'sunny_hours',
                'avg_cloud_cover_pct']
corr_labels  = ['Revenue', 'Orders', 'High Temp', 'Low Temp',
                'Precip', 'Sunny Hrs', 'Cloud %']
corr_matrix  = df_weather[weather_cols].corr()
corr_matrix.index   = corr_labels
corr_matrix.columns = corr_labels

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, ax=ax, mask=mask,
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 8},
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8})
ax.set_title('Weather-Sales Correlation Matrix')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
plt.setp(ax.yaxis.get_majorticklabels(), fontsize=8)

plt.suptitle('Weather Effects on Sales', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/tmp/chart5_weather.png', dpi=150, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this means
# MAGIC
# MAGIC **Scatter plots** (top row): each dot is one day. The dashed line is the
# MAGIC best-fit trend. The **r value** measures correlation strength:
# MAGIC - r close to +1.0 = strong positive relationship (warmer = more revenue)
# MAGIC - r close to -1.0 = strong negative relationship (more rain = less revenue)
# MAGIC - r close to 0 = no relationship
# MAGIC
# MAGIC **Correlation heatmap** (bottom right): green = positive correlation,
# MAGIC red = negative correlation, white = no correlation. Read it like a grid —
# MAGIC find Revenue on one axis and any weather variable on the other to see
# MAGIC how strongly they're related.
# MAGIC
# MAGIC The key insight for the forecast model: which weather variables are
# MAGIC actually worth including? Anything with |r| > 0.15 is worth considering.
# MAGIC Anything below that is probably noise.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6 — Outlier Detection
# MAGIC
# MAGIC Outliers are days that are so unusual they could mislead our forecast
# MAGIC model if not handled carefully. We have two types:
# MAGIC
# MAGIC - **Event-driven volume spikes** (Town Stroll): lots of normal tickets,
# MAGIC   real customer behavior, but not representative of a typical day
# MAGIC - **Revenue distortions** (Wine Tasting, catering): one or a few
# MAGIC   very large tickets that inflate the daily total artificially

# COMMAND ----------

# ── CHART 6: Outlier detection ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Left: Z-score outlier detection ──────────────────────────────────────────
ax1 = axes[0]

# Z-score measures how many standard deviations a day is from the average.
# Days beyond ±2.5 standard deviations are statistical outliers.
df['z_revenue'] = stats.zscore(df['net_revenue'])
df['z_orders']  = stats.zscore(df['order_count'])
ZSCORE_THRESHOLD = 2.5

outliers     = df[np.abs(df['z_revenue']) > ZSCORE_THRESHOLD]
non_outliers = df[np.abs(df['z_revenue']) <= ZSCORE_THRESHOLD]

ax1.scatter(non_outliers['business_date'], non_outliers['net_revenue'],
            color=COLOR_PRIMARY, alpha=0.5, s=30, label='Normal days', zorder=2)
ax1.scatter(outliers['business_date'], outliers['net_revenue'],
            color=COLOR_ACCENT, s=80, zorder=3,
            label=f'Outliers (|z| > {ZSCORE_THRESHOLD})', edgecolors='white', linewidth=1)

# Label each outlier
for _, row in outliers.iterrows():
    label = row.get('event_label', '') or f"{row['day_name'][:3]} {row['business_date'].strftime('%b %d')}"
    ax1.annotate(label,
                xy=(row['business_date'], row['net_revenue']),
                xytext=(0, 12), textcoords='offset points',
                ha='center', fontsize=8.5, fontweight='bold',
                color=COLOR_ACCENT,
                arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=1))

# Add threshold lines
mean_rev = df['net_revenue'].mean()
std_rev  = df['net_revenue'].std()
ax1.axhline(mean_rev + ZSCORE_THRESHOLD * std_rev,
            color=COLOR_ACCENT, linestyle=':', alpha=0.5, linewidth=1.5,
            label=f'±{ZSCORE_THRESHOLD}σ threshold')
ax1.axhline(mean_rev - ZSCORE_THRESHOLD * std_rev,
            color=COLOR_ACCENT, linestyle=':', alpha=0.5, linewidth=1.5)

ax1.set_title('Revenue Outlier Detection (Z-Score Method)', pad=12)
ax1.set_ylabel('Net Revenue ($)')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, ha='right')
ax1.legend(fontsize=8, loc='upper left')

# ── Right: ticket size distribution for catering detection ───────────────────
ax2 = axes[1]

# Look at the distribution of average ticket sizes per day
ax2.hist(df['avg_ticket_size'], bins=40, color=COLOR_PRIMARY,
         alpha=0.7, edgecolor='white', linewidth=0.5)

ax2.axvline(CATERING_THRESHOLD, color=COLOR_ACCENT, linewidth=2.5,
            linestyle='--', label=f'Catering threshold (${CATERING_THRESHOLD})')

# Highlight days above the threshold
catering_days = df[df['avg_ticket_size'] > CATERING_THRESHOLD]
if len(catering_days) > 0:
    ax2.axvspan(CATERING_THRESHOLD, df['avg_ticket_size'].max() + 50,
                alpha=0.1, color=COLOR_ACCENT)
    for _, row in catering_days.iterrows():
        label = row.get('event_label', '') or row['business_date'].strftime('%b %d')
        ax2.annotate(f"{label}\n${row['avg_ticket_size']:.0f} avg",
                    xy=(row['avg_ticket_size'], 1),
                    xytext=(0, 20), textcoords='offset points',
                    ha='center', fontsize=8.5, fontweight='bold',
                    color=COLOR_ACCENT,
                    arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT))

ax2.set_title('Distribution of Average Daily Ticket Size\n(catering days have unusually high averages)',
              pad=12)
ax2.set_xlabel('Average Ticket Size ($)')
ax2.set_ylabel('Number of Days')
ax2.legend()

plt.tight_layout()
plt.savefig('/tmp/chart6_outliers.png', dpi=150, bbox_inches='tight')
plt.show()

# Print outlier summary with treatment guidance
print("\nOutlier days detected (|z-score| > 2.5):")
print(f"{'Date':<14} {'Day':<12} {'Revenue':>10} {'Z':>6} {'Event':<22} {'Treatment'}")
print("-" * 85)

TREATMENT_MAP = {
    'PLANNED_EVENT':      'Keep — real signal, model learns pattern',
    'REVENUE_DISTORTION': 'Adjust — remove catering excess revenue',
    'ORGANIC_EVENT':      'Exclude — unpredictable, widens interval',
    None:                 'Investigate — may be real seasonal spike',
}

for _, row in outliers.sort_values('net_revenue', ascending=False).iterrows():
    label     = row.get('event_label') or ''
    etype     = row.get('event_type')
    treatment = TREATMENT_MAP.get(etype, TREATMENT_MAP[None])
    # Flag holiday window days as real signal
    if row.get('is_holiday_window') and not etype:
        label     = label or 'Holiday window'
        treatment = 'Keep — real seasonal demand (holiday wine/gifts)'
    print(f"{str(row['business_date'].date()):<14} {row['day_name']:<12} "
          f"${row['net_revenue']:>9,.2f} {row['z_revenue']:>+5.1f}  "
          f"{label:<22} {treatment}")

print(f"\nDays above catering threshold (avg ticket > ${CATERING_THRESHOLD}):")
if len(catering_days) > 0:
    for _, row in catering_days.iterrows():
        label = row.get('event_label', '') or 'Unknown'
        print(f"  {str(row['business_date'].date())}  {row['day_name']:<12}  "
              f"avg ticket: ${row['avg_ticket_size']:.2f}  ({label})")
else:
    print("  None detected at the daily average level.")
    print("  Note: catering detection at the order level requires checking Silver data.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this means
# MAGIC
# MAGIC **Z-score** measures how unusual a day is in terms of standard deviations
# MAGIC from the average. A z-score of +2.5 means that day's revenue was 2.5
# MAGIC standard deviations above average — something that would only happen by
# MAGIC chance about 1% of the time on a truly random day. These are the days
# MAGIC we need to flag before training.
# MAGIC
# MAGIC The **ticket size distribution** (right chart) shows the shape of
# MAGIC average ticket sizes across all days. Most days cluster around the
# MAGIC typical average. Days far to the right of the distribution are candidates
# MAGIC for catering/event revenue distortion.
# MAGIC
# MAGIC **Important:** the catering detection here is at the daily average level.
# MAGIC In the feature engineering notebook, we'll do a more precise check at
# MAGIC the individual order level using the Silver data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7 — Autocorrelation (Does Yesterday Predict Today?)
# MAGIC
# MAGIC One of the questions the forecast model needs to answer is: *"How much
# MAGIC does knowing yesterday's revenue tell us about today's revenue?"*
# MAGIC This is called **autocorrelation** — the correlation of a variable with
# MAGIC its own past values. It directly informs which lag features to include
# MAGIC in the LightGBM model.

# COMMAND ----------

# ── CHART 7: Autocorrelation analysis ────────────────────────────────────────

from pandas.plotting import autocorrelation_plot

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: revenue lagged scatter plots ───────────────────────────────────────
ax1 = axes[0]

rev = df_clean['net_revenue'].values
lag1_corr = np.corrcoef(rev[:-1], rev[1:])[0, 1]
lag7_corr = np.corrcoef(rev[:-7], rev[7:])[0, 1]

ax1.scatter(rev[:-7], rev[7:], alpha=0.4, color=COLOR_PRIMARY, s=30,
            label=f'Lag 7 days (r={lag7_corr:.2f})')
ax1.scatter(rev[:-1], rev[1:], alpha=0.4, color=COLOR_ACCENT, s=30,
            label=f'Lag 1 day (r={lag1_corr:.2f})')

ax1.set_xlabel('Revenue on Day N ($)')
ax1.set_ylabel('Revenue on Day N+k ($)')
ax1.set_title('Does Past Revenue Predict Future Revenue?', pad=12)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.legend(fontsize=9)

# ── Right: lag correlation bar chart ─────────────────────────────────────────
ax2 = axes[1]

lags     = list(range(1, 15))
lag_corrs = []
for lag in lags:
    if len(rev) > lag:
        r = np.corrcoef(rev[:-lag], rev[lag:])[0, 1]
        lag_corrs.append(r)
    else:
        lag_corrs.append(0)

colors = [COLOR_ACCENT if lag == 7 else COLOR_PRIMARY for lag in lags]
bars = ax2.bar(lags, lag_corrs, color=colors, alpha=0.75,
               edgecolor='white', linewidth=0.5)

# Add significance threshold lines
n = len(rev)
sig_threshold = 1.96 / np.sqrt(n)
ax2.axhline(sig_threshold,  color='gray', linestyle='--', linewidth=1.5,
            alpha=0.7, label=f'95% significance (±{sig_threshold:.2f})')
ax2.axhline(-sig_threshold, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axhline(0, color='black', linewidth=0.5)

# Highlight lag 7 (same day last week)
ax2.get_children()[6].set_facecolor(COLOR_ACCENT)
ax2.annotate('Lag 7\n(same day\nlast week)',
             xy=(7, lag_corrs[6]),
             xytext=(9, lag_corrs[6] + 0.05),
             fontsize=8.5, color=COLOR_ACCENT, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT))

ax2.set_xlabel('Lag (days)')
ax2.set_ylabel('Correlation with current revenue')
ax2.set_title('Revenue Autocorrelation by Lag', pad=12)
ax2.set_xticks(lags)
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/tmp/chart7_autocorrelation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nKey lag correlations:")
for lag, r in zip(lags[:8], lag_corrs[:8]):
    sig = " ← significant" if abs(r) > sig_threshold else ""
    print(f"  Lag {lag:2d} day(s): r = {r:+.3f}{sig}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### What this means
# MAGIC
# MAGIC The bar chart shows how correlated today's revenue is with revenue
# MAGIC from 1, 2, 3... days ago. Bars above the dashed gray lines are
# MAGIC statistically meaningful correlations.
# MAGIC
# MAGIC **Lag 7** (highlighted in red) is particularly important — it represents
# MAGIC the same day of the previous week. A high lag-7 correlation means that
# MAGIC *"what happened last Tuesday is a good predictor of this Tuesday"* — which
# MAGIC makes intuitive sense for a retail store with strong weekly patterns.
# MAGIC
# MAGIC This tells us which lag features to include in the LightGBM model:
# MAGIC any lag with a significant bar is worth including as a feature.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 8 — EDA Summary & Model Recommendations
# MAGIC
# MAGIC Based on everything we've seen, here's what the data is telling us
# MAGIC about how to build the forecast model.

# COMMAND ----------

# ── Summary statistics table ──────────────────────────────────────────────────

print("=" * 65)
print("EDA SUMMARY — KEY FINDINGS FOR THE FORECAST MODEL")
print("=" * 65)

# Day of week multipliers vs Monday baseline
monday_median = df_clean[df_clean['day_name'] == 'Monday']['net_revenue'].median()
print("\n1. DAY-OF-WEEK MULTIPLIERS (vs Monday baseline)")
print(f"   {'Day':<12} {'Multiplier':>12} {'Signal strength'}")
print("   " + "-" * 40)
for day in DAY_ORDER:
    med = df_clean[df_clean['day_name'] == day]['net_revenue'].median()
    mult = med / monday_median
    strength = "████" if mult > 2 else "███" if mult > 1.5 else "██" if mult > 1.2 else "█"
    print(f"   {day:<12} {mult:>10.2f}x  {strength}")

# Weather correlations
print("\n2. WEATHER FEATURE IMPORTANCE")
print(f"   {'Feature':<20} {'r with revenue':>15} {'Include?'}")
print("   " + "-" * 50)
weather_features = {
    'High Temp (°F)':  df_weather['weather_high_f'].corr(df_weather['net_revenue']),
    'Low Temp (°F)':   df_weather['weather_low_f'].corr(df_weather['net_revenue']),
    'Precipitation':   df_weather['total_precip_in'].corr(df_weather['net_revenue']),
    'Sunny Hours':     df_weather['sunny_hours'].corr(df_weather['net_revenue']),
    'Cloud Cover %':   df_weather['avg_cloud_cover_pct'].corr(df_weather['net_revenue']),
}
for feature, r in sorted(weather_features.items(), key=lambda x: abs(x[1]), reverse=True):
    include = "YES ✓" if abs(r) > 0.1 else "maybe" if abs(r) > 0.05 else "no"
    print(f"   {feature:<20} {r:>+14.3f}  {include}")

# Outliers
print("\n3. OUTLIER HANDLING — EVENT TYPE TAXONOMY")
print(f"   Outlier days detected:    {len(outliers)}")
print(f"   Of which:")
for etype, treatment in TREATMENT_MAP.items():
    if etype:
        count = len(outliers[outliers['event_type'] == etype])
        if count > 0:
            print(f"     {etype:<22} {count} day(s) — {treatment}")
holiday_outliers = outliers[outliers['is_holiday_window'] & outliers['event_type'].isna()]
if len(holiday_outliers) > 0:
    print(f"     {'Seasonal spike':<22} {len(holiday_outliers)} day(s) — Keep (real holiday demand)")
print(f"   Catering threshold:       ${CATERING_THRESHOLD} per ticket")

# Bread delivery
print("\n4. BREAD DELIVERY DAY EFFECT")
print(f"   Revenue lift:  +{lift_rev:.1f}%")
print(f"   Order lift:    +{lift_ord:.1f}%")
print(f"   Significant:   {'YES ✓' if p_val < 0.05 else 'Not yet (need more data)'}")

# Lag features
sig_lags = [lag for lag, r in zip(lags, lag_corrs) if abs(r) > sig_threshold]
print(f"\n5. RECOMMENDED LAG FEATURES FOR LIGHTGBM")
print(f"   Significant lags: {sig_lags}")
print(f"   Always include: 7 (same day last week), 14 (2 weeks ago)")

print("\n6. RECOMMENDED MODEL APPROACH")
print("   Primary:  Prophet (handles seasonality + special events natively)")
print("   Challenger: LightGBM (with day-of-week, weather, lag features)")
print("   Baseline: Weighted 4-week historical average by day-of-week")
print("=" * 65)