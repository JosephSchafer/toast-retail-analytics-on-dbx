# Databricks notebook source

# MAGIC %md
# MAGIC # Validation: Tips Incorrectly Included in Net Revenue
# MAGIC
# MAGIC **Business Rule:**
# MAGIC - Tips should only appear on **Online** and **Toast Pickup App** orders
# MAGIC - Tips must be **excluded** from Net Revenue
# MAGIC
# MAGIC **What this notebook checks:**
# MAGIC 1. Tips by order source — are any in-store orders carrying tips?
# MAGIC 2. Is `tip_amount` embedded inside `total_amount` in Silver?
# MAGIC 3. Does Gold `net_revenue` include tips? (compare Silver vs Gold)
# MAGIC 4. Magnitude of discrepancy — cumulative and by month
# MAGIC 5. Verdict and recommended fix

# COMMAND ----------

CATALOG = "3sp_analytics_workspace"
SILVER  = f"{CATALOG}.silver_sales.orders_silver"
GOLD    = f"{CATALOG}.gold.daily_sales_summary"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 1 — Tips by Order Source
# MAGIC
# MAGIC Tips should **only** exist on `Online` and `Toast Pickup App` orders.
# MAGIC Any other source with `tip_amount > 0` is a data quality issue independent of the revenue calculation.

# COMMAND ----------

tips_by_source = spark.sql(f"""
    SELECT
        source,
        COUNT(*)                                          AS order_count,
        SUM(CASE WHEN tip_amount > 0 THEN 1 ELSE 0 END)  AS orders_with_tips,
        ROUND(SUM(tip_amount), 2)                         AS total_tip_amount,
        ROUND(AVG(CASE WHEN tip_amount > 0 THEN tip_amount END), 2) AS avg_tip_when_present
    FROM {SILVER}
    WHERE voided = false
      AND approval_status = 'APPROVED'
    GROUP BY source
    ORDER BY total_tip_amount DESC
""")
display(tips_by_source)

unexpected_tip_sources = spark.sql(f"""
    SELECT source, COUNT(*) AS order_count, ROUND(SUM(tip_amount), 2) AS total_tips
    FROM {SILVER}
    WHERE voided = false
      AND approval_status = 'APPROVED'
      AND tip_amount > 0
      AND source NOT IN ('Online', 'Toast Pickup App')
    GROUP BY source
""")

count = unexpected_tip_sources.count()
if count > 0:
    print(f"⚠️  FAIL — {count} non-online source(s) carry tips:")
    display(unexpected_tip_sources)
else:
    print("✅  PASS — Tips only appear on Online / Toast Pickup App orders")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 2 — Is `tip_amount` Embedded in `total_amount`?
# MAGIC
# MAGIC Test: `total_amount ≈ gross_amount - total_discount_amount + tax_amount + tip_amount`
# MAGIC
# MAGIC If the residual is ~$0 on orders with tips, tips ARE baked into `total_amount`,
# MAGIC which means any Gold metric derived from `total_amount` will include tips.

# COMMAND ----------

tip_embed_check = spark.sql(f"""
    SELECT
        ROUND(AVG(ABS(
            total_amount
            - (gross_amount - total_discount_amount + tax_amount + tip_amount)
        )), 4)  AS avg_residual_with_tip_included,
        ROUND(AVG(ABS(
            total_amount
            - (gross_amount - total_discount_amount + tax_amount)
        )), 4)  AS avg_residual_without_tip,
        COUNT(*) AS orders_sampled
    FROM {SILVER}
    WHERE tip_amount > 0
      AND voided = false
      AND approval_status = 'APPROVED'
""")
display(tip_embed_check)

row = tip_embed_check.collect()[0]
if row['avg_residual_with_tip_included'] < 0.01:
    print("❌  tip_amount IS embedded in total_amount — tips will flow into net_revenue")
else:
    print("✅  tip_amount is NOT in total_amount — Gold net_revenue should be clean")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 3 — Gold Net Revenue vs. Correct Net Revenue (tip-excluded)
# MAGIC
# MAGIC Reconstruct what `net_revenue` *should* be from Silver, then compare to Gold.
# MAGIC
# MAGIC **Correct net revenue** = `SUM(total_amount - tip_amount)` per business date

# COMMAND ----------

comparison = spark.sql(f"""
    WITH silver_agg AS (
        SELECT
            business_date,
            ROUND(SUM(total_amount), 2)              AS silver_net_revenue_with_tips,
            ROUND(SUM(total_amount - tip_amount), 2) AS silver_net_revenue_correct,
            ROUND(SUM(tip_amount), 2)                AS total_tips
        FROM {SILVER}
        WHERE voided = false
          AND approval_status = 'APPROVED'
        GROUP BY business_date
    ),
    gold_agg AS (
        SELECT
            business_date,
            ROUND(net_revenue, 2) AS gold_net_revenue
        FROM {GOLD}
        WHERE record_type = 'actual'
    )
    SELECT
        g.business_date,
        g.gold_net_revenue,
        s.silver_net_revenue_with_tips,
        s.silver_net_revenue_correct,
        s.total_tips,
        ROUND(g.gold_net_revenue - s.silver_net_revenue_correct, 2) AS discrepancy,
        ROUND(
            (g.gold_net_revenue - s.silver_net_revenue_correct)
            / NULLIF(s.silver_net_revenue_correct, 0) * 100, 3
        ) AS discrepancy_pct
    FROM gold_agg g
    JOIN silver_agg s USING (business_date)
    ORDER BY g.business_date DESC
""")
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 4 — Cumulative & Monthly Discrepancy Magnitude

# COMMAND ----------

cumulative = spark.sql(f"""
    WITH silver_agg AS (
        SELECT
            business_date,
            SUM(total_amount)              AS silver_net_with_tips,
            SUM(total_amount - tip_amount) AS silver_net_correct,
            SUM(tip_amount)                AS total_tips
        FROM {SILVER}
        WHERE voided = false
          AND approval_status = 'APPROVED'
        GROUP BY business_date
    ),
    gold_agg AS (
        SELECT business_date, net_revenue AS gold_net_revenue
        FROM {GOLD}
        WHERE record_type = 'actual'
    )
    SELECT
        ROUND(SUM(g.gold_net_revenue), 2)                         AS total_gold_net_revenue,
        ROUND(SUM(s.silver_net_correct), 2)                       AS total_correct_net_revenue,
        ROUND(SUM(g.gold_net_revenue - s.silver_net_correct), 2)  AS total_discrepancy,
        ROUND(
            SUM(g.gold_net_revenue - s.silver_net_correct)
            / NULLIF(SUM(s.silver_net_correct), 0) * 100, 3
        )                                                         AS discrepancy_pct,
        ROUND(SUM(s.total_tips), 2)                               AS total_tips_in_period
    FROM gold_agg g
    JOIN silver_agg s USING (business_date)
""")
display(cumulative)

# COMMAND ----------

monthly = spark.sql(f"""
    WITH silver_agg AS (
        SELECT
            DATE_TRUNC('month', business_date) AS month,
            SUM(total_amount)              AS silver_net_with_tips,
            SUM(total_amount - tip_amount) AS silver_net_correct,
            SUM(tip_amount)                AS total_tips
        FROM {SILVER}
        WHERE voided = false
          AND approval_status = 'APPROVED'
        GROUP BY DATE_TRUNC('month', business_date)
    ),
    gold_agg AS (
        SELECT
            DATE_TRUNC('month', business_date) AS month,
            SUM(net_revenue) AS gold_net_revenue
        FROM {GOLD}
        WHERE record_type = 'actual'
        GROUP BY DATE_TRUNC('month', business_date)
    )
    SELECT
        g.month,
        ROUND(g.gold_net_revenue, 2)                              AS gold_net_revenue,
        ROUND(s.silver_net_correct, 2)                            AS correct_net_revenue,
        ROUND(g.gold_net_revenue - s.silver_net_correct, 2)       AS tip_overstatement,
        ROUND(
            (g.gold_net_revenue - s.silver_net_correct)
            / NULLIF(s.silver_net_correct, 0) * 100, 3
        )                                                         AS overstatement_pct,
        ROUND(s.total_tips, 2)                                    AS total_tips
    FROM gold_agg g
    JOIN silver_agg s USING (month)
    ORDER BY g.month DESC
""")
display(monthly)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check 5 — Online Order Tip Detail
# MAGIC
# MAGIC Confirms which online order sub-sources carry tips and validates
# MAGIC that no in-store tips are leaking into the discrepancy.

# COMMAND ----------

online_tip_detail = spark.sql(f"""
    SELECT
        source,
        DATE_TRUNC('month', business_date)          AS month,
        COUNT(*)                                    AS order_count,
        SUM(CASE WHEN tip_amount > 0 THEN 1 END)    AS orders_with_tips,
        ROUND(SUM(tip_amount), 2)                   AS total_tips,
        ROUND(AVG(CASE WHEN tip_amount > 0 THEN tip_amount END), 2) AS avg_tip
    FROM {SILVER}
    WHERE voided = false
      AND approval_status = 'APPROVED'
      AND tip_amount > 0
    GROUP BY source, DATE_TRUNC('month', business_date)
    ORDER BY month DESC, total_tips DESC
""")
display(online_tip_detail)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verdict & Recommended Fix
# MAGIC
# MAGIC | Check | Expected | Result |
# MAGIC |---|---|---|
# MAGIC | Tips only on Online orders | ✅ Yes | See Check 1 |
# MAGIC | `tip_amount` embedded in `total_amount` | Tips included | See Check 2 |
# MAGIC | Gold `net_revenue` excludes tips | ✅ Should be 0 discrepancy | See Check 3/4 |
# MAGIC
# MAGIC ### If discrepancy > $0:
# MAGIC
# MAGIC Tips are flowing into Gold `net_revenue` because the Gold pipeline sums `total_amount`
# MAGIC without stripping `tip_amount` first.
# MAGIC
# MAGIC **Fix — update the Gold pipeline aggregation:**
# MAGIC ```sql
# MAGIC -- Change:
# MAGIC SUM(total_amount)              AS net_revenue   -- ❌ includes tips
# MAGIC
# MAGIC -- To:
# MAGIC SUM(total_amount - tip_amount) AS net_revenue   -- ✅ correct
# MAGIC ```
# MAGIC
# MAGIC **Impact:** Net revenue is overstated by the cumulative tip total shown in Check 4.
# MAGIC Forecasting models trained on this data will have a small upward bias on Online-heavy days.
