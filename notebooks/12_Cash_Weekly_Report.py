# Databricks notebook source
# MAGIC %md
# MAGIC # Cash Audit — Weekly Email Report
# MAGIC
# MAGIC Builds a weekly cash reconciliation summary and sends it to joe@threesistersprovisions.com.
# MAGIC
# MAGIC **Covers:** the 7-day period ending yesterday (Sunday through Saturday, sent Monday AM).
# MAGIC
# MAGIC **Report sections:**
# MAGIC 1. Week summary — declared vs deposited, variance, match rate
# MAGIC 2. Running total since inception — cumulative delta
# MAGIC 3. Unmatched closes — dates where no deposit was found (action required)
# MAGIC 4. Needs review — deposit candidates not yet confirmed in Monarch
# MAGIC 5. Matched with variance — matched but amount didn't tie exactly
# MAGIC
# MAGIC **Sent via:** Databricks built-in email (dbutils.notebook.run is not used —
# MAGIC we use the Databricks Jobs email notification, or direct SMTP via secrets).

# COMMAND ----------

import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from pyspark.sql import functions as F

# COMMAND ----------

CATALOG       = "3sp_analytics_workspace"
MATCHES_TABLE = f"{CATALOG}.gold.cash_audit_matches"
CLOSES_TABLE  = f"{CATALOG}.gold.cash_register_closes"

RECIPIENT     = "joe@threesistersprovisions.com"
SUBJECT       = "Weekly Cash Audit Report — Three Sisters Provisions"

# Report window: last 7 days ending yesterday
yesterday  = datetime.date.today() - datetime.timedelta(days=1)
week_start = yesterday - datetime.timedelta(days=6)

# COMMAND ----------

# ── WEEK SUMMARY ─────────────────────────────────────────────────────────────

week_summary = spark.sql(f"""
    SELECT
        COUNT(*)                                                    AS total_closes,
        ROUND(SUM(declared_amount), 2)                             AS total_declared,
        ROUND(SUM(CASE WHEN match_status IN ('MATCHED','PARTIAL')
              THEN bank_deposit_amount ELSE 0 END), 2)             AS total_deposited,
        ROUND(SUM(CASE WHEN match_status IN ('MATCHED','PARTIAL')
              THEN variance_dollars ELSE 0 END), 2)                AS week_variance,
        SUM(CASE WHEN match_status = 'MATCHED'      THEN 1 ELSE 0 END) AS matched,
        SUM(CASE WHEN match_status = 'PARTIAL'      THEN 1 ELSE 0 END) AS partial,
        SUM(CASE WHEN match_status = 'UNMATCHED'    THEN 1 ELSE 0 END) AS unmatched,
        SUM(CASE WHEN match_status = 'NEEDS_REVIEW' THEN 1 ELSE 0 END) AS needs_review,
        SUM(CASE WHEN match_status = 'NO_CASH'      THEN 1 ELSE 0 END) AS no_cash
    FROM {MATCHES_TABLE}
    WHERE close_date BETWEEN DATE('{week_start}') AND DATE('{yesterday}')
      AND match_status != 'NO_CASH'
""").collect()[0]

# ── INCEPTION-TO-DATE RUNNING TOTAL ──────────────────────────────────────────

itd = spark.sql(f"""
    SELECT
        MIN(close_date)                                            AS first_date,
        COUNT(DISTINCT close_date)                                 AS total_close_dates,
        ROUND(SUM(declared_amount), 2)                            AS total_declared_itd,
        ROUND(SUM(CASE WHEN match_status IN ('MATCHED','PARTIAL')
              THEN bank_deposit_amount ELSE 0 END), 2)            AS total_deposited_itd,
        ROUND(SUM(CASE WHEN match_status IN ('MATCHED','PARTIAL')
              THEN variance_dollars ELSE 0 END), 2)               AS cumulative_variance,
        ROUND(SUM(CASE WHEN match_status = 'UNMATCHED'
              THEN declared_amount ELSE 0 END), 2)                AS total_unmatched_declared
    FROM {MATCHES_TABLE}
    WHERE match_status != 'NO_CASH'
""").collect()[0]

# ── UNMATCHED CLOSES (action required) ───────────────────────────────────────

unmatched = spark.sql(f"""
    SELECT
        close_date,
        ROUND(declared_amount, 2) AS declared_amount,
        notes,
        DATEDIFF(current_date(), close_date) AS days_ago
    FROM {MATCHES_TABLE}
    WHERE match_status = 'UNMATCHED'
    ORDER BY close_date DESC
    LIMIT 30
""").collect()

# ── NEEDS REVIEW ──────────────────────────────────────────────────────────────

needs_review = spark.sql(f"""
    SELECT
        close_date,
        ROUND(declared_amount, 2)    AS declared_amount,
        ROUND(bank_deposit_amount, 2) AS deposit_amount,
        ROUND(variance_dollars, 2)   AS variance,
        bank_deposit_date,
        account_name,
        original_statement,
        bank_deposit_id              AS monarch_id
    FROM {MATCHES_TABLE}
    WHERE match_status = 'NEEDS_REVIEW'
    ORDER BY close_date DESC
    LIMIT 20
""").collect()

# ── MATCHED WITH VARIANCE ─────────────────────────────────────────────────────

partial = spark.sql(f"""
    SELECT
        close_date,
        ROUND(declared_amount, 2)    AS declared_amount,
        ROUND(bank_deposit_amount, 2) AS deposit_amount,
        ROUND(variance_dollars, 2)   AS variance,
        bank_deposit_date,
        account_name
    FROM {MATCHES_TABLE}
    WHERE match_status = 'PARTIAL'
    ORDER BY ABS(variance_dollars) DESC
    LIMIT 10
""").collect()

# COMMAND ----------

# ── BUILD EMAIL ───────────────────────────────────────────────────────────────

def fmt_dollar(val):
    if val is None:
        return "—"
    sign = "-" if val < 0 else ""
    return f"{sign}${abs(val):,.2f}"


def variance_label(val):
    """Positive variance = declared more than deposited (cash went missing)."""
    if val is None:
        return "—"
    if val > 0:
        return f"+{fmt_dollar(val)} (short)"
    elif val < 0:
        return f"{fmt_dollar(val)} (over)"
    return "$0.00"


lines = []
lines.append(f"WEEKLY CASH AUDIT — Three Sisters Provisions")
lines.append(f"Period: {week_start.strftime('%b %d')} – {yesterday.strftime('%b %d, %Y')}")
lines.append("=" * 60)

lines.append("")
lines.append("THIS WEEK")
lines.append(f"  Business dates with cash:  {week_summary['total_closes']}")
lines.append(f"  Total declared:            {fmt_dollar(week_summary['total_declared'])}")
lines.append(f"  Total deposited (matched): {fmt_dollar(week_summary['total_deposited'])}")
lines.append(f"  Week variance:             {variance_label(week_summary['week_variance'])}")
lines.append(f"  Matched:       {week_summary['matched']}")
lines.append(f"  Partial match: {week_summary['partial']}")
lines.append(f"  Unmatched:     {week_summary['unmatched']}")
lines.append(f"  Needs review:  {week_summary['needs_review']}")

lines.append("")
lines.append("RUNNING TOTAL (inception to date)")
lines.append(f"  Since:                     {itd['first_date']}")
lines.append(f"  Total declared (all time): {fmt_dollar(itd['total_declared_itd'])}")
lines.append(f"  Total deposited (matched): {fmt_dollar(itd['total_deposited_itd'])}")
lines.append(f"  Cumulative variance:       {variance_label(itd['cumulative_variance'])}")
lines.append(f"  Total unmatched declared:  {fmt_dollar(itd['total_unmatched_declared'])}")

if unmatched:
    lines.append("")
    lines.append(f"UNMATCHED CLOSES — {len(unmatched)} dates with no deposit found")
    lines.append("  (Cash was declared removed from register but no bank deposit matched)")
    lines.append("")
    lines.append(f"  {'Date':<12} {'Declared':>10}  {'Days Ago':>8}")
    lines.append(f"  {'-'*12} {'-'*10}  {'-'*8}")
    for r in unmatched:
        lines.append(f"  {str(r['close_date']):<12} {fmt_dollar(r['declared_amount']):>10}  {r['days_ago']:>8}")
else:
    lines.append("")
    lines.append("UNMATCHED CLOSES — none. All declared cash has a matching deposit.")

if needs_review:
    lines.append("")
    lines.append(f"NEEDS REVIEW — {len(needs_review)} deposits not yet confirmed in Monarch")
    lines.append("  (A deposit was found near the right amount, but it hasn't been")
    lines.append("   categorized as 'Cash Deposit' in Monarch yet.)")
    lines.append("  To confirm: open Monarch, find the transaction, set category = Cash Deposit.")
    lines.append("  To dismiss: set any other category and it will stop appearing here.")
    lines.append("")
    lines.append(f"  {'Close':<12} {'Declared':>10} {'Deposit':>10} {'Variance':>10}  {'Account':<28}  Monarch ID")
    lines.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}  {'-'*28}  {'-'*30}")
    for r in needs_review:
        lines.append(
            f"  {str(r['close_date']):<12} "
            f"{fmt_dollar(r['declared_amount']):>10} "
            f"{fmt_dollar(r['deposit_amount']):>10} "
            f"{variance_label(r['variance']):>10}  "
            f"{(r['account_name'] or '')[:28]:<28}  "
            f"{r['monarch_id']}"
        )

if partial:
    lines.append("")
    lines.append(f"PARTIAL MATCHES — {len(partial)} closes matched but amounts differ")
    lines.append("")
    lines.append(f"  {'Close':<12} {'Declared':>10} {'Deposited':>10} {'Variance':>10}  Account")
    lines.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}  {'-'*28}")
    for r in partial:
        lines.append(
            f"  {str(r['close_date']):<12} "
            f"{fmt_dollar(r['declared_amount']):>10} "
            f"{fmt_dollar(r['deposit_amount']):>10} "
            f"{variance_label(r['variance']):>10}  "
            f"{(r['account_name'] or '')[:28]}"
        )

lines.append("")
lines.append("=" * 60)
lines.append("View full audit: 3sp_analytics_workspace.gold.cash_audit_matches")
lines.append("Monarch: https://app.monarchmoney.com/transactions")

body = "\n".join(lines)
print(body)

# COMMAND ----------

# ── SEND EMAIL ────────────────────────────────────────────────────────────────

try:
    smtp_user     = dbutils.secrets.get(scope="email_api", key="smtp_user")
    smtp_password = dbutils.secrets.get(scope="email_api", key="smtp_password")
    smtp_host     = dbutils.secrets.get(scope="email_api", key="smtp_host")
    smtp_port     = int(dbutils.secrets.get(scope="email_api", key="smtp_port"))

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{SUBJECT} ({yesterday.strftime('%b %d')})"
    msg["From"]    = smtp_user
    msg["To"]      = RECIPIENT

    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, RECIPIENT, msg.as_string())

    print(f"\n✓ Report sent to {RECIPIENT}")

except Exception as e:
    print(f"\n✗ Email send failed: {e}")
    print("Report body printed above. Configure SMTP secrets to enable sending.")
    print("Required: scope='email_api', keys: smtp_user, smtp_password, smtp_host, smtp_port")
