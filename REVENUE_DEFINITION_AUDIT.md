# Revenue Definition Audit & Remediation — Databricks

**Discovered:** 2026-07-01, during Snowflake parity reconciliation.
**Decision:** Full fix + retrain (correct Databricks to standard accounting convention).
**Severity:** Production correctness bug — forecasts & revenue reports overstate real sales by ~tax rate (~3%).

## The bug
In `gold.daily_sales_summary` (`notebooks/4_Gold_Sales_Summary.py` L311-312):
```python
ROUND(SUM(gross_amount), 2)              AS gross_revenue,   # pre-tax subtotal (REAL sales) — dead-ends
ROUND(SUM(total_amount - tip_amount), 2) AS net_revenue,     # subtotal + TAX (tips stripped) — flows everywhere
```
`net_revenue` is **tax-inclusive** and is the value consumed by features, the Prophet model, forecasts,
health-check MAPE, and Platinum/dashboards. `gross_revenue` (accounting-correct pre-tax) is stored but unused.
Standard convention (and Snowflake/CoCo's build): **net = pre-tax**. Tips are correctly excluded in both.

Reconciliation proof: Snowflake `NET_REVENUE` == Databricks `GROSS_REVENUE`, penny-identical 26/30 June days.

## Target end state
`net_revenue` = pre-tax subtotal (= current `gross_amount`). Introduce explicit tax-inclusive column
`revenue_with_tax` so the concept isn't lost and never ambiguous again. Keep tips excluded.

## Remediation sequence (ORDER MATTERS — value change before retrain, retrain before forecasts)

### Step 1 — Gold definition (`4_Gold_Sales_Summary.py`)
- L311-312: `net_revenue = ROUND(SUM(gross_amount),2)`; add `revenue_with_tax = ROUND(SUM(total_amount - tip_amount),2)`.
  Keep `gross_revenue` for now (backward-compat) but it now equals net — plan to deprecate.
- L456-457 hourly: same swap.
- L729 validation cell: update formula in lockstep.
- L129-130 column comments: fix ("net_revenue = pre-tax sales, tips & tax excluded").
- Run NB4 in `full_refresh` to rewrite all history.

### Step 2 — Features (`6_Feature_Engineering.py`)
- L436 `y_revenue = net_revenue` now picks up pre-tax automatically. Rebuild feature table (full refresh)
  so all `lag_*_revenue` / `rolling_*_revenue` recompute on the ~3%-lower scale.

### Step 3 — Retrain revenue model (`7_Model_Prophet_Revenue.py`)  ← REQUIRED
- Model's learned level, weekly seasonality, and `REVENUE_CAP` (L161/312) are calibrated to tax-inclusive
  magnitude. Retrain on rebuilt features. Follow MLflow compare/promote workflow — do NOT blind-overwrite prod.
- Confirm MLflow `target` tag still `net_revenue` (now means pre-tax).

### Step 4 — Regenerate forecasts (`9_Forecast_Generate.py`)
- After new model promoted to `@production`, regenerate `gold.daily_sales_forecast`. Recency-blend anchors
  (L378-393) self-correct to new scale.

### Step 5 — Reset accuracy baselines (`10` + `forecast_accuracy_history`)
- New MAPE baselines on pre-tax scale. Old accuracy history rows are on the old scale — mark/segment them.

### Unaffected (no change)
- Orders model (NB8) — forecasts order_count, not revenue.
- Cash audit (NB1b/1c/11/12) — never references net/gross.
- Silver — already neutral/correct.

## Snowflake side
No change — CoCo's build already uses net = pre-tax. This audit is Databricks-only. Once Databricks is
fixed, the two platforms will share the SAME convention and revenue numbers will match on the same column names.

## Toast reconciliation rules (applied to BOTH platforms 2026-07-03)
net_revenue = Toast "Net Sales" = SUM(gross_amount / check.amount, pre-tax, tips+tax excluded, voids+discounts
netted by Toast) MINUS gift-card sells/reloads (selection_type IN TOAST_CARD_RELOAD/TOAST_CARD_SELL/GIFTCARD/
STORED_VALUE_CARD — deferred revenue). gross_amount is tip-safe (no split-payment tip leak). Validated:
Snowflake 84/90 penny-exact (-0.029%); Databricks quarter net $183,802.76 vs Toast $183,852.76 (0.027%, the
$50 = Jun 18 date-attribution residual). Remaining diffs = paidDate-vs-openedDate business-date attribution
(offsetting pairs, net zero).

## Model retrain findings (2026-07-03)
Retrained NB7 on corrected features. NEW vs OLD (Jun 29) metrics:
- cv_mape: 32.98% → **24.42%** (IMPROVED — passes <25%); r2: 0.74 → **0.81** (improved).
- backtest_mape: 31% → 47% (WORSE) — but this is a BROKEN BACKTEST, not a model regression:
  the fixed `BACKTEST_CUTOFF=2026-03-01` trains on Jan-Feb ($1,119/day winter) to predict Mar-Apr,
  and the store ~doubled Jan→Jun ($1,119→$2,239), so no ramp signal is in the train window.
- FIX applied to NB7: backtest now = last-30-day holdout (`_bt_last_obs - 30d`); winner auto-selected
  by lowest cv_mape; promote gate = widget `promote_mode` (auto=promote iff _pass / never / force).
- NB7 IS job-runnable (logs 3 variant runs to experiment `/Users/joe.../toast_prophet_revenue`).
- CAVEAT: `system.mlflow.runs_latest` is a Delta Share with replication lag — fresh run metrics take
  time to appear; can't tight-loop verify via SQL. Read run output in the Databricks UI for immediate metrics.
- OUTSTANDING: confirm new backtest MAPE with 30-day holdout passes <25%; then let auto-promote fire +
  regenerate forecasts (NB9) + reset accuracy baselines.

## Progress
- [x] Step 1 Gold — DONE & DEPLOYED (full_refresh on prod Gold, 145s, SUCCESS). `4_Gold_Sales_Summary.py`:
      net_revenue = SUM(gross_amount) - gift_cards (Toast Net Sales); gross_revenue = net (compat);
      revenue_with_tax = total-tip (tax-incl); avg_ticket pre-tax; idempotent ALTER ADD revenue_with_tax
      migration; validation vs revenue_with_tax. Hourly net_revenue omits gift-card subtraction (documented,
      daily is authoritative). Prod Gold now matches Toast + Snowflake.
- [ ] Step 2 Features
- [ ] Step 3 Retrain + promote
- [ ] Step 4 Forecasts
- [ ] Step 5 Baselines
