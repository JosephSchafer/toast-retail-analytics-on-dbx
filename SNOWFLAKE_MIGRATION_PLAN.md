# Three Sisters Provisions - Databricks → Snowflake Replication Plan

**Goal:** Replicate the *business purpose* of the Databricks pipeline on Snowflake, using Snowflake-native
best practices, at a monthly spend at or below the ~$50/mo Databricks baseline. Run both platforms in
parallel and decide on cutover later.

**Account:** `ATQGHSA-MQ92949` · DB `THREE_SISTERS_ANALYTICS` · user `VIRIDIAJOE` · WH `TSP_PIPELINE_WH` (X-Small)

---

## 1. Guiding architecture decisions

| Databricks pattern | Snowflake-native replacement | Why |
|---|---|---|
| Python notebooks calling Toast/Monarch/Open-Meteo REST | **Stored procedures + External Access Integration + Secrets** (`external_access_integrations`) | No cluster; secrets managed in-DB; already prototyped (`INGEST_TOAST_*`, `INGEST_WEATHER_*`) |
| Serverless Spark SQL transforms (NB2/4/5/11) | **SQL stored procs / views** on `TSP_PIPELINE_WH` | Pure SQL; identical logic |
| Prophet + MLflow (NB7/8/9) | **`SNOWFLAKE.ML.FORECAST`** built-in model | No Python/MLflow; already prototyped (`REFRESH_FORECASTS`) |
| Feature engineering pandas lags/rolling (NB6) | **SQL window functions** (`LAG`, `AVG OVER`) | Snowflake windowing covers all NB6 lags/rolling; cyclical encoding = `SIN/COS` in SQL |
| Databricks Job (task DAG, cron) | **Snowflake Task DAG** (serverless or WH tasks) | Native scheduling; per-task warehouse; auto-suspend |
| Email notebooks | **`SYSTEM$SEND_EMAIL`** procs (`SEND_*` already stubbed) | Native; requires verified notification integration |
| Delta MERGE | `MERGE INTO` on standard Snowflake tables | Direct equivalent |

**Cost model:** X-Small WH = 1 credit/hr, billed per-second (60s min per resume), auto-suspend 60s.
Observed `TSP_PIPELINE_WH` usage ≈ 0.1–0.2 credits/day. A full daily DAG that runs a few minutes
should stay ≈ 0.2–0.4 credits/day ≈ **$12–24/mo** at ~$2/credit + `SNOWFLAKE.ML.FORECAST` serverless
credits (small). Comfortably under $50. Guardrail: a **Resource Monitor** capping monthly credits.

---

## 2. Current state (what CoCo already built - reuse, don't rebuild)

**Schemas:** BRONZE, SILVER_SALES, GOLD, PLATINUM, REFERENCE (all described) ✅
**Ingestion procs (External Access + Secrets):** `INGEST_TOAST_ORDERS`, `INGEST_TOAST_INVENTORY`,
`INGEST_WEATHER_ARCHIVE`, `INGEST_WEATHER_FORECAST`, diagnostics `DIAG_TOAST_*` ✅
**Transform procs:** `REFRESH_TRANSFORMS`, `REFRESH_INVENTORY_VELOCITY`, `REFRESH_FORECASTS` ✅
**Forecasting:** `SNOWFLAKE.ML.FORECAST` revenue + order models; `MODEL_RUNS` history; training views
`V_FORECAST_TRAINING`, `V_ORDER_FORECAST_TRAINING` ✅
**Email stubs:** `SEND_DAILY_SALES_REPORT`, `SEND_WEEKLY_COST_REPORT`, `SEND_DAILY_STOCKOUT_REPORT`,
`SEND_DAILY_AGED_INVENTORY_REPORT`, `SEND_WEEKLY_FORECAST_ACCURACY_REPORT` ✅
**Warehouse:** `TSP_PIPELINE_WH` X-Small, auto-suspend 60s ✅

**Gaps / defects:**
- ❌ **No Tasks** - nothing scheduled; pipeline is manual (root cause of the stale data).
- ⚠️ `GOLD.DAILY_SALES_BY_CATEGORY` wrong grain - only `Uncategorized` (no item-catalog join); 262 rows
  (day-level) vs Databricks 5,459 (day×category). Needs REFERENCE.ITEM_CATALOG + category join.
- ⚠️ `GOLD.DAILY_SALES_FORECAST` holds only the latest run (31 rows) - no per-run history, so no
  accuracy-by-horizon analysis (Databricks keeps every run keyed by created-at + horizon).
- ❌ No cash audit (Toast cash + Monarch reconciliation).
- ❌ No DQ schema (par levels, missing cost/supplier/barcode, category drift).
- ❌ No `HOURLY_SALES_SUMMARY`.
- ❌ `REFERENCE.ITEM_CATALOG`, `REFERENCE.STORE_EVENTS` not populated.
- ⚠️ Forecast currently trains on bare net-revenue series only - no weather/event regressors that
  Databricks Prophet used. Acceptable v1; note as a known accuracy gap.

---

## 3. Target end-state object map (parity)

```
BRONZE
  toast_orders_raw            (have)      toast_cash_entries_raw      (NEW)
  toast_inventory_snapshot    (have)      toast_cash_deposits_raw     (NEW)
  toast_menu_items_raw        (have)      monarch_bank_deposits_raw   (NEW)
  toast_purchase_orders_raw   (have)      ingestion_watermark         (have)
  weather_hourly              (have)
SILVER_SALES
  orders_silver               (have)      order_items_silver          (have)
GOLD
  daily_sales_summary         (have)      hourly_sales_summary        (NEW)
  daily_sales_by_category     (FIX grain) forecast_features           (NEW - for regressors)
  daily_sales_forecast        (FIX hist)  forecast_accuracy_history   (NEW)
  inventory_velocity          (have)      cash_register_closes        (NEW)
  model_runs                  (have)      cash_audit_matches          (NEW)
REFERENCE
  item_catalog                (populate)  store_events                (populate)
DQ  (NEW schema)
  par_level_suggestions, missing_cost, missing_supplier, missing_barcode,
  category_drift, suggested_discontinued, nomenclature_fix, inverted_margin
PLATINUM
  daily_sales_combined        (view - fix forecast-history dedup)
```

---

## 4. Phased execution plan

### Phase 0 - Foundation & guardrails (½ day)
- [ ] Create **Resource Monitor** on the account/WH: monthly credit quota (e.g. 25 credits ≈ $50) with
      NOTIFY at 75%, SUSPEND at 100%. This is the hard spend guarantee.
- [ ] Confirm `TSP_EXTERNAL_ACCESS` integration + secrets (Toast id/secret/guid) are valid; add a
      **Monarch** secret + external-access rule for cash audit.
- [ ] Set `TSP_PIPELINE_WH` auto-suspend to 60s (already), STATEMENT_TIMEOUT sane default.
- [ ] Create **email notification integration** verified to joe@threesistersprovisions.com (needed by SEND_* procs).
- [ ] Snapshot current Databricks Gold row counts/values as the **parity baseline** (this file's audit table).

### Phase 1 - Fix & complete Sales + Forecasting (core, mostly done) (1–2 days)
- [x] **DONE 2026-07-01:** Populate `REFERENCE.ITEM_CATALOG` (1,091 items, matches DBX exactly). Built
      reusable Snowflake-native loader mirroring NB3: stage `REFERENCE.TOAST_EXPORTS` + file format
      `FF_TOAST_CSV` + proc `REFERENCE.LOAD_ITEM_CATALOG('file.csv')` (repo: `snowflake/procedures/load_item_catalog.sql`).
      Workflow: PUT Toast retail-items CSV to stage → CALL loader (full-replace).
- [x] **DONE 2026-07-01:** Fixed `DAILY_SALES_BY_CATEGORY` grain via ITEM_CATALOG join (item_guid=item_id):
      5,455 rows vs DBX 5,459 (99.9%); top-category revenues within 0.3% of DBX. 41 real categories.
- [ ] Populate `REFERENCE.STORE_EVENTS` (closures/events) - needed for accurate closed-day handling & forecast.
- [ ] **Fix `DAILY_SALES_BY_CATEGORY`**: rewrite `REFRESH_TRANSFORMS` to join items→item_catalog on
      category, grain = day × sales_category. Validate rowcount trends toward Databricks' 5,459.
- [x] **DONE 2026-07-01:** Added `HOURLY_SALES_SUMMARY` (3,754 rows/262 days, net $387,202 reconciles).
      Repo: `snowflake/transforms/hourly_sales_summary.sql`.
- [ ] Ensure `DAILY_SALES_SUMMARY` emits **explicit zero-revenue closed-day rows** via a date spine
      (Databricks does; Snowflake proc currently does NOT - fold into forecast-features/closed-day work).
- [x] **DONE 2026-07-01:** Fixed `DAILY_SALES_FORECAST` history - added `FORECAST_CREATED_AT` +
      `FORECAST_HORIZON_DAYS`; `REFRESH_FORECASTS` now deletes only the same-version future rows (keeps
      prior runs for accuracy-by-horizon). Created `FORECAST_ACCURACY_HISTORY` table (scoring proc TBD).
      Repo: `snowflake/transforms/forecast_and_platinum.sql`.
- [ ] Upgrade `SNOWFLAKE.ML.FORECAST` calls to pass **exogenous regressors** (weather, bread-day, events)
      via the multi-series/exog input, matching Prophet regressors. (v1 shipped without; KNOWN GAP.)
- [x] **DONE 2026-07-01:** Fixed `PLATINUM.DAILY_SALES_COMBINED` - now UNIONs actuals + latest-run
      forecasts (future dates only, deduped by MAX(forecast_created_at)). Verified 262 actual + 30 forecast rows.

### Phase 2 - Cash audit - TOAST SIDE ONLY (Monarch backlogged) (1 day)
> **Monarch bank-deposit ingestion is BACKLOGGED (2026-07-01).** Monarch's MCP/data-portability is
> paused pending a partner data-portability issue, and it's a one-off worth revisiting later. Until then
> reconciliation can't complete the bank side, so we build only the Toast-side cash capture now and
> defer the match/report.
- [ ] **Toast cash ingestion**: extend ingest procs for `/cashmgmt/entries` + `/deposits` → bronze cash tables.
- [ ] Build `cash_register_closes` (declared closes/deposits aggregated per business_date) - the Toast side.
- [ ] Stub `cash_audit_matches` schema; leave reconciliation + `SEND_WEEKLY_*` cash report for when Monarch returns.

### BACKLOG (revisit later)
- [ ] **Monarch bank-deposit ingestion** (NB1c) - GraphQL pull, 14-day lookback, candidate scoring.
      Blocked on Monarch data-portability pause. When unblocked: port to external-access stored proc,
      then finish NB11 reconciliation (7-day window / $5 tolerance) + Monday cash email.

### Phase 3 - Inventory / DQ - DONE 2026-07-01
- [x] Created `DQ` schema + `DQ.REFRESH_DQ()` proc building all 8 tables (repo: `snowflake/procedures/refresh_dq.sql`).
- [x] `PAR_LEVEL_SUGGESTIONS` = 1,068 (EXACT match DBX); `inverted_margin` 4 (exact); `category_drift` 138 (~135).
- [ ] KNOWN VARIANCES (advisory tables, acceptable but note): missing_cost 28 vs DBX 2 (DBX supplements cost
      from RECEIVE records - deferred join); missing_supplier 402 vs 318 (NULL vs '' handling);
      suggested_discontinued 386 vs 146 + missing_barcode 105 vs 25 + nomenclature_fix 171 vs 227
      (original DBX logic NOT version-controlled - filters inferred from schemas; refine if exact parity needed).
- [ ] Wire `SEND_DAILY_STOCKOUT_REPORT` / `SEND_DAILY_AGED_INVENTORY_REPORT` (in Phase 4 email work).

### Phase 4 - Orchestration (Tasks) - DONE 2026-07-01
> **KEY FINDING:** CoCo had ALREADY built the full daily Task DAG (rooted at `BRONZE.DAILY_INGEST_WEATHER`,
> cron `0 5 * * * America/New_York`, all `started`): ingest weather→forecast→orders→inventory →
> REFRESH_TRANSFORMS → REFRESH_INVENTORY_VELOCITY → REFRESH_FORECASTS → SEND_{SALES,STOCKOUT,AGED} emails.
> The "stale data" in the original audit was just the account being 1 day old, not a broken pipeline.
> Because my Phase 1/3 fixes live in the PROCEDURES the tasks call, they wired in automatically.
- [x] Patched `REFRESH_TRANSFORMS` (category fix + hourly) - DAG now produces correct tables.
- [x] Spliced in `BRONZE.DAILY_REFRESH_DQ` (calls `DQ.REFRESH_DQ`) after velocity.
- [x] Added `GOLD.PIPELINE_HEALTH_CHECK()` proc (freshness + forecast-horizon, emails on issue). NOT yet
      wired as a task in CoCo's DAG - TODO if desired (CoCo's DAG has no health-check node).
- [x] Triggered full DAG manually: 11/11 tasks SUCCEEDED, data now fresh to 2026-07-01, all fixes held.
- Dropped my duplicate `GOLD.T_*` DAG (redundant with CoCo's).

#### (original Phase 4 checklist - superseded by the above)
- [ ] Build the **Task DAG** on `TSP_PIPELINE_WH` mirroring the Databricks task order:
      ```
      t_ingest_toast_orders  ─┐
      t_ingest_weather        ├─► t_transform_silver ─► t_gold_sales ─► t_features ─► t_forecast ─► t_health_check
      t_ingest_cash ──────────┘                                                   └─► t_reconcile_cash
      t_ingest_monarch ───────┘
      ```
- [ ] Root task cron = daily ~06:30 UTC (matches Databricks ~01:30 ET window); child tasks via AFTER.
- [ ] Weekly tasks: cash report (Mon), cost/forecast-accuracy reports.
- [ ] Add a **health-check task** (NB10 equivalent): freshness, forecast horizon ≥25d, feature nulls;
      email on failure.
- [ ] `ALTER TASK ... RESUME` the DAG; verify one full unattended run.

### Phase 5 - Parity validation & parallel run (ongoing)
- [ ] Build a **reconciliation query set** comparing Snowflake vs Databricks Gold on the overlap window:
      row counts AND net-revenue/order dollar values per date (row counts can match while values drift).
- [ ] Run both platforms daily for N weeks; track deltas. Target: |net_revenue delta| < 1% per day.
- [ ] Decision gate: if parity holds + Snowflake monthly spend < $50 → cutover candidate; else iterate.

---

## 5. Spend controls (the $50 guarantee)
1. **Resource Monitor** hard cap (Phase 0) - suspends WH at quota; nothing can silently overrun.
2. X-Small only; auto-suspend 60s; no multi-cluster.
3. Tasks run serially, minutes/day → keep WH active-time low.
4. `SNOWFLAKE.ML.FORECAST` retrain is cheap at this data size (263 rows); retrain daily is fine, or
   throttle to 2–3×/week if credits trend high.
5. Weekly review of `WAREHOUSE_METERING_HISTORY` + `METERING_DAILY_HISTORY` for the first month.

## 5b. PARITY FINDING (2026-07-01) - net/gross naming inversion
Reconciliation of June daily revenue: Snowflake `NET_REVENUE` == Databricks `GROSS_REVENUE`,
**penny-identical on 26/30 days** (the 4 diffs are event/discount days, <1%). Order counts match ±0-2.
**There is NO revenue data discrepancy.** The apparent ~3.2% gap was a DEFINITION INVERSION:
- **Databricks:** `net_revenue` = tax-INCLUSIVE total; `gross_revenue` = pre-tax subtotal (NON-STANDARD).
- **Snowflake:** `NET_REVENUE` = pre-tax subtotal (standard); `GROSS_REVENUE` = total.
Snowflake's convention is the accounting-correct one (net = what you keep, pre-tax).

**USER DECISION (2026-07-01):** Keep Snowflake as-is; CORRECT DATABRICKS to standard convention.
Before changing Databricks: full medallion trace in progress (agent) to map blast radius - must know
whether the tax-inclusive value only mislabels at gold, or PROPAGATES into forecasts/emails/features
(which would mean forecasts predict tax-inclusive $ and reports overstate real sales ~3%). Do NOT edit
Databricks until that trace is reviewed. See `REVENUE_DEFINITION_AUDIT.md` (to be written).

## 6. Known risks / watch-items
- **Monarch API** (Phase 2) is the highest-risk port (unofficial GraphQL + auth in a stored proc).
- **Forecast regressors**: v1 SNOWFLAKE.ML.FORECAST on bare revenue < Prophet-with-regressors accuracy;
  closing this gap is the main modeling task.
- **Item-catalog dependency**: several Gold/DQ fixes are blocked until ITEM_CATALOG is populated.
- **Email integration** must be verified before SEND_* procs work.
- Secrets for Toast/Monarch live in Snowflake - treat rotation + access-grants carefully.
```
```
