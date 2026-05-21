# Three Sisters Provisions — Databricks Project

## Project Overview
Daily data pipeline for Three Sisters Provisions (Cohasset, MA coastal grocery/café).
Source: Toast POS REST API. Stack: Bronze → Silver → Gold → Platinum → Prophet forecasting.
All tables in catalog `3sp_analytics_workspace`.

## CLI Profile
Always pass `-p "joe@threesistersprovisions.com"` to every `databricks` CLI command.
Workspace: `https://4261785904697758.8.gcp.databricks.com`

## Notebook Upload Pattern
```bash
MSYS_NO_PATHCONV=1 databricks workspace import "/Workspace/Shared/Sales & Weather Repo/NOTEBOOK_NAME" \
  --file "C:/Users/Joseph Schafer/databricks-project/notebooks/NOTEBOOK_NAME.py" \
  --format SOURCE --language PYTHON --overwrite -p "joe@threesistersprovisions.com"
```
Notebooks live in `/Workspace/Shared/Sales & Weather Repo/`. Local path is `notebooks/`.

## Main Pipeline
**Job ID:** `601248779031413` — "Daily Sales Ingestion & Transformation"
**Task order:** NB1 (Toast ingest) → NB2 (weather) → NB4 (Gold sales) → NB6 (features) → NB9 (forecast) → NB10 (health check)
**Trigger a full run:** `manage_job_runs` action=run_now job_id=601248779031413
**Check status:** `manage_job_runs` action=get run_id=<run_id>

### Notebook Paths
The daily job runs notebooks from the Git Repos path (synced via GitHub Actions on push):
`/Repos/joe@threesistersprovisions.com/3sp-analytics/notebooks/`
NB10 is at `/Shared/Sales & Weather Repo/10_Pipeline_Health_Check` (uploaded directly; move to Repos path after first git push of that file).
NB7/NB8 (retrain — not in the daily job) live at `/Shared/Sales & Weather Repo/` and are run manually or via temp jobs.

## Key Tables

| Table | Layer | Description |
|-------|-------|-------------|
| `bronze.toast_orders_raw` | Bronze | Raw Toast orders JSON |
| `bronze.toast_inventory_history_raw` | Bronze | Inventory adjustments; RECEIVE events have `costInfo.capturedCost` in raw_json |
| `bronze.weather_hourly` | Bronze | Hourly weather (Open-Meteo) |
| `silver_sales.order_items_silver` | Silver | Line items: `item_guid`, `display_name`, `unit_price`, `quantity`, `voided` |
| `silver_sales.orders_silver` | Silver | Order headers |
| `gold.daily_sales_summary` | Gold | NB4 output — daily aggregated sales + weather |
| `gold.forecast_features` | Gold | NB6 output — ML feature table, one row/day |
| `gold.daily_sales_forecast` | Gold | NB9 output — Prophet 30-day forward forecasts; one row per (business_date, forecast_created_at::date); `forecast_horizon_days` enables accuracy-by-horizon analysis |
| `gold.forecast_accuracy_history` | Gold | NB7/NB8 output — one row per retrain run; tracks train_mape, cv_mape, cv_mape_7d/14d, backtest_mape; enables Phase 2 retraining decisions |
| `platinum.daily_sales_combined` | Platinum | VIEW joining actuals + latest forecasts |
| `dq.par_level_suggestions` | DQ | Active SKUs with inventory status, par levels, velocity |
| `reference.store_events` | Reference | Special event flags (holiday closures, events) |

## Key Data Joins
**Item cost from receiving records:**
```sql
WITH cost_latest AS (
    SELECT item_version_id,
           CAST(get_json_object(raw_json, '$.costInfo.capturedCost') AS DOUBLE) AS unit_cost,
           ROW_NUMBER() OVER (PARTITION BY item_version_id ORDER BY created_date DESC) AS rn
    FROM 3sp_analytics_workspace.bronze.toast_inventory_history_raw
    WHERE adjustment_type = 'RECEIVE'
      AND get_json_object(raw_json, '$.costInfo.capturedCost') IS NOT NULL
),
cost AS (SELECT item_version_id, unit_cost FROM cost_latest WHERE rn = 1),
prices AS (
    SELECT item_guid, display_name, ROUND(AVG(unit_price), 2) AS avg_unit_price
    FROM 3sp_analytics_workspace.silver_sales.order_items_silver
    WHERE business_date >= date_sub(current_date(), 60) AND voided = false AND quantity > 0
    GROUP BY item_guid, display_name
)
-- Join: cost.item_version_id = prices.item_guid (they are the same Toast item GUID)
```

## Temp Job Pattern for One-Off Notebook Runs
Use when a notebook needs non-default parameters (e.g. `run_mode=full_rebuild`):
1. `manage_jobs` action=create, name=`Temp_DescriptiveName`, tasks with `base_parameters`
2. `manage_job_runs` action=run_now, job_id=<new_id>
3. `manage_job_runs` action=wait, run_id=<run_id> (timeout=600, poll_interval=20)
4. `manage_jobs` action=delete, job_id=<new_id> — always clean up
Serverless compute is used by default — no cluster config needed.

## Known Gotchas

### Delta MERGE Does NOT Add New Columns
`DeltaTable.merge()` (with `whenMatchedUpdateAll`) silently ignores DataFrame columns absent from
the target schema. Even a "full_rebuild" MERGE run will succeed but drop new columns.

**Fix — two steps:**
1. `ALTER TABLE <table> ADD COLUMNS (col_name DOUBLE COMMENT '...')` — adds column with NULLs
2. `UPDATE <table> SET col_name = <formula> WHERE col_name IS NULL` — backfills existing rows
Then the next incremental MERGE will keep the column populated going forward.

### spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled") is Blocked on Serverless
Raises `[CONFIG_NOT_AVAILABLE] SQLSTATE: 42K0I`. Do not attempt it.
Use the ALTER TABLE + UPDATE pattern above instead.

### Lakeview Categorical Axis Sort is Always Alphabetical
Lakeview v3 bar/scatter charts have no `sort` encoding option for categorical axes.
The axis always sorts alphabetically.

**Workaround** — encode sort order into the label with a zero-padded rank prefix:
```sql
LPAD(CAST(ROW_NUMBER() OVER (ORDER BY sort_col DESC) AS STRING), 2, '0') || '. ' || LEFT(name, 28) AS item
```
Result: `01. Highest Item`, `02. Second Item`, etc. Alphabetical = rank order.

### Lakeview "Send Now" Has No Public REST API
`POST /api/2.0/lakeview/dashboards/{id}/schedules/{sid}/execute` → 404.
No variation of the endpoint works (api/2.0, api/2.1, /run, /send-now, etc.).
Manual send is **UI-only**: Dashboard → ⋮ menu → Schedule → "Send now".

### Lakeview Schedule Subscriber Requires user_id, Not user_name
`{"subscriber": {"user_subscriber": {"user_name": "..."}}}` — `user_name` is silently ignored.
Look up the integer user_id first:
```bash
databricks users list --filter "userName eq \"email@domain.com\"" \
  -p "joe@threesistersprovisions.com" -o json
```
Then use: `{"subscriber": {"user_subscriber": {"user_id": <integer>}}}`

### REST API Calls from Windows: Use PowerShell, Not Bash
Bash + grep/python3 for JSON parsing is unreliable in Git Bash on Windows.
Use PowerShell with `ConvertFrom-Json`:
```powershell
$token = (databricks auth token -p "joe@threesistersprovisions.com" --output json 2>$null | ConvertFrom-Json).access_token
$headers = @{"Authorization" = "Bearer $token"; "Content-Type" = "application/json"}
Invoke-RestMethod -Method POST -Uri "https://4261785904697758.8.gcp.databricks.com/api/2.0/..." -Headers $headers -Body "{}"
```
