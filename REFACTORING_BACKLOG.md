# Refactoring Backlog - Toast Retail Analytics on Databricks
## Goal: Open-Source as "Toast Retail Analytics Starter"

A community-deployable Databricks Asset Bundle for Toast Retail operators.
Target audience: small independent retailers running Toast Retail who want
a production-grade analytics pipeline without building it from scratch.

**License recommendation: Business Source License 1.1 (BUSL-1.1)**
- Free to use, modify, and self-host for any non-commercial or single-business use
- Cannot be resold as a managed service or SaaS product without a commercial license
- Converts to Apache 2.0 after 4 years (standard BUSL terms)
- Rationale: lets any Toast operator use it freely; protects your JosephSchafer.com
  consulting angle (you can offer paid deployment/customization without competitors
  cloning the work and undercutting you); respectable open-source citizenship

---

## Backlog Structure

Items are grouped by phase. Within each phase, items are ordered by dependency
(things that block other things come first). Estimated effort is in half-days (HD).

---

## Phase 1 - Centralize Configuration (blocks everything else)

> Right now, store-specific constants are scattered across 7+ notebooks.
> A new deployer would have to hunt through every file to stand up their instance.
> One config file fixes this and makes the project bundle-ready.

### 1.1 - Create `config/store_config.py` (or a widget-driven config notebook)
**Effort:** 1 HD

Extract all store-specific constants into one place:
- `CATALOG` name (`YOUR_CATALOG` → parameterized)
- `LATITUDE`, `LONGITUDE`, `TIMEZONE` ([your city]-specific in NB2)
- `STORE_OPEN_HOUR`, `STORE_CLOSE_HOUR` (hardcoded in NB4)
- `BACKFILL_START` date (hardcoded in NB2 as `"YOUR_TOAST_GOLIVE_DATE"`)
- `LOCAL_TZ` (repeated in NB4, NB9)
- `PRIOR_MONTHLY_LINEARITY` dict (NB8 - highly store-specific seasonality priors)
- `PROMOTION_THRESHOLD` (NB8 - auto-promote if new model beats production by this %)
- Prophet hyperparameter defaults (changepoint_prior_scale, weekday/weekend prior scales)

Pattern: a `notebooks/0_Config.py` notebook that defines a `CONFIG` dict and
is `%run`-included by all downstream notebooks, with every value also settable
via Databricks widget or bundle variable override.

### 1.2 - Replace all hardcoded secret scope/key names with config variables
**Effort:** 0.5 HD

Currently hardcoded in NB1:
```python
dbutils.secrets.get(scope="toast_api", key="toast_client_id")
```
Move scope name and key names into `store_config.py`. Document the required
secret scope structure in `SETUP.md`.

### 1.3 - Standardize `run_mode` widget naming across notebooks
**Effort:** 0.5 HD

Inconsistency today:
- NB1: `"incremental"` / `"backfill"`
- NB4: `"incremental"` / `"full_refresh"`
- NB6: `"incremental"` / `"full_rebuild"`
- NB9: `"scheduled"` / `"backfill"`

Pick one convention (`incremental` / `full_rebuild`) and apply it everywhere.
Update job definitions and CLAUDE.md accordingly.

---

## Phase 2 - Code Cleanup

### 2.1 - Remove duplicated `_prior_seasonal_index` function
**Effort:** 0.5 HD

This function is defined identically in both NB7 (`7_Model_Prophet_Revenue.py`)
and NB9 (`9_Forecast_Generate.py`). It should live once in `0_Config.py` or a
shared `notebooks/utils.py` and be `%run`-included. Duplication means changes
to the prior logic must be made in two places and will silently diverge.

### 2.2 - Fix NB9 changelog formatting
**Effort:** 0.5 HD

The v6/v2 changelog entries in NB9 are missing the `# MAGIC ` prefix so they
render as raw Python comments in the notebook rather than Markdown. Inconsistent
with v7/v5/v4/v3/v1 entries. Fix the formatting.

### 2.3 - Audit and remove retired/diagnostic notebooks from the active tree
**Effort:** 0.5 HD

Files present in the repo that need a decision:
- `_RETIRED_README.py` - keep as reference or delete
- `Diagnostic_Toast_Retail_API.py` - useful for setup but shouldn't be in pipeline
- `Test_Toast_Retail_API_Access.py` - same
- `validate_tips_in_net_revenue.py` - one-off analysis, not pipeline
- `8_Model_Prophet_Orders.py` - is this used? NB9 loads `toast_orders_prophet` but
  there's no corresponding training notebook in the daily job. Needs a decision.
- `5_EDA_Sales_Forecast.py` - exploratory, not pipeline

Resolution: move non-pipeline notebooks to a `notebooks/tools/` subdirectory.
Keep them in git (they're useful) but make the pipeline boundary clear.

### 2.4 - Standardize notebook header format
**Effort:** 0.5 HD

NB1 and NB2 have excellent headers (table of modes, schedule, design decisions).
NB7 and NB9 have changelogs. NB6 has a detailed feature category explanation.
NB8 is sparse. Apply a consistent header template across all pipeline notebooks:

```
# Title - What layer and what it does
## Purpose (2-3 sentences)
## Run modes (table)
## Schedule (when it runs, what depends on it)
## Reads from / Writes to (table)
## Change log (version | date | author | change)
```

### 2.5 - Remove [store abbreviation]-specific item lists from Kitchen Staffing dashboard queries
**Effort:** 1 HD  *(you noted the dashboard wasn't super useful - defer or drop)*

The hardcoded item names (Turkey Cheddar, Chicken Salad Wrap, etc.) in the
Kitchen Staffing dashboard SQL are entirely [store abbreviation]-specific. For open-source
publication, this dashboard either needs to be dropped, or the item list needs
to come from a reference table (`reference.kitchen_items` with a
`staffing_category` column) that a deployer populates for their own menu.

---

## Phase 3 - Documentation

### 3.1 - Write `README.md`
**Effort:** 1 HD

The repo has `CLAUDE.md` (internal instructions) but no user-facing README.
Needs:
- What this is (one paragraph pitch)
- Architecture diagram: Bronze → Silver → Gold → Platinum → Prophet
- Prerequisites: Databricks workspace, Unity Catalog enabled, Toast Retail API access
- Quick-start: 5 steps from clone to first pipeline run
- Table of all notebooks with purpose and schedule
- Table of all tables with layer, grain, and description
- Known limitations / first-year seasonality caveat for Prophet

### 3.2 - Write `SETUP.md` - Deployer's Guide
**Effort:** 1 HD

Step-by-step for a new Toast Retail operator:
1. Clone the repo
2. Create Unity Catalog and set `CATALOG` in config
3. Create the Databricks secret scope `toast_api` with required keys
   (document exact key names and where to find values in Toast Portal)
4. Run `0_Config.py` to validate configuration
5. Run NB1 in `backfill` mode from your Toast go-live date
6. Run NB2 in backfill mode
7. Run NB4, NB6, NB7, NB9 in sequence
8. Create the daily job (or deploy the bundle)
9. Tune Prophet priors after ~90 days of data

### 3.3 - Write `SEASONALITY_TUNING.md`
**Effort:** 1 HD

The hardest part of deploying this for a new store is that the Prophet priors
(`PRIOR_MONTHLY_LINEARITY`, `ne_prior_weekend`, `ne_prior_weekday`,
`changepoint_prior_scale`) are tuned to a coastal Massachusetts seasonal business.
A Texas BBQ joint or a Denver coffee shop has a completely different curve.

Document:
- What each prior controls and how to read the CV MAPE to know if it's wrong
- The DOW bias diagnostic in NB8 and how to interpret its output
- The monthly divergence diagnostic and what >20% means
- Recommended tuning process: run with defaults for 90 days, then use the
  Forecast Accuracy dashboard to identify systematic bias, then adjust priors

### 3.4 - Add `LICENSE` file
**Effort:** 0.5 HD

Add BUSL-1.1 license text. The BUSL requires specifying:
- `Change Date`: recommend 4 years from first public release
- `Change License`: Apache 2.0
- `Additional Use Grant`: "You may use the software for your own internal
  business operations at a single retail location without restriction."

---

## Phase 4 - Bundle (DABs)

### 4.1 - Create `databricks.yml` - root bundle definition
**Effort:** 1 HD

```yaml
bundle:
  name: toast-retail-analytics

variables:
  catalog:
    description: Unity Catalog name for this deployment
    default: toast_retail_analytics
  store_name:
    description: Human-readable store name (used in dashboard titles)
  secret_scope:
    description: Databricks secret scope containing Toast API credentials
    default: toast_api

workspace:
  host: ${var.workspace_host}

targets:
  dev:
    mode: development
  prod:
    mode: production
```

### 4.2 - Define the daily pipeline job in bundle YAML
**Effort:** 1 HD

Convert job `YOUR_DAILY_JOB_ID` into `resources/jobs/daily_pipeline.yml`:
- Task order: NB1 → NB2 → NB4 → NB6 → NB9 → NB10
- Schedules, retries, email notifications
- Serverless compute (no cluster config needed)
- All `CATALOG` references replaced with `${var.catalog}`

### 4.3 - Define the weekly retrain job in bundle YAML
**Effort:** 0.5 HD

Convert the weekly retrain job (NB7 → NB8, Monday 4am ET) into
`resources/jobs/weekly_retrain.yml`.

### 4.4 - Parameterize notebook paths in bundle
**Effort:** 0.5 HD

All notebook task paths in the bundle should reference the repo-relative path
(`./notebooks/1_Bronze_Ingest_Toast_Orders`) so they work regardless of
which workspace path the bundle is deployed to.

---

## Phase 5 - Testing & Validation

### 5.1 - Add a `0_Bootstrap_Validate.py` notebook
**Effort:** 1 HD

A setup-validation notebook that a new deployer runs once to confirm their
environment is correctly configured before running any pipeline:
- Can reach Toast API (auth test with the configured credentials)
- Can reach Open-Meteo (no key required, but network access check)
- Secret scope exists and all required keys are present
- Unity Catalog exists and the pipeline user has CREATE TABLE permissions
- MLflow experiment path is writable

Exits with a pass/fail summary and clear error messages for each check.

### 5.2 - Add NB10 health check coverage for new tables
**Effort:** 0.5 HD

`10_Pipeline_Health_Check.py` should verify:
- `gold.forecast_accuracy_history` is being written (NB7/NB8 output)
- `gold.retrain_review_history` is being written (NB8 output)
- Latest forecast covers at least 25 of the next 30 days
- Model registry has a `@production` alias for both revenue and orders models

---

## Phase 6 - Post-Open-Source

### 6.1 - GitHub repository hygiene
**Effort:** 0.5 HD

- Rename repo from `3sp-analytics` to `toast-retail-analytics` (or similar)
- Add `.github/ISSUE_TEMPLATE/` with bug report and feature request templates
- Add `CONTRIBUTING.md` (how to submit a PR, coding style, test requirements)
- Update `CLAUDE.md` to remove [store abbreviation]-specific operational details that shouldn't
  be public (job IDs, workspace URL, store-specific schedule notes)
- Add `.env.example` documenting required environment variables

### 6.2 - Scrub [store abbreviation]-specific constants from all notebooks
**Effort:** 1 HD

A full pass to replace every remaining [store abbreviation]-specific hardcoded value with a
config variable or a clear `# CONFIGURE: replace with your value` comment:
- `YOUR_LATITUDE` / `YOUR_LONGITUDE` ([your city] coordinates in NB2)
- `"YOUR_TOAST_GOLIVE_DATE"` (Toast go-live date)
- `YOUR_CATALOG` catalog name (should all be via config by Phase 1)
- `PRIOR_MONTHLY_LINEARITY` values in NB8
- Any remaining references to "[your store name]" or "[store abbreviation]" in notebook text

### 6.3 - JosephSchafer.com positioning
**Effort:** ongoing

The README should include a tasteful "Professional Deployment" section:
> This project is maintained by [Joseph Schafer](https://josephschafer.com).
> If you'd like help deploying this for your store, tuning the forecasting model,
> or extending it for your specific needs, professional services are available.

Keep it brief. The open-source work speaks for itself; the consulting CTA just
makes the connection explicit.

---

## Future Ideas (not scheduled - explore when data warrants)

### F.1 - LightGBM gradient-boosted revenue model
**Context:** The feature engineering notebook (NB6) was originally designed with a
LightGBM model in mind - lag features, cyclical DOW encoding, and training_weight
were all built for it. The model was researched but never implemented. All the
infrastructure is already in place.

**When it makes sense:** After ~18 months of data (roughly Jan 2027). Prophet is
the right tool for the first year because it handles sparse data and encodes
priors well. Once you have enough data for gradient boosting to generalize,
LightGBM typically outperforms Prophet on short-horizon (1-7 day) forecasts.

**What it would involve:**
- A new `7b_Model_LightGBM_Revenue.py` notebook reading from `gold.forecast_features`
- Use `lag_*` and rolling features that Prophet ignores
- Add to NB8 review logic to compare MAPE against the Prophet production model
- The winner of NB7 vs NB7b competes for `@production` alias

**Effort when ready:** ~2 HD for the model notebook, 0.5 HD to wire into NB8.

### F.2 - Resolve `8_Model_Prophet_Orders.py` status
**Context:** NB9 loads a `toast_orders_prophet@production` model from the registry,
but `8_Model_Prophet_Orders.py` (the training notebook for it) is not in any
scheduled job and has never been updated with the multi-variant / NB8-review pattern
applied to the revenue model (NB7). The orders model is likely stale.

**Decision needed:**
- If order count forecasting matters: bring NB8_Orders up to the same standard as
  NB7 (multi-variant loop, CV horizon=30d, accuracy history write, NB8 review),
  add it to the weekly retrain job
- If it doesn't matter: remove the orders model load from NB9 and simplify the
  forecast output to revenue-only

---

## Summary - Estimated Total Effort

| Phase | Items | Effort |
|-------|-------|--------|
| 1 - Centralize Config | 3 items | 2 HD |
| 2 - Code Cleanup | 5 items | 3 HD |
| 3 - Documentation | 4 items | 3.5 HD |
| 4 - Bundle (DABs) | 4 items | 3 HD |
| 5 - Testing | 2 items | 1.5 HD |
| 6 - Post-Open-Source | 3 items | 1.5 HD |
| **Total** | **21 items** | **~14 HD (7 focused days)** |

## Recommended Sequencing

If you want to do this in chunks without blocking [store abbreviation] operations:

1. **Phase 1 first** - config centralization is the dependency for everything
   else and also directly benefits your day-to-day work on [store abbreviation]
2. **Phase 3 in parallel** - documentation can be written before the bundle
   exists and helps clarify what the bundle needs to do
3. **Phase 2 next** - cleanup is satisfying and low-risk
4. **Phase 4 when you have a clean base** - bundle on top of clean code
5. **Phase 5 and 6 last** - polish before public announcement

## What to Skip / Defer

- **Kitchen Staffing dashboard** - you noted it wasn't very useful. Don't
  invest in open-sourcing it until the concept is more proven. Leave it out
  of the bundle v1.
- **`8_Model_Prophet_Orders.py`** - needs a decision before open-sourcing.
  If `toast_orders_prophet` is actually used by NB9, this notebook needs the
  same multi-variant + NB8-review treatment as NB7. If it's orphaned, delete it.
- **LightGBM** - NB6 mentions "Both the Prophet and LightGBM models read from
  this table" but there's no LightGBM notebook. Don't promise it in the README
  until it exists.
