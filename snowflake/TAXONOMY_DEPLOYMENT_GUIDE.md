# Taxonomy Classification on Snowflake — How It's Hosted & Deployed

Companion to the field use-case brief. This explains **how Cortex models actually run on Snowflake**
(the "where is the model hosted?" question), and the exact steps to deploy this on the 3SP account.

---

## 1. The mental model: there is no model to host

This is the crux of the Snowflake AI story, and the most common thing customers get wrong.

With a traditional stack you would: pick a model → provision a GPU/inference server (or call an external
API) → containerize → deploy an endpoint → keep it patched and paid-for → move data to it.

**On Snowflake Cortex, none of that exists.** The LLMs (`llama3.1-8b`, `mistral-large`, `snowflake-arctic`,
Claude/GPT where regionally available, etc.) are **pre-hosted and managed by Snowflake inside the region**.
You invoke them as SQL functions:

```sql
SELECT AI_COMPLETE('llama3.1-8b', 'your prompt here');
```

The function call is scheduled onto Snowflake-managed GPU capacity, runs **in-region against data that
never leaves the account**, and bills as **AI_FUNCTIONS serverless credits** (token-based) — not against
your warehouse. Your warehouse (`TSP_PIPELINE_WH`, X-Small) only does the surrounding SQL: reading the
catalog, parsing JSON, writing results.

So "deploying the model" = **writing a stored procedure that calls the function.** That's it.

### What you DON'T manage
- No model weights, no GPU, no container, no serving endpoint, no autoscaling config.
- No data egress / no external DPA — the item catalog is read in place.
- No separate uptime: if the warehouse is up for the query, the model is available.

### What you DO manage
- The **prompt** (the real engineering — see the AI_CLASSIFY vs AI_COMPLETE comparison).
- The **valid label set** (derived from your own taxonomy so answers are always real categories).
- A **guardrail** (validate suggestions against the allowed set; flag off-taxonomy answers).
- The **human-in-the-loop** review + the curated override table.

---

## 2. The objects that make up this solution

| Object | Type | Role |
|---|---|---|
| `REFERENCE.ITEM_CATALOG` | table | Source of truth — items + current taxonomy (loaded from Toast export) |
| `DQ.SUGGEST_TAXONOMY_V2()` | stored proc | Scores all active items via `AI_COMPLETE` (llama3.1-8b) → confidence + reason |
| `DQ.TAXONOMY_SUGGESTIONS` | table | One row/item: suggestion, confidence, reason, and human-decision columns |
| `DQ.TAXONOMY_REVIEW_APP` | Streamlit | Native in-Snowflake review UI (accept/reject/edit) |
| `REFERENCE.ITEM_CATEGORY_OVERRIDES` | table | Human-approved category fixes the pipeline joins over the catalog |
| `DQ.STREAMLIT_APPS` | stage | Holds the app's `.py` file |

Everything is a managed Snowflake object. Nothing runs outside the account.

---

## 3. Deploy steps (reproducible)

### a. Classifier (the "model")
```sql
-- Deploy + run. Scores every active item; ~$0.83 in AI_FUNCTIONS credits for ~1,000 items.
-- File: snowflake/procedures/suggest_taxonomy_v2.sql
CALL THREE_SISTERS_ANALYTICS.DQ.SUGGEST_TAXONOMY_V2();
```
Model selection is literally the first argument to `AI_COMPLETE`. To try a stronger model, change
`'llama3.1-8b'` to e.g. `'mistral-large2'` and re-run — no redeploy of infrastructure.

### b. Streamlit review app (native)
```sql
-- 1. stage exists
CREATE STAGE IF NOT EXISTS THREE_SISTERS_ANALYTICS.DQ.STREAMLIT_APPS DIRECTORY=(ENABLE=TRUE);
-- 2. upload the app file (snow CLI):
--    snow sql -q "PUT file://.../taxonomy_review_app.py @...DQ.STREAMLIT_APPS/taxonomy_review OVERWRITE=TRUE"
-- 3. create the app object
CREATE OR REPLACE STREAMLIT THREE_SISTERS_ANALYTICS.DQ.TAXONOMY_REVIEW_APP
  ROOT_LOCATION='@THREE_SISTERS_ANALYTICS.DQ.STREAMLIT_APPS/taxonomy_review'
  MAIN_FILE='taxonomy_review_app.py'
  QUERY_WAREHOUSE='TSP_PIPELINE_WH';
```
Open it from Snowsight → Projects → Streamlit. No hosting, no URL to manage, runs on the same X-Small WH.

### c. (Optional) schedule a re-score
Add a task so new/edited items get re-classified after each catalog reload — mirrors the existing
`DAILY_*` DAG pattern:
```sql
CREATE OR REPLACE TASK THREE_SISTERS_ANALYTICS.DQ.WEEKLY_RESCORE_TAXONOMY
  WAREHOUSE=TSP_PIPELINE_WH SCHEDULE='USING CRON 0 8 * * MON UTC'
  AS CALL THREE_SISTERS_ANALYTICS.DQ.SUGGEST_TAXONOMY_V2();
ALTER TASK THREE_SISTERS_ANALYTICS.DQ.WEEKLY_RESCORE_TAXONOMY RESUME;
```

---

## 4. How human decisions flow back into the pipeline
1. App writes the decision to `TAXONOMY_SUGGESTIONS` (`human_decision`, `human_category`, `reviewed_at`).
2. On accept/edit, it MERGEs the chosen category into `REFERENCE.ITEM_CATEGORY_OVERRIDES`.
3. Downstream (e.g. `DAILY_SALES_BY_CATEGORY`) LEFT JOINs overrides over the catalog:
   `COALESCE(override.category, catalog.category)` — so curated fixes always win, and the raw Toast
   export stays untouched.

This is the durable value: the model proposes at scale, the human curates the contested ~30%, and the
curated layer compounds over time.

---

## 5. Measured results (live account, Jul 2026)
- 1,068 active items scored. Model agrees with current taxonomy on **733 (69%)**.
- **308 actionable, valid, high-confidence (≥0.80)** reclassifications surfaced for review.
- Avg confidence **0.92**; off-taxonomy hallucinations **0.94%** (auto-flagged, never shown as fixes).
- Cost: **0.638 AI_FUNCTIONS credits (~$1.66) for TWO full-catalog runs.**
- `AI_COMPLETE` + confidence beat one-line `AI_CLASSIFY` (which mis-moved 55 NA-Drinks items to Café Drinks).

## 6. Model options & when to escalate
| Model | Use when |
|---|---|
| `llama3.1-8b` (used here) | Default — cheap, fast, 99%+ valid JSON, great for short-text classification |
| `mistral-large2` / `llama3.1-70b` | Ambiguous items, longer descriptions, or when you want fewer "needs a call" rows |
| `claude-*` / `openai-*` (region-dependent) | Highest reasoning; check `AI_COMPLETE` model availability in AWS_CA_CENTRAL_1 |
| Deterministic + `EMBED_TEXT` similarity | If you need full explainability / zero hallucination — match to nearest labeled item by vector distance |

Swapping models is a one-token change in the proc. Start cheap, escalate only for the rows that need it.
