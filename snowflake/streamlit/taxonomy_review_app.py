# ============================================================================
# Three Sisters — Taxonomy Review (Streamlit in Snowflake)
# Human-in-the-loop review of Cortex taxonomy suggestions. Runs natively inside
# Snowflake (no hosting/deploy infra). Reads DQ.TAXONOMY_SUGGESTIONS, captures
# accept/reject/edit decisions back to the same table, and can push accepted
# categories to a curated override table the pipeline consumes.
# ============================================================================
import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd

st.set_page_config(page_title="Taxonomy Review", layout="wide")
session = get_active_session()

DB = "THREE_SISTERS_ANALYTICS"
SUGG = f"{DB}.DQ.TAXONOMY_SUGGESTIONS"
OVERRIDE = f"{DB}.REFERENCE.ITEM_CATEGORY_OVERRIDES"

st.title("🏷️  Product Taxonomy Review")
st.caption("AI suggestions from Snowflake Cortex (llama3.1-8b). You have the final say — "
           "every decision trains the review queue and writes a clean override the pipeline uses.")

# --- Valid categories (the store's own taxonomy) for the edit dropdown ---
@st.cache_data(ttl=600)
def load_categories():
    df = session.sql(
        f"SELECT DISTINCT category FROM {DB}.REFERENCE.ITEM_CATALOG "
        f"WHERE category IS NOT NULL AND TRIM(category)<>'' ORDER BY category"
    ).to_pandas()
    return df["CATEGORY"].tolist()

VALID_CATEGORIES = load_categories()

# --- Sidebar: what to review ---
st.sidebar.header("Review queue")
mode = st.sidebar.radio(
    "Show items where…",
    ["Model disagrees (high confidence)", "Model disagrees (needs judgment)",
     "Off-taxonomy / flagged", "Subcategory gaps", "All unreviewed"],
)
min_conf = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.8, 0.05)

where = "human_decision IS NULL"
if mode == "Model disagrees (high confidence)":
    where += f" AND is_reclassification AND confidence >= {min_conf} " \
             f"AND suggested_category IN (SELECT DISTINCT category FROM {DB}.REFERENCE.ITEM_CATALOG)"
elif mode == "Model disagrees (needs judgment)":
    where += " AND is_reclassification AND confidence < 0.8"
elif mode == "Off-taxonomy / flagged":
    where += f" AND suggested_category NOT IN (SELECT DISTINCT category FROM {DB}.REFERENCE.ITEM_CATALOG)"
elif mode == "Subcategory gaps":
    where += " AND (subcategory IS NULL OR TRIM(subcategory) = '')"

@st.cache_data(ttl=60)
def load_queue(where_clause):
    return session.sql(
        f"SELECT item_id, name, brand, category_group, current_category, "
        f"suggested_category, confidence, reason, price "
        f"FROM {SUGG} WHERE {where_clause} ORDER BY confidence DESC NULLS LAST, name LIMIT 200"
    ).to_pandas()

queue = load_queue(where)

# --- Progress metrics ---
stats = session.sql(
    f"SELECT COUNT(*) t, COUNT_IF(human_decision IS NOT NULL) reviewed, "
    f"COUNT_IF(human_decision='accepted') acc, COUNT_IF(human_decision='rejected') rej, "
    f"COUNT_IF(human_decision='edited') ed FROM {SUGG}"
).to_pandas().iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Items in catalog", int(stats["T"]))
c2.metric("Reviewed", int(stats["REVIEWED"]))
c3.metric("Accepted", int(stats["ACC"]))
c4.metric("Rejected / Edited", int(stats["REJ"]) + int(stats["ED"]))
st.progress(min(1.0, (stats["REVIEWED"] / stats["T"]) if stats["T"] else 0.0))

st.divider()
st.subheader(f"{len(queue)} items to review — {mode}")

def record(item_id, decision, chosen_category):
    session.sql(f"""
        UPDATE {SUGG}
        SET human_decision = '{decision}',
            human_category = {'NULL' if chosen_category is None else "'" + chosen_category.replace("'","''") + "'"},
            reviewed_at = CURRENT_TIMESTAMP()
        WHERE item_id = '{item_id}'
    """).collect()
    # Write the curated override the pipeline reads (accepted suggestion or human edit).
    if chosen_category is not None:
        session.sql(f"""
            MERGE INTO {OVERRIDE} t
            USING (SELECT '{item_id}' item_id, '{chosen_category.replace("'","''")}' category) s
            ON t.item_id = s.item_id
            WHEN MATCHED THEN UPDATE SET category = s.category, updated_at = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT (item_id, category, updated_at)
                VALUES (s.item_id, s.category, CURRENT_TIMESTAMP())
        """).collect()

# --- Review cards ---
for _, r in queue.iterrows():
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**{r['NAME']}**  \n"
                        f"<span style='color:#888'>{r['BRAND'] or '—'} · {r['CATEGORY_GROUP']} · "
                        f"${r['PRICE'] if pd.notna(r['PRICE']) else '—'}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"Current: `{r['CURRENT_CATEGORY']}`  →  "
                        f"**AI suggests: `{r['SUGGESTED_CATEGORY']}`**")
            conf = r["CONFIDENCE"] or 0
            st.markdown(f"Confidence: **{conf:.0%}** · _{r['REASON'] or ''}_")
        with right:
            b1, b2, b3 = st.columns(3)
            if b1.button("✅ Accept", key=f"a_{r['ITEM_ID']}"):
                record(r["ITEM_ID"], "accepted", r["SUGGESTED_CATEGORY"]); st.rerun()
            if b2.button("❌ Reject", key=f"r_{r['ITEM_ID']}"):
                record(r["ITEM_ID"], "rejected", None); st.rerun()
            new_cat = b3.selectbox("Edit", ["—"] + VALID_CATEGORIES, key=f"e_{r['ITEM_ID']}",
                                   label_visibility="collapsed")
            if new_cat != "—" and b3.button("Save edit", key=f"s_{r['ITEM_ID']}"):
                record(r["ITEM_ID"], "edited", new_cat); st.rerun()

if queue.empty:
    st.success("Nothing in this queue — try a different filter or lower the confidence threshold.")
