-- ============================================================================
-- Product Taxonomy Classifier v2 — Cortex AI_COMPLETE (llama3.1-8b) with
-- confidence + reasoning. More accurate than AI_CLASSIFY (which over-reclassifies
-- cross-group-ambiguous items like NA Drinks). Structured JSON output powers the
-- human-in-the-loop Streamlit feedback app.
-- In-warehouse, open-source LLM, no infra to host.
-- ============================================================================
CREATE OR REPLACE PROCEDURE THREE_SISTERS_ANALYTICS.DQ.SUGGEST_TAXONOMY_V2()
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS CALLER
AS
$$
DECLARE
  v_rows NUMBER;
  v_label_csv STRING;
BEGIN
  -- Build the valid-category CSV from the store's own taxonomy (so suggestions are always real).
  SELECT LISTAGG(DISTINCT category, ', ') WITHIN GROUP (ORDER BY category)
    INTO :v_label_csv
  FROM THREE_SISTERS_ANALYTICS.REFERENCE.ITEM_CATALOG
  WHERE category IS NOT NULL AND TRIM(category) <> '';

  CREATE OR REPLACE TABLE THREE_SISTERS_ANALYTICS.DQ.TAXONOMY_SUGGESTIONS AS
  WITH raw AS (
    SELECT
      ic.item_id, ic.name, ic.brand, ic.category_group, ic.category AS current_category,
      ic.subcategory, ic.sales_category, ic.price,
      AI_COMPLETE('llama3.1-8b',
        CONCAT(
          'You categorize retail items for a coastal grocery and cafe. ',
          'Choose EXACTLY ONE category from this list: ', :v_label_csv, '. ',
          'Item name: "', ic.name, '"',
          COALESCE(CONCAT(', brand: "', ic.brand, '"'), ''),
          COALESCE(CONCAT(', description: "', ic.description, '"'), ''), '. ',
          'Current category is "', COALESCE(ic.category, 'NONE'), '". ',
          'Return ONLY valid JSON, no prose: ',
          '{"category":"<exact category from the list>","confidence":<number 0.0 to 1.0>,"reason":"<max 12 words>"}'
        )
      ) AS ai_raw
    FROM THREE_SISTERS_ANALYTICS.REFERENCE.ITEM_CATALOG ic
    WHERE ic.is_discontinued = FALSE
  )
  SELECT
    item_id, name, brand, category_group, current_category, subcategory, sales_category, price,
    TRY_PARSE_JSON(ai_raw):category::STRING       AS suggested_category,
    TRY_PARSE_JSON(ai_raw):confidence::FLOAT      AS confidence,
    TRY_PARSE_JSON(ai_raw):reason::STRING         AS reason,
    (TRY_PARSE_JSON(ai_raw):category::STRING <> current_category) AS is_reclassification,
    (current_category IS NULL OR TRIM(current_category)='' OR sales_category IS NULL) AS is_gap_fill,
    NULL::STRING  AS human_decision,   -- filled by Streamlit app: 'accepted'|'rejected'|'edited'
    NULL::STRING  AS human_category,   -- the human-chosen category if edited
    NULL::TIMESTAMP_NTZ AS reviewed_at,
    ai_raw        AS _ai_raw,
    CURRENT_TIMESTAMP() AS _scored_at,
    'ai_complete_llama31_8b' AS _method
  FROM raw;

  SELECT COUNT(*) INTO :v_rows FROM THREE_SISTERS_ANALYTICS.DQ.TAXONOMY_SUGGESTIONS;
  RETURN 'Taxonomy v2 scored: ' || :v_rows || ' active items (llama3.1-8b + confidence)';
END;
$$;

CALL THREE_SISTERS_ANALYTICS.DQ.SUGGEST_TAXONOMY_V2();
