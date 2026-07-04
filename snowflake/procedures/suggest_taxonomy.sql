-- ============================================================================
-- Product Taxonomy Classifier (Snowflake Cortex AI_CLASSIFY)
-- Scans ITEM_CATALOG, suggests category (and flags disagreements) using the
-- store's OWN existing taxonomy as the label set. In-warehouse, no infra.
-- ============================================================================

-- Valid category label set is derived from the existing catalog so suggestions
-- always map to a real Toast category. Stored as a helper view.
CREATE OR REPLACE VIEW THREE_SISTERS_ANALYTICS.DQ.V_TAXONOMY_LABELS AS
SELECT ARRAY_AGG(DISTINCT category) AS categories,
       ARRAY_AGG(DISTINCT category_group) AS groups
FROM THREE_SISTERS_ANALYTICS.REFERENCE.ITEM_CATALOG
WHERE category IS NOT NULL AND TRIM(category) <> '';

-- Suggestion table: one row per item scored. Re-runnable (full replace).
CREATE OR REPLACE PROCEDURE THREE_SISTERS_ANALYTICS.DQ.SUGGEST_TAXONOMY()
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS CALLER
AS
$$
DECLARE
  v_labels ARRAY;
  v_rows NUMBER;
BEGIN
  SELECT categories INTO :v_labels FROM THREE_SISTERS_ANALYTICS.DQ.V_TAXONOMY_LABELS;

  CREATE OR REPLACE TABLE THREE_SISTERS_ANALYTICS.DQ.TAXONOMY_SUGGESTIONS AS
  WITH scored AS (
    SELECT
      ic.item_id, ic.name, ic.brand, ic.category_group, ic.category AS current_category,
      ic.subcategory, ic.sales_category,
      AI_CLASSIFY(
        CONCAT('Retail item to categorize for a coastal grocery + cafe. ',
               'Name: ', ic.name,
               COALESCE(CONCAT(' | Brand: ', ic.brand), ''),
               COALESCE(CONCAT(' | Description: ', ic.description), '')),
        (SELECT categories FROM THREE_SISTERS_ANALYTICS.DQ.V_TAXONOMY_LABELS)
      ) AS cls
    FROM THREE_SISTERS_ANALYTICS.REFERENCE.ITEM_CATALOG ic
    WHERE ic.is_discontinued = FALSE
  )
  SELECT
    item_id, name, brand, category_group, current_category, subcategory, sales_category,
    cls:labels[0]::STRING AS suggested_category,
    -- disagreement flag: model suggests a different category than currently set
    (cls:labels[0]::STRING <> current_category) AS is_reclassification,
    -- gap flag: item had no meaningful current category signal
    (current_category IS NULL OR TRIM(current_category) = '' OR sales_category IS NULL) AS is_gap_fill,
    CURRENT_TIMESTAMP() AS _scored_at,
    'ai_classify_cortex' AS _method
  FROM scored;

  SELECT COUNT(*) INTO :v_rows FROM THREE_SISTERS_ANALYTICS.DQ.TAXONOMY_SUGGESTIONS;
  RETURN 'Taxonomy scored: ' || :v_rows || ' active items';
END;
$$;

CALL THREE_SISTERS_ANALYTICS.DQ.SUGGEST_TAXONOMY();
