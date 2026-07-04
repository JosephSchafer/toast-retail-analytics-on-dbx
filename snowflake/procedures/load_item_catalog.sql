-- File format for Toast BOH exports: quoted fields, multi-line descriptions, header row.
CREATE OR REPLACE FILE FORMAT THREE_SISTERS_ANALYTICS.REFERENCE.FF_TOAST_CSV
  TYPE = CSV
  PARSE_HEADER = FALSE
  SKIP_HEADER = 1
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  FIELD_DELIMITER = ','
  MULTI_LINE = TRUE
  TRIM_SPACE = FALSE
  EMPTY_FIELD_AS_NULL = TRUE
  ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE;

-- Reusable loader: mirrors Databricks NB3 (3_Reference_Item_Catalog.py).
-- Usage: CALL REFERENCE.LOAD_ITEM_CATALOG('retail-items-YYYY-MM-DD-HHMMSS.csv');
-- Drop the Toast "Menu > Items > Export CSV" file into stage REFERENCE.TOAST_EXPORTS first (PUT).
CREATE OR REPLACE PROCEDURE THREE_SISTERS_ANALYTICS.REFERENCE.LOAD_ITEM_CATALOG(CSV_FILENAME STRING)
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS CALLER
AS
$$
DECLARE
  v_exported_at TIMESTAMP_NTZ;
  v_rows NUMBER;
BEGIN
  -- Derive export timestamp from filename (retail-items-YYYY-MM-DD-...), else now().
  v_exported_at := COALESCE(
    TRY_TO_TIMESTAMP_NTZ(REGEXP_SUBSTR(:CSV_FILENAME, '\\d{4}-\\d{2}-\\d{2}'), 'YYYY-MM-DD'),
    CURRENT_TIMESTAMP()::TIMESTAMP_NTZ
  );

  -- Stage the raw CSV positionally into a scratch table (76 Toast columns, all TEXT).
  CREATE OR REPLACE TEMPORARY TABLE THREE_SISTERS_ANALYTICS.REFERENCE.TMP_ITEM_RAW (
    item_id STRING, item_multi_location_id STRING, name STRING, category_group STRING, category STRING,
    subcategory STRING, inventory_status STRING, inventory_quantity STRING, par_min STRING, par_max STRING,
    par_target STRING, on_order STRING, inventory_cost STRING, inventory_value STRING, inventory_days_on_hand STRING,
    last_received_from STRING, inventory_last_received STRING, size STRING, package_unit STRING, applicable_taxes STRING,
    tax_rates STRING, price STRING, last_price_update STRING, cost STRING, cost_change STRING,
    margin_change STRING, gross_profit STRING, gross_margin STRING, gross_markup STRING, last_7_day_sales STRING,
    last_30_day_sales STRING, last_90_day_sales STRING, last_7_day_orders STRING, last_30_day_orders STRING, last_90_day_orders STRING,
    barcode STRING, item_type STRING, output_inventory_group_id STRING, source_inventory_group_ids STRING, depletion_type STRING,
    last_counted_time STRING, target STRING, owner STRING, last_cost_update STRING, supplier STRING,
    supplier_item_id STRING, receiving_units STRING, receiving_unit_quantities STRING, plu STRING, sales_category STRING,
    brand STRING, shelf_locations STRING, storage_locations STRING, pricing_strategy STRING, visibility STRING,
    tax_inclusion STRING, takeout_delivery_tax_exempt STRING, discount_eligible STRING, rewards_eligible STRING, prep_stations STRING,
    tare_weight STRING, barcode_scanner_config STRING, unit_of_measure STRING, prompt_for_quantity STRING, selling_strategy STRING,
    pos_name STRING, contains_alcohol STRING, assistance_programs STRING, shelf_label_print_time STRING, primary_unit STRING,
    created STRING, last_updated STRING, discontinued STRING, discontinued_reason STRING, image_url STRING, description STRING
  );

  -- Match the file by base name, tolerant of a .gz suffix that PUT may add.
  EXECUTE IMMEDIATE
    'COPY INTO THREE_SISTERS_ANALYTICS.REFERENCE.TMP_ITEM_RAW ' ||
    'FROM @THREE_SISTERS_ANALYTICS.REFERENCE.TOAST_EXPORTS ' ||
    'PATTERN = ''.*' || REGEXP_REPLACE(:CSV_FILENAME, '\\.csv$', '') || '\\.csv(\\.gz)?'' ' ||
    'FILE_FORMAT = (FORMAT_NAME = THREE_SISTERS_ANALYTICS.REFERENCE.FF_TOAST_CSV) ' ||
    'ON_ERROR = ''CONTINUE'' FORCE = TRUE';

  -- Transform -> final catalog (full replace). clean_currency + bool parse + category_drift.
  CREATE OR REPLACE TABLE THREE_SISTERS_ANALYTICS.REFERENCE.ITEM_CATALOG AS
  SELECT
    item_id, item_multi_location_id, name, pos_name, brand, description, item_type,
    category_group, category, subcategory, sales_category, applicable_taxes, tax_rates, tax_inclusion,
    IFF(UPPER(TRIM(contains_alcohol)) = 'YES', TRUE, FALSE) AS contains_alcohol,
    takeout_delivery_tax_exempt,
    TRY_TO_DOUBLE(REGEXP_REPLACE(price, '[$,]', '')) AS price,
    TRY_TO_DOUBLE(REGEXP_REPLACE(cost,  '[$,]', '')) AS cost,
    gross_margin, supplier, supplier_item_id, barcode, plu, size, package_unit,
    receiving_units, receiving_unit_quantities, unit_of_measure, inventory_status, inventory_quantity,
    par_min, par_max, visibility, discount_eligible, rewards_eligible,
    IFF(discontinued IS NOT NULL AND TRIM(discontinued) <> '', TRUE, FALSE) AS is_discontinued,
    discontinued_reason, last_7_day_sales, last_30_day_sales, last_90_day_sales, created, last_updated,
    CASE WHEN sales_category IS NULL OR category IS NULL THEN NULL
         ELSE (LOWER(TRIM(sales_category)) <> LOWER(TRIM(category))) END AS category_drift,
    :v_exported_at AS _catalog_exported_at,
    :CSV_FILENAME  AS _source_filename
  FROM THREE_SISTERS_ANALYTICS.REFERENCE.TMP_ITEM_RAW
  WHERE item_id IS NOT NULL AND TRIM(item_id) <> '';

  SELECT COUNT(*) INTO :v_rows FROM THREE_SISTERS_ANALYTICS.REFERENCE.ITEM_CATALOG;
  RETURN 'ITEM_CATALOG loaded: ' || :v_rows || ' items from ' || :CSV_FILENAME;
END;
$$;
