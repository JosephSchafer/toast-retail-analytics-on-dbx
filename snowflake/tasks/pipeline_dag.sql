-- ============================================================================
-- [your store name] daily pipeline — Snowflake Task DAG on TSP_PIPELINE_WH.
-- Mirrors the Databricks job order. Serverless-free: uses the X-Small WH
-- (auto-suspend 60s) so credit use stays minimal. Resource Monitor caps spend.
--
-- Order:
--   T_INGEST_ORDERS (root, cron)
--     ├─ T_INGEST_WEATHER_ARCHIVE
--     ├─ T_INGEST_WEATHER_FORECAST
--     └─ T_INGEST_INVENTORY
--          → T_TRANSFORMS (silver+gold+category+hourly)
--               → T_VELOCITY → T_DQ
--               → T_FORECASTS
--                    → T_HEALTH_CHECK (+ daily sales email)
-- ============================================================================
USE DATABASE THREE_SISTERS_ANALYTICS;
USE SCHEMA GOLD;

-- --- Health-check proc (NB10 equivalent): freshness + forecast horizon, email on issue ---
CREATE OR REPLACE PROCEDURE GOLD.PIPELINE_HEALTH_CHECK()
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS CALLER
AS
$$
DECLARE
  v_max_actual DATE;
  v_fc_horizon NUMBER;
  v_issues STRING DEFAULT '';
BEGIN
  SELECT MAX(BUSINESS_DATE) INTO :v_max_actual FROM GOLD.DAILY_SALES_SUMMARY WHERE NET_REVENUE > 0;
  SELECT DATEDIFF('day', CURRENT_DATE(), MAX(FORECAST_DATE)) INTO :v_fc_horizon
    FROM GOLD.DAILY_SALES_FORECAST
    WHERE FORECAST_CREATED_AT = (SELECT MAX(FORECAST_CREATED_AT) FROM GOLD.DAILY_SALES_FORECAST);

  IF (:v_max_actual < DATEADD('day',-2,CURRENT_DATE())) THEN
    v_issues := v_issues || 'STALE ACTUALS (latest=' || :v_max_actual || '); ';
  END IF;
  IF (:v_fc_horizon < 25) THEN
    v_issues := v_issues || 'SHORT FORECAST HORIZON (' || :v_fc_horizon || 'd); ';
  END IF;

  IF (v_issues <> '') THEN
    CALL SYSTEM$SEND_EMAIL('TSP_EMAIL_REPORTS','YOUR_EMAIL',
      'TSP Snowflake pipeline health WARNING',
      'Pipeline health check found issues: ' || :v_issues);
    RETURN 'WARN: ' || :v_issues;
  END IF;
  RETURN 'OK: actuals=' || :v_max_actual || ', forecast_horizon=' || :v_fc_horizon || 'd';
END;
$$;

-- ============================ TASK DAG ============================
-- Suspend-recreate pattern: create all suspended, resume root last.

CREATE OR REPLACE TASK GOLD.T_INGEST_ORDERS
  WAREHOUSE = TSP_PIPELINE_WH
  SCHEDULE = 'USING CRON 30 6 * * * UTC'   -- ~01:30 ET daily
  COMMENT = 'Root: ingest yesterday Toast orders (incremental)'
AS CALL BRONZE.INGEST_TOAST_ORDERS('incremental');

CREATE OR REPLACE TASK GOLD.T_INGEST_WEATHER_ARCHIVE
  WAREHOUSE = TSP_PIPELINE_WH AFTER GOLD.T_INGEST_ORDERS
AS CALL BRONZE.INGEST_WEATHER_ARCHIVE('incremental');

CREATE OR REPLACE TASK GOLD.T_INGEST_WEATHER_FORECAST
  WAREHOUSE = TSP_PIPELINE_WH AFTER GOLD.T_INGEST_ORDERS
AS CALL BRONZE.INGEST_WEATHER_FORECAST();

CREATE OR REPLACE TASK GOLD.T_INGEST_INVENTORY
  WAREHOUSE = TSP_PIPELINE_WH AFTER GOLD.T_INGEST_ORDERS
AS CALL BRONZE.INGEST_TOAST_INVENTORY();

-- Transforms wait on all ingest children
CREATE OR REPLACE TASK GOLD.T_TRANSFORMS
  WAREHOUSE = TSP_PIPELINE_WH
  AFTER GOLD.T_INGEST_WEATHER_ARCHIVE, GOLD.T_INGEST_WEATHER_FORECAST, GOLD.T_INGEST_INVENTORY
AS CALL GOLD.REFRESH_TRANSFORMS();

-- Fan-out after transforms
CREATE OR REPLACE TASK GOLD.T_VELOCITY
  WAREHOUSE = TSP_PIPELINE_WH AFTER GOLD.T_TRANSFORMS
AS CALL GOLD.REFRESH_INVENTORY_VELOCITY();

CREATE OR REPLACE TASK GOLD.T_DQ
  WAREHOUSE = TSP_PIPELINE_WH AFTER GOLD.T_VELOCITY
AS CALL DQ.REFRESH_DQ();

CREATE OR REPLACE TASK GOLD.T_FORECASTS
  WAREHOUSE = TSP_PIPELINE_WH AFTER GOLD.T_TRANSFORMS
AS CALL GOLD.REFRESH_FORECASTS();

-- Health check + daily sales email waits on the two leaf branches
CREATE OR REPLACE TASK GOLD.T_HEALTH_CHECK
  WAREHOUSE = TSP_PIPELINE_WH
  AFTER GOLD.T_DQ, GOLD.T_FORECASTS
AS BEGIN
     CALL GOLD.PIPELINE_HEALTH_CHECK();
     CALL GOLD.SEND_DAILY_SALES_REPORT();
   END;

-- Resume children first, root last (Snowflake requirement)
ALTER TASK GOLD.T_HEALTH_CHECK RESUME;
ALTER TASK GOLD.T_FORECASTS RESUME;
ALTER TASK GOLD.T_DQ RESUME;
ALTER TASK GOLD.T_VELOCITY RESUME;
ALTER TASK GOLD.T_TRANSFORMS RESUME;
ALTER TASK GOLD.T_INGEST_INVENTORY RESUME;
ALTER TASK GOLD.T_INGEST_WEATHER_FORECAST RESUME;
ALTER TASK GOLD.T_INGEST_WEATHER_ARCHIVE RESUME;
ALTER TASK GOLD.T_INGEST_ORDERS RESUME;

SHOW TASKS IN DATABASE THREE_SISTERS_ANALYTICS;
