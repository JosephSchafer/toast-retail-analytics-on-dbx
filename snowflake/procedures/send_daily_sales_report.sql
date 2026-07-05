CREATE OR REPLACE PROCEDURE "SEND_DAILY_SALES_REPORT"()
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS CALLER
AS '
DECLARE
  v_html VARCHAR;
  v_today_rev FLOAT;
  v_today_orders NUMBER;
  v_yesterday_rev FLOAT DEFAULT 0;
  v_mtd_rev FLOAT;
  v_avg_7d FLOAT;
  v_avg_14d FLOAT;
  v_avg_21d FLOAT;
  v_max_date DATE;
  v_month_start DATE;
  v_pct_change FLOAT DEFAULT 0;
  v_table_rows VARCHAR DEFAULT '''';
  v_dow_rows VARCHAR DEFAULT '''';
  v_forecast_rows VARCHAR DEFAULT '''';
  v_today_forecast FLOAT DEFAULT 0;
  v_next7_forecast FLOAT DEFAULT 0;
  v_proj_month FLOAT DEFAULT 0;
  v_latest_run TIMESTAMP_NTZ;
BEGIN
  SELECT MAX(BUSINESS_DATE) INTO :v_max_date FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_SUMMARY;
  
  SELECT NET_REVENUE, ORDER_COUNT 
  INTO :v_today_rev, :v_today_orders
  FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_SUMMARY 
  WHERE BUSINESS_DATE = :v_max_date;
  
  SELECT COALESCE(MAX(NET_REVENUE), 0) INTO :v_yesterday_rev
  FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_SUMMARY 
  WHERE BUSINESS_DATE = DATEADD(day, -1, :v_max_date);
  
  v_month_start := DATE_TRUNC(''month'', :v_max_date)::DATE;
  SELECT SUM(NET_REVENUE) INTO :v_mtd_rev
  FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_SUMMARY 
  WHERE BUSINESS_DATE >= :v_month_start AND BUSINESS_DATE <= :v_max_date;
  
  SELECT AVG(NET_REVENUE) INTO :v_avg_7d
  FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_SUMMARY 
  WHERE BUSINESS_DATE > DATEADD(day, -7, :v_max_date) AND BUSINESS_DATE <= :v_max_date;
  
  SELECT AVG(NET_REVENUE) INTO :v_avg_14d
  FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_SUMMARY 
  WHERE BUSINESS_DATE > DATEADD(day, -14, :v_max_date) AND BUSINESS_DATE <= :v_max_date;
  
  SELECT AVG(NET_REVENUE) INTO :v_avg_21d
  FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_SUMMARY 
  WHERE BUSINESS_DATE > DATEADD(day, -21, :v_max_date) AND BUSINESS_DATE <= :v_max_date;

  IF (v_yesterday_rev > 0) THEN
    v_pct_change := ROUND((v_today_rev - v_yesterday_rev) / v_yesterday_rev * 100, 1);
  END IF;

  -- Forecast KPIs. The forecast table now keeps MULTIPLE runs per date (history-preserving),
  -- so every read MUST pin to the latest run or SUMs multiply by the run count.
  SELECT MAX(FORECAST_CREATED_AT) INTO :v_latest_run FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_FORECAST;

  SELECT COALESCE(FORECAST_REVENUE, 0) INTO :v_today_forecast
  FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_FORECAST
  WHERE FORECAST_DATE = DATEADD(day, 1, :v_max_date) AND FORECAST_CREATED_AT = :v_latest_run
  ORDER BY GENERATED_AT DESC LIMIT 1;

  SELECT COALESCE(SUM(FORECAST_REVENUE), 0) INTO :v_next7_forecast
  FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_FORECAST
  WHERE FORECAST_DATE > :v_max_date AND FORECAST_DATE <= DATEADD(day, 7, :v_max_date)
    AND FORECAST_CREATED_AT = :v_latest_run;

  -- Projected month = MTD actuals + remaining days forecast (latest run only)
  SELECT :v_mtd_rev + COALESCE(SUM(FORECAST_REVENUE), 0) INTO :v_proj_month
  FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_FORECAST
  WHERE FORECAST_DATE > :v_max_date AND FORECAST_DATE <= LAST_DAY(:v_max_date)
    AND FORECAST_CREATED_AT = :v_latest_run;
  
  -- 14-day detail
  SELECT LISTAGG(
    ''<tr><td>'' || BUSINESS_DATE || ''</td><td>'' || DAY_NAME || ''</td><td>$'' || TO_CHAR(NET_REVENUE,''999,999.00'') || ''</td><td>'' || ORDER_COUNT || ''</td><td>$'' || TO_CHAR(AVG_TICKET,''99.00'') || ''</td><td>'' || COALESCE(WEATHER_CATEGORY,''-'') || ''</td><td>'' || COALESCE(WEATHER_HIGH_F::VARCHAR,''-'') || ''</td></tr>''
    , '''') WITHIN GROUP (ORDER BY BUSINESS_DATE DESC)
  INTO :v_table_rows
  FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_SUMMARY 
  WHERE BUSINESS_DATE > DATEADD(day, -14, :v_max_date);

  -- DOW averages
  SELECT LISTAGG(''<tr><td>'' || DAY_NAME || ''</td><td>$'' || AVG_REV || ''</td><td>'' || AVG_ORD || ''</td></tr>'', '''')
    WITHIN GROUP (ORDER BY DOW)
  INTO :v_dow_rows
  FROM (
    SELECT DAY_NAME, MIN(DAY_OF_WEEK) AS DOW, ROUND(AVG(NET_REVENUE),0) AS AVG_REV, ROUND(AVG(ORDER_COUNT),0) AS AVG_ORD
    FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_SUMMARY 
    WHERE BUSINESS_DATE > DATEADD(day, -56, :v_max_date)
    GROUP BY DAY_NAME
  );

  -- 7-day forecast detail
  SELECT COALESCE(LISTAGG(
    ''<tr><td>'' || FORECAST_DATE || ''</td><td>'' || DAYNAME(FORECAST_DATE) || ''</td><td>$'' || ROUND(FORECAST_REVENUE, 0) || ''</td><td>'' || COALESCE(FORECAST_ORDERS::VARCHAR, ''-'') || ''</td><td>'' || MODEL_VERSION || ''</td></tr>''
    , '''') WITHIN GROUP (ORDER BY FORECAST_DATE), '''')
  INTO :v_forecast_rows
  FROM THREE_SISTERS_ANALYTICS.GOLD.DAILY_SALES_FORECAST
  WHERE FORECAST_DATE > :v_max_date AND FORECAST_DATE <= DATEADD(day, 7, :v_max_date)
    AND FORECAST_CREATED_AT = :v_latest_run;

  v_html := ''<!DOCTYPE html><html><head><style>
body{font-family:Arial,sans-serif;margin:0;padding:20px;background:#f5f5f5}
.c{max-width:700px;margin:0 auto;background:#fff;border-radius:8px;padding:24px;box-shadow:0 2px 4px rgba(0,0,0,.1)}
h1{color:#1a1a2e;margin-bottom:4px}
.sub{color:#666;margin-bottom:20px}
.kr{display:flex;justify-content:space-between;margin-bottom:20px}
.k{text-align:center;padding:12px;background:#f8f9fa;border-radius:6px;flex:1;margin:0 4px}
.kv{font-size:22px;font-weight:bold;color:#1a1a2e}
.kl{font-size:10px;color:#666;text-transform:uppercase}
.kc{font-size:11px}
.up{color:#28a745}.dn{color:#dc3545}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:12px}
th{background:#1a1a2e;color:#fff;padding:8px;text-align:left}
td{padding:6px 8px;border-bottom:1px solid #eee}
tr:nth-child(even){background:#f8f9fa}
.s{margin-top:24px}
.st{font-size:14px;font-weight:bold;color:#1a1a2e;border-bottom:2px solid #4361ee;padding-bottom:4px;margin-bottom:8px}
.ft{margin-top:24px;padding-top:12px;border-top:1px solid #eee;font-size:11px;color:#999;text-align:center}
.fc{border-left:3px solid #ff9800;background:#fff8e1;padding:8px;margin:4px 0;font-size:12px}
</style></head><body><div class="c">
<h1>[your store name]</h1>
<p class="sub">Sales Actuals &amp; Forecast &mdash; '' || :v_max_date || ''</p>
<div class="kr">
<div class="k"><div class="kl">Projected Month</div><div class="kv">$'' || TO_CHAR(:v_proj_month, ''999,990'') || ''</div><div class="kc">actuals + forecast</div></div>
<div class="k"><div class="kl">MTD Revenue</div><div class="kv">$'' || TO_CHAR(:v_mtd_rev, ''999,990'') || ''</div><div class="kc">'' || DAYOFMONTH(:v_max_date) || '' days</div></div>
<div class="k"><div class="kl">Last Day</div><div class="kv">$'' || TO_CHAR(:v_today_rev, ''9,999,990'') || ''</div><div class="kc '' || CASE WHEN :v_pct_change >= 0 THEN ''up'' ELSE ''dn'' END || ''">'' || CASE WHEN :v_pct_change >= 0 THEN ''&#9650;+'' ELSE ''&#9660;'' END || :v_pct_change || ''%</div></div>
<div class="k"><div class="kl">Tomorrow Forecast</div><div class="kv">$'' || TO_CHAR(:v_today_forecast, ''9,999,990'') || ''</div><div class="kc">predicted</div></div>
<div class="k"><div class="kl">Next 7 Days</div><div class="kv">$'' || TO_CHAR(:v_next7_forecast, ''999,990'') || ''</div><div class="kc">forecast</div></div>
</div>
<div class="kr">
<div class="k"><div class="kl">7-Day Avg</div><div class="kv">$'' || TO_CHAR(:v_avg_7d, ''9,999,990'') || ''</div></div>
<div class="k"><div class="kl">14-Day Avg</div><div class="kv">$'' || TO_CHAR(:v_avg_14d, ''9,999,990'') || ''</div></div>
<div class="k"><div class="kl">21-Day Avg</div><div class="kv">$'' || TO_CHAR(:v_avg_21d, ''9,999,990'') || ''</div></div>
</div>
<div class="s"><div class="st">7-Day Forecast</div>
<table><tr><th>Date</th><th>Day</th><th>Forecast Revenue</th><th>Forecast Orders</th><th>Model</th></tr>''
|| :v_forecast_rows || ''</table></div>
<div class="s"><div class="st">Last 14 Days Actuals</div>
<table><tr><th>Date</th><th>Day</th><th>Revenue</th><th>Orders</th><th>Avg Ticket</th><th>Weather</th><th>High</th></tr>''
|| :v_table_rows || ''</table></div>
<div class="s"><div class="st">Day-of-Week Averages (8 wks)</div>
<table><tr><th>Day</th><th>Avg Revenue</th><th>Avg Orders</th></tr>''
|| :v_dow_rows || ''</table></div>
<div class="s"><div class="st">Pipeline Freshness</div>
<table><tr><th>Layer</th><th>Last Updated</th></tr>
<tr><td>Bronze Orders</td><td>'' || :v_max_date || ''</td></tr>
<tr><td>Gold Summary</td><td>'' || :v_max_date || ''</td></tr>
<tr><td>Forecast Model</td><td>'' || CURRENT_DATE() || ''</td></tr>
</table></div>
<div class="ft">Generated by Snowflake &bull; [your store name] Analytics<br>'' || CURRENT_TIMESTAMP()::VARCHAR || ''</div>
</div></body></html>'';
  
  CALL SYSTEM$SEND_SNOWFLAKE_NOTIFICATION(
    SNOWFLAKE.NOTIFICATION.TEXT_HTML(:v_html),
    SNOWFLAKE.NOTIFICATION.EMAIL_INTEGRATION_CONFIG(''TSP_EMAIL_REPORTS'', ''[your store name] - Daily Sales Report'', ARRAY_CONSTRUCT(''YOUR_EMAIL''))
  );
  
  RETURN ''Sales report sent for '' || :v_max_date || '' (with forecasts)'';
END;
';