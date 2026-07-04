CREATE OR REPLACE PROCEDURE "INGEST_TOAST_ORDERS"("RUN_MODE" VARCHAR)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python','requests')
HANDLER = 'run'
EXTERNAL_ACCESS_INTEGRATIONS = (TSP_EXTERNAL_ACCESS)
SECRETS = ('restaurant_guid'=THREE_SISTERS_ANALYTICS.BRONZE.TOAST_RESTAURANT_GUID,'client_secret'=THREE_SISTERS_ANALYTICS.BRONZE.TOAST_CLIENT_SECRET,'client_id'=THREE_SISTERS_ANALYTICS.BRONZE.TOAST_CLIENT_ID)
EXECUTE AS CALLER
AS '
import requests
import json
import time
import uuid
import base64
from datetime import datetime, timedelta, date
from snowflake.snowpark import Session
import _snowflake

AUTH_URL = "https://ws.toasttab.com/authentication/v1/authentication/login"
ORDERS_URL = "https://ws.toasttab.com/orders/v2/ordersBulk"
PAGE_SIZE = 100
BACKFILL_START = "2025-09-01"
THROTTLE_SECONDS = 0.25

def get_token(client_id, client_secret):
    resp = requests.post(AUTH_URL, json={
        "clientId": client_id,
        "clientSecret": client_secret,
        "userAccessType": "TOAST_MACHINE_CLIENT"
    }, headers={"Content-Type": "application/json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    token_obj = data.get("token", data)
    return token_obj.get("accessToken", "")

def fetch_orders_for_date(token, restaurant_guid, d):
    headers = {
        "Authorization": f"Bearer {token}",
        "Toast-Restaurant-External-ID": restaurant_guid
    }
    start_ts = f"{d.isoformat()}T00:00:00.000+0000"
    end_ts = f"{d.isoformat()}T23:59:59.999+0000"
    all_orders = []
    page = 1
    while True:
        params = {"startDate": start_ts, "endDate": end_ts, "pageSize": str(PAGE_SIZE), "page": str(page)}
        resp = requests.get(ORDERS_URL, headers=headers, params=params, timeout=60)
        if resp.status_code == 401:
            return None
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        orders = resp.json()
        if not orders:
            break
        all_orders.extend(orders)
        if len(orders) < PAGE_SIZE:
            break
        page += 1
        time.sleep(THROTTLE_SECONDS)
    return all_orders

def run(session: Session, run_mode: str) -> str:
    run_mode = run_mode.lower().strip()
    client_id = _snowflake.get_generic_secret_string(''client_id'')
    client_secret = _snowflake.get_generic_secret_string(''client_secret'')
    restaurant_guid = _snowflake.get_generic_secret_string(''restaurant_guid'')

    token = get_token(client_id, client_secret)
    if not token:
        return "ERROR: Failed to authenticate with Toast API"

    if run_mode == ''backfill'':
        start_date = date.fromisoformat(BACKFILL_START)
    else:
        max_row = session.sql(
            "SELECT COALESCE(MAX(SOURCE_DATE), ''2025-09-01''::DATE) AS MD FROM THREE_SISTERS_ANALYTICS.BRONZE.TOAST_ORDERS_RAW"
        ).collect()
        start_date = max_row[0][''MD'']
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        # Tiered lookback (2026-07-03): re-pull recent days so late-arriving orders AND
        # edits (discounts/voids/tips applied after first pull) reconcile. run_mode carries
        # the depth: "incremental"/"daily"=3d, "weekly"=14d, "quarterly"=90d. MERGE upsert
        # below makes re-pulls fully idempotent (updates existing rows, not just inserts).
        LOOKBACK_MAP = {"incremental": 3, "daily": 3, "weekly": 14, "quarterly": 90}
        lookback_days = LOOKBACK_MAP.get(run_mode, 3)
        start_date = start_date - timedelta(days=lookback_days)

    # Include today: today''s in-progress business day should refresh each run.
    end_date = date.today()
    if start_date > end_date:
        return f"No new dates to ingest. Last: {start_date}"

    batch_id = str(uuid.uuid4())[:8]
    total_orders = 0
    total_days = 0
    failed_dates = []
    token_time = time.time()

    current_date = start_date
    while current_date <= end_date:
        if time.time() - token_time > 3000:
            token = get_token(client_id, client_secret)
            token_time = time.time()

        try:
            orders = fetch_orders_for_date(token, restaurant_guid, current_date)
            if orders is None:
                token = get_token(client_id, client_secret)
                token_time = time.time()
                orders = fetch_orders_for_date(token, restaurant_guid, current_date)

            if orders:
                # Batch entire day: encode all orders as one JSON array, then FLATTEN
                all_orders_json = json.dumps(orders)
                b64_payload = base64.b64encode(all_orders_json.encode(''utf-8'')).decode(''ascii'')

                # MERGE upsert on ORDER_GUID: inserts new orders AND updates existing ones
                # whose RAW_JSON changed (late discounts/voids/tip edits). This is what makes
                # re-pulled days converge exactly to Toast. Idempotent on re-run.
                merge_sql = f"""
                MERGE INTO THREE_SISTERS_ANALYTICS.BRONZE.TOAST_ORDERS_RAW AS t
                USING (
                    SELECT
                        f.value:guid::STRING              AS ORDER_GUID,
                        f.value                           AS RAW_JSON,
                        ''{current_date.isoformat()}''::DATE AS SOURCE_DATE
                    FROM TABLE(FLATTEN(input => PARSE_JSON(BASE64_DECODE_STRING(''{b64_payload}'')))) f
                ) AS s
                ON t.ORDER_GUID = s.ORDER_GUID
                WHEN MATCHED AND t.RAW_JSON != s.RAW_JSON THEN UPDATE SET
                    t.RAW_JSON = s.RAW_JSON,
                    t.INGESTED_AT = CURRENT_TIMESTAMP(),
                    t.INGESTION_BATCH_ID = ''{batch_id}'',
                    t.SOURCE_DATE = s.SOURCE_DATE
                WHEN NOT MATCHED THEN INSERT
                    (ORDER_GUID, BUSINESS_DATE, RAW_JSON, INGESTED_AT, INGESTION_BATCH_ID, SOURCE_DATE, RESTAURANT_GUID)
                    VALUES (s.ORDER_GUID, s.SOURCE_DATE, s.RAW_JSON, CURRENT_TIMESTAMP(),
                            ''{batch_id}'', s.SOURCE_DATE, ''{restaurant_guid}'')
                """
                result = session.sql(merge_sql).collect()
                total_orders += len(orders)
            total_days += 1

        except Exception as e:
            failed_dates.append(f"{current_date}: {str(e)[:100]}")

        current_date += timedelta(days=1)
        time.sleep(THROTTLE_SECONDS)

    session.sql(f"""
        MERGE INTO THREE_SISTERS_ANALYTICS.BRONZE.INGESTION_WATERMARK t
        USING (SELECT ''toast_orders'' AS PIPELINE_NAME, ''{end_date.isoformat()}''::DATE AS LAST_SUCCESSFUL_DATE,
               CURRENT_TIMESTAMP() AS LAST_RUN_AT, ''SUCCESS'' AS STATUS, {total_orders} AS ROWS_INGESTED,
               ''Days: {total_days}, Failed: {len(failed_dates)}'' AS NOTES) s
        ON t.PIPELINE_NAME = s.PIPELINE_NAME
        WHEN MATCHED THEN UPDATE SET LAST_SUCCESSFUL_DATE=s.LAST_SUCCESSFUL_DATE, LAST_RUN_AT=s.LAST_RUN_AT, STATUS=s.STATUS, ROWS_INGESTED=s.ROWS_INGESTED, NOTES=s.NOTES
        WHEN NOT MATCHED THEN INSERT VALUES (s.PIPELINE_NAME, s.LAST_SUCCESSFUL_DATE, s.LAST_RUN_AT, s.STATUS, s.ROWS_INGESTED, s.NOTES)
    """).collect()

    result = f"Toast orders ingestion complete: {start_date} to {end_date}. Orders: {total_orders}, Days: {total_days}"
    if failed_dates:
        result += f"\\nFailed ({len(failed_dates)}): " + "; ".join(failed_dates[:5])
    return result
';