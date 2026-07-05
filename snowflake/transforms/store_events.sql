-- REFERENCE.STORE_EVENTS — hand-curated special events / closures (mirrors Databricks).
-- Manually maintained: INSERT/UPDATE rows as events are planned. Feeds forecast training weights.
CREATE TABLE IF NOT EXISTS THREE_SISTERS_ANALYTICS.REFERENCE.STORE_EVENTS (
  EVENT_DATE DATE,
  EVENT_NAME STRING,
  EVENT_TYPE STRING,   -- PLANNED_EVENT | REVENUE_DISTORTION | ORGANIC_EVENT | FUTURE_CLOSURE
  LOWER_WINDOW NUMBER, -- Prophet-style window (days before) affected
  UPPER_WINDOW NUMBER, -- days after affected
  NOTES STRING,
  IS_ACTIVE BOOLEAN,
  CREATED_AT TIMESTAMP_NTZ,
  UPDATED_AT TIMESTAMP_NTZ
) COMMENT='Special events & closures. Manually maintained. Drives forecast training weights / holiday flags.';

-- Seed with the current Databricks rows (idempotent: clear + reload the curated set)
DELETE FROM THREE_SISTERS_ANALYTICS.REFERENCE.STORE_EVENTS;
INSERT INTO THREE_SISTERS_ANALYTICS.REFERENCE.STORE_EVENTS
  (EVENT_DATE, EVENT_NAME, EVENT_TYPE, LOWER_WINDOW, UPPER_WINDOW, NOTES, IS_ACTIVE, CREATED_AT, UPDATED_AT)
VALUES
  ('2025-12-11','December event / unknown','ORGANIC_EVENT',0,0,'Anomalous revenue spike in December. Cause unknown. Excluded from training as unpredictable one-off.',TRUE, '2026-03-27T21:26:12'::TIMESTAMP_NTZ,'2026-03-27T21:29:24'::TIMESTAMP_NTZ),
  ('2025-12-13','[your city] Village Town Stroll','PLANNED_EVENT',-1,0,'Annual holiday event. High foot traffic, many small tickets. Real demand signal — keep in training.',TRUE,'2026-03-27T21:26:12'::TIMESTAMP_NTZ,'2026-03-27T21:29:24'::TIMESTAMP_NTZ),
  ('2026-03-02','Databricks Wine Tasting Event','REVENUE_DISTORTION',0,0,'Single large catering ticket inflated revenue. Excluded from training.',FALSE,'2026-03-27T21:26:12'::TIMESTAMP_NTZ,'2026-03-27T21:31:25'::TIMESTAMP_NTZ),
  ('2026-03-03','Databricks Wine Tasting Event','REVENUE_DISTORTION',0,0,'Single large catering ticket inflated revenue. Excluded from training.',TRUE,'2026-03-27T21:29:24'::TIMESTAMP_NTZ,'2026-03-27T21:29:24'::TIMESTAMP_NTZ),
  ('2026-11-26','Thanksgiving 2026','FUTURE_CLOSURE',0,0,'Store closed. Forecast rows labeled likely_closed.',TRUE,'2026-03-27T21:26:12'::TIMESTAMP_NTZ,'2026-03-27T21:29:24'::TIMESTAMP_NTZ),
  ('2026-12-25','Christmas Day 2026','FUTURE_CLOSURE',0,0,'Store closed. Forecast rows labeled likely_closed.',TRUE,'2026-03-27T21:26:12'::TIMESTAMP_NTZ,'2026-03-27T21:29:24'::TIMESTAMP_NTZ);

SELECT COUNT(*) AS event_ct, COUNT_IF(IS_ACTIVE) AS active_ct FROM THREE_SISTERS_ANALYTICS.REFERENCE.STORE_EVENTS;
