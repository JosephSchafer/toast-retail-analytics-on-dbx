# Parity Baseline - Databricks Gold (authoritative), captured 2026-07-01

Snapshot of Databricks `YOUR_CATALOG.gold` used to validate Snowflake replication.
Databricks is the source of truth; Snowflake numbers should converge to these (net_revenue delta < 1%/day).

## daily_sales_summary (open days only, net_revenue > 0)
| metric | value |
|---|---|
| row_count | 266 |
| total_net_revenue | 399,170.08 |
| total_gross_revenue | 388,545.85 |
| total_orders | 17,041 |
| date range | 2025-09-18 → 2026-06-30 |

> Note: net_revenue > gross here because gross excludes tax while net includes some components differently
> per NB4 definitions - treat these two totals as fixed fingerprints, not a derivation to re-check.

## Per-day net_revenue & orders - June 2026 (the reconciliation window)
| business_date | net_revenue | order_count |
|---|---:|---:|
| 2026-06-01 | 1523.07 | 67 |
| 2026-06-02 | 1886.17 | 102 |
| 2026-06-03 | 1623.42 | 78 |
| 2026-06-04 | 1992.07 | 89 |
| 2026-06-05 | 2648.12 | 120 |
| 2026-06-06 | 3338.72 | 130 |
| 2026-06-07 | 3986.87 | 151 |
| 2026-06-08 | 1416.71 | 87 |
| 2026-06-09 | 2328.70 | 96 |
| 2026-06-10 | 2056.31 | 106 |
| 2026-06-11 | 2314.70 | 105 |
| 2026-06-12 | 2842.90 | 110 |
| 2026-06-13 | 3534.81 | 151 |
| 2026-06-14 | 3190.24 | 144 |
| 2026-06-15 | 1850.73 | 89 |
| 2026-06-16 | 2033.06 | 106 |
| 2026-06-17 | 2548.53 | 96 |
| 2026-06-18 | 2318.19 | 96 |
| 2026-06-19 | 3590.71 | 163 |
| 2026-06-20 | 3434.72 | 151 |
| 2026-06-21 | 3862.15 | 177 |
| 2026-06-22 | 1473.06 | 70 |
| 2026-06-23 | 2011.02 | 99 |
| 2026-06-24 | 2386.68 | 102 |
| 2026-06-25 | 2985.06 | 138 |
| 2026-06-26 | 2955.02 | 130 |
| 2026-06-27 | 3090.80 | 120 |
| 2026-06-28 | 4581.37 | 173 |
| 2026-06-29 | 2800.84 | 151 |
| 2026-06-30 | 1830.25 | 100 |

## Row-count fingerprints (from earlier audit, 2026-07-01)
| table | Databricks rows |
|---|---:|
| bronze.toast_orders_raw | 17,164 |
| silver_sales.orders_silver | 17,164 |
| silver_sales.order_items_silver | 41,889 |
| gold.daily_sales_summary (all rows) | 286 |
| gold.daily_sales_by_category | 5,459 |
| gold.weather_hourly (bronze) | 9,120 |
