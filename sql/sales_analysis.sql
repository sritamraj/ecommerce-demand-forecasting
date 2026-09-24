-- ============================================================
-- E-Commerce Demand Forecasting -- SQL Business Analysis
-- ============================================================
-- Table: sales(date, product_id, category, quantity, price,
--              promotion, discount, store, region, holiday)
-- This is the cleaned table produced by notebooks/01_data_quality.ipynb.
--
-- Dialect note: written and tested against SQLite (via
-- src/run_sql_analysis.py) for a zero-setup, file-based demo.
-- Portable to Postgres/Redshift/MySQL with trivial edits:
--   * strftime('%Y-%m', date)  ->  TO_CHAR(date, 'YYYY-MM')      (Postgres/Redshift)
--   * strftime('%w', date)     ->  EXTRACT(DOW FROM date)         (Postgres/Redshift)
-- ============================================================


-- Q1. What is total sales (revenue and units) by month?
SELECT
    strftime('%Y-%m', date)                AS year_month,
    SUM(quantity)                          AS total_units,
    ROUND(SUM(quantity * price), 2)        AS total_revenue
FROM sales
GROUP BY year_month
ORDER BY year_month;


-- Q2. What are the top products (by units sold)?
SELECT
    product_id,
    category,
    SUM(quantity)                          AS total_units,
    ROUND(SUM(quantity * price), 2)        AS total_revenue
FROM sales
GROUP BY product_id, category
ORDER BY total_units DESC
LIMIT 10;


-- Q3. What are the top categories (by revenue)?
SELECT
    category,
    SUM(quantity)                          AS total_units,
    ROUND(SUM(quantity * price), 2)        AS total_revenue,
    COUNT(DISTINCT product_id)             AS n_products
FROM sales
GROUP BY category
ORDER BY total_revenue DESC;


-- Q4. Which products have the most volatile demand?
-- (coefficient of variation = stdev / mean of daily units, higher = more volatile)
SELECT
    product_id,
    category,
    ROUND(AVG(quantity), 2)                AS avg_daily_units,
    ROUND(
        (
            SUM(quantity * quantity) * 1.0 / COUNT(*)
            - (SUM(quantity) * 1.0 / COUNT(*)) * (SUM(quantity) * 1.0 / COUNT(*))
        ), 2
    )                                       AS variance_daily_units
FROM sales
GROUP BY product_id, category
ORDER BY variance_daily_units DESC
LIMIT 10;


-- Q5. What is average daily demand by product?
SELECT
    product_id,
    category,
    ROUND(AVG(quantity), 2)                AS avg_daily_units,
    MIN(quantity)                          AS min_daily_units,
    MAX(quantity)                          AS max_daily_units
FROM sales
GROUP BY product_id, category
ORDER BY avg_daily_units DESC
LIMIT 15;


-- Q6. What is the monthly growth rate (month-over-month, total units)?
WITH monthly AS (
    SELECT
        strftime('%Y-%m', date)            AS year_month,
        SUM(quantity)                      AS total_units
    FROM sales
    GROUP BY year_month
)
SELECT
    year_month,
    total_units,
    LAG(total_units) OVER (ORDER BY year_month)  AS prev_month_units,
    ROUND(
        100.0 * (total_units - LAG(total_units) OVER (ORDER BY year_month))
        / LAG(total_units) OVER (ORDER BY year_month), 2
    )                                              AS mom_growth_pct
FROM monthly
ORDER BY year_month;


-- Q7. What does demand look like by weekday?
SELECT
    CASE CAST(strftime('%w', date) AS INTEGER)
        WHEN 0 THEN 'Sunday'    WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'   WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'  WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END                                     AS weekday,
    CAST(strftime('%w', date) AS INTEGER)   AS weekday_num,
    ROUND(AVG(quantity), 2)                 AS avg_units,
    SUM(quantity)                           AS total_units
FROM sales
GROUP BY weekday, weekday_num
ORDER BY weekday_num;


-- Q8. What products have declining demand?
-- Compare average daily demand in the most recent 90 days of data
-- against the prior 90-day window; negative delta = declining.
WITH bounds AS (
    SELECT MAX(date) AS max_date FROM sales
),
recent AS (
    SELECT product_id, AVG(quantity) AS avg_recent
    FROM sales, bounds
    WHERE date > date(bounds.max_date, '-90 day')
    GROUP BY product_id
),
prior AS (
    SELECT product_id, AVG(quantity) AS avg_prior
    FROM sales, bounds
    WHERE date <= date(bounds.max_date, '-90 day')
      AND date >  date(bounds.max_date, '-180 day')
    GROUP BY product_id
)
SELECT
    recent.product_id,
    ROUND(prior.avg_prior, 2)   AS avg_daily_prior_90d,
    ROUND(recent.avg_recent, 2) AS avg_daily_recent_90d,
    ROUND(recent.avg_recent - prior.avg_prior, 2) AS change,
    ROUND(100.0 * (recent.avg_recent - prior.avg_prior) / prior.avg_prior, 2) AS pct_change
FROM recent
JOIN prior ON recent.product_id = prior.product_id
ORDER BY pct_change ASC
LIMIT 10;
