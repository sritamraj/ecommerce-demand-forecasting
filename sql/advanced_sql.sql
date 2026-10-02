-- Q9. Which products rank highest within each category?
WITH product_sales AS (
    SELECT
        product_id,
        category,
        SUM(quantity) AS total_units,
        ROUND(SUM(quantity * price), 2) AS total_revenue
    FROM sales
    GROUP BY product_id, category
)
SELECT
    product_id,
    category,
    total_units,
    total_revenue,
    RANK() OVER (
        PARTITION BY category
        ORDER BY total_units DESC
    ) AS category_rank
FROM product_sales
ORDER BY category, category_rank, product_id;


-- Q10. What is the 7-day rolling demand for each product?
WITH daily_demand AS (
    SELECT
        date,
        product_id,
        quantity AS daily_units
    FROM daily_product_panel
)
SELECT
    date,
    product_id,
    daily_units,
    ROUND(
        AVG(daily_units) OVER (
            PARTITION BY product_id
            ORDER BY date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS rolling_7d_avg_units
FROM daily_demand
ORDER BY product_id, date;


-- Q11. What is each product's month-over-month demand growth?
WITH monthly_product AS (
    SELECT
        strftime('%Y-%m', date) AS year_month,
        product_id,
        category,
        SUM(quantity) AS total_units
    FROM sales
    GROUP BY year_month, product_id, category
),
with_previous AS (
    SELECT
        year_month,
        product_id,
        category,
        total_units,
        LAG(total_units) OVER (
            PARTITION BY product_id
            ORDER BY year_month
        ) AS prev_month_units
    FROM monthly_product
)
SELECT
    year_month,
    product_id,
    category,
    total_units,
    prev_month_units,
    ROUND(
        100.0 * (total_units - prev_month_units)
        / NULLIF(prev_month_units, 0),
        2
    ) AS mom_growth_pct
FROM with_previous
ORDER BY product_id, year_month;