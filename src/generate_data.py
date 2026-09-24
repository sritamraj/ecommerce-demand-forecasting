"""
Generates a synthetic but realistic e-commerce sales dataset.

Why synthetic data:
This environment has no internet access, so a Kaggle/public dataset
cannot be downloaded directly. Instead we generate a dataset with the
same statistical properties a real retail dataset would have: trend,
weekly seasonality, yearly seasonality, promotion effects, price
elasticity, holiday spikes, and realistic noise -- plus deliberately
injected messiness (missing values, duplicates, negative quantities,
a few bad dates) so the data-quality step in the pipeline has real
issues to catch. Swap in a real dataset (e.g. Kaggle "Store Item Demand
Forecasting Challenge", or a company export with the same columns) by
replacing this script's output at data/sales_raw.csv -- nothing
downstream needs to change.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

START_DATE = "2023-01-01"
END_DATE = "2025-06-30"
dates = pd.date_range(START_DATE, END_DATE, freq="D")
n_days = len(dates)

CATEGORIES = ["Electronics", "Home & Kitchen", "Apparel", "Beauty", "Sports"]
N_PRODUCTS = 40
REGIONS = ["North", "South", "East", "West"]
STORES = [f"STORE_{i}" for i in range(1, 6)]

# US-style holiday-ish spike dates (kept simple/deterministic, not a real calendar lib
# to avoid extra dependencies)
HOLIDAYS = pd.to_datetime([
    "2023-01-01", "2023-07-04", "2023-11-24", "2023-12-25",
    "2024-01-01", "2024-07-04", "2024-11-29", "2024-12-25",
    "2025-01-01", "2025-07-04",
])

products = []
for i in range(1, N_PRODUCTS + 1):
    category = CATEGORIES[(i - 1) % len(CATEGORIES)]
    base_demand = rng.uniform(15, 120)          # average daily units
    trend_slope = rng.normal(0.01, 0.02)         # slow drift, some products declining
    yearly_amp = rng.uniform(0.1, 0.5)           # seasonality strength
    weekly_amp = rng.uniform(0.05, 0.35)
    price = round(rng.uniform(8, 250), 2)
    noise_scale = rng.uniform(0.12, 0.30)
    products.append({
        "product_id": f"P{i:03d}",
        "category": category,
        "base_demand": base_demand,
        "trend_slope": trend_slope,
        "yearly_amp": yearly_amp,
        "weekly_amp": weekly_amp,
        "price": price,
        "noise_scale": noise_scale,
    })
products_df = pd.DataFrame(products)

rows = []
for _, p in products_df.iterrows():
    t = np.arange(n_days)
    day_of_week = dates.dayofweek.values                 # Mon=0 ... Sun=6
    day_of_year = dates.dayofyear.values

    trend = p["base_demand"] + p["trend_slope"] * t
    yearly_season = 1 + p["yearly_amp"] * np.sin(2 * np.pi * day_of_year / 365.25)
    # weekend lift for most categories, weekday lift for a few (Electronics: paydays)
    weekly_pattern = np.where(day_of_week >= 5, 1 + p["weekly_amp"], 1 - p["weekly_amp"] * 0.4)

    is_holiday = dates.isin(HOLIDAYS).astype(float)
    holiday_boost = 1 + is_holiday * rng.uniform(0.6, 1.8)

    # promotions: random ~6% of days, boosts demand, comes with a discount
    promo_flag = (rng.random(n_days) < 0.06).astype(int)
    discount = np.where(promo_flag == 1, rng.uniform(0.05, 0.30, n_days).round(2), 0.0)
    promo_boost = 1 + promo_flag * (discount * 2.5)

    mean_demand = trend * yearly_season * weekly_pattern * holiday_boost * promo_boost
    mean_demand = np.clip(mean_demand, 1, None)

    noise = rng.normal(0, p["noise_scale"], n_days)
    quantity = np.round(mean_demand * (1 + noise)).astype(int)
    quantity = np.clip(quantity, 0, None)

    # occasional random spikes (viral/stockup events) -- ~0.5% of days
    spike_days = rng.random(n_days) < 0.005
    quantity = np.where(spike_days, quantity + rng.integers(30, 150, n_days), quantity)

    region = rng.choice(REGIONS, size=n_days)
    store = rng.choice(STORES, size=n_days)
    unit_price = p["price"] * (1 - discount)

    df = pd.DataFrame({
        "date": dates,
        "product_id": p["product_id"],
        "category": p["category"],
        "quantity": quantity,
        "price": unit_price.round(2),
        "promotion": promo_flag,
        "discount": discount,
        "store": store,
        "region": region,
        "holiday": is_holiday.astype(int),
    })
    rows.append(df)

sales = pd.concat(rows, ignore_index=True)

# ---------------------------------------------------------------
# Inject realistic messiness for the data-quality step to discover
# ---------------------------------------------------------------
sales = sales.sample(frac=1.0, random_state=7).reset_index(drop=True)

n = len(sales)
idx = rng.permutation(n)

# 1) missing quantity / price (~0.4%)
missing_qty_idx = idx[: int(0.002 * n)]
missing_price_idx = idx[int(0.002 * n): int(0.004 * n)]
sales.loc[missing_qty_idx, "quantity"] = np.nan
sales.loc[missing_price_idx, "price"] = np.nan

# 2) a handful of negative values (data entry errors -- returns miscoded)
neg_idx = idx[int(0.004 * n): int(0.0045 * n)]
sales.loc[neg_idx, "quantity"] = -sales.loc[neg_idx, "quantity"].abs() - 1

neg_price_idx = idx[int(0.0045 * n): int(0.005 * n)]
sales.loc[neg_price_idx, "price"] = -sales.loc[neg_price_idx, "price"].abs()

# 3) exact duplicate rows (~0.15%)
dup_sample = sales.sample(frac=0.0015, random_state=11)
sales = pd.concat([sales, dup_sample], ignore_index=True)

# 4) a few invalid / future dates as raw strings would look in a CSV export
sales["date"] = sales["date"].dt.strftime("%Y-%m-%d").astype(object)
bad_date_idx = rng.choice(sales.index, size=25, replace=False)
sales.loc[bad_date_idx[:10], "date"] = "2025-13-40"          # invalid date
sales.loc[bad_date_idx[10:20], "date"] = "2026-01-15"        # future date beyond dataset window
sales.loc[bad_date_idx[20:], "date"] = None                  # missing date

sales.to_csv("/home/claude/ecommerce-demand-forecasting/data/sales_raw.csv", index=False)
print("Rows:", len(sales))
print(sales.head(10).to_string())
print("\nDtypes:\n", sales.dtypes)
