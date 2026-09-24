"""
Feature engineering for daily product-level demand forecasting.

LEAKAGE RULE (read this before touching this file):
Every feature for date T must be computable using only information
that existed at or before date T. Lags and rolling windows are
therefore always shifted by at least 1 day before any window
aggregation is taken, and calendar features use only the date itself.
Never add a feature derived from `quantity` on a date >= the row's
own date.
"""
import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per (product_id, date): daily demand + same-day
    business attributes (price/promotion/discount/holiday), with every
    product_id x date combination present (missing days filled with 0
    demand) so lag/rolling windows are computed on a complete calendar."""
    daily = (
        df.groupby(["product_id", "category", "date"])
        .agg(
            quantity=("quantity", "sum"),
            price=("price", "mean"),
            promotion=("promotion", "max"),
            discount=("discount", "mean"),
            holiday=("holiday", "max"),
        )
        .reset_index()
    )

    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    panels = []
    for pid, g in daily.groupby("product_id"):
        g = g.set_index("date").reindex(full_dates)
        g["product_id"] = pid
        g["category"] = g["category"].ffill().bfill()
        g["quantity"] = g["quantity"].fillna(0)
        g["price"] = g["price"].ffill().bfill()
        g["promotion"] = g["promotion"].fillna(0)
        g["discount"] = g["discount"].fillna(0)
        g["holiday"] = g["holiday"].fillna(0)
        g.index.name = "date"
        panels.append(g.reset_index())

    panel = pd.concat(panels, ignore_index=True)
    return panel.sort_values(["product_id", "date"]).reset_index(drop=True)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Adds lag, rolling, and calendar features. Operates per-product so
    one product's history never leaks into another's lag features."""
    out = []
    for pid, g in panel.groupby("product_id"):
        g = g.sort_values("date").copy()

        for lag in LAGS:
            g[f"lag_{lag}"] = g["quantity"].shift(lag)

        # rolling stats computed on lag_1 (i.e. shifted by 1 day first) so the
        # window for predicting day T never includes day T's own demand
        shifted = g["quantity"].shift(1)
        for w in ROLLING_WINDOWS:
            g[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
            g[f"rolling_std_{w}"] = shifted.rolling(w).std()

        g["day_of_week"] = g["date"].dt.dayofweek
        g["month"] = g["date"].dt.month
        g["week_of_year"] = g["date"].dt.isocalendar().week.astype(int)
        g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)

        out.append(g)

    result = pd.concat(out, ignore_index=True)
    return result.sort_values(["product_id", "date"]).reset_index(drop=True)


FEATURE_COLUMNS = (
    [f"lag_{l}" for l in LAGS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + ["day_of_week", "month", "week_of_year", "is_weekend",
       "price", "promotion", "discount", "holiday"]
)
TARGET_COLUMN = "quantity"
