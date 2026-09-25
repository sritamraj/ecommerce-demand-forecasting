"""
Feature engineering for daily product-level demand forecasting.

LEAKAGE RULE (read this before touching this file):
Every feature for date T must be computable using only information
that existed at or before date T. Lags and rolling windows are
therefore always shifted by at least 1 day before any window
aggregation is taken, and calendar features use only the date itself.

Same-day business attributes (price, promotion, discount, holiday)
are treated as known/planned inputs available at forecast time.
Never add a feature derived from `quantity` on a date >= the row's
own date.
"""

import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per (product_id, date).

    Creates a complete product-date calendar so lag and rolling
    features are computed on a continuous daily timeline.

    Missing demand is treated as zero. Historical business attributes
    are forward-filled only; future values are never used to fill
    earlier dates.
    """
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

    full_dates = pd.date_range(
        daily["date"].min(),
        daily["date"].max(),
        freq="D",
    )

    panels = []

    for pid, g in daily.groupby("product_id"):
        g = g.set_index("date").reindex(full_dates)

        g["product_id"] = pid

        # Forward-fill only: never use future values to populate
        # earlier dates.
        g["category"] = g["category"].ffill()

        # Missing calendar days represent zero observed demand.
        g["quantity"] = g["quantity"].fillna(0)

        # Historical price only; never backfill from the future.
        g["price"] = g["price"].ffill()

        g["promotion"] = g["promotion"].fillna(0)
        g["discount"] = g["discount"].fillna(0)
        g["holiday"] = g["holiday"].fillna(0)

        g.index.name = "date"
        panels.append(g.reset_index())

    panel = pd.concat(panels, ignore_index=True)

    return panel.sort_values(
        ["product_id", "date"]
    ).reset_index(drop=True)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and calendar features.

    Operations are performed independently per product so one
    product's history cannot leak into another product's features.
    """
    out = []

    for pid, g in panel.groupby("product_id"):
        g = g.sort_values("date").copy()

        # Historical demand lags.
        for lag in LAGS:
            g[f"lag_{lag}"] = g["quantity"].shift(lag)

        # Rolling statistics use lagged demand, so the current day's
        # target is never included in its own features.
        shifted = g["quantity"].shift(1)

        for w in ROLLING_WINDOWS:
            g[f"rolling_mean_{w}"] = shifted.rolling(w).mean()
            g[f"rolling_std_{w}"] = shifted.rolling(w).std()

        # Calendar features are deterministic from the date itself.
        g["day_of_week"] = g["date"].dt.dayofweek
        g["month"] = g["date"].dt.month
        g["week_of_year"] = g["date"].dt.isocalendar().week.astype(int)
        g["is_weekend"] = (g["day_of_week"] >= 5).astype(int)

        out.append(g)

    result = pd.concat(out, ignore_index=True)

    return result.sort_values(
        ["product_id", "date"]
    ).reset_index(drop=True)


FEATURE_COLUMNS = (
    [f"lag_{l}" for l in LAGS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + [
        "day_of_week",
        "month",
        "week_of_year",
        "is_weekend",
        "price",
        "promotion",
        "discount",
        "holiday",
    ]
)

TARGET_COLUMN = "quantity"