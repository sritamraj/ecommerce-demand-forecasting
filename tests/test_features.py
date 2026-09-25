import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, add_features, build_panel


def test_build_panel_creates_complete_product_date_grid():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-03", "2025-01-01", "2025-01-03"]
            ),
            "product_id": ["P1", "P1", "P2", "P2"],
            "category": ["A", "A", "B", "B"],
            "quantity": [10, 30, 20, 40],
            "price": [100, 100, 200, 200],
            "promotion": [0, 0, 1, 1],
            "discount": [0, 0, 10, 10],
            "holiday": [0, 0, 0, 0],
        }
    )

    panel = build_panel(raw)

    assert len(panel) == 6
    assert panel["product_id"].nunique() == 2
    assert panel["date"].nunique() == 3

    missing_day = panel[
        (panel["product_id"] == "P1")
        & (panel["date"] == pd.Timestamp("2025-01-02"))
    ].iloc[0]

    assert missing_day["quantity"] == 0


def test_add_features_uses_only_prior_target_values():
    dates = pd.date_range("2025-01-01", periods=35, freq="D")

    df = pd.DataFrame(
        {
            "date": dates,
            "product_id": ["P1"] * len(dates),
            "category": ["A"] * len(dates),
            "quantity": np.arange(1, len(dates) + 1, dtype=float),
            "price": [100.0] * len(dates),
            "promotion": [0] * len(dates),
            "discount": [0.0] * len(dates),
            "holiday": [0] * len(dates),
        }
    )

    featured = add_features(df)

    row = featured.iloc[28]

    # At day 29, lag_1 must equal day 28's target,
    # lag_7 must equal day 22's target, and lag_14 must
    # equal day 15's target.
    assert row["lag_1"] == 28
    assert row["lag_7"] == 22
    assert row["lag_14"] == 15
    assert row["lag_28"] == 1

    # The rolling mean is based on shifted targets, so the
    # current target (29) must not enter the 7-day window.
    expected_rolling_7 = np.mean(np.arange(22, 29))
    assert np.isclose(row["rolling_mean_7"], expected_rolling_7)

    # The current target is never one of the model features.
    assert "quantity" not in FEATURE_COLUMNS


def test_feature_columns_are_present_after_feature_engineering():
    dates = pd.date_range("2025-01-01", periods=40, freq="D")

    df = pd.DataFrame(
        {
            "date": dates,
            "product_id": ["P1"] * len(dates),
            "category": ["A"] * len(dates),
            "quantity": np.arange(1, len(dates) + 1, dtype=float),
            "price": [100.0] * len(dates),
            "promotion": [0] * len(dates),
            "discount": [0.0] * len(dates),
            "holiday": [0] * len(dates),
        }
    )

    featured = add_features(df)

    assert set(FEATURE_COLUMNS).issubset(featured.columns)