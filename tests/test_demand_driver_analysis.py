from pathlib import Path

import pandas as pd

from src.demand_driver_analysis import (
    forecast_error_analysis,
    grouped_demand_patterns,
    load_data,
    numeric_associations,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def test_numeric_demand_associations():
    features, _ = load_data()

    result = numeric_associations(features)

    assert result.shape[0] == 7
    assert set(
        [
            "variable",
            "pearson_correlation_with_demand",
            "absolute_correlation",
        ]
    ).issubset(result.columns)

    assert result["pearson_correlation_with_demand"].notna().all()
    assert result["absolute_correlation"].notna().all()


def test_grouped_demand_patterns():
    features, _ = load_data()

    result = grouped_demand_patterns(features)

    assert result.shape[0] == 9

    assert set(
        [
            "segment",
            "segment_value",
            "avg_demand",
            "total_units",
            "observations",
        ]
    ).issubset(result.columns)

    assert result["avg_demand"].notna().all()
    assert result["total_units"].notna().all()


def test_forecast_error_analysis():
    features, predictions = load_data()

    (
        merged,
        product_error,
        category_error,
        condition_error,
    ) = forecast_error_analysis(
        features,
        predictions,
    )

    # Final test contains 40 products x 60 days.
    assert len(merged) == 2400
    assert merged["product_id"].nunique() == 40
    assert merged["date"].nunique() == 60

    # Every product should have exactly 60 observations.
    product_counts = merged.groupby("product_id").size()
    assert (product_counts == 60).all()

    # Five product categories.
    assert category_error["category"].nunique() == 5

    # Error metrics must exist and be finite.
    for column in [
        "mae",
        "bias",
        "rmse",
        "p90_absolute_error",
        "p95_absolute_error",
    ]:
        assert product_error[column].notna().all()
        assert category_error[column].notna().all()

    assert (product_error["mae"] >= 0).all()
    assert (category_error["mae"] >= 0).all()


def test_generated_demand_driver_reports_exist():
    expected_files = [
        "demand_driver_associations.csv",
        "demand_driver_segments.csv",
        "forecast_error_by_product.csv",
        "forecast_error_by_category.csv",
        "forecast_error_by_condition.csv",
        "final_test_predictions_with_error_analysis.csv",
    ]

    for filename in expected_files:
        path = REPORTS / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_final_test_report_has_expected_rows():
    path = REPORTS / "final_test_predictions_with_error_analysis.csv"

    df = pd.read_csv(path)

    assert len(df) == 2400
    assert df["product_id"].nunique() == 40

    required_columns = [
        "date",
        "product_id",
        "y_true",
        "y_pred",
        "error",
        "absolute_error",
        "squared_error",
    ]

    for column in required_columns:
        assert column in df.columns

    assert df["y_true"].notna().all()
    assert df["y_pred"].notna().all()
    assert df["error"].notna().all()
    assert df["absolute_error"].notna().all()