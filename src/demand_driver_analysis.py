"""
Demand driver and forecast error analysis.

Purpose:
    Identify observable associations between demand and business variables,
    then segment forecast errors to understand where the model performs
    differently.

Important:
    These analyses are observational. Results are reported as associations
    and predictive patterns, not causal effects.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "model_features.csv"
PREDICTIONS_PATH = ROOT / "reports" / "final_test_predictions.csv"

OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load model features and final-test predictions."""

    features = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["date"],
    )

    return features, predictions


def numeric_associations(df):
    """
    Measure Pearson correlations between demand and numeric business
    variables.

    Correlation is descriptive/associational, not causal.
    """

    columns = [
        "quantity",
        "price",
        "promotion",
        "discount",
        "holiday",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    available = [c for c in columns if c in df.columns]

    corr = (
        df[available]
        .corr(numeric_only=True)["quantity"]
        .drop("quantity")
        .sort_values(
            key=lambda s: s.abs(),
            ascending=False,
        )
        .reset_index()
    )

    corr.columns = [
        "variable",
        "pearson_correlation_with_demand",
    ]

    corr["absolute_correlation"] = (
        corr["pearson_correlation_with_demand"].abs()
    )

    return corr


def grouped_demand_patterns(df):
    """Summarize demand across observable business segments."""

    results = []

    if "promotion" in df.columns:
        promotion = (
            df.groupby("promotion")["quantity"]
            .agg(
                avg_demand="mean",
                total_units="sum",
                observations="count",
            )
            .reset_index()
        )

        promotion["segment"] = "promotion"
        promotion["segment_value"] = promotion["promotion"].astype(str)

        results.append(
            promotion[
                [
                    "segment",
                    "segment_value",
                    "avg_demand",
                    "total_units",
                    "observations",
                ]
            ]
        )

    if "holiday" in df.columns:
        holiday = (
            df.groupby("holiday")["quantity"]
            .agg(
                avg_demand="mean",
                total_units="sum",
                observations="count",
            )
            .reset_index()
        )

        holiday["segment"] = "holiday"
        holiday["segment_value"] = holiday["holiday"].astype(str)

        results.append(
            holiday[
                [
                    "segment",
                    "segment_value",
                    "avg_demand",
                    "total_units",
                    "observations",
                ]
            ]
        )

    if "category" in df.columns:
        category = (
            df.groupby("category")["quantity"]
            .agg(
                avg_demand="mean",
                total_units="sum",
                observations="count",
            )
            .reset_index()
        )

        category["segment"] = "category"
        category["segment_value"] = category["category"].astype(str)

        results.append(
            category[
                [
                    "segment",
                    "segment_value",
                    "avg_demand",
                    "total_units",
                    "observations",
                ]
            ]
        )

    if not results:
        return pd.DataFrame(
            columns=[
                "segment",
                "segment_value",
                "avg_demand",
                "total_units",
                "observations",
            ]
        )

    return pd.concat(
        results,
        ignore_index=True,
    )


def forecast_error_analysis(features, predictions):
    """
    Analyze final-test forecast errors by product, category, and
    observable conditions.

    The final-test prediction file uses:
        quantity   = actual demand
        prediction = model forecast
        error      = actual - prediction
        abs_error  = absolute error
    """

    prediction_columns = [
        "date",
        "product_id",
        "quantity",
        "prediction",
    ]

    missing = [
        column
        for column in prediction_columns
        if column not in predictions.columns
    ]

    if missing:
        raise ValueError(
            "Missing prediction columns: "
            f"{missing}. Expected columns include: "
            f"{prediction_columns}"
        )

    pred = predictions[prediction_columns].copy()

    # Rename into analysis-friendly names internally.
    pred = pred.rename(
        columns={
            "quantity": "y_true",
            "prediction": "y_pred",
        }
    )

    # Forecast error:
    # positive error = model under-forecasted
    # negative error = model over-forecasted
    pred["error"] = pred["y_true"] - pred["y_pred"]

    pred["absolute_error"] = pred["error"].abs()

    pred["squared_error"] = pred["error"] ** 2

    merged = pred.merge(
        features,
        on=["date", "product_id"],
        how="left",
        suffixes=("", "_feature"),
    )

    # ------------------------------------------------------------------
    # Product-level forecast error
    # ------------------------------------------------------------------

    product_error = (
        merged.groupby("product_id")
        .agg(
            observations=("absolute_error", "count"),
            mae=("absolute_error", "mean"),
            bias=("error", "mean"),
            rmse=(
                "squared_error",
                lambda x: np.sqrt(np.mean(x)),
            ),
            p90_absolute_error=(
                "absolute_error",
                lambda x: np.percentile(x, 90),
            ),
            p95_absolute_error=(
                "absolute_error",
                lambda x: np.percentile(x, 95),
            ),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    # ------------------------------------------------------------------
    # Category-level forecast error
    # ------------------------------------------------------------------

    category_error = (
        merged.groupby("category")
        .agg(
            observations=("absolute_error", "count"),
            mae=("absolute_error", "mean"),
            bias=("error", "mean"),
            rmse=(
                "squared_error",
                lambda x: np.sqrt(np.mean(x)),
            ),
            p90_absolute_error=(
                "absolute_error",
                lambda x: np.percentile(x, 90),
            ),
            p95_absolute_error=(
                "absolute_error",
                lambda x: np.percentile(x, 95),
            ),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    # ------------------------------------------------------------------
    # Forecast error by observable business condition
    # ------------------------------------------------------------------

    condition_results = []

    for column in [
        "promotion",
        "holiday",
        "is_weekend",
    ]:
        if column not in merged.columns:
            continue

        grouped = (
            merged.groupby(column)
            .agg(
                mae=("absolute_error", "mean"),
                bias=("error", "mean"),
                observations=("absolute_error", "count"),
            )
            .reset_index()
        )

        grouped["condition"] = column
        grouped["condition_value"] = grouped[column].astype(str)

        condition_results.append(
            grouped[
                [
                    "condition",
                    "condition_value",
                    "mae",
                    "bias",
                    "observations",
                ]
            ]
        )

    if condition_results:
        condition_error = pd.concat(
            condition_results,
            ignore_index=True,
        )
    else:
        condition_error = pd.DataFrame(
            columns=[
                "condition",
                "condition_value",
                "mae",
                "bias",
                "observations",
            ]
        )

    return (
        merged,
        product_error,
        category_error,
        condition_error,
    )


def main():
    print("=" * 80)
    print("DEMAND DRIVER AND FORECAST ERROR ANALYSIS")
    print("=" * 80)

    features, predictions = load_data()

    print(f"Features:      {features.shape}")
    print(f"Predictions:   {predictions.shape}")

    # ------------------------------------------------------------------
    # Demand associations
    # ------------------------------------------------------------------

    print()
    print("-" * 80)
    print("NUMERIC DEMAND ASSOCIATIONS")
    print("-" * 80)

    associations = numeric_associations(features)

    print(associations.to_string(index=False))

    # ------------------------------------------------------------------
    # Grouped demand patterns
    # ------------------------------------------------------------------

    print()
    print("-" * 80)
    print("GROUPED DEMAND PATTERNS")
    print("-" * 80)

    grouped_patterns = grouped_demand_patterns(features)

    print(grouped_patterns.to_string(index=False))

    # ------------------------------------------------------------------
    # Forecast error analysis
    # ------------------------------------------------------------------

    print()
    print("-" * 80)
    print("FINAL-TEST FORECAST ERROR ANALYSIS")
    print("-" * 80)

    (
        merged_predictions,
        product_error,
        category_error,
        condition_error,
    ) = forecast_error_analysis(
        features,
        predictions,
    )

    print()
    print("Highest product-level MAE:")
    print(
        product_error.head(10).to_string(index=False)
    )

    print()
    print("Category-level error:")
    print(
        category_error.to_string(index=False)
    )

    print()
    print("Error by observable condition:")
    print(
        condition_error.to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------------

    associations.to_csv(
        OUTPUT_DIR / "demand_driver_associations.csv",
        index=False,
    )

    grouped_patterns.to_csv(
        OUTPUT_DIR / "demand_driver_segments.csv",
        index=False,
    )

    product_error.to_csv(
        OUTPUT_DIR / "forecast_error_by_product.csv",
        index=False,
    )

    category_error.to_csv(
        OUTPUT_DIR / "forecast_error_by_category.csv",
        index=False,
    )

    condition_error.to_csv(
        OUTPUT_DIR / "forecast_error_by_condition.csv",
        index=False,
    )

    merged_predictions.to_csv(
        OUTPUT_DIR / "final_test_predictions_with_error_analysis.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Completion message
    # ------------------------------------------------------------------

    print()
    print("-" * 80)
    print("SAVED REPORTS")
    print("-" * 80)

    print(
        "reports/demand_driver_associations.csv"
    )
    print(
        "reports/demand_driver_segments.csv"
    )
    print(
        "reports/forecast_error_by_product.csv"
    )
    print(
        "reports/forecast_error_by_category.csv"
    )
    print(
        "reports/forecast_error_by_condition.csv"
    )
    print(
        "reports/final_test_predictions_with_error_analysis.csv"
    )

    print()
    print("Demand driver analysis complete.")


if __name__ == "__main__":
    main()

