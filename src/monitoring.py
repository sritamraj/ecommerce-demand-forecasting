from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_FILE = (
    ROOT
    / "data"
    / "benchmark"
    / "public_benchmark_daily.csv"
)

PREDICTIONS_FILE = (
    ROOT
    / "reports"
    / "benchmark"
    / "benchmark_final_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "monitoring"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "forecast_monitoring_report.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "monitoring_summary.txt"
)

MONITORING_COLUMNS = [
    "demand",
    "price",
    "discount",
    "promotion",
    "holiday",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
]


def safe_mean(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def safe_std(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) <= 1:
        return np.nan

    return float(values.std())


def safe_min(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(values.min())


def safe_max(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(values.max())


def calculate_distribution_shift(
    reference,
    monitored,
):
    reference = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    monitored = pd.to_numeric(
        monitored,
        errors="coerce",
    ).dropna()

    if len(reference) == 0 or len(monitored) == 0:
        return np.nan

    reference_mean = reference.mean()
    reference_std = reference.std()

    monitored_mean = monitored.mean()

    if (
        not np.isfinite(reference_std)
        or reference_std == 0
    ):
        return 0.0

    return float(
        abs(
            monitored_mean
            - reference_mean
        )
        / reference_std
    )


def calculate_performance(
    actual,
    prediction,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    error = (
        prediction
        - actual
    )

    absolute_error = np.abs(
        error
    )

    squared_error = (
        error ** 2
    )

    denominator = (
        np.abs(actual)
        + np.abs(prediction)
    )

    smape_values = np.zeros_like(
        denominator,
        dtype=float,
    )

    numerator = (
        2.0
        * np.abs(error)
    )

    np.divide(
        numerator,
        denominator,
        out=smape_values,
        where=denominator != 0,
    )

    total_actual = np.sum(
        np.abs(actual)
    )

    if total_actual == 0:
        wape = np.nan
    else:
        wape = (
            np.sum(absolute_error)
            / total_actual
            * 100.0
        )

    return {
        "MAE": float(
            np.mean(absolute_error)
        ),
        "RMSE": float(
            np.sqrt(
                np.mean(
                    squared_error
                )
            )
        ),
        "sMAPE_%": float(
            np.mean(
                smape_values
            )
            * 100.0
        ),
        "WAPE_%": float(
            wape
        ),
        "prediction_mean": float(
            np.mean(prediction)
        ),
        "prediction_std": float(
            np.std(prediction)
        ),
        "prediction_min": float(
            np.min(prediction)
        ),
        "prediction_max": float(
            np.max(prediction)
        ),
        "negative_prediction_count": int(
            np.sum(prediction < 0)
        ),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading benchmark dataset..."
    )

    benchmark = pd.read_csv(
        BENCHMARK_FILE,
        parse_dates=["date"],
    )

    print(
        "Loading final predictions..."
    )

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        parse_dates=["date"],
    )

    print(
        "\nMONITORING INPUT VALIDATION"
    )

    required_benchmark_columns = {
        "id",
        "date",
        "demand",
        "is_final_test",
    }

    required_prediction_columns = {
        "id",
        "date",
        "demand",
        "prediction",
        "model",
    }

    missing_benchmark = (
        required_benchmark_columns
        - set(benchmark.columns)
    )

    missing_predictions = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_benchmark:
        raise ValueError(
            "Missing benchmark columns: "
            f"{sorted(missing_benchmark)}"
        )

    if missing_predictions:
        raise ValueError(
            "Missing prediction columns: "
            f"{sorted(missing_predictions)}"
        )

    print(
        "  Required benchmark columns: PASS"
    )

    print(
        "  Required prediction columns: PASS"
    )

    # ---------------------------------------------------------------
    # DATA QUALITY
    # ---------------------------------------------------------------

    print(
        "\nDATA QUALITY CHECKS"
    )

    data_quality_rows = []

    for column in MONITORING_COLUMNS:

        if column not in benchmark.columns:
            continue

        missing_count = int(
            benchmark[column].isna().sum()
        )

        negative_count = 0

        numeric_values = pd.to_numeric(
            benchmark[column],
            errors="coerce",
        )

        if column in {
            "demand",
            "price",
            "discount",
        }:
            negative_count = int(
                (
                    numeric_values
                    < 0
                ).sum()
            )

        data_quality_rows.append(
            {
                "check_type": (
                    "data_quality"
                ),
                "metric": column,
                "missing_count": (
                    missing_count
                ),
                "negative_count": (
                    negative_count
                ),
                "mean": safe_mean(
                    benchmark[column]
                ),
                "std": safe_std(
                    benchmark[column]
                ),
                "min": safe_min(
                    benchmark[column]
                ),
                "max": safe_max(
                    benchmark[column]
                ),
            }
        )

    # ---------------------------------------------------------------
    # FEATURE DISTRIBUTION MONITORING
    # ---------------------------------------------------------------

    print(
        "\nFEATURE DISTRIBUTION MONITORING"
    )

    reference = benchmark[
        benchmark["is_final_test"] == 0
    ].copy()

    monitored = benchmark[
        benchmark["is_final_test"] == 1
    ].copy()

    for column in MONITORING_COLUMNS:

        if column not in benchmark.columns:
            continue

        shift = calculate_distribution_shift(
            reference[column],
            monitored[column],
        )

        data_quality_rows.append(
            {
                "check_type": (
                    "distribution_shift"
                ),
                "metric": column,
                "missing_count": (
                    int(
                        monitored[
                            column
                        ].isna().sum()
                    )
                ),
                "negative_count": (
                    int(
                        (
                            pd.to_numeric(
                                monitored[
                                    column
                                ],
                                errors="coerce",
                            )
                            < 0
                        ).sum()
                    )
                ),
                "mean": safe_mean(
                    monitored[column]
                ),
                "std": safe_std(
                    monitored[column]
                ),
                "min": safe_min(
                    monitored[column]
                ),
                "max": safe_max(
                    monitored[column]
                ),
                "standardized_mean_shift": (
                    shift
                ),
            }
        )

    # ---------------------------------------------------------------
    # PREDICTION MONITORING
    # ---------------------------------------------------------------

    print(
        "\nPREDICTION MONITORING"
    )

    performance_rows = []

    for model_name in sorted(
        predictions["model"].unique()
    ):

        model_predictions = predictions[
            predictions["model"]
            == model_name
        ].copy()

        metrics = calculate_performance(
            model_predictions[
                "demand"
            ],
            model_predictions[
                "prediction"
            ],
        )

        performance_rows.append(
            {
                "check_type": (
                    "model_performance"
                ),
                "metric": model_name,
                "missing_count": int(
                    model_predictions[
                        "prediction"
                    ].isna().sum()
                ),
                "negative_count": (
                    metrics[
                        "negative_prediction_count"
                    ]
                ),
                "mean": (
                    metrics[
                        "prediction_mean"
                    ]
                ),
                "std": (
                    metrics[
                        "prediction_std"
                    ]
                ),
                "min": (
                    metrics[
                        "prediction_min"
                    ]
                ),
                "max": (
                    metrics[
                        "prediction_max"
                    ]
                ),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "sMAPE_%": (
                    metrics["sMAPE_%"]
                ),
                "WAPE_%": (
                    metrics["WAPE_%"]
                ),
            }
        )

        print(
            f"  {model_name}:"
        )

        print(
            f"    MAE: "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"    RMSE: "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"    WAPE: "
            f"{metrics['WAPE_%']:.4f}%"
        )

        print(
            f"    Negative predictions: "
            f"{metrics['negative_prediction_count']}"
        )

    report = pd.DataFrame(
        data_quality_rows
        + performance_rows
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    total_rows = len(
        benchmark
    )

    holdout_rows = int(
        benchmark[
            "is_final_test"
        ].sum()
    )

    negative_predictions = int(
        predictions[
            "prediction"
        ].lt(0).sum()
    )

    total_missing_predictions = int(
        predictions[
            "prediction"
        ].isna().sum()
    )

    summary_lines = [
        "FORECAST MONITORING SUMMARY",
        "===========================",
        "",
        f"Benchmark rows: {total_rows}",
        f"Final-test rows: {holdout_rows}",
        f"Prediction rows: {len(predictions)}",
        "",
        "DATA QUALITY",
        f"Missing predictions: {total_missing_predictions}",
        f"Negative predictions: {negative_predictions}",
        "",
        "MONITORING DESIGN",
        "- Reference period: pre-holdout benchmark data",
        "- Monitored period: frozen final holdout",
        "- Checks: missingness, negative values, distribution shift",
        "- Model checks: MAE, RMSE, sMAPE, WAPE, prediction range",
        "",
        "The monitoring layer is diagnostic.",
        "It does not modify model parameters or retrain the model.",
        "",
        "MONITORING COMPLETE",
    ]

    SUMMARY_FILE.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print(
        "\nMONITORING REPORT SAVED:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nMONITORING SUMMARY SAVED:"
    )

    print(
        SUMMARY_FILE
    )

    print(
        "\nFORECAST MONITORING COMPLETE"
    )


if __name__ == "__main__":
    main()