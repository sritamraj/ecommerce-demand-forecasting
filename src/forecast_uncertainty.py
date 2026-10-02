
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "reports"
    / "benchmark"
    / "benchmark_cv_results.csv"
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
    / "uncertainty"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "forecast_uncertainty_results.csv"
)

PREDICTIONS_OUTPUT_FILE = (
    OUTPUT_DIR
    / "forecast_predictions_with_intervals.csv"
)

CONFIDENCE_LEVEL = 0.90


def empirical_residual_quantile(
    residuals,
    confidence_level,
):
    """
    Estimate a symmetric empirical residual interval.

    This uses absolute residuals and therefore provides
    an empirical prediction-interval width.

    It is not presented as a formal conformal guarantee.
    """
    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    residuals = residuals[
        np.isfinite(residuals)
    ]

    if len(residuals) == 0:
        raise ValueError(
            "No valid residuals available "
            "for uncertainty calibration."
        )

    alpha = 1.0 - confidence_level

    quantile = np.quantile(
        np.abs(residuals),
        1.0 - alpha,
    )

    return float(quantile)


def calculate_interval_metrics(
    actual,
    prediction,
    interval_width,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    lower = np.maximum(
        0.0,
        prediction - interval_width,
    )

    upper = (
        prediction
        + interval_width
    )

    covered = (
        (actual >= lower)
        & (actual <= upper)
    )

    coverage = (
        np.mean(covered)
        * 100.0
    )

    mean_width = np.mean(
        upper - lower
    )

    mean_absolute_error = np.mean(
        np.abs(
            actual - prediction
        )
    )

    return {
        "coverage_%": float(
            coverage
        ),
        "mean_interval_width": float(
            mean_width
        ),
        "mean_absolute_error": float(
            mean_absolute_error
        ),
        "n_observations": int(
            len(actual)
        ),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading benchmark CV results..."
    )

    cv_results = pd.read_csv(
        INPUT_FILE
    )

    print(
        "Loading final predictions..."
    )

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        parse_dates=["date"],
    )

    required_cv_columns = {
        "model",
        "MAE",
        "RMSE",
        "sMAPE_%",
        "WAPE_%",
    }

    missing_cv = (
        required_cv_columns
        - set(cv_results.columns)
    )

    if missing_cv:
        raise ValueError(
            "Missing CV columns: "
            f"{sorted(missing_cv)}"
        )

    required_prediction_columns = {
        "id",
        "date",
        "demand",
        "prediction",
        "model",
    }

    missing_predictions = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_predictions:
        raise ValueError(
            "Missing prediction columns: "
            f"{sorted(missing_predictions)}"
        )

    print(
        "\nUNCERTAINTY CALIBRATION"
    )

    print(
        "Confidence level: "
        f"{CONFIDENCE_LEVEL:.0%}"
    )

    print(
        "\nImportant:"
    )

    print(
        "The final holdout is used only "
        "for evaluation."
    )

    print(
        "No final-test residuals are used "
        "to calibrate the interval."
    )

    # ------------------------------------------------------------------
    # Because benchmark_cv_results.csv contains only aggregate metrics,
    # we use the saved final predictions only to construct the final
    # interval demonstration. The uncertainty width is calibrated from
    # a conservative residual scale derived from CV model performance.
    #
    # The calibration uses the maximum observed CV RMSE for the selected
    # model as an uncertainty scale. This avoids using final-test errors
    # to tune the interval.
    # ------------------------------------------------------------------

    model_names = [
        "Seasonal Naive",
        "Gradient Boosting",
    ]

    result_rows = []
    output_frames = []

    for model_name in model_names:

        print(
            f"\nProcessing {model_name}..."
        )

        model_cv = cv_results[
            cv_results["model"]
            == model_name
        ].copy()

        model_predictions = predictions[
            predictions["model"]
            == model_name
        ].copy()

        if model_cv.empty:
            raise ValueError(
                f"No CV results found for "
                f"{model_name}."
            )

        if model_predictions.empty:
            raise ValueError(
                f"No final predictions found "
                f"for {model_name}."
            )

        # Use the largest CV RMSE as a conservative
        # uncertainty scale. This is derived only
        # from historical CV results.
        interval_width = float(
            model_cv["RMSE"].max()
        )

        actual = (
            model_predictions[
                "demand"
            ].to_numpy(
                dtype=float
            )
        )

        point_prediction = (
            model_predictions[
                "prediction"
            ].to_numpy(
                dtype=float
            )
        )

        lower = np.maximum(
            0.0,
            point_prediction
            - interval_width,
        )

        upper = (
            point_prediction
            + interval_width
        )

        covered = (
            (actual >= lower)
            & (actual <= upper)
        )

        coverage = (
            np.mean(covered)
            * 100.0
        )

        mean_width = np.mean(
            upper - lower
        )

        mean_absolute_error = np.mean(
            np.abs(
                actual
                - point_prediction
            )
        )

        result_rows.append(
            {
                "model": model_name,
                "confidence_level_%": (
                    CONFIDENCE_LEVEL
                    * 100.0
                ),
                "calibration_source": (
                    "historical time-series CV"
                ),
                "interval_width": (
                    interval_width
                ),
                "final_holdout_coverage_%": (
                    coverage
                ),
                "mean_interval_width": (
                    mean_width
                ),
                "final_holdout_MAE": (
                    mean_absolute_error
                ),
                "final_holdout_rows": (
                    len(actual)
                ),
            }
        )

        model_predictions[
            "prediction_lower"
        ] = lower

        model_predictions[
            "prediction_upper"
        ] = upper

        model_predictions[
            "interval_width"
        ] = interval_width

        model_predictions[
            "interval_covered"
        ] = covered

        output_frames.append(
            model_predictions
        )

        print(
            f"  Calibration width: "
            f"{interval_width:.4f}"
        )

        print(
            f"  Final holdout coverage: "
            f"{coverage:.2f}%"
        )

        print(
            f"  Mean interval width: "
            f"{mean_width:.4f}"
        )

        print(
            f"  Final holdout MAE: "
            f"{mean_absolute_error:.4f}"
        )

    results = pd.DataFrame(
        result_rows
    )

    interval_predictions = pd.concat(
        output_frames,
        ignore_index=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    interval_predictions.to_csv(
        PREDICTIONS_OUTPUT_FILE,
        index=False,
    )

    print(
        "\nUNCERTAINTY RESULTS SAVED:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nPREDICTIONS WITH INTERVALS SAVED:"
    )

    print(
        PREDICTIONS_OUTPUT_FILE
    )

    print(
        "\nUNCERTAINTY SUMMARY:"
    )

    print(
        results.to_string(
            index=False
        )
    )

    print(
        "\nFORECAST UNCERTAINTY "
        "ANALYSIS COMPLETE"
    )


if __name__ == "__main__":
    main()

