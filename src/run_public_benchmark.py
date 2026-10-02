
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "benchmark"
    / "public_benchmark_daily.csv"
)

OUTPUT_DIR = ROOT / "reports" / "benchmark"

CV_RESULTS_FILE = (
    OUTPUT_DIR
    / "benchmark_cv_results.csv"
)

FINAL_RESULTS_FILE = (
    OUTPUT_DIR
    / "benchmark_final_results.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "benchmark_final_predictions.csv"
)


FORECAST_HORIZON = 28
N_CV_FOLDS = 4

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "month",
    "wday",
    "is_weekend",
    "has_event",
    "event_type_present",
    "snap",
    "item_code",
    "store_code",
    "dept_code",
    "cat_code",
    "state_code",
]


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error.

    When both actual and predicted demand are zero,
    the contribution is defined as zero.

    Uses np.divide(..., where=...) to avoid
    divide-by-zero warnings.
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    numerator = (
        2.0
        * np.abs(y_pred - y_true)
    )

    values = np.zeros_like(
        numerator,
        dtype=float,
    )

    np.divide(
        numerator,
        denominator,
        out=values,
        where=denominator != 0,
    )

    return float(
        np.mean(values) * 100.0
    )


def wape(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    denominator = np.sum(
        np.abs(y_true)
    )

    if denominator == 0:
        return np.nan

    return float(
        np.sum(
            np.abs(y_true - y_pred)
        )
        / denominator
        * 100.0
    )


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mae = np.mean(
        np.abs(
            y_true - y_pred
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                y_true - y_pred
            ) ** 2
        )
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "sMAPE_%": smape(
            y_true,
            y_pred,
        ),
        "WAPE_%": wape(
            y_true,
            y_pred,
        ),
    }


def build_feature_row(
    history,
    calendar_row,
    metadata,
):
    history = np.asarray(
        history,
        dtype=float,
    )

    if len(history) < 28:
        raise ValueError(
            "At least 28 historical observations "
            "are required."
        )

    lag_1 = history[-1]
    lag_7 = history[-7]
    lag_14 = history[-14]
    lag_28 = history[-28]

    rolling_7 = history[-7:]
    rolling_14 = history[-14:]
    rolling_28 = history[-28:]

    return {
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "rolling_mean_7": np.mean(
            rolling_7
        ),
        "rolling_mean_14": np.mean(
            rolling_14
        ),
        "rolling_mean_28": np.mean(
            rolling_28
        ),
        "rolling_std_7": np.std(
            rolling_7,
            ddof=1,
        ),
        "rolling_std_28": np.std(
            rolling_28,
            ddof=1,
        ),
        "month": int(
            calendar_row["month"]
        ),
        "wday": int(
            calendar_row["wday"]
        ),
        "is_weekend": int(
            calendar_row["is_weekend"]
        ),
        "has_event": int(
            calendar_row["has_event"]
        ),
        "event_type_present": int(
            calendar_row[
                "event_type_present"
            ]
        ),
        "snap": int(
            calendar_row["snap"]
        ),
        "item_code": metadata[
            "item_code"
        ],
        "store_code": metadata[
            "store_code"
        ],
        "dept_code": metadata[
            "dept_code"
        ],
        "cat_code": metadata[
            "cat_code"
        ],
        "state_code": metadata[
            "state_code"
        ],
    }


def recursive_forecast(
    model,
    history_df,
    future_df,
    metadata,
):
    history = list(
        history_df[
            "demand"
        ].astype(float)
    )

    predictions = []

    for _, calendar_row in future_df.iterrows():

        feature_row = build_feature_row(
            history,
            calendar_row,
            metadata,
        )

        X = pd.DataFrame(
            [feature_row],
            columns=FEATURE_COLUMNS,
        )

        prediction = float(
            model.predict(X)[0]
        )

        # Demand cannot be negative.
        prediction = max(
            0.0,
            prediction,
        )

        predictions.append(
            prediction
        )

        # Recursive forecasting:
        # use the model prediction as
        # the next historical observation.
        history.append(
            prediction
        )

    return np.asarray(
        predictions,
        dtype=float,
    )


def prepare_data():
    print(
        "Loading benchmark dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    required_columns = {
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "date",
        "demand",
        "month",
        "wday",
        "is_weekend",
        "has_event",
        "event_type_present",
        "snap",
        "is_final_test",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.sort_values(
        [
            "id",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    # Stable categorical encodings.
    for column in [
        "item_id",
        "store_id",
        "dept_id",
        "cat_id",
        "state_id",
    ]:
        df[
            f"{column.replace('_id', '')}_code"
        ] = pd.Categorical(
            df[column]
        ).codes

    return df


def get_feature_training_data(
    df,
    train_end_date,
):
    train = df[
        df["date"] <= train_end_date
    ].copy()

    # Remove rows where lag/rolling features
    # cannot yet be calculated.
    train = train.dropna(
        subset=[
            "lag_1",
            "lag_7",
            "lag_14",
            "lag_28",
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_mean_28",
            "rolling_std_7",
            "rolling_std_28",
        ]
    ).copy()

    X = train[
        FEATURE_COLUMNS
    ].copy()

    y = train[
        "demand"
    ].astype(float)

    return X, y


def train_model(
    df,
    train_end_date,
):
    X, y = get_feature_training_data(
        df,
        train_end_date,
    )

    print(
        f"  Training rows: {len(X):,}"
    )

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(
        X,
        y,
    )

    return model


def evaluate_window(
    df,
    train_end_date,
    validation_start,
    validation_end,
    model_name,
):
    model = None

    if model_name == "Gradient Boosting":
        model = train_model(
            df,
            train_end_date,
        )

    validation = df[
        (
            df["date"]
            >= validation_start
        )
        & (
            df["date"]
            <= validation_end
        )
    ].copy()

    all_predictions = []

    metadata_columns = [
        "item_code",
        "store_code",
        "dept_code",
        "cat_code",
        "state_code",
    ]

    for series_id, future in validation.groupby(
        "id",
        sort=False,
    ):
        future = future.sort_values(
            "date"
        ).copy()

        history = df[
            (df["id"] == series_id)
            & (
                df["date"]
                <= train_end_date
            )
        ].sort_values(
            "date"
        )

        if len(history) < 28:
            raise ValueError(
                f"Insufficient history for "
                f"series {series_id}"
            )

        metadata = (
            history.iloc[0][
                metadata_columns
            ].to_dict()
        )

        if model_name == "Seasonal Naive":
            history_values = list(
                history[
                    "demand"
                ].astype(float)
            )

            predictions = []

            for _ in range(
                len(future)
            ):
                prediction = (
                    history_values[-7]
                )

                prediction = max(
                    0.0,
                    float(prediction),
                )

                predictions.append(
                    prediction
                )

                history_values.append(
                    prediction
                )

        else:
            predictions = recursive_forecast(
                model=model,
                history_df=history,
                future_df=future,
                metadata=metadata,
            )

        series_result = future[
            [
                "id",
                "date",
                "demand",
            ]
        ].copy()

        series_result[
            "prediction"
        ] = predictions

        all_predictions.append(
            series_result
        )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics = calculate_metrics(
        predictions_df[
            "demand"
        ],
        predictions_df[
            "prediction"
        ],
    )

    metrics["model"] = model_name

    return (
        metrics,
        predictions_df,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = prepare_data()

    final_test_start = df.loc[
        df["is_final_test"] == 1,
        "date",
    ].min()

    final_test_end = df.loc[
        df["is_final_test"] == 1,
        "date",
    ].max()

    print(
        "\nFINAL HOLDOUT:"
    )

    print(
        f"  {final_test_start.date()} "
        f"to "
        f"{final_test_end.date()}"
    )

    print(
        "\nBuilding CV windows..."
    )

    cv_windows = []

    validation_end = (
        final_test_start
        - pd.Timedelta(days=1)
    )

    for fold in range(
        N_CV_FOLDS,
        0,
        -1,
    ):
        validation_start = (
            validation_end
            - pd.Timedelta(days=27)
        )

        train_end_date = (
            validation_start
            - pd.Timedelta(days=1)
        )

        cv_windows.append(
            {
                "fold": (
                    N_CV_FOLDS
                    - fold
                    + 1
                ),
                "train_end": (
                    train_end_date
                ),
                "validation_start": (
                    validation_start
                ),
                "validation_end": (
                    validation_end
                ),
            }
        )

        validation_end = (
            validation_start
            - pd.Timedelta(days=1)
        )

    print(
        "\nCV WINDOWS:"
    )

    for window in cv_windows:
        print(
            f"  Fold {window['fold']}: "
            f"train <= "
            f"{window['train_end'].date()}, "
            f"validate "
            f"{window['validation_start'].date()} "
            f"to "
            f"{window['validation_end'].date()}"
        )

    cv_rows = []

    print(
        "\nRUNNING TIME-SERIES CV..."
    )

    for window in cv_windows:

        fold = window["fold"]

        print(
            f"\n=== FOLD {fold} ==="
        )

        for model_name in [
            "Seasonal Naive",
            "Gradient Boosting",
        ]:

            print(
                f"Evaluating {model_name}..."
            )

            metrics, _ = evaluate_window(
                df=df,
                train_end_date=(
                    window["train_end"]
                ),
                validation_start=(
                    window["validation_start"]
                ),
                validation_end=(
                    window["validation_end"]
                ),
                model_name=model_name,
            )

            row = {
                "fold": fold,
                "train_end": (
                    window["train_end"].date()
                ),
                "validation_start": (
                    window[
                        "validation_start"
                    ].date()
                ),
                "validation_end": (
                    window[
                        "validation_end"
                    ].date()
                ),
                **metrics,
            }

            cv_rows.append(
                row
            )

            print(
                f"  MAE: "
                f"{metrics['MAE']:.4f}"
            )

            print(
                f"  RMSE: "
                f"{metrics['RMSE']:.4f}"
            )

            print(
                f"  sMAPE: "
                f"{metrics['sMAPE_%']:.4f}%"
            )

            print(
                f"  WAPE: "
                f"{metrics['WAPE_%']:.4f}%"
            )

    cv_results = pd.DataFrame(
        cv_rows
    )

    cv_results.to_csv(
        CV_RESULTS_FILE,
        index=False,
    )

    print(
        "\nCV RESULTS SAVED:"
    )

    print(
        CV_RESULTS_FILE
    )

    print(
        "\nCV SUMMARY:"
    )

    summary = (
        cv_results
        .groupby("model")[
            [
                "MAE",
                "RMSE",
                "sMAPE_%",
                "WAPE_%",
            ]
        ]
        .mean()
        .sort_values("MAE")
    )

    print(
        summary.to_string()
    )

    print(
        "\nRUNNING FROZEN FINAL HOLDOUT..."
    )

    final_train_end = (
        final_test_start
        - pd.Timedelta(days=1)
    )

    final_rows = []
    final_prediction_frames = []

    for model_name in [
        "Seasonal Naive",
        "Gradient Boosting",
    ]:

        print(
            f"\nEvaluating {model_name} "
            "on frozen final test..."
        )

        metrics, predictions = (
            evaluate_window(
                df=df,
                train_end_date=(
                    final_train_end
                ),
                validation_start=(
                    final_test_start
                ),
                validation_end=(
                    final_test_end
                ),
                model_name=model_name,
            )
        )

        final_rows.append(
            {
                "model": model_name,
                "train_end": (
                    final_train_end.date()
                ),
                "final_test_start": (
                    final_test_start.date()
                ),
                "final_test_end": (
                    final_test_end.date()
                ),
                **metrics,
            }
        )

        predictions = predictions.copy()

        predictions[
            "model"
        ] = model_name

        final_prediction_frames.append(
            predictions
        )

        print(
            f"  MAE: "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"  RMSE: "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"  sMAPE: "
            f"{metrics['sMAPE_%']:.4f}%"
        )

        print(
            f"  WAPE: "
            f"{metrics['WAPE_%']:.4f}%"
        )

    final_results = pd.DataFrame(
        final_rows
    )

    final_results.to_csv(
        FINAL_RESULTS_FILE,
        index=False,
    )

    final_predictions = pd.concat(
        final_prediction_frames,
        ignore_index=True,
    )

    final_predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        "\nFINAL RESULTS SAVED:"
    )

    print(
        FINAL_RESULTS_FILE
    )

    print(
        "\nFINAL PREDICTIONS SAVED:"
    )

    print(
        PREDICTIONS_FILE
    )

    print(
        "\nFINAL HOLDOUT RESULTS:"
    )

    print(
        final_results.to_string(
            index=False
        )
    )

    print(
        "\nPUBLIC BENCHMARK "
        "EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()

