"""Batch inference using the trained production demand model."""

import sys
sys.path.insert(0, ".")

from pathlib import Path

import pandas as pd

from features import FEATURE_COLUMNS, TARGET_COLUMN
from production_model import load_production_model, predict_demand


INPUT_PATH = Path("data/model_features.csv")
OUTPUT_PATH = Path("reports/batch_predictions.csv")


def run_batch_inference():
    """Generate batch demand predictions."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"],
    )

    missing = set(FEATURE_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required features: {sorted(missing)}"
        )

    model = load_production_model()

    predictions = predict_demand(
        df,
        model=model,
    )

    result = df[
        [
            "date",
            "product_id",
            "category",
            TARGET_COLUMN,
        ]
    ].copy()

    result["prediction"] = predictions

    result["error"] = (
        result[TARGET_COLUMN]
        - result["prediction"]
    )

    result["absolute_error"] = (
        result["error"].abs()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 70)
    print("BATCH INFERENCE")
    print("=" * 70)

    print(f"Input rows:       {len(df):,}")
    print(f"Output rows:      {len(result):,}")
    print(f"Products:         {result['product_id'].nunique():,}")

    print(
        f"Date range:       "
        f"{result['date'].min().date()} "
        f"to "
        f"{result['date'].max().date()}"
    )

    print(
        f"Prediction min:   "
        f"{result['prediction'].min():.4f}"
    )

    print(
        f"Prediction max:   "
        f"{result['prediction'].max():.4f}"
    )

    print(
        f"Prediction mean:  "
        f"{result['prediction'].mean():.4f}"
    )

    print(
        f"Negative predictions: "
        f"{(result['prediction'] < 0).sum()}"
    )

    print()
    print("Sample predictions:")
    print(
        result.head(10).to_string(
            index=False
        )
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")

    return result


if __name__ == "__main__":
    run_batch_inference()