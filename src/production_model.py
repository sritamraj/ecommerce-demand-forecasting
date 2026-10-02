"""Production training and inference utilities for demand forecasting."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURE_COLUMNS, TARGET_COLUMN


MODEL_PATH = Path("models/gradient_boosting_demand_model.joblib")
FEATURE_PATH = Path("models/feature_columns.txt")

TEST_DAYS = 60


def build_production_model():
    """Build the same Gradient Boosting model used during CV."""
    return HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=1.0,
        random_state=42,
    )


def train_production_model(
    data_path="data/model_features.csv",
    test_days=TEST_DAYS,
):
    """Train the final production model on development data only."""

    df = pd.read_csv(
        data_path,
        parse_dates=["date"],
    )

    max_date = df["date"].max()

    test_start = (
        max_date
        - pd.Timedelta(days=test_days - 1)
    )

    development = df[
        df["date"] < test_start
    ].copy()

    final_test = df[
        df["date"] >= test_start
    ].copy()

    if development.empty:
        raise ValueError("Development dataset is empty.")

    if final_test.empty:
        raise ValueError("Final test dataset is empty.")

    if development["date"].max() >= final_test["date"].min():
        raise ValueError(
            "Development data overlaps the final test period."
        )

    X_train = development[FEATURE_COLUMNS]
    y_train = development[TARGET_COLUMN]

    model = build_production_model()

    model.fit(
        X_train,
        y_train,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    FEATURE_PATH.write_text(
        "\n".join(FEATURE_COLUMNS),
        encoding="utf-8",
    )

    print("=" * 70)
    print("PRODUCTION MODEL TRAINING")
    print("=" * 70)

    print(f"Full data:       {df.shape}")
    print(f"Development:     {development.shape}")
    print(f"Final test:      {final_test.shape}")

    print(
        f"Development end: "
        f"{development['date'].max().date()}"
    )

    print(
        f"Final test:      "
        f"{final_test['date'].min().date()} "
        f"to "
        f"{final_test['date'].max().date()}"
    )

    print()
    print("Model:")
    print(model)

    print()
    print(f"Saved model:   {MODEL_PATH}")
    print(f"Saved features: {FEATURE_PATH}")

    return model


def load_production_model(
    model_path=MODEL_PATH,
):
    """Load the trained production model."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Production model not found: {model_path}. "
            "Run train_production_model() first."
        )

    return joblib.load(model_path)


def predict_demand(
    df,
    model=None,
):
    """Generate non-negative demand predictions."""

    if model is None:
        model = load_production_model()

    missing = set(FEATURE_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required features: {sorted(missing)}"
        )

    predictions = model.predict(
        df[FEATURE_COLUMNS]
    )

    return np.clip(
        predictions,
        0,
        None,
    )


if __name__ == "__main__":
    train_production_model()