"""Tests for production batch inference."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from features import FEATURE_COLUMNS
from production_model import (
    MODEL_PATH,
    load_production_model,
    predict_demand,
)


INPUT_PATH = Path("data/model_features.csv")
OUTPUT_PATH = Path("reports/batch_predictions.csv")


def test_production_model_exists():
    assert MODEL_PATH.exists()


def test_production_model_loads():
    model = load_production_model()

    assert model is not None


def test_batch_predictions_exist():
    assert OUTPUT_PATH.exists()


def test_batch_prediction_schema():
    df = pd.read_csv(OUTPUT_PATH)

    required_columns = {
        "date",
        "product_id",
        "category",
        "quantity",
        "prediction",
        "error",
        "absolute_error",
    }

    assert required_columns.issubset(
        df.columns
    )


def test_batch_prediction_row_count():
    input_df = pd.read_csv(INPUT_PATH)
    output_df = pd.read_csv(OUTPUT_PATH)

    assert len(output_df) == len(input_df)


def test_batch_predictions_are_non_negative():
    df = pd.read_csv(OUTPUT_PATH)

    assert (
        df["prediction"] >= 0
    ).all()


def test_batch_predictions_are_finite():
    df = pd.read_csv(OUTPUT_PATH)

    assert (
        df["prediction"].notna()
    ).all()

    assert (
        df["prediction"].apply(
            lambda x: abs(x) != float("inf")
        )
    ).all()


def test_batch_has_all_products():
    df = pd.read_csv(OUTPUT_PATH)

    assert (
        df["product_id"].nunique()
        == 40
    )


def test_prediction_function():
    input_df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"],
    )

    sample = input_df.head(20)

    model = load_production_model()

    predictions = predict_demand(
        sample,
        model=model,
    )

    assert len(predictions) == 20
    assert predictions.min() >= 0
    assert predictions.dtype.kind in "fc"


def test_all_required_features_exist():
    input_df = pd.read_csv(INPUT_PATH)

    missing = (
        set(FEATURE_COLUMNS)
        - set(input_df.columns)
    )

    assert not missing