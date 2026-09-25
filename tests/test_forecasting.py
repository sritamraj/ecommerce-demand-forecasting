import numpy as np
import pandas as pd
import pytest

from src.forecasting import seasonal_naive_predict, time_series_folds


def test_seasonal_naive_uses_previous_season():
    dates = pd.date_range("2025-01-01", periods=14, freq="D")

    df = pd.DataFrame(
        {
            "date": dates,
            "product_id": ["P1"] * 14,
            "quantity": np.arange(10, 24, dtype=float),
        }
    )

    predictions = seasonal_naive_predict(df)

    assert predictions.iloc[7] == 10
    assert predictions.iloc[13] == 16


def test_seasonal_naive_respects_product_boundaries():
    dates = pd.date_range("2025-01-01", periods=8, freq="D")

    df = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "product_id": ["P1"] * 8 + ["P2"] * 8,
            "quantity": list(range(1, 9)) + list(range(101, 109)),
        }
    )

    predictions = seasonal_naive_predict(df)

    assert predictions.iloc[8 + 7] == 101


def test_time_series_folds_are_chronological_and_non_overlapping():
    dates = pd.Series(
        pd.date_range("2024-01-01", "2025-05-01", freq="D")
    )

    folds = list(
        time_series_folds(
            dates,
            n_folds=4,
            test_size_days=60,
        )
    )

    assert len(folds) == 4

    previous_test_end = None

    for train_mask, test_mask, metadata in folds:
        assert train_mask.dtype == bool
        assert test_mask.dtype == bool

        assert train_mask.any()
        assert test_mask.any()

        assert not np.any(train_mask & test_mask)

        train_dates = dates[train_mask]
        test_dates = dates[test_mask]

        train_end = pd.Timestamp(train_dates.max())
        test_start = pd.Timestamp(test_dates.min())
        test_end = pd.Timestamp(test_dates.max())

        assert train_end < test_start
        assert test_start == pd.Timestamp(metadata["test_start"])
        assert test_end == pd.Timestamp(metadata["test_end"])
        assert train_end == pd.Timestamp(metadata["train_end"])

        if previous_test_end is not None:
            assert test_start > previous_test_end

        previous_test_end = test_end


def test_time_series_folds_use_expanding_training_windows():
    dates = pd.Series(
        pd.date_range("2024-01-01", "2025-05-01", freq="D")
    )

    folds = list(
        time_series_folds(
            dates,
            n_folds=4,
            test_size_days=60,
        )
    )

    training_lengths = [
        int(train_mask.sum())
        for train_mask, _, _ in folds
    ]

    assert training_lengths == sorted(training_lengths)
    assert len(set(training_lengths)) == len(training_lengths)


def test_time_series_folds_require_sufficient_history():
    dates = pd.Series(
        pd.date_range("2025-01-01", periods=299, freq="D")
    )

    with pytest.raises(ValueError, match="Not enough dates"):
        list(
            time_series_folds(
                dates,
                n_folds=4,
                test_size_days=60,
            )
        )