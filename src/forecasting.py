"""Baseline and model-training utilities for demand forecasting."""

import numpy as np
import pandas as pd


def seasonal_naive_predict(
    df,
    target_col="quantity",
    group_col="product_id",
    season=7,
):
    """Generate a seasonal-naive prediction using prior seasonal demand."""
    return df.groupby(group_col)[target_col].shift(season)


def time_series_folds(
    dates: pd.Series,
    n_folds: int = 4,
    test_size_days: int = 60,
):
    """Yield expanding-window chronological CV folds.

    The supplied dates define the development period only.
    The caller must exclude the final untouched holdout before
    calling this function.
    """
    dates = pd.Series(dates)

    unique_dates = np.sort(dates.dropna().unique())

    required_days = (n_folds + 1) * test_size_days
    if len(unique_dates) < required_days:
        raise ValueError(
            f"Not enough dates for {n_folds} folds of "
            f"{test_size_days} days. "
            f"Need at least {required_days} unique dates, "
            f"found {len(unique_dates)}."
        )

    max_date = unique_dates[-1]

    folds = []

    for i in range(n_folds, 0, -1):
        test_end = (
            max_date
            - np.timedelta64(test_size_days * (i - 1), "D")
        )

        test_start = (
            test_end
            - np.timedelta64(test_size_days - 1, "D")
        )

        train_end = (
            test_start
            - np.timedelta64(1, "D")
        )

        folds.append(
            (
                train_end,
                test_start,
                test_end,
            )
        )

    for train_end, test_start, test_end in folds:
        train_mask = dates.values <= train_end

        test_mask = (
            (dates.values >= test_start)
            & (dates.values <= test_end)
        )

        if not train_mask.any():
            raise ValueError(
                f"Empty training set for fold ending {test_end}."
            )

        if not test_mask.any():
            raise ValueError(
                f"Empty validation set for fold "
                f"{test_start} to {test_end}."
            )

        yield (
            train_mask,
            test_mask,
            {
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            },
        )