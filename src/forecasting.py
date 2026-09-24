"""Baseline and model-training utilities for demand forecasting."""
import numpy as np
import pandas as pd


def seasonal_naive_predict(df: pd.DataFrame, target_col: str = "quantity",
                            group_col: str = "product_id", season: int = 7) -> pd.Series:
    """Predicts day T's demand as the same product's demand `season` days
    earlier (already computed as `lag_7` in the feature table when season=7).
    This uses no model fitting at all -- it's the bar every ML model must clear."""
    return df.groupby(group_col)[target_col].shift(season)


def time_series_folds(dates: pd.Series, n_folds: int = 4, test_size_days: int = 60):
    """Yields (train_mask, test_mask) boolean arrays using an expanding-window
    time-series split: each fold's test period is a contiguous future block,
    and training data is everything strictly before it. This mirrors
    sklearn.model_selection.TimeSeriesSplit but works directly on dates
    across a multi-product panel (all products share the same fold boundaries)."""
    unique_dates = np.sort(dates.unique())
    max_date = unique_dates[-1]

    folds = []
    for i in range(n_folds, 0, -1):
        test_end = max_date - np.timedelta64(test_size_days * (i - 1), "D")
        test_start = test_end - np.timedelta64(test_size_days - 1, "D")
        train_end = test_start - np.timedelta64(1, "D")
        folds.append((train_end, test_start, test_end))

    for train_end, test_start, test_end in folds:
        train_mask = dates.values <= train_end
        test_mask = (dates.values >= test_start) & (dates.values <= test_end)
        yield train_mask, test_mask, (train_end, test_start, test_end)
