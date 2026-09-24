"""Evaluation metrics for demand forecasts."""
import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def smape(y_true, y_pred, eps: float = 1.0):
    """Symmetric MAPE with an epsilon floor in the denominator so
    near-zero-demand days don't blow the metric up to infinity."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom) * 100)


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": round(mae(y_true, y_pred), 3),
        "RMSE": round(rmse(y_true, y_pred), 3),
        "sMAPE_%": round(smape(y_true, y_pred), 2),
    }
