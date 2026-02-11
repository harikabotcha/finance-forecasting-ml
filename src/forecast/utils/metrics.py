"""Evaluation metrics for forecasting model comparison."""

import numpy as np
import pandas as pd


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Returns value as percentage (0-100)."""
    mask = actual != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error. Returns value as percentage (0-100)."""
    denominator = np.abs(actual) + np.abs(predicted)
    mask = denominator != 0
    if mask.sum() == 0:
        return 0.0
    return float(
        np.mean(2.0 * np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100
    )


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Percentage of times the predicted direction matches actual direction.

    Returns:
        Accuracy as a percentage (0-100).
    """
    if len(actual) < 2:
        return 0.0
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    return float(np.mean(actual_direction == predicted_direction) * 100)


def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute all evaluation metrics at once.

    Args:
        actual: Array of actual values.
        predicted: Array of predicted values.

    Returns:
        Dictionary of metric names to values.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "rmse": round(rmse(actual, predicted), 4),
        "mae": round(mae(actual, predicted), 4),
        "mape": round(mape(actual, predicted), 4),
        "smape": round(smape(actual, predicted), 4),
        "directional_accuracy": round(directional_accuracy(actual, predicted), 4),
    }
