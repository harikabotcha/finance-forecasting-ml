"""Visualization utilities for forecast results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forecast.logger import get_logger

logger = get_logger(__name__)


def plot_forecast_comparison(
    actual_dates: np.ndarray,
    actual_values: np.ndarray,
    predictions: dict[str, pd.DataFrame],
    title: str = "Forecast Comparison",
    save_path: str | Path | None = None,
) -> None:
    """Plot actual vs predicted values for multiple models.

    Args:
        actual_dates: Array of dates for the actual values.
        actual_values: Array of actual revenue values.
        predictions: Dict mapping model name to forecast DataFrame.
        title: Chart title.
        save_path: Path to save the figure (optional).
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(actual_dates, actual_values, label="Actual", color="black", linewidth=2)

    colors = {"arima": "#e74c3c", "prophet": "#3498db", "xgboost": "#2ecc71", "ensemble": "#9b59b6"}
    for name, pred_df in predictions.items():
        color = colors.get(name, None)
        ax.plot(
            pred_df["date"].values,
            pred_df["predicted_revenue"].values,
            label=name.upper(),
            color=color,
            linewidth=1.5,
            linestyle="--",
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("Forecast plot saved to %s", save_path)

    plt.close(fig)
