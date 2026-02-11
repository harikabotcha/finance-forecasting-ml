"""Model evaluation and comparison framework."""

import numpy as np
import pandas as pd

from src.forecast.logger import get_logger
from src.forecast.utils.metrics import compute_all_metrics

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate and compare multiple forecasting models."""

    def __init__(self):
        self.results: dict[str, dict[str, float]] = {}

    def evaluate_model(
        self,
        model_name: str,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate a single model's predictions.

        Args:
            model_name: Identifier for the model.
            actual: Array of actual values.
            predicted: Array of predicted values.

        Returns:
            Dictionary of metric values.
        """
        metrics = compute_all_metrics(actual, predicted)
        self.results[model_name] = metrics

        logger.info(
            "Model '%s' — RMSE: %.2f, MAE: %.2f, MAPE: %.2f%%, DA: %.1f%%",
            model_name,
            metrics["rmse"],
            metrics["mae"],
            metrics["mape"],
            metrics["directional_accuracy"],
        )
        return metrics

    def compare_models(self) -> pd.DataFrame:
        """Generate a comparison table across all evaluated models.

        Returns:
            DataFrame with models as rows and metrics as columns, sorted by RMSE.
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet.")

        comparison = pd.DataFrame(self.results).T
        comparison.index.name = "model"
        comparison = comparison.sort_values("rmse")

        logger.info("\n=== Model Comparison ===\n%s", comparison.to_string())
        return comparison

    def get_best_model(self, metric: str = "rmse") -> str:
        """Return the name of the best-performing model.

        Args:
            metric: Metric to use for ranking (lower is better for rmse/mae/mape).

        Returns:
            Name of the best model.
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet.")

        if metric == "directional_accuracy":
            # Higher is better
            best = max(self.results, key=lambda k: self.results[k][metric])
        else:
            # Lower is better
            best = min(self.results, key=lambda k: self.results[k][metric])

        logger.info("Best model by %s: %s", metric, best)
        return best
