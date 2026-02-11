"""Ensemble model combining predictions from multiple forecasters."""

import numpy as np
import pandas as pd

from src.forecast.logger import get_logger
from src.forecast.utils.metrics import rmse

logger = get_logger(__name__)


class EnsembleForecaster:
    """Weighted ensemble that combines predictions from multiple models.

    Weights are computed from inverse RMSE on validation data — models with
    lower error receive higher weight.
    """

    def __init__(self):
        self.weights: dict[str, float] = {}
        self.model_names: list[str] = []

    def compute_weights(
        self,
        actual: np.ndarray,
        predictions: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Compute ensemble weights based on inverse RMSE.

        Args:
            actual: Array of actual values (validation set).
            predictions: Dict mapping model name to predicted arrays.

        Returns:
            Dictionary of model weights that sum to 1.0.
        """
        rmse_scores = {}
        for name, preds in predictions.items():
            score = rmse(actual, preds)
            rmse_scores[name] = score
            logger.info("Model '%s' RMSE: %.4f", name, score)

        # Inverse RMSE weighting (lower RMSE = higher weight)
        inverse_scores = {name: 1.0 / (score + 1e-8) for name, score in rmse_scores.items()}
        total = sum(inverse_scores.values())
        self.weights = {name: score / total for name, score in inverse_scores.items()}
        self.model_names = list(self.weights.keys())

        logger.info("Ensemble weights: %s", {k: round(v, 4) for k, v in self.weights.items()})
        return self.weights

    def predict(self, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate ensemble prediction by weighted averaging.

        Args:
            predictions: Dict mapping model name to forecast DataFrame.
                Each DataFrame must have 'date' and 'predicted_revenue' columns.

        Returns:
            DataFrame with date and ensemble predicted_revenue.
        """
        if not self.weights:
            # Equal weighting if no weights have been computed
            n = len(predictions)
            self.weights = {name: 1.0 / n for name in predictions}
            logger.warning("No weights computed — using equal weighting")

        # Get dates from first model's predictions
        first_model = list(predictions.keys())[0]
        dates = predictions[first_model]["date"].values

        ensemble_values = np.zeros(len(dates))
        for name, pred_df in predictions.items():
            weight = self.weights.get(name, 0)
            ensemble_values += weight * pred_df["predicted_revenue"].values

        return pd.DataFrame(
            {
                "date": dates,
                "predicted_revenue": np.maximum(ensemble_values, 0),
            }
        )
