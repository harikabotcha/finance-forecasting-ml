"""Model training orchestration — trains all models and evaluates on test data."""

import numpy as np
import pandas as pd

from src.forecast.config import settings
from src.forecast.logger import get_logger
from src.forecast.models.arima_model import ARIMAForecaster
from src.forecast.models.ensemble import EnsembleForecaster
from src.forecast.models.evaluator import ModelEvaluator
from src.forecast.models.prophet_model import ProphetForecaster
from src.forecast.models.xgboost_model import XGBoostForecaster

logger = get_logger(__name__)


class TrainingResult:
    """Container for training outputs."""

    def __init__(
        self,
        models: dict,
        evaluator: ModelEvaluator,
        ensemble: EnsembleForecaster,
        predictions: dict[str, pd.DataFrame],
    ):
        self.models = models
        self.evaluator = evaluator
        self.ensemble = ensemble
        self.predictions = predictions


def train_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    product_id: str,
    target_column: str = "revenue",
) -> TrainingResult:
    """Train ARIMA, Prophet, and XGBoost models, then build an ensemble.

    Args:
        train_df: Training data with engineered features.
        test_df: Test data for evaluation.
        product_id: Product identifier for saving artifacts.
        target_column: Name of the target column.

    Returns:
        TrainingResult containing all models, evaluator, and predictions.
    """
    logger.info("=" * 60)
    logger.info("Training all models for product: %s", product_id)
    logger.info("=" * 60)

    models = {}
    predictions = {}
    test_horizon = len(test_df)
    actual = test_df[target_column].values

    # 1. ARIMA
    logger.info("--- Training ARIMA ---")
    try:
        arima = ARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        arima.fit(train_df, target_column=target_column)
        arima_preds = arima.predict(horizon=test_horizon)
        models["arima"] = arima
        predictions["arima"] = arima_preds
        arima.save(settings.models_dir / product_id / "arima")
    except Exception as e:
        logger.warning("ARIMA training failed: %s", e)

    # 2. Prophet
    logger.info("--- Training Prophet ---")
    try:
        prophet = ProphetForecaster()
        prophet.fit(train_df, target_column=target_column)
        prophet_preds = prophet.predict(horizon=test_horizon)
        models["prophet"] = prophet
        predictions["prophet"] = prophet_preds
        prophet.save(settings.models_dir / product_id / "prophet")
    except Exception as e:
        logger.warning("Prophet training failed: %s", e)

    # 3. XGBoost
    logger.info("--- Training XGBoost ---")
    try:
        xgb = XGBoostForecaster(n_estimators=200, max_depth=6, learning_rate=0.1)
        xgb.fit(train_df, target_column=target_column)
        xgb_preds = xgb.predict(horizon=test_horizon)
        models["xgboost"] = xgb
        predictions["xgboost"] = xgb_preds
        xgb.save(settings.models_dir / product_id / "xgboost")
    except Exception as e:
        logger.warning("XGBoost training failed: %s", e)

    # 4. Evaluate all models
    evaluator = ModelEvaluator()
    for name, pred_df in predictions.items():
        pred_values = pred_df["predicted_revenue"].values[:len(actual)]
        evaluator.evaluate_model(name, actual[:len(pred_values)], pred_values)

    # 5. Ensemble
    ensemble = EnsembleForecaster()
    if len(predictions) > 1:
        pred_arrays = {
            name: pred_df["predicted_revenue"].values[:len(actual)]
            for name, pred_df in predictions.items()
        }
        ensemble.compute_weights(actual, pred_arrays)
        ensemble_preds = ensemble.predict(predictions)
        predictions["ensemble"] = ensemble_preds
        evaluator.evaluate_model(
            "ensemble",
            actual[:len(ensemble_preds)],
            ensemble_preds["predicted_revenue"].values[:len(actual)],
        )

    # Summary
    comparison = evaluator.compare_models()
    logger.info("Best model: %s", evaluator.get_best_model())

    return TrainingResult(
        models=models,
        evaluator=evaluator,
        ensemble=ensemble,
        predictions=predictions,
    )
