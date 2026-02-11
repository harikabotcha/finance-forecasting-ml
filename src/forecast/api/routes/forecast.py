"""Forecast endpoints — trigger predictions and retrieve results."""

from fastapi import APIRouter, HTTPException

from src.forecast.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    ModelPrediction,
    PredictionPoint,
)
from src.forecast.config import settings
from src.forecast.data.generator import PRODUCT_PROFILES
from src.forecast.logger import get_logger
from src.forecast.models.arima_model import ARIMAForecaster
from src.forecast.models.ensemble import EnsembleForecaster
from src.forecast.models.prophet_model import ProphetForecaster
from src.forecast.models.xgboost_model import XGBoostForecaster

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Forecasting"])


def _load_and_predict(model_class, model_dir, horizon):
    """Helper to load a model and generate predictions."""
    model = model_class()
    model.load(model_dir)
    return model.predict(horizon=horizon)


@router.post("/forecast/predict", response_model=ForecastResponse)
async def predict(request: ForecastRequest) -> ForecastResponse:
    """Generate forecasts for a product using all trained models.

    Returns predictions from ARIMA, Prophet, XGBoost, and an ensemble.
    """
    if request.product_id not in PRODUCT_PROFILES:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{request.product_id}' not found. Available: {list(PRODUCT_PROFILES.keys())}",
        )

    product_dir = settings.models_dir / request.product_id
    if not product_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained models found for '{request.product_id}'. Run the training pipeline first.",
        )

    model_predictions: list[ModelPrediction] = []
    raw_predictions = {}

    # Try loading each model
    model_configs = [
        ("arima", ARIMAForecaster, product_dir / "arima"),
        ("prophet", ProphetForecaster, product_dir / "prophet"),
        ("xgboost", XGBoostForecaster, product_dir / "xgboost"),
    ]

    for name, cls, model_dir in model_configs:
        try:
            pred_df = _load_and_predict(cls, model_dir, request.horizon)
            points = [
                PredictionPoint(
                    date=row["date"],
                    predicted_revenue=round(row["predicted_revenue"], 2),
                    lower_ci=round(row.get("lower_ci", 0), 2) if "lower_ci" in row else None,
                    upper_ci=round(row.get("upper_ci", 0), 2) if "upper_ci" in row else None,
                )
                for _, row in pred_df.iterrows()
            ]
            model_predictions.append(ModelPrediction(model_name=name, predictions=points))
            raw_predictions[name] = pred_df
        except Exception as e:
            logger.warning("Could not load %s model: %s", name, e)

    if not model_predictions:
        raise HTTPException(status_code=500, detail="No models could be loaded.")

    # Ensemble
    ensemble_points = None
    if len(raw_predictions) > 1:
        ensemble = EnsembleForecaster()
        ensemble_df = ensemble.predict(raw_predictions)
        ensemble_points = [
            PredictionPoint(
                date=row["date"],
                predicted_revenue=round(row["predicted_revenue"], 2),
            )
            for _, row in ensemble_df.iterrows()
        ]

    return ForecastResponse(
        product_id=request.product_id,
        horizon=request.horizon,
        models=model_predictions,
        ensemble=ensemble_points,
    )
