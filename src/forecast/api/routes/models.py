"""Model information and metrics endpoints."""

from fastapi import APIRouter, HTTPException

from src.forecast.api.schemas import ModelInfo, ModelMetrics
from src.forecast.config import settings
from src.forecast.data.generator import PRODUCT_PROFILES
from src.forecast.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Models"])

MODEL_NAMES = ["arima", "prophet", "xgboost"]


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """List all available models and their training status."""
    models = []
    for product_id in PRODUCT_PROFILES:
        for model_name in MODEL_NAMES:
            model_dir = settings.models_dir / product_id / model_name
            models.append(
                ModelInfo(
                    name=model_name,
                    product_id=product_id,
                    is_trained=model_dir.exists(),
                )
            )
    return models


@router.get("/models/{product_id}/{model_name}", response_model=ModelInfo)
async def get_model_info(product_id: str, model_name: str) -> ModelInfo:
    """Get information about a specific model."""
    if product_id not in PRODUCT_PROFILES:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found.")
    if model_name not in MODEL_NAMES:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {MODEL_NAMES}",
        )

    model_dir = settings.models_dir / product_id / model_name
    return ModelInfo(
        name=model_name,
        product_id=product_id,
        is_trained=model_dir.exists(),
    )
