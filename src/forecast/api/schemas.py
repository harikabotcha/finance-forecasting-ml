"""Pydantic request/response models for the API."""

from datetime import date
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request body for generating a forecast."""

    product_id: str = Field(..., description="Product identifier (e.g. PROD_A)")
    horizon: int = Field(default=30, ge=1, le=365, description="Forecast horizon in days")


class PredictionPoint(BaseModel):
    """A single forecasted data point."""

    date: date
    predicted_revenue: float
    lower_ci: float | None = None
    upper_ci: float | None = None


class ModelPrediction(BaseModel):
    """Predictions from a single model."""

    model_name: str
    predictions: list[PredictionPoint]


class ForecastResponse(BaseModel):
    """Response containing forecasts from all models."""

    product_id: str
    horizon: int
    models: list[ModelPrediction]
    ensemble: list[PredictionPoint] | None = None


class ModelMetrics(BaseModel):
    """Performance metrics for a model."""

    model_name: str
    rmse: float
    mae: float
    mape: float
    smape: float
    directional_accuracy: float


class ModelInfo(BaseModel):
    """Information about a trained model."""

    name: str
    product_id: str
    is_trained: bool = False
    metrics: ModelMetrics | None = None


class HealthResponse(BaseModel):
    """API health check response."""

    status: str
    version: str
    models_loaded: int
