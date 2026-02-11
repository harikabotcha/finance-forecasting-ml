"""Health check endpoint."""

from fastapi import APIRouter

from src.forecast.api.schemas import HealthResponse
from src.forecast.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Check API health and report status."""
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        models_loaded=0,
    )
