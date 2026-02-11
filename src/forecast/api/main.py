"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.forecast.api.middleware import RequestLoggingMiddleware
from src.forecast.api.routes import forecast, health, models
from src.forecast.config import settings
from src.forecast.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Starting Finance Forecasting API v%s", settings.api_version)
    logger.info("Environment: %s | Debug: %s", settings.app_env, settings.debug)
    yield
    logger.info("Shutting down Finance Forecasting API")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Machine learning-based revenue forecasting system with ARIMA, Prophet, and XGBoost models.",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(models.router)


@app.get("/", tags=["Root"])
async def root():
    """API root — provides basic info and links."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health",
    }
