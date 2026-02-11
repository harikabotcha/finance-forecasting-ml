"""Centralized configuration management using Pydantic settings."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql://forecast_user:forecast_pass@localhost:5432/forecast_db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Finance Forecasting API"
    api_version: str = "1.0.0"

    # ML Configuration
    forecast_horizon: int = 30
    test_size: float = 0.2
    random_seed: int = 42

    # Data Paths
    data_raw_path: str = "data/raw"
    data_processed_path: str = "data/processed"
    model_artifacts_path: str = "data/models"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def raw_dir(self) -> Path:
        path = Path(self.data_raw_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def processed_dir(self) -> Path:
        path = Path(self.data_processed_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def models_dir(self) -> Path:
        path = Path(self.model_artifacts_path)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
