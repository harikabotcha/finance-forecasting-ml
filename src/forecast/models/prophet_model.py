"""Facebook Prophet forecasting model implementation."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet

from src.forecast.logger import get_logger
from src.forecast.models.base import BaseForecaster

logger = get_logger(__name__)


class ProphetForecaster(BaseForecaster):
    """Prophet model for time series forecasting with multiple seasonalities."""

    name = "prophet"

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        self.last_date = None

    def fit(self, train_df: pd.DataFrame, target_column: str = "revenue") -> None:
        """Fit Prophet model on training data.

        Prophet expects columns named 'ds' (date) and 'y' (target).

        Args:
            train_df: Training DataFrame with 'date' and target column.
            target_column: Name of the target column.
        """
        logger.info("Fitting Prophet model")

        prophet_df = pd.DataFrame(
            {
                "ds": pd.to_datetime(train_df["date"]),
                "y": train_df[target_column].values,
            }
        )

        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )

        # Suppress Prophet's verbose logging
        import logging as _logging
        _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)
        _logging.getLogger("prophet").setLevel(_logging.WARNING)

        self.model.fit(prophet_df)
        self.last_date = prophet_df["ds"].max()

        logger.info("Prophet model fitted (train size: %d)", len(prophet_df))

    def predict(self, horizon: int = 30) -> pd.DataFrame:
        """Generate future forecasts with uncertainty intervals.

        Args:
            horizon: Number of days to forecast.

        Returns:
            DataFrame with date, predicted_revenue, lower_ci, upper_ci.
        """
        if self.model is None:
            raise RuntimeError("Model must be fit before predicting.")

        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)

        # Take only the forecast period (last `horizon` rows)
        forecast_period = forecast.tail(horizon)

        return pd.DataFrame(
            {
                "date": forecast_period["ds"].dt.date.values,
                "predicted_revenue": np.maximum(forecast_period["yhat"].values, 0),
                "lower_ci": np.maximum(forecast_period["yhat_lower"].values, 0),
                "upper_ci": np.maximum(forecast_period["yhat_upper"].values, 0),
            }
        )

    def save(self, path: Path) -> None:
        """Save the trained Prophet model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "prophet_model.joblib")
        meta = {
            "last_date": str(self.last_date),
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
        }
        with open(path / "prophet_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Prophet model saved to %s", path)

    def load(self, path: Path) -> None:
        """Load a previously trained Prophet model."""
        path = Path(path)
        self.model = joblib.load(path / "prophet_model.joblib")
        with open(path / "prophet_meta.json") as f:
            meta = json.load(f)
        self.last_date = pd.Timestamp(meta["last_date"])
        logger.info("Prophet model loaded from %s", path)
