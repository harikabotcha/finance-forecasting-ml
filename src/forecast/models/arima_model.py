"""SARIMA forecasting model implementation."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.forecast.logger import get_logger
from src.forecast.models.base import BaseForecaster

logger = get_logger(__name__)


class ARIMAForecaster(BaseForecaster):
    """Seasonal ARIMA (SARIMA) model for univariate time series forecasting."""

    name = "arima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7),
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None
        self.last_date = None
        self.train_series = None

    def fit(self, train_df: pd.DataFrame, target_column: str = "revenue") -> None:
        """Fit SARIMA model on training data.

        Args:
            train_df: Training DataFrame with 'date' and target column.
            target_column: Name of the target column.
        """
        logger.info("Fitting SARIMA%s x %s", self.order, self.seasonal_order)

        series = train_df.set_index(pd.to_datetime(train_df["date"]))[target_column]
        series = series.asfreq("D", method="ffill")
        self.train_series = series

        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = self.model.fit(disp=False, maxiter=200)
        self.last_date = series.index[-1]

        logger.info("SARIMA fitted — AIC: %.2f, BIC: %.2f", self.model_fit.aic, self.model_fit.bic)

    def predict(self, horizon: int = 30) -> pd.DataFrame:
        """Generate future forecasts with confidence intervals.

        Args:
            horizon: Number of days to forecast.

        Returns:
            DataFrame with date, predicted_revenue, lower_ci, upper_ci.
        """
        if self.model_fit is None:
            raise RuntimeError("Model must be fit before predicting.")

        forecast = self.model_fit.get_forecast(steps=horizon)
        predicted = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)

        dates = pd.date_range(start=self.last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        return pd.DataFrame(
            {
                "date": dates.date,
                "predicted_revenue": np.maximum(predicted.values, 0),
                "lower_ci": np.maximum(conf_int.iloc[:, 0].values, 0),
                "upper_ci": np.maximum(conf_int.iloc[:, 1].values, 0),
            }
        )

    def save(self, path: Path) -> None:
        """Save the trained model artifacts."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model_fit, path / "arima_model.joblib")
        meta = {
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "last_date": str(self.last_date),
        }
        with open(path / "arima_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("ARIMA model saved to %s", path)

    def load(self, path: Path) -> None:
        """Load model artifacts from disk."""
        path = Path(path)
        self.model_fit = joblib.load(path / "arima_model.joblib")
        with open(path / "arima_meta.json") as f:
            meta = json.load(f)
        self.order = tuple(meta["order"])
        self.seasonal_order = tuple(meta["seasonal_order"])
        self.last_date = pd.Timestamp(meta["last_date"])
        logger.info("ARIMA model loaded from %s", path)
