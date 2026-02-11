"""Abstract base class for all forecasting models."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseForecaster(ABC):
    """Common interface that all forecasting models must implement."""

    name: str = "base"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, target_column: str = "revenue") -> None:
        """Train the model on historical data.

        Args:
            train_df: Training DataFrame.
            target_column: Name of the target column.
        """

    @abstractmethod
    def predict(self, horizon: int = 30) -> pd.DataFrame:
        """Generate forecasts for the specified horizon.

        Args:
            horizon: Number of future periods to forecast.

        Returns:
            DataFrame with columns: date, predicted_revenue.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to disk.

        Args:
            path: Directory to save model artifacts.
        """

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load a previously trained model from disk.

        Args:
            path: Directory containing model artifacts.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
