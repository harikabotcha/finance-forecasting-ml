"""XGBoost forecasting model using engineered features."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.forecast.logger import get_logger
from src.forecast.models.base import BaseForecaster

logger = get_logger(__name__)

# Features used for XGBoost training (must match feature engineering output)
FEATURE_COLUMNS = [
    "revenue_lag_1",
    "revenue_lag_7",
    "revenue_lag_14",
    "revenue_lag_30",
    "revenue_rolling_mean_7",
    "revenue_rolling_std_7",
    "revenue_rolling_mean_30",
    "revenue_rolling_std_30",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "week_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "revenue_yoy_change",
]


class XGBoostForecaster(BaseForecaster):
    """XGBoost gradient boosting model for tabular time series forecasting."""

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.feature_columns = FEATURE_COLUMNS
        self.last_row = None
        self.last_date = None

    def fit(self, train_df: pd.DataFrame, target_column: str = "revenue") -> None:
        """Fit XGBoost on engineered features.

        Args:
            train_df: Training DataFrame with engineered features.
            target_column: Name of the target column.
        """
        available_features = [c for c in self.feature_columns if c in train_df.columns]
        logger.info("Fitting XGBoost with %d features", len(available_features))

        X = train_df[available_features].values
        y = train_df[target_column].values

        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbosity=0,
        )
        self.model.fit(X, y)
        self.feature_columns = available_features
        self.last_row = train_df.iloc[-1].to_dict()
        self.last_date = pd.to_datetime(train_df["date"]).max()

        # Log feature importance
        importances = dict(zip(available_features, self.model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top 5 features: %s", top_features)

    def predict(self, horizon: int = 30) -> pd.DataFrame:
        """Generate forecasts using recursive multi-step prediction.

        For multi-step ahead forecasting, each prediction is fed back as input
        for subsequent steps (using lag features).

        Args:
            horizon: Number of days to forecast.

        Returns:
            DataFrame with date and predicted_revenue.
        """
        if self.model is None:
            raise RuntimeError("Model must be fit before predicting.")

        predictions = []
        current = self.last_row.copy()
        current_date = self.last_date

        for step in range(horizon):
            current_date = current_date + pd.Timedelta(days=1)

            # Update date features
            current["day_of_week"] = current_date.dayofweek
            current["day_of_month"] = current_date.day
            current["month"] = current_date.month
            current["quarter"] = (current_date.month - 1) // 3 + 1
            current["week_of_year"] = current_date.isocalendar()[1]
            current["is_weekend"] = int(current_date.dayofweek >= 5)
            current["is_month_start"] = int(current_date.day == 1)
            current["is_month_end"] = int(
                current_date.day == pd.Timestamp(current_date).days_in_month
            )

            # Build feature vector
            X = np.array([[current.get(f, 0) for f in self.feature_columns]])
            pred = float(self.model.predict(X)[0])
            pred = max(pred, 0)
            predictions.append({"date": current_date.date(), "predicted_revenue": pred})

            # Shift lag features for next step
            if "revenue_lag_30" in current:
                current["revenue_lag_30"] = current.get("revenue_lag_14", pred)
            if "revenue_lag_14" in current:
                current["revenue_lag_14"] = current.get("revenue_lag_7", pred)
            if "revenue_lag_7" in current:
                current["revenue_lag_7"] = current.get("revenue_lag_1", pred)
            if "revenue_lag_1" in current:
                current["revenue_lag_1"] = pred

        return pd.DataFrame(predictions)

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores.

        Returns:
            Dictionary mapping feature names to importance values.
        """
        if self.model is None:
            raise RuntimeError("Model must be fit first.")
        return dict(zip(self.feature_columns, self.model.feature_importances_))

    def save(self, path: Path) -> None:
        """Save XGBoost model and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "xgboost_model.joblib")
        meta = {
            "feature_columns": self.feature_columns,
            "last_date": str(self.last_date),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }
        with open(path / "xgboost_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save last row for recursive prediction
        last_row_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in self.last_row.items() if isinstance(v, (int, float, np.integer, np.floating))}
        with open(path / "xgboost_last_row.json", "w") as f:
            json.dump(last_row_serializable, f, indent=2)

        logger.info("XGBoost model saved to %s", path)

    def load(self, path: Path) -> None:
        """Load XGBoost model and metadata."""
        path = Path(path)
        self.model = joblib.load(path / "xgboost_model.joblib")
        with open(path / "xgboost_meta.json") as f:
            meta = json.load(f)
        self.feature_columns = meta["feature_columns"]
        self.last_date = pd.Timestamp(meta["last_date"])

        with open(path / "xgboost_last_row.json") as f:
            self.last_row = json.load(f)

        logger.info("XGBoost model loaded from %s", path)
