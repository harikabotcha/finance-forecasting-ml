"""Shared test fixtures."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_dates():
    """Generate 365 days of dates."""
    start = date(2022, 1, 1)
    return [start + timedelta(days=i) for i in range(365)]


@pytest.fixture
def sample_timeseries(sample_dates):
    """Generate a simple synthetic time series DataFrame."""
    n = len(sample_dates)
    rng = np.random.default_rng(42)
    revenue = 5000 + np.arange(n) * 2 + rng.normal(0, 200, n)
    revenue = np.maximum(revenue, 0)
    price = rng.uniform(45, 55, n)
    units = (revenue / price).astype(int)

    return pd.DataFrame(
        {
            "date": sample_dates,
            "product_id": "TEST_PROD",
            "revenue": np.round(revenue, 2),
            "units_sold": units,
            "price": np.round(price, 2),
        }
    )


@pytest.fixture
def sample_timeseries_with_features(sample_timeseries):
    """Time series with engineered features added."""
    from src.forecast.pipeline.feature_engineering import engineer_features
    from src.forecast.pipeline.preprocessor import preprocess

    df = preprocess(sample_timeseries)
    df = engineer_features(df)
    return df


@pytest.fixture
def train_test_split(sample_timeseries_with_features):
    """Pre-split train/test data."""
    from src.forecast.pipeline.data_splitter import time_based_split

    return time_based_split(sample_timeseries_with_features, test_size=0.2)


@pytest.fixture
def api_client():
    """FastAPI test client."""
    from src.forecast.api.main import app

    return TestClient(app)
