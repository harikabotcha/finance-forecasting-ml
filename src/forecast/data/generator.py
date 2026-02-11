"""Synthetic financial data generator for reproducible development and testing."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.forecast.config import settings
from src.forecast.data.schemas import DatasetMetadata
from src.forecast.logger import get_logger

logger = get_logger(__name__)

PRODUCT_PROFILES = {
    "PROD_A": {
        "base_revenue": 5000.0,
        "trend_slope": 2.5,
        "yearly_amplitude": 1500.0,
        "weekly_amplitude": 800.0,
        "noise_std": 300.0,
        "base_price": 49.99,
    },
    "PROD_B": {
        "base_revenue": 12000.0,
        "trend_slope": 5.0,
        "yearly_amplitude": 3000.0,
        "weekly_amplitude": 1200.0,
        "noise_std": 600.0,
        "base_price": 129.99,
    },
    "PROD_C": {
        "base_revenue": 2500.0,
        "trend_slope": 1.0,
        "yearly_amplitude": 800.0,
        "weekly_amplitude": 400.0,
        "noise_std": 200.0,
        "base_price": 19.99,
    },
}


def generate_product_series(
    product_id: str,
    start_date: date,
    end_date: date,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic revenue time series for a single product.

    Creates daily data with trend, yearly seasonality, weekly seasonality, and noise.

    Args:
        product_id: Product identifier (must be in PRODUCT_PROFILES).
        start_date: Start date for the series.
        end_date: End date for the series.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: date, product_id, revenue, units_sold, price.
    """
    rng = np.random.default_rng(seed)
    profile = PRODUCT_PROFILES[product_id]

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)
    day_index = np.arange(n_days)

    # Trend component: gradual linear growth
    trend = profile["base_revenue"] + profile["trend_slope"] * day_index

    # Yearly seasonality: peaks in Q4 (holiday season)
    yearly = profile["yearly_amplitude"] * np.sin(
        2 * np.pi * (day_index - 60) / 365.25
    )

    # Weekly seasonality: higher on weekdays, dip on weekends
    day_of_week = dates.dayofweek.values
    weekly = np.where(
        day_of_week < 5,
        profile["weekly_amplitude"] * 0.3,
        -profile["weekly_amplitude"] * 0.7,
    )

    # Random noise
    noise = rng.normal(0, profile["noise_std"], n_days)

    # Combine and ensure non-negative
    revenue = np.maximum(trend + yearly + weekly + noise, 0)

    # Derive units sold and price
    price_variation = rng.uniform(0.95, 1.05, n_days)
    price = profile["base_price"] * price_variation
    units_sold = np.maximum((revenue / price).astype(int), 0)

    # Recalculate revenue from units * price for consistency
    revenue = units_sold * price

    return pd.DataFrame(
        {
            "date": dates.date,
            "product_id": product_id,
            "revenue": np.round(revenue, 2),
            "units_sold": units_sold,
            "price": np.round(price, 2),
        }
    )


def generate_dataset(
    start_date: date | None = None,
    end_date: date | None = None,
    products: list[str] | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Generate a complete synthetic financial dataset for all products.

    Args:
        start_date: Start date (default: 3 years ago from today).
        end_date: End date (default: today).
        products: List of product IDs (default: all defined profiles).
        save: Whether to save the dataset to CSV.

    Returns:
        Tuple of (DataFrame, DatasetMetadata).
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=3 * 365)
    if end_date is None:
        end_date = date.today()
    if products is None:
        products = list(PRODUCT_PROFILES.keys())

    logger.info(
        "Generating synthetic data for %d products from %s to %s",
        len(products),
        start_date,
        end_date,
    )

    frames = []
    for i, product_id in enumerate(products):
        df = generate_product_series(
            product_id=product_id,
            start_date=start_date,
            end_date=end_date,
            seed=settings.random_seed + i,
        )
        frames.append(df)

    dataset = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "product_id"]
    ).reset_index(drop=True)

    metadata = DatasetMetadata(
        num_records=len(dataset),
        num_products=len(products),
        date_start=start_date,
        date_end=end_date,
        products=products,
        generated_at=pd.Timestamp.now().isoformat(),
    )

    if save:
        output_path = settings.raw_dir / "financial_data.csv"
        dataset.to_csv(output_path, index=False)
        logger.info("Saved dataset to %s (%d records)", output_path, len(dataset))

    return dataset, metadata
