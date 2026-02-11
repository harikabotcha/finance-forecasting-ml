"""Data loading utilities for financial datasets."""

from datetime import date
from pathlib import Path

import pandas as pd

from src.forecast.config import settings
from src.forecast.logger import get_logger

logger = get_logger(__name__)


def load_csv(
    filepath: str | Path | None = None,
    product_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Load financial data from CSV with optional filtering.

    Args:
        filepath: Path to CSV file. Defaults to raw data directory.
        product_id: Filter to a specific product.
        start_date: Filter records on or after this date.
        end_date: Filter records on or before this date.

    Returns:
        Filtered DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if filepath is None:
        filepath = settings.raw_dir / "financial_data.csv"

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath, parse_dates=["date"])

    # Apply filters
    if product_id is not None:
        df = df[df["product_id"] == product_id]
        logger.info("Filtered to product %s: %d records", product_id, len(df))

    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]

    if end_date is not None:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    df = df.sort_values(["date", "product_id"]).reset_index(drop=True)
    logger.info("Loaded %d records", len(df))
    return df
