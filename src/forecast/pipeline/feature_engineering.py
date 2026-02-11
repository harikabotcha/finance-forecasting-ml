"""Feature engineering for financial time series forecasting."""

import pandas as pd
import numpy as np

from src.forecast.logger import get_logger

logger = get_logger(__name__)


def add_lag_features(
    df: pd.DataFrame, column: str = "revenue", lags: list[int] | None = None
) -> pd.DataFrame:
    """Create lagged versions of the target column.

    Args:
        df: Input DataFrame sorted by date.
        column: Column to create lags for.
        lags: List of lag periods in days. Default: [1, 7, 14, 30].

    Returns:
        DataFrame with new lag columns.
    """
    if lags is None:
        lags = [1, 7, 14, 30]

    df = df.copy()
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)

    logger.info("Added %d lag features: %s", len(lags), lags)
    return df


def add_rolling_features(
    df: pd.DataFrame, column: str = "revenue", windows: list[int] | None = None
) -> pd.DataFrame:
    """Create rolling window statistics (mean, std) for the target column.

    Args:
        df: Input DataFrame sorted by date.
        column: Column to compute rolling stats for.
        windows: List of window sizes in days. Default: [7, 30].

    Returns:
        DataFrame with new rolling feature columns.
    """
    if windows is None:
        windows = [7, 30]

    df = df.copy()
    for window in windows:
        df[f"{column}_rolling_mean_{window}"] = (
            df[column].rolling(window=window, min_periods=1).mean()
        )
        df[f"{column}_rolling_std_{window}"] = (
            df[column].rolling(window=window, min_periods=1).std().fillna(0)
        )

    logger.info("Added rolling features for windows: %s", windows)
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar-based features from the date column.

    Creates: day_of_week, day_of_month, month, quarter, is_weekend,
    week_of_year, is_month_start, is_month_end.

    Args:
        df: Input DataFrame with a 'date' column.

    Returns:
        DataFrame with new date feature columns.
    """
    df = df.copy()
    dates = pd.to_datetime(df["date"])

    df["day_of_week"] = dates.dt.dayofweek
    df["day_of_month"] = dates.dt.day
    df["month"] = dates.dt.month
    df["quarter"] = dates.dt.quarter
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dates.dt.is_month_start.astype(int)
    df["is_month_end"] = dates.dt.is_month_end.astype(int)

    logger.info("Added 8 date-based features")
    return df


def add_yoy_change(df: pd.DataFrame, column: str = "revenue") -> pd.DataFrame:
    """Add year-over-year percentage change.

    Args:
        df: Input DataFrame sorted by date.
        column: Column to compute YoY change for.

    Returns:
        DataFrame with a YoY change column.
    """
    df = df.copy()
    df[f"{column}_yoy_change"] = df[column].pct_change(periods=365).fillna(0)
    logger.info("Added year-over-year change feature")
    return df


def engineer_features(df: pd.DataFrame, target_column: str = "revenue") -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Args:
        df: Preprocessed DataFrame for a single product.
        target_column: The revenue column to derive features from.

    Returns:
        DataFrame with all engineered features. Rows with NaN from lagging are dropped.
    """
    logger.info("Engineering features for %d records", len(df))

    df = add_lag_features(df, column=target_column)
    df = add_rolling_features(df, column=target_column)
    df = add_date_features(df)
    df = add_yoy_change(df, column=target_column)

    # Drop rows with NaN values introduced by lagging
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows with NaN from lag/rolling features", dropped)

    logger.info("Feature engineering complete: %d records, %d features", len(df), len(df.columns))
    return df
