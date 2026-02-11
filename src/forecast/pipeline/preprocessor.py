"""Data preprocessing: cleaning, normalization, and outlier handling."""

import pandas as pd
import numpy as np

from src.forecast.logger import get_logger

logger = get_logger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using forward fill, then backward fill for leading NaNs.

    Args:
        df: Input DataFrame with potential missing values.

    Returns:
        DataFrame with missing values filled.
    """
    before = df.isnull().sum().sum()
    df = df.sort_values("date")
    df = df.ffill().bfill()
    after = df.isnull().sum().sum()
    if before > 0:
        logger.info("Filled %d missing values (remaining: %d)", before - after, after)
    return df


def remove_outliers(
    df: pd.DataFrame, column: str = "revenue", iqr_multiplier: float = 3.0
) -> pd.DataFrame:
    """Cap outliers using the IQR method (winsorization).

    Values beyond Q1 - multiplier*IQR or Q3 + multiplier*IQR are clipped.

    Args:
        df: Input DataFrame.
        column: Column to check for outliers.
        iqr_multiplier: Multiplier for IQR bounds.

    Returns:
        DataFrame with outliers capped.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    outlier_count = ((df[column] < lower) | (df[column] > upper)).sum()
    if outlier_count > 0:
        logger.info("Capping %d outliers in '%s' to [%.2f, %.2f]", outlier_count, column, lower, upper)

    df = df.copy()
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def normalize_column(
    df: pd.DataFrame, column: str, method: str = "minmax"
) -> tuple[pd.DataFrame, dict]:
    """Normalize a column and return scaling parameters for inverse transform.

    Args:
        df: Input DataFrame.
        column: Column to normalize.
        method: 'minmax' for [0,1] scaling or 'zscore' for standard normal.

    Returns:
        Tuple of (DataFrame with normalized column, scaling parameters dict).
    """
    df = df.copy()
    params: dict = {"method": method, "column": column}

    if method == "minmax":
        min_val = df[column].min()
        max_val = df[column].max()
        range_val = max_val - min_val
        if range_val == 0:
            df[column] = 0.0
        else:
            df[column] = (df[column] - min_val) / range_val
        params["min"] = float(min_val)
        params["max"] = float(max_val)
    elif method == "zscore":
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val == 0:
            df[column] = 0.0
        else:
            df[column] = (df[column] - mean_val) / std_val
        params["mean"] = float(mean_val)
        params["std"] = float(std_val)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return df, params


def preprocess(df: pd.DataFrame, target_column: str = "revenue") -> pd.DataFrame:
    """Run the full preprocessing pipeline on a single-product DataFrame.

    Steps: sort by date → fill missing → remove outliers.

    Args:
        df: Raw data for a single product.
        target_column: The revenue column to clean.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    logger.info("Preprocessing %d records", len(df))

    df = df.sort_values("date").reset_index(drop=True)
    df = handle_missing_values(df)
    df = remove_outliers(df, column=target_column)

    logger.info("Preprocessing complete: %d records", len(df))
    return df
