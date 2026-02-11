"""Time-based train/test splitting for time series data."""

import pandas as pd

from src.forecast.config import settings
from src.forecast.logger import get_logger

logger = get_logger(__name__)


def time_based_split(
    df: pd.DataFrame,
    test_size: float | None = None,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets using a time-based cutoff.

    Unlike random splitting, this preserves temporal ordering — the test set
    always comes after the training set chronologically.

    Args:
        df: Input DataFrame sorted by date.
        test_size: Fraction of data to use for testing (default from config).
        date_column: Name of the date column.

    Returns:
        Tuple of (train_df, test_df).
    """
    if test_size is None:
        test_size = settings.test_size

    df = df.sort_values(date_column).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    logger.info(
        "Train/test split: %d train, %d test (%.0f%% test)",
        len(train_df),
        len(test_df),
        test_size * 100,
    )

    return train_df, test_df
