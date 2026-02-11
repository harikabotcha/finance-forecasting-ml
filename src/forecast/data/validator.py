"""Data validation and quality checks for financial time series."""

import pandas as pd

from src.forecast.data.schemas import ValidationReport
from src.forecast.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["date", "product_id", "revenue", "units_sold", "price"]


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Run comprehensive validation checks on a financial dataset.

    Checks include: required columns, missing values, duplicates, date gaps,
    and statistical outliers.

    Args:
        df: DataFrame to validate.

    Returns:
        ValidationReport with results and any warnings/errors.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return ValidationReport(
            is_valid=False,
            total_records=len(df),
            errors=errors,
        )

    # Check missing values
    missing_values = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
    total_missing = sum(missing_values.values())
    if total_missing > 0:
        warnings.append(f"Found {total_missing} total missing values across columns")

    # Check duplicates (same date + product_id)
    duplicate_count = df.duplicated(subset=["date", "product_id"]).sum()
    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate date-product pairs")

    # Check date gaps per product
    date_gaps: list[str] = []
    for product_id in df["product_id"].unique():
        product_df = df[df["product_id"] == product_id].sort_values("date")
        dates = pd.to_datetime(product_df["date"])
        if len(dates) > 1:
            diffs = dates.diff().dropna()
            gaps = diffs[diffs > pd.Timedelta(days=1)]
            for idx in gaps.index:
                gap_start = dates.loc[idx - 1] if (idx - 1) in dates.index else "unknown"
                date_gaps.append(f"{product_id}: gap after {gap_start} ({gaps.loc[idx].days} days)")

    if date_gaps:
        warnings.append(f"Found {len(date_gaps)} date gaps in the time series")

    # Check for negative values
    if (df["revenue"] < 0).any():
        errors.append("Negative revenue values detected")
    if (df["units_sold"] < 0).any():
        errors.append("Negative units_sold values detected")
    if (df["price"] <= 0).any():
        errors.append("Non-positive price values detected")

    # Check outliers using IQR method on revenue
    outlier_count = 0
    for product_id in df["product_id"].unique():
        product_revenue = df[df["product_id"] == product_id]["revenue"]
        q1 = product_revenue.quantile(0.25)
        q3 = product_revenue.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        n_outliers = ((product_revenue < lower) | (product_revenue > upper)).sum()
        outlier_count += n_outliers

    if outlier_count > 0:
        warnings.append(f"Found {outlier_count} statistical outliers (3x IQR)")

    is_valid = len(errors) == 0
    if is_valid:
        logger.info("Data validation passed — %d records, %d warnings", len(df), len(warnings))
    else:
        logger.error("Data validation failed — %d errors", len(errors))

    return ValidationReport(
        is_valid=is_valid,
        total_records=len(df),
        missing_values=missing_values,
        duplicate_count=duplicate_count,
        date_gaps=date_gaps,
        outlier_count=outlier_count,
        warnings=warnings,
        errors=errors,
    )
