"""Pipeline orchestrator — coordinates the full data processing workflow."""

import pandas as pd

from src.forecast.config import settings
from src.forecast.data.loader import load_csv
from src.forecast.data.validator import validate_dataset
from src.forecast.logger import get_logger
from src.forecast.pipeline.data_splitter import time_based_split
from src.forecast.pipeline.feature_engineering import engineer_features
from src.forecast.pipeline.preprocessor import preprocess

logger = get_logger(__name__)


class PipelineResult:
    """Container for pipeline outputs."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame,
        product_id: str,
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.full_df = full_df
        self.product_id = product_id

    def __repr__(self) -> str:
        return (
            f"PipelineResult(product={self.product_id}, "
            f"train={len(self.train_df)}, test={len(self.test_df)})"
        )


def run_pipeline(
    product_id: str,
    filepath: str | None = None,
) -> PipelineResult:
    """Execute the end-to-end data pipeline for a single product.

    Steps:
        1. Load data (from CSV)
        2. Validate data quality
        3. Preprocess (clean, handle outliers)
        4. Engineer features (lags, rolling stats, calendar)
        5. Split into train/test sets

    Args:
        product_id: Product to process.
        filepath: Optional path to CSV file.

    Returns:
        PipelineResult with train/test DataFrames.

    Raises:
        ValueError: If data validation fails.
    """
    logger.info("=" * 60)
    logger.info("Running pipeline for product: %s", product_id)
    logger.info("=" * 60)

    # Step 1: Load
    df = load_csv(filepath=filepath, product_id=product_id)
    logger.info("Step 1/5 — Loaded %d records", len(df))

    # Step 2: Validate
    report = validate_dataset(df)
    if not report.is_valid:
        raise ValueError(f"Data validation failed: {report.errors}")
    logger.info("Step 2/5 — Validation passed (%d warnings)", len(report.warnings))

    # Step 3: Preprocess
    df = preprocess(df, target_column="revenue")
    logger.info("Step 3/5 — Preprocessing complete")

    # Step 4: Feature engineering
    df = engineer_features(df, target_column="revenue")
    logger.info("Step 4/5 — Feature engineering complete (%d features)", len(df.columns))

    # Step 5: Split
    train_df, test_df = time_based_split(df)
    logger.info("Step 5/5 — Split complete: train=%d, test=%d", len(train_df), len(test_df))

    # Save processed data
    output_path = settings.processed_dir / f"{product_id}_processed.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved processed data to %s", output_path)

    return PipelineResult(
        train_df=train_df,
        test_df=test_df,
        full_df=df,
        product_id=product_id,
    )


def run_all_pipelines(
    products: list[str] | None = None,
) -> dict[str, PipelineResult]:
    """Run the pipeline for all products.

    Args:
        products: List of product IDs. Defaults to all available.

    Returns:
        Dictionary mapping product_id to PipelineResult.
    """
    if products is None:
        from src.forecast.data.generator import PRODUCT_PROFILES
        products = list(PRODUCT_PROFILES.keys())

    results = {}
    for product_id in products:
        result = run_pipeline(product_id)
        results[product_id] = result
        logger.info("Pipeline result: %s", result)

    logger.info("All pipelines complete: %d products processed", len(results))
    return results
