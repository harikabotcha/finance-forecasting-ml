"""Standalone script to train all forecasting models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forecast.data.generator import PRODUCT_PROFILES
from src.forecast.logger import get_logger
from src.forecast.models.trainer import train_all_models
from src.forecast.pipeline.orchestrator import run_pipeline

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== Model Training Pipeline ===")

    for product_id in PRODUCT_PROFILES:
        logger.info("Processing product: %s", product_id)
        result = run_pipeline(product_id)
        training_result = train_all_models(
            train_df=result.train_df,
            test_df=result.test_df,
            product_id=product_id,
        )
        comparison = training_result.evaluator.compare_models()
        logger.info("Results for %s:\n%s", product_id, comparison.to_string())

    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    main()
