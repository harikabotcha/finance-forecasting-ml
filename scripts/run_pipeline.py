"""End-to-end pipeline: generate data → process → train → evaluate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forecast.data.generator import PRODUCT_PROFILES, generate_dataset
from src.forecast.data.validator import validate_dataset
from src.forecast.logger import get_logger
from src.forecast.models.trainer import train_all_models
from src.forecast.pipeline.orchestrator import run_pipeline

logger = get_logger(__name__)


def main() -> None:
    logger.info("=" * 60)
    logger.info("FINANCE FORECASTING ML — END-TO-END PIPELINE")
    logger.info("=" * 60)

    # Step 1: Generate data
    logger.info("\n>>> Step 1: Generating synthetic data...")
    dataset, metadata = generate_dataset(save=True)
    report = validate_dataset(dataset)
    if not report.is_valid:
        logger.error("Data validation failed: %s", report.errors)
        sys.exit(1)
    logger.info("Data generated: %d records for %d products", metadata.num_records, metadata.num_products)

    # Step 2: Process and train for each product
    all_results = {}
    for product_id in PRODUCT_PROFILES:
        logger.info("\n>>> Processing %s...", product_id)

        pipeline_result = run_pipeline(product_id)
        training_result = train_all_models(
            train_df=pipeline_result.train_df,
            test_df=pipeline_result.test_df,
            product_id=product_id,
        )
        all_results[product_id] = training_result

    # Step 3: Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for product_id, result in all_results.items():
        best = result.evaluator.get_best_model()
        logger.info("  %s — Best model: %s", product_id, best)

    logger.info("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
