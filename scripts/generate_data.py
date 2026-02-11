"""Standalone script to generate synthetic financial data."""

import sys
from pathlib import Path

# Allow running as python -m scripts.generate_data
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forecast.data.generator import generate_dataset
from src.forecast.data.validator import validate_dataset
from src.forecast.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== Generating Synthetic Financial Data ===")

    dataset, metadata = generate_dataset(save=True)
    logger.info("Generated dataset: %s", metadata.model_dump_json(indent=2))

    report = validate_dataset(dataset)
    logger.info("Validation result: valid=%s, warnings=%d", report.is_valid, len(report.warnings))

    if not report.is_valid:
        logger.error("Validation errors: %s", report.errors)
        sys.exit(1)

    logger.info("=== Data Generation Complete ===")


if __name__ == "__main__":
    main()
