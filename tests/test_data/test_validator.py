"""Tests for data validation."""

import pandas as pd
import numpy as np

from src.forecast.data.validator import validate_dataset


class TestValidateDataset:
    def test_valid_dataset_passes(self, sample_timeseries):
        report = validate_dataset(sample_timeseries)
        assert report.is_valid is True
        assert len(report.errors) == 0

    def test_missing_columns_fails(self):
        df = pd.DataFrame({"date": ["2023-01-01"], "product_id": ["A"]})
        report = validate_dataset(df)
        assert report.is_valid is False
        assert any("Missing required columns" in e for e in report.errors)

    def test_negative_revenue_fails(self, sample_timeseries):
        df = sample_timeseries.copy()
        df.loc[0, "revenue"] = -100
        report = validate_dataset(df)
        assert report.is_valid is False

    def test_missing_values_warning(self, sample_timeseries):
        df = sample_timeseries.copy()
        df.loc[5, "revenue"] = np.nan
        report = validate_dataset(df)
        # Missing values produce a warning, not an error
        assert report.missing_values["revenue"] >= 1

    def test_duplicate_detection(self, sample_timeseries):
        df = pd.concat([sample_timeseries, sample_timeseries.head(3)], ignore_index=True)
        report = validate_dataset(df)
        assert report.duplicate_count >= 3
