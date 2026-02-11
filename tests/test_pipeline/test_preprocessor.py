"""Tests for the preprocessor module."""

import numpy as np
import pandas as pd

from src.forecast.pipeline.preprocessor import (
    handle_missing_values,
    normalize_column,
    preprocess,
    remove_outliers,
)


class TestHandleMissingValues:
    def test_fills_missing_values(self, sample_timeseries):
        df = sample_timeseries.copy()
        df.loc[5, "revenue"] = np.nan
        df.loc[10, "revenue"] = np.nan
        result = handle_missing_values(df)
        assert result["revenue"].isnull().sum() == 0

    def test_no_change_when_no_missing(self, sample_timeseries):
        result = handle_missing_values(sample_timeseries)
        assert result["revenue"].isnull().sum() == 0


class TestRemoveOutliers:
    def test_caps_extreme_values(self):
        df = pd.DataFrame({"revenue": [100, 110, 105, 108, 112, 10000, 95]})
        result = remove_outliers(df, column="revenue", iqr_multiplier=1.5)
        assert result["revenue"].max() < 10000

    def test_no_change_when_no_outliers(self, sample_timeseries):
        original_max = sample_timeseries["revenue"].max()
        result = remove_outliers(sample_timeseries, iqr_multiplier=5.0)
        # With a generous multiplier, values shouldn't change much
        assert result["revenue"].max() <= original_max


class TestNormalizeColumn:
    def test_minmax_normalizes_to_0_1(self, sample_timeseries):
        result, params = normalize_column(sample_timeseries, "revenue", method="minmax")
        assert result["revenue"].min() >= 0.0
        assert result["revenue"].max() <= 1.0
        assert params["method"] == "minmax"

    def test_zscore_centers_at_zero(self, sample_timeseries):
        result, params = normalize_column(sample_timeseries, "revenue", method="zscore")
        assert abs(result["revenue"].mean()) < 0.01
        assert params["method"] == "zscore"


class TestPreprocess:
    def test_full_preprocessing_pipeline(self, sample_timeseries):
        result = preprocess(sample_timeseries.copy())
        assert len(result) > 0
        assert result["revenue"].isnull().sum() == 0
        # Should be sorted by date
        dates = pd.to_datetime(result["date"])
        assert dates.is_monotonic_increasing
