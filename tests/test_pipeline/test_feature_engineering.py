"""Tests for the feature engineering module."""

import pandas as pd

from src.forecast.pipeline.feature_engineering import (
    add_date_features,
    add_lag_features,
    add_rolling_features,
    add_yoy_change,
    engineer_features,
)


class TestAddLagFeatures:
    def test_creates_lag_columns(self, sample_timeseries):
        result = add_lag_features(sample_timeseries, lags=[1, 7])
        assert "revenue_lag_1" in result.columns
        assert "revenue_lag_7" in result.columns

    def test_lag_values_are_correct(self, sample_timeseries):
        result = add_lag_features(sample_timeseries, lags=[1])
        # The lag_1 value at index 5 should be the revenue at index 4
        assert result["revenue_lag_1"].iloc[5] == result["revenue"].iloc[4]


class TestAddRollingFeatures:
    def test_creates_rolling_columns(self, sample_timeseries):
        result = add_rolling_features(sample_timeseries, windows=[7])
        assert "revenue_rolling_mean_7" in result.columns
        assert "revenue_rolling_std_7" in result.columns


class TestAddDateFeatures:
    def test_creates_all_date_features(self, sample_timeseries):
        result = add_date_features(sample_timeseries)
        expected = [
            "day_of_week", "day_of_month", "month", "quarter",
            "week_of_year", "is_weekend", "is_month_start", "is_month_end",
        ]
        for col in expected:
            assert col in result.columns

    def test_weekend_detection(self, sample_timeseries):
        result = add_date_features(sample_timeseries)
        # Check that weekends are correctly identified
        assert result["is_weekend"].isin([0, 1]).all()


class TestAddYoYChange:
    def test_creates_yoy_column(self, sample_timeseries):
        result = add_yoy_change(sample_timeseries)
        assert "revenue_yoy_change" in result.columns


class TestEngineerFeatures:
    def test_full_feature_pipeline(self, sample_timeseries):
        from src.forecast.pipeline.preprocessor import preprocess

        df = preprocess(sample_timeseries.copy())
        result = engineer_features(df)
        # Should have more columns than original
        assert len(result.columns) > len(sample_timeseries.columns)
        # Should have no NaN values
        assert result.isnull().sum().sum() == 0
        # Should have fewer rows due to lag dropping
        assert len(result) < len(df)
