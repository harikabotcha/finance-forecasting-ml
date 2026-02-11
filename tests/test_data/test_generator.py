"""Tests for the synthetic data generator."""

from datetime import date

import pandas as pd

from src.forecast.data.generator import generate_dataset, generate_product_series


class TestGenerateProductSeries:
    def test_generates_correct_date_range(self):
        start = date(2023, 1, 1)
        end = date(2023, 12, 31)
        df = generate_product_series("PROD_A", start, end)
        assert df["date"].iloc[0] == start
        assert df["date"].iloc[-1] == end

    def test_generates_expected_columns(self):
        df = generate_product_series("PROD_A", date(2023, 1, 1), date(2023, 3, 31))
        expected_cols = {"date", "product_id", "revenue", "units_sold", "price"}
        assert set(df.columns) == expected_cols

    def test_no_negative_revenue(self):
        df = generate_product_series("PROD_A", date(2023, 1, 1), date(2023, 12, 31))
        assert (df["revenue"] >= 0).all()

    def test_reproducibility_with_seed(self):
        df1 = generate_product_series("PROD_A", date(2023, 1, 1), date(2023, 6, 30), seed=42)
        df2 = generate_product_series("PROD_A", date(2023, 1, 1), date(2023, 6, 30), seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_product_series("PROD_A", date(2023, 1, 1), date(2023, 6, 30), seed=1)
        df2 = generate_product_series("PROD_A", date(2023, 1, 1), date(2023, 6, 30), seed=2)
        assert not df1["revenue"].equals(df2["revenue"])


class TestGenerateDataset:
    def test_generates_all_products(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.forecast.data.generator.settings.data_raw_path", str(tmp_path))
        df, meta = generate_dataset(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 3, 31),
            save=True,
        )
        assert meta.num_products == 3
        assert set(df["product_id"].unique()) == {"PROD_A", "PROD_B", "PROD_C"}

    def test_metadata_is_accurate(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.forecast.data.generator.settings.data_raw_path", str(tmp_path))
        df, meta = generate_dataset(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            products=["PROD_A"],
            save=False,
        )
        assert meta.num_records == len(df)
        assert meta.num_products == 1
        assert meta.date_start == date(2023, 1, 1)
        assert meta.date_end == date(2023, 1, 31)
