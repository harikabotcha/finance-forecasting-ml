"""Tests for the XGBoost forecasting model."""

from src.forecast.models.xgboost_model import XGBoostForecaster


class TestXGBoostForecaster:
    def test_fit_and_predict(self, train_test_split):
        train_df, test_df = train_test_split
        model = XGBoostForecaster(n_estimators=50, max_depth=3)
        model.fit(train_df, target_column="revenue")

        predictions = model.predict(horizon=10)
        assert len(predictions) == 10
        assert "date" in predictions.columns
        assert "predicted_revenue" in predictions.columns
        assert (predictions["predicted_revenue"] >= 0).all()

    def test_feature_importance(self, train_test_split):
        train_df, _ = train_test_split
        model = XGBoostForecaster(n_estimators=50)
        model.fit(train_df)

        importance = model.get_feature_importance()
        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())

    def test_save_and_load(self, train_test_split, tmp_path):
        train_df, _ = train_test_split
        model = XGBoostForecaster(n_estimators=50)
        model.fit(train_df)

        model.save(tmp_path / "xgb")
        loaded = XGBoostForecaster()
        loaded.load(tmp_path / "xgb")

        preds1 = model.predict(horizon=5)
        preds2 = loaded.predict(horizon=5)
        assert preds1["predicted_revenue"].tolist() == preds2["predicted_revenue"].tolist()
