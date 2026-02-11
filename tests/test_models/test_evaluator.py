"""Tests for evaluation metrics and model evaluator."""

import numpy as np

from src.forecast.models.evaluator import ModelEvaluator
from src.forecast.utils.metrics import compute_all_metrics, directional_accuracy, mae, mape, rmse


class TestMetrics:
    def test_rmse_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert rmse(actual, actual) == 0.0

    def test_rmse_known_value(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.5, 2.5, 3.5])
        assert abs(rmse(actual, predicted) - 0.5) < 1e-6

    def test_mae_perfect_prediction(self):
        actual = np.array([10.0, 20.0, 30.0])
        assert mae(actual, actual) == 0.0

    def test_mape_known_value(self):
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([110.0, 220.0, 330.0])
        # 10% off on each = 10% MAPE
        assert abs(mape(actual, predicted) - 10.0) < 1e-6

    def test_directional_accuracy(self):
        actual = np.array([1, 3, 2, 5])
        predicted = np.array([1, 4, 1, 6])
        # Directions: actual=[up, down, up], predicted=[up, down, up] → 100%
        assert directional_accuracy(actual, predicted) == 100.0

    def test_compute_all_metrics(self):
        actual = np.array([100.0, 110.0, 105.0])
        predicted = np.array([102.0, 108.0, 106.0])
        result = compute_all_metrics(actual, predicted)
        assert "rmse" in result
        assert "mae" in result
        assert "mape" in result
        assert "smape" in result
        assert "directional_accuracy" in result


class TestModelEvaluator:
    def test_evaluate_and_compare(self):
        evaluator = ModelEvaluator()
        actual = np.array([100, 110, 105, 115, 120])
        evaluator.evaluate_model("model_a", actual, np.array([102, 108, 106, 113, 121]))
        evaluator.evaluate_model("model_b", actual, np.array([90, 100, 95, 105, 110]))

        comparison = evaluator.compare_models()
        assert len(comparison) == 2
        assert comparison.index[0] == "model_a"  # Should be sorted by RMSE (lower first)

    def test_get_best_model(self):
        evaluator = ModelEvaluator()
        actual = np.array([100, 110, 105])
        evaluator.evaluate_model("good", actual, np.array([101, 109, 106]))
        evaluator.evaluate_model("bad", actual, np.array([50, 50, 50]))

        assert evaluator.get_best_model("rmse") == "good"
