"""Tests for the forecast API endpoint."""


class TestForecastEndpoint:
    def test_predict_unknown_product_returns_404(self, api_client):
        response = api_client.post(
            "/api/v1/forecast/predict",
            json={"product_id": "UNKNOWN", "horizon": 7},
        )
        assert response.status_code == 404

    def test_predict_requires_product_id(self, api_client):
        response = api_client.post(
            "/api/v1/forecast/predict",
            json={"horizon": 7},
        )
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_horizon(self, api_client):
        response = api_client.post(
            "/api/v1/forecast/predict",
            json={"product_id": "PROD_A", "horizon": 0},
        )
        assert response.status_code == 422

    def test_models_list(self, api_client):
        response = api_client.get("/api/v1/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
