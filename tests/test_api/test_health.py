"""Tests for the health check endpoint."""


class TestHealthEndpoint:
    def test_health_returns_200(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, api_client):
        response = api_client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_root_endpoint(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "docs" in data
