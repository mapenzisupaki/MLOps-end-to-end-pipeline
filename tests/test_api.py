from fastapi.testclient import TestClient

from api.main import app


def test_health_reports_model_readiness():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_ready" in response.json()
