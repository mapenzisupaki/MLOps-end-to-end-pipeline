from fastapi.testclient import TestClient

from api.main import app


def test_health_reports_model_readiness():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "model_ready" in payload
    assert payload["explainability"] == "xgboost_tree_shap_pred_contribs"

