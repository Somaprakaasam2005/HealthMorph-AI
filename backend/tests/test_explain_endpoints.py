import json

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_explain_shap_minimal_vector():
    features = [10, 20, 30]
    payload = {"features_json": json.dumps(features)}

    resp = client.post("/explain/shap", data=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_type"] == "shap"
    assert data["input_dim"] == len(features)
    shap = data.get("shap", {})
    assert isinstance(shap.get("shap_values"), list)
    assert len(shap.get("shap_values", [])) == len(features)


def test_explain_lime_minimal_vector():
    features = [5, 15, 25]
    payload = {"features_json": json.dumps(features)}

    resp = client.post("/explain/lime", data=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_type"] == "lime"
    assert data["input_dim"] == len(features)
    lime = data.get("lime", {})
    assert "local_features" in lime
    assert lime.get("decision_boundary") == 0.5


def test_explain_counterfactual_recovery():
    features = [60, 40, 20]
    payload = {
        "features_json": json.dumps(features),
        "analysis_type": "recovery",
    }

    resp = client.post("/explain/counterfactual", data=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_type"] == "recovery"
    assert data["input_dim"] == len(features)
    cf = data.get("counterfactual", {})
    assert "current_risk_score" in cf or "boundary_score" in cf
