import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def valid_prediction_payload():
    return {
        "product_id": "P001",
        "category": "Electronics",
        "price": 100,
        "promotion": 0,
        "discount": 0,
        "holiday": 0,
        "lag_1": 98,
        "lag_7": 130,
        "lag_14": 120,
        "lag_28": 110,
        "rolling_mean_7": 120,
        "rolling_std_7": 30,
        "rolling_mean_14": 118,
        "rolling_std_14": 32,
        "rolling_mean_28": 115,
        "rolling_std_28": 35,
        "day_of_week": 2,
        "month": 6,
        "week_of_year": 23,
        "is_weekend": 0,
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "HistGradientBoostingRegressor"
    assert data["feature_count"] == 18
    assert len(data["features"]) == 18


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json=valid_prediction_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["product_id"] == "P001"
    assert data["category"] == "Electronics"
    assert "predicted_demand" in data
    assert data["predicted_demand"] >= 0


def test_predict_endpoint_returns_deterministic_result():
    payload = valid_prediction_payload()

    response_1 = client.post("/predict", json=payload)
    response_2 = client.post("/predict", json=payload)

    assert response_1.status_code == 200
    assert response_2.status_code == 200

    prediction_1 = response_1.json()["predicted_demand"]
    prediction_2 = response_2.json()["predicted_demand"]

    assert prediction_1 == prediction_2


def test_predict_endpoint_rejects_invalid_promotion():
    payload = valid_prediction_payload()
    payload["promotion"] = 2

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_month():
    payload = valid_prediction_payload()
    payload["month"] = 13

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_missing_feature():
    payload = valid_prediction_payload()
    del payload["lag_7"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422