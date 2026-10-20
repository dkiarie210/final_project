from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    payload = {
        "watch_hours": 8.0,
        "last_login_days": 30.0,
        "monthly_fee": 14.99,
        "number_of_profiles": 3.0,
        "avg_watch_time_per_day": 0.3,
        "subscription_type": "Standard",
        "region": "Africa",
        "device": "Mobile",
        "payment_method": "Credit Card",
        "favorite_genre": "Drama"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction_label"] in {"churned", "retained"}
