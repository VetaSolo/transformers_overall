from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_predict_positive():
    response = client.post("/predict", json={"text": "This movie was amazing and great"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert 0.0 <= body["score"] <= 1.0


def test_predict_negative():
    response = client.post("/predict", json={"text": "Terrible waste, horrible experience"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "negative"
    assert 0.0 <= body["score"] <= 1.0


def test_predict_empty_rejected():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
