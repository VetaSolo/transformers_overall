from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.data_loading import load_sentiment_csv
from src.main import MODEL_DIR, app

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample_assignment.csv"
IMDB_CSV = ROOT / "data" / "IMDB Dataset.csv"


def test_load_assignment_format_text_label():
    df = load_sentiment_csv(SAMPLE_CSV)
    assert list(df.columns) == ["text", "label"]
    assert set(df["label"].unique()).issubset({0, 1, 2})
    assert len(df) >= 2


@pytest.mark.skipif(not IMDB_CSV.exists(), reason="IMDB CSV not present")
def test_load_imdb_aliases_review_sentiment():
    df = load_sentiment_csv(IMDB_CSV)
    assert list(df.columns) == ["text", "label"]
    assert set(df["label"].unique()) == {0, 1}


client = TestClient(app)


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="fine_tuned_model/ missing — run Day 5")
def test_root_uses_finetuned_backend():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"] == "fine_tuned"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="fine_tuned_model/ missing — run Day 5")
def test_predict_positive():
    response = client.post("/predict", json={"text": "This movie was amazing and great"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert 0.0 <= body["score"] <= 1.0


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="fine_tuned_model/ missing — run Day 5")
def test_predict_negative():
    response = client.post("/predict", json={"text": "Terrible waste, horrible experience"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "negative"
    assert 0.0 <= body["score"] <= 1.0


def test_predict_empty_rejected():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_missing_model_returns_503(monkeypatch, tmp_path):
    """API must not report ok without loadable weights."""
    import src.main as main_mod

    empty = tmp_path / "fine_tuned_model"
    empty.mkdir()
    (empty / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(main_mod, "MODEL_DIR", empty)
    main_mod._model = None
    main_mod._tokenizer = None
    c = TestClient(main_mod.app)
    response = c.get("/")
    assert response.status_code == 503
    response = c.post("/predict", json={"text": "hello great"})
    assert response.status_code == 503


def test_root_rejects_dir_without_weights(monkeypatch, tmp_path):
    import src.main as main_mod

    bare = tmp_path / "fine_tuned_model"
    bare.mkdir()
    (bare / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main_mod, "MODEL_DIR", bare)
    main_mod._model = None
    main_mod._tokenizer = None
    assert TestClient(main_mod.app).get("/").status_code == 503
