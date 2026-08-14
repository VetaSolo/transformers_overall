"""FastAPI demo: GET / and POST /predict using the fine-tuned model only."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model_io import assert_finetuned_model_ready, model_weight_files
from src.models import PredictRequest, PredictResponse

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "fine_tuned_model"
LABEL_MAP = {0: "negative", 1: "positive", 2: "neutral"}

app = FastAPI(title="Sentiment API", version="0.3.1")

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=f"{exc}. Run: python -m src.finetune",
    )


def _load_model() -> None:
    global _tokenizer, _model
    if _model is not None:
        return
    try:
        assert_finetuned_model_ready(MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()
        _model.to(_device)
    except Exception as exc:  # noqa: BLE001 — surface as 503 for API clients
        _tokenizer = None
        _model = None
        raise _model_unavailable(exc) from exc


def predict_with_finetuned(text: str) -> tuple[str, float]:
    _load_model()
    assert _model is not None and _tokenizer is not None

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits[0]
        probs = F.softmax(logits, dim=-1)
    pred = int(torch.argmax(probs).item())
    label = LABEL_MAP.get(pred, str(pred))
    return label, float(probs[pred].item())


@app.get("/")
def root() -> dict:
    """Health-check: directory AND weight files must exist; model must load."""
    try:
        weight = assert_finetuned_model_ready(MODEL_DIR)
        _load_model()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _model_unavailable(exc) from exc

    return {
        "status": "ok",
        "message": "Sentiment API is running",
        "backend": "fine_tuned",
        "model_dir": str(MODEL_DIR),
        "weights": weight.name,
        "weight_files": [p.name for p in model_weight_files(MODEL_DIR)],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    label, score = predict_with_finetuned(payload.text)
    return PredictResponse(label=label, score=score)
