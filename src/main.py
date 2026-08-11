"""FastAPI: GET / and POST /predict using the fine-tuned model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models import PredictRequest, PredictResponse
from src.utils import predict_sentiment as heuristic_predict

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "fine_tuned_model"
LABEL_MAP = {0: "negative", 1: "positive"}

app = FastAPI(title="Sentiment API", version="0.2.0")

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    if not MODEL_DIR.exists():
        return
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.eval()
    _model.to(_device)


def predict_with_finetuned(text: str) -> tuple[str, float]:
    _load_model()
    if _model is None or _tokenizer is None:
        return heuristic_predict(text)

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits[0]
        probs = F.softmax(logits, dim=-1)
    pred = int(torch.argmax(probs).item())
    return LABEL_MAP[pred], float(probs[pred].item())


@app.get("/")
def root() -> dict[str, str]:
    backend = "fine_tuned" if MODEL_DIR.exists() else "heuristic_stub"
    return {"status": "ok", "message": "Sentiment API is running", "backend": backend}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    label, score = predict_with_finetuned(payload.text)
    return PredictResponse(label=label, score=score)
