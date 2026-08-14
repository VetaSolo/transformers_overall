"""Gradio demo for fine-tuned sentiment model."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "fine_tuned_model"

# Binary IMDB labels (not 3-class)
LABEL_MAP = {0: "Negative", 1: "Positive"}

from src.model_io import assert_finetuned_model_ready

assert_finetuned_model_ready(MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_sentiment(text: str) -> dict[str, float]:
    """Return class probabilities for Gradio Label component."""
    if not text or not text.strip():
        return {LABEL_MAP[0]: 0.0, LABEL_MAP[1]: 0.0}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

    return {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Введите текст для анализа..."),
    outputs=gr.Label(num_top_classes=2, label="Sentiment"),
    title="Sentiment Analysis с DistilBERT",
    description="Fine-tuned DistilBERT на IMDB (25% subset). Binary: Negative / Positive.",
    examples=[
        "This movie was absolutely fantastic!",
        "Terrible, waste of my time.",
        "It was okay, nothing special.",
        "Best film I've seen this year!",
        "Boring and too long.",
    ],
)


if __name__ == "__main__":
    demo.launch()
