"""Inference helpers for fine-tuned and baseline models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.embeddings import get_cls_embeddings


def predict_fine_tuned(texts, model, tokenizer, max_length: int = 128, batch_size: int = 32):
    """Predict with a fine-tuned AutoModelForSequenceClassification."""
    if isinstance(texts, str):
        texts = [texts]

    model.eval()
    device = next(model.parameters()).device
    predictions = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        for i, text in enumerate(batch_texts):
            predictions.append(
                {
                    "text": text,
                    "prediction": int(preds[i].item()),
                    "probabilities": probs[i].cpu().numpy(),
                }
            )

    return predictions


def predict_baseline(texts, model, embed_fn=None, clean_func=None, batch_size: int = 32):
    """Predict with LogisticRegression on CLS embeddings (Day 4 baseline).

    Unlike classic TF-IDF baselines, our Day-4 pipeline uses frozen DistilBERT
    CLS vectors. ``embed_fn`` defaults to ``get_cls_embeddings``.
    """
    if isinstance(texts, str):
        texts = [texts]

    if clean_func:
        texts = [clean_func(t) for t in texts]

    if embed_fn is None:
        embed_fn = get_cls_embeddings

    X = embed_fn(texts, batch_size=batch_size)
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    results = []
    for i, text in enumerate(texts):
        results.append(
            {
                "text": text,
                "prediction": int(preds[i]),
                "probabilities": probs[i] if probs is not None else None,
            }
        )
    return results


LABEL_NAMES = {0: "negative", 1: "positive"}


def format_probs(probs) -> str:
    if probs is None:
        return "n/a"
    arr = np.asarray(probs)
    return ", ".join(f"{LABEL_NAMES.get(i, i)}={arr[i]:.3f}" for i in range(len(arr)))
