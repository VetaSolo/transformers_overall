"""Baseline helpers: tokenization and CLS embeddings (no transformer fine-tuning)."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)


def tokenize_texts(texts: list[str], max_length: int = 128):
    """Tokenize a list of texts into a PyTorch batch."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


@torch.no_grad()
def get_cls_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Extract CLS embeddings for all texts in batches."""
    all_embeddings: list[np.ndarray] = []
    total = (len(texts) + batch_size - 1) // batch_size

    for i, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenize_texts(batch_texts)
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        outputs = model(**encoded)
        cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls)
        if i == 1 or i % 20 == 0 or i == total:
            print(f"  CLS batches: {i}/{total}", flush=True)

    return np.vstack(all_embeddings)
