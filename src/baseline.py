"""
Day 4 — Baseline without fine-tuning the transformer.

CLS embeddings (frozen DistilBERT) + LogisticRegression.
Saves baseline_model.pkl for Day 6 comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.data_loading import (
    DEFAULT_FRACTION,
    RANDOM_STATE,
    load_sentiment_csv,
    subsample_stratified,
)
from src.embeddings import get_cls_embeddings, tokenize_texts

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "baseline_results.txt"
BASELINE_PKL = ROOT / "baseline_model.pkl"
EMBEDDINGS_CACHE = ROOT / "data" / "cls_embeddings.npy"
LABELS_CACHE = ROOT / "data" / "cls_labels.npy"


def demo_tokenization() -> None:
    print("=" * 60)
    print("ЗАДАЧА 1: Токенизация")
    print("=" * 60)

    samples = [
        "This movie was absolutely amazing!",
        "Terrible experience, would not recommend.",
        "An average film with a few memorable scenes.",
    ]
    batch = tokenize_texts(samples, max_length=128)

    print(f"texts: {len(samples)}")
    print(f"input_ids shape:      {tuple(batch['input_ids'].shape)}")
    print(f"attention_mask shape: {tuple(batch['attention_mask'].shape)}")
    print(f"input_ids[0][:20]:    {batch['input_ids'][0][:20].tolist()}")
    print()


def demo_cls_embeddings() -> None:
    print("=" * 60)
    print("ЗАДАЧА 2: CLS-эмбеддинги")
    print("=" * 60)

    samples = [
        "This movie was absolutely amazing!",
        "Terrible experience, would not recommend.",
        "An average film with a few memorable scenes.",
    ]
    emb = get_cls_embeddings(samples, batch_size=2)
    print(f"embeddings shape: {emb.shape}  (n_texts, hidden_size)")
    print(f"dtype: {emb.dtype}")
    print()


def run_baseline(
    data_path: str | None = None,
    max_samples: int | None = None,
    fraction: float | None = DEFAULT_FRACTION,
    batch_size: int = 32,
) -> float:
    print("=" * 60)
    print("ЗАДАЧА 3: Logistic Regression на эмбеддингах")
    print("=" * 60)

    df = load_sentiment_csv(data_path)
    source = df.attrs.get("source_path", data_path)
    print(f"Loaded: {source} ({len(df)} rows)")

    df = subsample_stratified(df, fraction=fraction, max_samples=max_samples)
    print(f"Using: {len(df)} rows for baseline")

    texts = df["text"].tolist()
    y = df["label"].to_numpy()

    cache_ok = (
        max_samples is None
        and fraction == DEFAULT_FRACTION
        and EMBEDDINGS_CACHE.exists()
        and LABELS_CACHE.exists()
        and np.load(LABELS_CACHE).shape[0] == len(texts)
    )

    if cache_ok:
        print(f"Loading cached embeddings: {EMBEDDINGS_CACHE}")
        X = np.load(EMBEDDINGS_CACHE)
        y = np.load(LABELS_CACHE)
    else:
        print(f"Extracting CLS embeddings (batch_size={batch_size})...")
        X = get_cls_embeddings(texts, batch_size=batch_size)
        if max_samples is None and fraction == DEFAULT_FRACTION:
            np.save(EMBEDDINGS_CACHE, X)
            np.save(LABELS_CACHE, y)
            print(f"Saved embeddings cache -> {EMBEDDINGS_CACHE}")

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Day 6 must load THIS model — not retrain inside compare.py
    joblib.dump(clf, BASELINE_PKL)
    print(f"Saved baseline model -> {BASELINE_PKL}")

    report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(report)
    print(f"macro F1: {f1:.4f}")

    RESULTS_PATH.write_text(
        "Baseline: DistilBERT CLS embeddings + LogisticRegression\n"
        f"data: {source}\n"
        f"model: distilbert-base-uncased (frozen)\n"
        f"n_samples: {len(y)}\n"
        f"train/test: 80/20, stratify=y, random_state={RANDOM_STATE}\n"
        f"classifier: LogisticRegression(max_iter=1000, n_jobs=-1)\n"
        f"artifact: {BASELINE_PKL.name}\n\n"
        f"{report}\n"
        f"macro F1: {f1:.4f}\n",
        encoding="utf-8",
    )
    print(f"Saved: {RESULTS_PATH}")
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline: CLS + LogisticRegression")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="CSV path (text,label or review,sentiment). Default: data/dataset.csv or IMDB.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-demo", action="store_true")
    args = parser.parse_args()

    if not args.skip_demo:
        demo_tokenization()
        demo_cls_embeddings()

    run_baseline(
        data_path=args.data,
        max_samples=args.max_samples,
        fraction=args.fraction,
        batch_size=args.batch_size,
    )
