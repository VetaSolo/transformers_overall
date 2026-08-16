"""
Day 4 — Baseline without fine-tuning the transformer.

CLS embeddings (frozen DistilBERT) + LogisticRegression.
Saves baseline_model.pkl (+ manifest) so Day 6 can reuse this exact model on a
test split the baseline has never been trained on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from src.data_loading import (
    DEFAULT_FRACTION,
    RANDOM_STATE,
    TEST_SIZE,
    data_manifest,
    make_split,
    prepare_dataset,
)
from src.embeddings import get_cls_embeddings, tokenize_texts

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "baseline_results.txt"
BASELINE_PKL = ROOT / "baseline_model.pkl"
BASELINE_MANIFEST = ROOT / "baseline_model.json"
CACHE_DIR = ROOT / "data"


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


def _embed_cached(texts: list[str], name: str, fingerprint: str, batch_size: int) -> np.ndarray:
    cache = CACHE_DIR / f"cls_{name}_{fingerprint}.npy"
    if cache.exists():
        cached = np.load(cache)
        if cached.shape[0] == len(texts):
            print(f"Loading cached embeddings: {cache.name}", flush=True)
            return cached
    print(f"Extracting CLS embeddings for {name} ({len(texts)} texts)...", flush=True)
    emb = get_cls_embeddings(texts, batch_size=batch_size)
    np.save(cache, emb)
    print(f"Saved cache -> {cache.name}", flush=True)
    return emb


def run_baseline(
    data_path: str | None = None,
    max_samples: int | None = None,
    fraction: float | None = DEFAULT_FRACTION,
    batch_size: int = 32,
) -> float:
    print("=" * 60)
    print("ЗАДАЧА 3: Logistic Regression на эмбеддингах")
    print("=" * 60)

    df = prepare_dataset(data_path, fraction=fraction, max_samples=max_samples)
    source = df.attrs.get("source_path", data_path)
    train_df, test_df = make_split(df)
    manifest = data_manifest(df, test_df, fraction=fraction, max_samples=max_samples)

    print(f"Loaded: {source} ({manifest['n_full']} rows)")
    print(f"Using: {len(df)} rows | train={len(train_df)} test={len(test_df)}")
    print(f"Split fingerprint: {manifest['split_fingerprint']}")

    fp = manifest["split_fingerprint"]
    X_train = _embed_cached(train_df["text"].tolist(), "train", fp, batch_size)
    X_test = _embed_cached(test_df["text"].tolist(), "test", fp, batch_size)
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Day 6 must load THIS model — not retrain inside compare.py
    joblib.dump(clf, BASELINE_PKL)
    BASELINE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved baseline model -> {BASELINE_PKL}")
    print(f"Saved manifest -> {BASELINE_MANIFEST}")

    report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(report)
    print(f"macro F1: {f1:.4f}")

    RESULTS_PATH.write_text(
        "Baseline: DistilBERT CLS embeddings + LogisticRegression\n"
        f"data: {source}\n"
        f"model: distilbert-base-uncased (frozen)\n"
        f"n_samples: {len(df)} (of {manifest['n_full']}), "
        f"fraction={fraction}, max_samples={max_samples}\n"
        f"train/test: {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, stratify, "
        f"random_state={RANDOM_STATE}\n"
        f"train={len(train_df)}, test={len(test_df)}\n"
        f"split_fingerprint: {manifest['split_fingerprint']}\n"
        f"classifier: LogisticRegression(max_iter=1000, n_jobs=-1)\n"
        f"artifacts: {BASELINE_PKL.name}, {BASELINE_MANIFEST.name}\n\n"
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
