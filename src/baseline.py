"""
Baseline без обучения трансформера

Задача 1 — токенизация
Задача 2 — CLS-эмбеддинги
Задача 3 — Logistic Regression на эмбеддингах
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.embeddings import get_cls_embeddings, tokenize_texts

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "IMDB Dataset.csv"
EMBEDDINGS_CACHE = ROOT / "data" / "imdb_cls_embeddings.npy"
LABELS_CACHE = ROOT / "data" / "imdb_labels.npy"
RESULTS_PATH = ROOT / "baseline_results.txt"


def load_imdb(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"review", "sentiment"}.issubset(df.columns):
        raise ValueError(f"Expected columns review, sentiment. Got: {list(df.columns)}")
    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)
    return df


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


def run_baseline(max_samples: int | None = None, batch_size: int = 32) -> float:
    print("=" * 60)
    print("ЗАДАЧА 3: Logistic Regression на эмбеддингах")
    print("=" * 60)

    df = load_imdb()
    if max_samples is not None:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Using subset: {len(df)} rows")
    else:
        print(f"Using full dataset: {len(df)} rows")

    texts = df["review"].astype(str).tolist()
    y = df["label"].to_numpy()

    cache_ok = (
        max_samples is None
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
        if max_samples is None:
            np.save(EMBEDDINGS_CACHE, X)
            np.save(LABELS_CACHE, y)
            print(f"Saved embeddings cache -> {EMBEDDINGS_CACHE}")

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(report)
    print(f"macro F1: {f1:.4f}")

    RESULTS_PATH.write_text(
        "Baseline: DistilBERT CLS embeddings + LogisticRegression\n"
        f"model: distilbert-base-uncased (frozen)\n"
        f"n_samples: {len(y)}\n"
        f"train/test: 80/20, stratify=y, random_state=42\n"
        f"classifier: LogisticRegression(max_iter=1000, n_jobs=-1)\n\n"
        f"{report}\n"
        f"macro F1: {f1:.4f}\n",
        encoding="utf-8",
    )
    print(f"Saved: {RESULTS_PATH}")
    return f1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transformer-free baseline on IMDB")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional subset size (default: full 50K). Useful for a quick CPU smoke test.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-demo", action="store_true", help="Skip tasks 1–2 demos")
    args = parser.parse_args()

    if not args.skip_demo:
        demo_tokenization()
        demo_cls_embeddings()

    run_baseline(max_samples=args.max_samples, batch_size=args.batch_size)
