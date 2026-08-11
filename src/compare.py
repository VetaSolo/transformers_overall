"""
Compare baseline (CLS + LogisticRegression) vs fine-tuned DistilBERT.

Day: inference & comparison.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.embeddings import get_cls_embeddings
from src.predict import LABEL_NAMES, format_probs, predict_baseline, predict_fine_tuned

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "IMDB Dataset.csv"
FT_DIR = ROOT / "fine_tuned_model"
BASELINE_PKL = ROOT / "baseline_model.pkl"
RESULTS_PATH = ROOT / "comparison_results.txt"
CM_FT_PATH = ROOT / "confusion_matrix_finetuned.png"
CM_BASE_PATH = ROOT / "confusion_matrix_baseline.png"

# Same fraction / seed as fine-tuning for a fair hold-out comparison
FRACTION = 0.25
RANDOM_STATE = 42

EXAMPLE_TEXTS = [
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "It was okay, nothing special.",
    "Best film I've seen this year!",
    "Boring and too long.",
]


def load_imdb_subset(fraction: float = FRACTION) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["text"] = df["review"].astype(str)
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)

    n_full = len(df)
    target_n = max(2, int(n_full * fraction))
    n_per = max(1, target_n // 2)
    parts = [
        g.sample(n=min(len(g), n_per), random_state=RANDOM_STATE)
        for _, g in df.groupby("label")
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def ensure_baseline_model(train_texts: list[str], train_labels: list[int]):
    """Load baseline LR from disk, or fit on CLS embeddings and save."""
    if BASELINE_PKL.exists():
        print(f"Loading baseline: {BASELINE_PKL}", flush=True)
        return joblib.load(BASELINE_PKL)

    print(
        f"Training baseline LogisticRegression on CLS embeddings ({len(train_texts)} texts)...",
        flush=True,
    )
    X_train = get_cls_embeddings(train_texts, batch_size=32)
    print(f"Train embeddings shape: {X_train.shape}", flush=True)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)
    joblib.dump(clf, BASELINE_PKL)
    print(f"Saved baseline -> {BASELINE_PKL}", flush=True)
    return clf


def plot_confusion(cm, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[LABEL_NAMES[0], LABEL_NAMES[1]],
        yticklabels=[LABEL_NAMES[0], LABEL_NAMES[1]],
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main() -> None:
    # --- Task 1: load fine-tuned ---
    if not FT_DIR.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {FT_DIR}. Run src.finetune first.")

    model_ft = AutoModelForSequenceClassification.from_pretrained(FT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FT_DIR)
    model_ft.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft.to(device)
    print(f"Loaded fine-tuned model from {FT_DIR} ({device})", flush=True)

    # --- data: same 25% subset + 80/20 split as fine-tune ---
    df = load_imdb_subset(FRACTION)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()
    print(f"Subset: {len(df)} | train: {len(train_texts)} | test: {len(test_texts)}", flush=True)

    baseline_model = ensure_baseline_model(train_texts, train_labels)

    # --- Task 4: qualitative examples ---
    print("\n" + "=" * 60, flush=True)
    print("Сравнение на примерах", flush=True)
    print("=" * 60, flush=True)
    preds_ft = predict_fine_tuned(EXAMPLE_TEXTS, model_ft, tokenizer)
    preds_baseline = predict_baseline(EXAMPLE_TEXTS, baseline_model)

    for i, text in enumerate(EXAMPLE_TEXTS):
        print(f"\nТекст: {text}")
        print(
            f"Fine-tuned: {LABEL_NAMES[preds_ft[i]['prediction']]} "
            f"(probs: {format_probs(preds_ft[i]['probabilities'])})"
        )
        print(
            f"Baseline:   {LABEL_NAMES[preds_baseline[i]['prediction']]} "
            f"(probs: {format_probs(preds_baseline[i]['probabilities'])})"
        )
        print(f"Совпадают: {preds_ft[i]['prediction'] == preds_baseline[i]['prediction']}")

    # --- Tasks 5–6: hold-out metrics + confusion matrices ---
    print("\n" + "=" * 60)
    print("Оценка на test set")
    print("=" * 60)

    print("Predicting fine-tuned...", flush=True)
    preds_ft_all = predict_fine_tuned(test_texts, model_ft, tokenizer, batch_size=32)
    y_pred_ft = [p["prediction"] for p in preds_ft_all]

    print("Predicting baseline...", flush=True)
    preds_baseline_all = predict_baseline(test_texts, baseline_model, batch_size=32)
    y_pred_base = [p["prediction"] for p in preds_baseline_all]

    cm_ft = confusion_matrix(test_labels, y_pred_ft)
    cm_base = confusion_matrix(test_labels, y_pred_base)
    plot_confusion(cm_ft, "Confusion Matrix - Fine-tuned Model", CM_FT_PATH)
    plot_confusion(cm_base, "Confusion Matrix - Baseline Model", CM_BASE_PATH)

    print("\n=== Fine-tuned Model ===")
    report_ft = classification_report(test_labels, y_pred_ft, digits=4)
    print(report_ft)
    f1_ft = f1_score(test_labels, y_pred_ft, average="macro")
    acc_ft = accuracy_score(test_labels, y_pred_ft)

    print("\n=== Baseline Model ===")
    report_base = classification_report(test_labels, y_pred_base, digits=4)
    print(report_base)
    f1_base = f1_score(test_labels, y_pred_base, average="macro")
    acc_base = accuracy_score(test_labels, y_pred_base)

    improvement = (f1_ft - f1_base) / f1_base * 100 if f1_base > 0 else float("nan")
    print("\n=== Сравнение ===")
    print(f"Fine-tuned F1: {f1_ft:.4f}, Accuracy: {acc_ft:.4f}")
    print(f"Baseline F1:   {f1_base:.4f}, Accuracy: {acc_base:.4f}")
    print(f"Улучшение F1:  {improvement:.2f}%")

    # --- Task 7 ---
    RESULTS_PATH.write_text(
        "=== Сравнение моделей ===\n\n"
        f"data: IMDB {FRACTION:.0%} subset, test_size=0.2, random_state={RANDOM_STATE}\n"
        f"n_test: {len(test_texts)}\n"
        "baseline: frozen DistilBERT CLS + LogisticRegression\n"
        "fine-tuned: DistilBERTForSequenceClassification\n\n"
        "Fine-tuned Model:\n"
        f"  F1 (macro): {f1_ft:.4f}\n"
        f"  Accuracy: {acc_ft:.4f}\n\n"
        f"{report_ft}\n"
        "Baseline Model:\n"
        f"  F1 (macro): {f1_base:.4f}\n"
        f"  Accuracy: {acc_base:.4f}\n\n"
        f"{report_base}\n"
        f"Улучшение F1: {improvement:.2f}%\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
