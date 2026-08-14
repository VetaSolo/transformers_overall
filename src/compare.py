"""
Day 6 — Compare Day-4 baseline (baseline_model.pkl) vs Day-5 fine-tuned model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loading import (
    DEFAULT_FRACTION,
    RANDOM_STATE,
    load_sentiment_csv,
    subsample_stratified,
)
from src.model_io import assert_finetuned_model_ready
from src.predict import LABEL_NAMES, format_probs, predict_baseline, predict_fine_tuned

ROOT = Path(__file__).resolve().parents[1]
FT_DIR = ROOT / "fine_tuned_model"
BASELINE_PKL = ROOT / "baseline_model.pkl"
RESULTS_PATH = ROOT / "comparison_results.txt"
CM_FT_PATH = ROOT / "confusion_matrix_finetuned.png"
CM_BASE_PATH = ROOT / "confusion_matrix_baseline.png"

EXAMPLE_TEXTS = [
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "It was okay, nothing special.",
    "Best film I've seen this year!",
    "Boring and too long.",
]


def load_baseline_from_day4():
    """Load LogisticRegression saved by Day 4 — do not retrain here."""
    if not BASELINE_PKL.exists():
        raise FileNotFoundError(
            f"Missing {BASELINE_PKL}. Run Day 4 first: python -m src.baseline"
        )
    print(f"Loading Day-4 baseline: {BASELINE_PKL}", flush=True)
    return joblib.load(BASELINE_PKL)


def plot_confusion(cm, title: str, path: Path) -> None:
    labels = sorted(LABEL_NAMES)
    tick = [LABEL_NAMES[i] for i in labels if i in LABEL_NAMES]
    # binary fallback
    if cm.shape[0] == 2:
        tick = [LABEL_NAMES[0], LABEL_NAMES[1]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=tick, yticklabels=tick)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if not FT_DIR.exists():
        raise FileNotFoundError(f"Fine-tuned model not found: {FT_DIR}. Run: python -m src.finetune")

    assert_finetuned_model_ready(FT_DIR)
    model_ft = AutoModelForSequenceClassification.from_pretrained(FT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FT_DIR)
    model_ft.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft.to(device)
    print(f"Loaded fine-tuned model from {FT_DIR} ({device})", flush=True)

    baseline_model = load_baseline_from_day4()

    df = load_sentiment_csv(args.data)
    source = df.attrs.get("source_path", args.data)
    df = subsample_stratified(df, fraction=args.fraction, max_samples=args.max_samples)
    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()
    print(f"Data: {source} | subset={len(df)} | test={len(test_texts)}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Сравнение на примерах", flush=True)
    print("=" * 60, flush=True)
    preds_ft = predict_fine_tuned(EXAMPLE_TEXTS, model_ft, tokenizer)
    preds_baseline = predict_baseline(EXAMPLE_TEXTS, baseline_model)

    for i, text in enumerate(EXAMPLE_TEXTS):
        print(f"\nТекст: {text}")
        print(
            f"Fine-tuned: {LABEL_NAMES.get(preds_ft[i]['prediction'], preds_ft[i]['prediction'])} "
            f"(probs: {format_probs(preds_ft[i]['probabilities'])})"
        )
        print(
            f"Baseline:   {LABEL_NAMES.get(preds_baseline[i]['prediction'], preds_baseline[i]['prediction'])} "
            f"(probs: {format_probs(preds_baseline[i]['probabilities'])})"
        )
        print(f"Совпадают: {preds_ft[i]['prediction'] == preds_baseline[i]['prediction']}")

    print("\n" + "=" * 60)
    print("Оценка на test set")
    print("=" * 60)

    print("Predicting fine-tuned...", flush=True)
    preds_ft_all = predict_fine_tuned(test_texts, model_ft, tokenizer, batch_size=32)
    y_pred_ft = [p["prediction"] for p in preds_ft_all]

    print("Predicting baseline (Day-4 model)...", flush=True)
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

    print("\n=== Baseline Model (from Day 4) ===")
    report_base = classification_report(test_labels, y_pred_base, digits=4)
    print(report_base)
    f1_base = f1_score(test_labels, y_pred_base, average="macro")
    acc_base = accuracy_score(test_labels, y_pred_base)

    improvement = (f1_ft - f1_base) / f1_base * 100 if f1_base > 0 else float("nan")
    print("\n=== Сравнение ===")
    print(f"Fine-tuned F1: {f1_ft:.4f}, Accuracy: {acc_ft:.4f}")
    print(f"Baseline F1:   {f1_base:.4f}, Accuracy: {acc_base:.4f}")
    print(f"Улучшение F1:  {improvement:.2f}%")

    RESULTS_PATH.write_text(
        "=== Сравнение моделей ===\n\n"
        f"data: {source}\n"
        f"subset fraction={args.fraction}, max_samples={args.max_samples}, "
        f"test_size=0.2, random_state={RANDOM_STATE}\n"
        f"n_test: {len(test_texts)}\n"
        "baseline: Day-4 artifact baseline_model.pkl "
        "(frozen DistilBERT CLS + LogisticRegression)\n"
        "fine-tuned: ./fine_tuned_model\n\n"
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
