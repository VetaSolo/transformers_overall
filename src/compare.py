"""
Day 6 — Compare Day-4 baseline (baseline_model.pkl) vs Day-5 fine-tuned model.

Both models are evaluated on the exact same held-out split, and the baseline
artifact is rejected unless it was trained on a matching split (otherwise its
training data would overlap this test set and the metrics would be inflated).
"""

from __future__ import annotations

import argparse
import json
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loading import (
    DEFAULT_FRACTION,
    RANDOM_STATE,
    data_manifest,
    make_split,
    prepare_dataset,
)
from src.model_io import assert_finetuned_model_ready
from src.predict import LABEL_NAMES, format_probs, predict_baseline, predict_fine_tuned

ROOT = Path(__file__).resolve().parents[1]
FT_DIR = ROOT / "fine_tuned_model"
BASELINE_PKL = ROOT / "baseline_model.pkl"
BASELINE_MANIFEST = ROOT / "baseline_model.json"
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


def load_baseline_from_day4(manifest: dict):
    """Load the Day-4 model, refusing artifacts built on a different split."""
    if not BASELINE_PKL.exists():
        raise FileNotFoundError(
            f"Missing {BASELINE_PKL.name}. Run Day 4 first: python -m src.baseline"
        )
    if not BASELINE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing {BASELINE_MANIFEST.name}: the baseline artifact predates split "
            "tracking, so its training data may overlap this test set. "
            "Re-run Day 4: python -m src.baseline"
        )

    saved = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if saved.get("split_fingerprint") != manifest["split_fingerprint"]:
        raise ValueError(
            "Baseline was trained on a different data split "
            f"(baseline: n_used={saved.get('n_used')}, "
            f"fingerprint={saved.get('split_fingerprint')}; "
            f"current: n_used={manifest['n_used']}, "
            f"fingerprint={manifest['split_fingerprint']}). "
            "Comparing them would leak baseline training data into the test set. "
            "Re-run Day 4 with the same flags: python -m src.baseline "
            f"--fraction {manifest['fraction']}"
        )

    print(f"Loading Day-4 baseline: {BASELINE_PKL.name} (split {saved['split_fingerprint']})")
    return joblib.load(BASELINE_PKL)


def plot_confusion(cm, title: str, path: Path) -> None:
    tick = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)][: cm.shape[0]]
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

    assert_finetuned_model_ready(FT_DIR)
    model_ft = AutoModelForSequenceClassification.from_pretrained(FT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FT_DIR)
    model_ft.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft.to(device)
    print(f"Loaded fine-tuned model from {FT_DIR} ({device})", flush=True)

    df = prepare_dataset(args.data, fraction=args.fraction, max_samples=args.max_samples)
    source = df.attrs.get("source_path", args.data)
    _, test_df = make_split(df)
    manifest = data_manifest(
        df, test_df, fraction=args.fraction, max_samples=args.max_samples
    )
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()
    print(
        f"Data: {source} | subset={len(df)} | test={len(test_texts)} "
        f"| split={manifest['split_fingerprint']}",
        flush=True,
    )

    baseline_model = load_baseline_from_day4(manifest)

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
        f"n_used: {manifest['n_used']} (of {manifest['n_full']}), "
        f"fraction={args.fraction}, max_samples={args.max_samples}\n"
        f"test_size={manifest['test_size']}, random_state={RANDOM_STATE}\n"
        f"n_test: {manifest['n_test']}\n"
        f"split_fingerprint: {manifest['split_fingerprint']} "
        "(identical for baseline, fine-tuned and this evaluation)\n"
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
