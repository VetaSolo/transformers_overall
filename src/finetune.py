"""
Day 5 — Fine-tuning DistilBERT for sentiment classification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW  # transformers.AdamW removed in recent versions
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loading import (
    DEFAULT_FRACTION,
    RANDOM_STATE,
    TEST_SIZE,
    data_manifest,
    make_split,
    prepare_dataset,
)
from src.dataset import SentimentDataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "fine_tuned_model"
RESULTS_PATH = ROOT / "fine_tuned_results.txt"
MANIFEST_PATH = ROOT / "fine_tuned_model.json"
MODEL_NAME = "distilbert-base-uncased"


def train_epoch(model, dataloader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    try:
        from tqdm import tqdm

        iterator = tqdm(dataloader, desc="train", leave=False)
    except ImportError:
        iterator = dataloader

    for step, batch in enumerate(iterator, start=1):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss.item():.4f}")
        elif step % 50 == 0:
            print(f"  step {step}/{len(dataloader)} loss={loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device) -> tuple[float, float]:
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")
    return accuracy, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on sentiment CSV")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="CSV with text,label (or IMDB review,sentiment)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--fraction",
        type=float,
        default=DEFAULT_FRACTION,
        help="Fraction of dataset to use (default: 0.25)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional absolute subset size (overrides --fraction)",
    )
    args = parser.parse_args()

    df = prepare_dataset(args.data, fraction=args.fraction, max_samples=args.max_samples)
    source = df.attrs.get("source_path", args.data)
    labels = df["label"].tolist()

    train_df, val_df = make_split(df)
    manifest = data_manifest(
        df, val_df, fraction=args.fraction, max_samples=args.max_samples
    )
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()

    print(f"Loaded: {source}")
    print(f"Using subset: {len(df)} / {manifest['n_full']} samples")
    print(f"Split fingerprint: {manifest['split_fingerprint']}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)

    num_labels = len(set(labels))
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device} | classes: {num_labels} | epochs: {args.epochs}")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    val_acc = 0.0
    val_f1 = 0.0
    history: list[str] = []

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)

        line = (
            f"Epoch {epoch + 1}/{args.epochs}\n"
            f"Train Loss: {train_loss:.4f}\n"
            f"Val Accuracy: {val_acc:.4f}\n"
            f"Val F1: {val_f1:.4f}\n"
            + "-" * 50
        )
        print(line)
        history.append(line)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR, max_shard_size="80MB")
    tokenizer.save_pretrained(MODEL_DIR)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    RESULTS_PATH.write_text(
        f"model: {MODEL_NAME}\n"
        f"data: {source}\n"
        f"n_samples: {len(df)} (of {manifest['n_full']})\n"
        f"fraction/max_samples: fraction={args.fraction}, max_samples={args.max_samples}\n"
        f"train/val: {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, stratify, "
        f"random_state={RANDOM_STATE}\n"
        f"train={len(train_texts)}, val={len(val_texts)}\n"
        f"split_fingerprint: {manifest['split_fingerprint']}\n"
        f"epochs: {args.epochs}\n"
        f"batch_size: {args.batch_size}\n"
        f"lr: {args.lr}\n"
        f"max_length: {args.max_length}\n"
        f"device: {device}\n\n"
        + "\n".join(history)
        + "\n"
        f"Final Validation F1: {val_f1:.4f}\n"
        f"Final Validation Accuracy: {val_acc:.4f}\n",
        encoding="utf-8",
    )

    print(f"Saved model -> {MODEL_DIR}")
    print(f"Saved metrics -> {RESULTS_PATH}")
    print(f"Final Validation F1: {val_f1:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
