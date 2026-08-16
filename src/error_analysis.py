"""Error analysis for the fine-tuned model (FP / FN)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_loading import (
    DEFAULT_FRACTION,
    RANDOM_STATE,
    data_manifest,
    make_split,
    prepare_dataset,
)
from src.model_io import assert_finetuned_model_ready
from src.predict import LABEL_NAMES, predict_fine_tuned

ROOT = Path(__file__).resolve().parents[1]
FT_DIR = ROOT / "fine_tuned_model"
OUT_PATH = ROOT / "error_analysis.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="FP/FN error analysis")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if not FT_DIR.exists():
        raise FileNotFoundError(f"Missing fine-tuned model: {FT_DIR}")

    assert_finetuned_model_ready(FT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(FT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FT_DIR)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df = prepare_dataset(args.data, fraction=args.fraction, max_samples=args.max_samples)
    source = df.attrs.get("source_path", args.data)
    _, test_df = make_split(df)
    manifest = data_manifest(
        df, test_df, fraction=args.fraction, max_samples=args.max_samples
    )
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    print(f"Running inference on {len(test_texts)} test texts...", flush=True)
    preds = predict_fine_tuned(test_texts, model, tokenizer, batch_size=32)
    y_pred_ft = [p["prediction"] for p in preds]
    scores = [float(max(p["probabilities"])) for p in preds]

    df_test = pd.DataFrame(
        {
            "text": test_texts,
            "true_label": test_labels,
            "pred_label": y_pred_ft,
            "confidence": scores,
        }
    )

    errors = df_test[df_test["true_label"] != df_test["pred_label"]].copy()
    fp = errors[(errors["pred_label"] == 1) & (errors["true_label"] == 0)]
    fn = errors[(errors["pred_label"] == 0) & (errors["true_label"] == 1)]

    print(f"Всего ошибок: {len(errors)}", flush=True)
    print(f"False Positives: {len(fp)}", flush=True)
    print(f"False Negatives: {len(fn)}", flush=True)

    print("\n=== FALSE POSITIVES (сказали positive, а было negative) ===", flush=True)
    for _, row in fp.head(5).iterrows():
        print(f"\nТекст: {row['text'][:100]}...", flush=True)
        print(
            f"Истинный класс: {row['true_label']} ({LABEL_NAMES.get(row['true_label'], row['true_label'])}), "
            f"Предсказан: {row['pred_label']} ({LABEL_NAMES.get(row['pred_label'], row['pred_label'])})",
            flush=True,
        )

    print("\n=== FALSE NEGATIVES (сказали negative, а было positive) ===", flush=True)
    for _, row in fn.head(5).iterrows():
        print(f"\nТекст: {row['text'][:100]}...", flush=True)
        print(
            f"Истинный класс: {row['true_label']} ({LABEL_NAMES.get(row['true_label'], row['true_label'])}), "
            f"Предсказан: {row['pred_label']} ({LABEL_NAMES.get(row['pred_label'], row['pred_label'])})",
            flush=True,
        )

    errors["text_length"] = errors["text"].str.len()
    mean_err_len = float(errors["text_length"].mean()) if len(errors) else 0.0
    mean_all_len = float(df_test["text"].str.len().mean())
    print(f"\nСредняя длина ошибочных текстов: {mean_err_len:.0f}", flush=True)
    print(f"Средняя длина всех текстов: {mean_all_len:.0f}", flush=True)

    mean_fp_conf = float(fp["confidence"].mean()) if len(fp) else 0.0
    mean_fn_conf = float(fn["confidence"].mean()) if len(fn) else 0.0
    mean_ok_conf = float(
        df_test.loc[df_test["true_label"] == df_test["pred_label"], "confidence"].mean()
    )

    observations = [
        f"Ошибки: {100 * len(errors) / max(len(df_test), 1):.1f}% теста ({len(errors)}/{len(df_test)}).",
        f"FP={len(fp)}, FN={len(fn)}.",
        f"Средняя длина ошибок ({mean_err_len:.0f}) vs всех ({mean_all_len:.0f}).",
        f"Уверенность: FP={mean_fp_conf:.3f}, FN={mean_fn_conf:.3f}, correct={mean_ok_conf:.3f}.",
        "Частые паттерны: ирония/смешанные отзывы, обрезка max_length=128, HTML (<br />) в IMDB.",
    ]

    lines: list[str] = [
        "=== АНАЛИЗ ОШИБОК ===\n",
        f"model: {FT_DIR}",
        f"data: {source}",
        f"subset fraction={args.fraction}, max_samples={args.max_samples}, random_state={RANDOM_STATE}",
        f"split_fingerprint: {manifest['split_fingerprint']}",
        f"n_test: {len(df_test)}\n",
        f"Всего ошибок: {len(errors)}",
        f"False Positives: {len(fp)}",
        f"False Negatives: {len(fn)}",
        f"Средняя длина ошибочных текстов: {mean_err_len:.0f}",
        f"Средняя длина всех текстов: {mean_all_len:.0f}\n",
        "=== ПРИМЕРЫ FALSE POSITIVES ===\n",
    ]

    for _, row in fp.head(5).iterrows():
        lines.append(f"Текст: {row['text']}")
        lines.append(
            f"Истинный: {row['true_label']}, Предсказан: {row['pred_label']}, "
            f"confidence={row['confidence']:.4f}\n"
        )

    lines.append("\n=== ПРИМЕРЫ FALSE NEGATIVES ===\n")
    for _, row in fn.head(5).iterrows():
        lines.append(f"Текст: {row['text']}")
        lines.append(
            f"Истинный: {row['true_label']}, Предсказан: {row['pred_label']}, "
            f"confidence={row['confidence']:.4f}\n"
        )

    lines.append("\n=== НАБЛЮДЕНИЯ ===\n")
    for obs in observations:
        lines.append(f"- {obs}")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
