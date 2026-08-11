"""Error analysis for the fine-tuned model (FP / FN)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from src.compare import FRACTION, RANDOM_STATE, load_imdb_subset
from src.predict import LABEL_NAMES, predict_fine_tuned

ROOT = Path(__file__).resolve().parents[1]
FT_DIR = ROOT / "fine_tuned_model"
OUT_PATH = ROOT / "error_analysis.txt"


def main() -> None:
    if not FT_DIR.exists():
        raise FileNotFoundError(f"Missing fine-tuned model: {FT_DIR}")

    model = AutoModelForSequenceClassification.from_pretrained(FT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FT_DIR)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df = load_imdb_subset(FRACTION)
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
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
            f"Истинный класс: {row['true_label']} ({LABEL_NAMES[row['true_label']]}), "
            f"Предсказан: {row['pred_label']} ({LABEL_NAMES[row['pred_label']]}), "
            f"conf={row['confidence']:.3f}",
            flush=True,
        )

    print("\n=== FALSE NEGATIVES (сказали negative, а было positive) ===", flush=True)
    for _, row in fn.head(5).iterrows():
        print(f"\nТекст: {row['text'][:100]}...", flush=True)
        print(
            f"Истинный класс: {row['true_label']} ({LABEL_NAMES[row['true_label']]}), "
            f"Предсказан: {row['pred_label']} ({LABEL_NAMES[row['pred_label']]}), "
            f"conf={row['confidence']:.3f}",
            flush=True,
        )

    errors["text_length"] = errors["text"].str.len()
    mean_err_len = errors["text_length"].mean() if len(errors) else 0.0
    mean_all_len = df_test["text"].str.len().mean()
    print(f"\nСредняя длина ошибочных текстов: {mean_err_len:.0f}", flush=True)
    print(f"Средняя длина всех текстов: {mean_all_len:.0f}", flush=True)

    # Heuristic observations from counts / length / confidence
    mean_fp_conf = float(fp["confidence"].mean()) if len(fp) else 0.0
    mean_fn_conf = float(fn["confidence"].mean()) if len(fn) else 0.0
    mean_ok_conf = float(
        df_test.loc[df_test["true_label"] == df_test["pred_label"], "confidence"].mean()
    )

    observations = [
        f"Ошибки составляют {100 * len(errors) / len(df_test):.1f}% тестовой выборки "
        f"({len(errors)}/{len(df_test)}).",
        f"FP={len(fp)}, FN={len(fn)} — баланс ошибок "
        + (
            "примерно равный."
            if abs(len(fp) - len(fn)) <= max(5, 0.15 * len(errors))
            else ("смещён в сторону FP (ложный positive)." if len(fp) > len(fn) else "смещён в сторону FN (ложный negative).")
        ),
        f"Средняя длина ошибок ({mean_err_len:.0f}) vs всех текстов ({mean_all_len:.0f}). "
        + (
            "Ошибки чаще на более длинных отзывах — модель хуже держит смешанный/саркастичный контекст при max_length=128."
            if mean_err_len > mean_all_len * 1.05
            else (
                "Ошибки чаще на более коротких текстах — мало сигнала для CLS."
                if mean_err_len < mean_all_len * 0.95
                else "Длина ошибочных и корректных текстов сопоставима."
            )
        ),
        f"Средняя уверенность на ошибках: FP={mean_fp_conf:.3f}, FN={mean_fn_conf:.3f}; "
        f"на верных предсказаниях: {mean_ok_conf:.3f}.",
        "Типичные паттерны IMDB: ирония/сарказм, смешанные отзывы (хвалит актёров, ругает сюжет), "
        "spoiler-heavy тексты и HTML-артефакты (<br />) в исходных данных.",
        "Ограничение max_length=128 обрезает длинные рецензии — важный вердикт может оказаться за окном.",
    ]

    lines: list[str] = [
        "=== АНАЛИЗ ОШИБОК ===\n",
        f"model: {FT_DIR}",
        f"data: IMDB {FRACTION:.0%} subset, test_size=0.2, random_state={RANDOM_STATE}",
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
            f"Истинный: {row['true_label']} ({LABEL_NAMES[int(row['true_label'])]}), "
            f"Предсказан: {row['pred_label']} ({LABEL_NAMES[int(row['pred_label'])]}), "
            f"confidence={row['confidence']:.4f}\n"
        )

    lines.append("\n=== ПРИМЕРЫ FALSE NEGATIVES ===\n")
    for _, row in fn.head(5).iterrows():
        lines.append(f"Текст: {row['text']}")
        lines.append(
            f"Истинный: {row['true_label']} ({LABEL_NAMES[int(row['true_label'])]}), "
            f"Предсказан: {row['pred_label']} ({LABEL_NAMES[int(row['pred_label'])]}), "
            f"confidence={row['confidence']:.4f}\n"
        )

    lines.append("\n=== НАБЛЮДЕНИЯ ===\n")
    for obs in observations:
        lines.append(f"- {obs}")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
