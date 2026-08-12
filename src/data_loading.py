"""Unified sentiment CSV loading for the assignment interface.

Accepted schemas:
1) Official assignment format: columns ``text``, ``label``
2) IMDB Kaggle aliases: columns ``review``, ``sentiment``
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_CANDIDATES = (
    DATA_DIR / "dataset.csv",
    DATA_DIR / "IMDB Dataset.csv",
)

RANDOM_STATE = 42
DEFAULT_FRACTION = 0.25


def resolve_data_path(path: str | Path | None = None) -> Path:
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        return p

    env = os.getenv("SENTIMENT_DATA")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
        raise FileNotFoundError(f"SENTIMENT_DATA points to missing file: {p}")

    for candidate in DEFAULT_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No dataset found. Put a CSV at data/dataset.csv with columns "
        "text,label (or data/IMDB Dataset.csv with review,sentiment)."
    )


def _normalize_label(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    mapping = {
        "negative": 0,
        "neg": 0,
        "0": 0,
        "positive": 1,
        "pos": 1,
        "1": 1,
        "neutral": 2,
        "neu": 2,
        "2": 2,
    }
    lowered = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(lowered) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown label values: {unknown}. Expected 0/1(/2) or pos/neg(/neu).")
    return lowered.map(mapping).astype(int)


def load_sentiment_csv(path: str | Path | None = None) -> pd.DataFrame:
    """Load CSV and return DataFrame with columns: text (str), label (int)."""
    csv_path = resolve_data_path(path)
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    if "text" in cols and "label" in cols:
        text_col, label_col = cols["text"], cols["label"]
    elif "review" in cols and "sentiment" in cols:
        text_col, label_col = cols["review"], cols["sentiment"]
    else:
        raise ValueError(
            "CSV must have columns text,label "
            f"(assignment format) or review,sentiment (IMDB). Got: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": _normalize_label(df[label_col]),
        }
    )
    out = out.dropna(subset=["text", "label"]).reset_index(drop=True)
    out.attrs["source_path"] = str(csv_path)
    return out


def subsample_stratified(
    df: pd.DataFrame,
    fraction: float | None = DEFAULT_FRACTION,
    max_samples: int | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Optional stratified subset (used to speed up CPU fine-tuning)."""
    n_full = len(df)
    if max_samples is not None:
        target_n = min(max_samples, n_full)
    elif fraction is None or fraction >= 1.0:
        return df.reset_index(drop=True)
    else:
        target_n = max(2, int(n_full * fraction))

    if target_n >= n_full:
        return df.reset_index(drop=True)

    n_per = max(1, target_n // 2)
    parts = [
        g.sample(n=min(len(g), n_per), random_state=random_state)
        for _, g in df.groupby("label")
    ]
    return (
        pd.concat(parts)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
