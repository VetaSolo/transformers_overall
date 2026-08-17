"""Unified sentiment CSV loading for the assignment interface.

Accepted schemas:
1) Official assignment format: columns ``text``, ``label``
2) IMDB Kaggle aliases: columns ``review``, ``sentiment``
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_CANDIDATES = (
    DATA_DIR / "dataset.csv",
    DATA_DIR / "IMDB Dataset.csv",
)

RANDOM_STATE = 42
DEFAULT_FRACTION = 0.25
TEST_SIZE = 0.2
# Default 25% subsample is only for large corpora (IMDB). Tiny CSVs like
# data/sample_assignment.csv must be kept whole so stratify still works.
MIN_ROWS_TO_SUBSAMPLE = 50
MIN_PER_CLASS = 2


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
    """Optional stratified subset (used to speed up CPU fine-tuning).

    Small datasets are never shrunk: 6-row ``sample_assignment.csv`` with the
    default fraction=0.25 would otherwise become 2 rows and break ``stratify``.
    """
    n_full = len(df)
    n_classes = max(int(df["label"].nunique()), 1)

    if max_samples is not None:
        target_n = min(max_samples, n_full)
    elif fraction is None or fraction >= 1.0:
        return df.reset_index(drop=True)
    elif n_full < MIN_ROWS_TO_SUBSAMPLE:
        return df.reset_index(drop=True)
    else:
        target_n = max(n_classes * MIN_PER_CLASS, int(n_full * fraction))

    if target_n >= n_full:
        return df.reset_index(drop=True)

    n_per = max(MIN_PER_CLASS, target_n // n_classes)
    parts = [
        g.sample(n=min(len(g), n_per), random_state=random_state)
        for _, g in df.groupby("label")
    ]
    out = (
        pd.concat(parts)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    if int(out["label"].value_counts().min()) < MIN_PER_CLASS:
        return df.reset_index(drop=True)
    return out


def prepare_dataset(
    path: str | Path | None = None,
    fraction: float | None = DEFAULT_FRACTION,
    max_samples: int | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    df = load_sentiment_csv(path)
    source = df.attrs.get("source_path")
    n_full = len(df)
    df = subsample_stratified(
        df, fraction=fraction, max_samples=max_samples, random_state=random_state
    )
    df.attrs["source_path"] = source
    df.attrs["n_full"] = n_full
    return df


def make_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The single train/test split shared by days 4, 5, 6 and 7.

    Every day must derive its data through prepare_dataset + make_split so the
    baseline never sees fine-tuning test examples during training.
    """
    n = len(df)
    if n < 2:
        raise ValueError(f"Need at least 2 rows to split, got {n}")

    n_classes = int(df["label"].nunique())
    min_count = int(df["label"].value_counts().min())
    can_stratify = min_count >= MIN_PER_CLASS

    n_test = max(1, int(round(n * test_size)))
    if can_stratify:
        n_test = max(n_test, n_classes)
        n_test = min(n_test, n - n_classes)
        if n_test < n_classes:
            n_test = max(n_classes, min(n // 2, n - n_classes))
    else:
        n_test = min(n_test, n - 1)

    stratify = df["label"] if can_stratify else None
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=n_test,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        train_df, test_df = train_test_split(
            df,
            test_size=max(1, min(n_test, n - 1)),
            random_state=random_state,
            stratify=None,
        )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_fingerprint(test_df: pd.DataFrame) -> str:
    """Stable hash of the test split, used to detect incompatible artifacts."""
    digest = hashlib.sha256()
    for text, label in zip(test_df["text"], test_df["label"]):
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(text.encode("utf-8", errors="replace"))
        digest.update(b"\x01")
    return digest.hexdigest()[:16]


def data_manifest(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    fraction: float | None,
    max_samples: int | None,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> dict:
    source = str(df.attrs.get("source_path", ""))
    try:
        source = str(Path(source).resolve().relative_to(ROOT.resolve()))
    except (ValueError, OSError):
        pass
    return {
        "source_path": source,
        "n_full": int(df.attrs.get("n_full", len(df))),
        "n_used": int(len(df)),
        "n_test": int(len(test_df)),
        "fraction": fraction,
        "max_samples": max_samples,
        "random_state": random_state,
        "test_size": test_size,
        "split_fingerprint": split_fingerprint(test_df),
    }
