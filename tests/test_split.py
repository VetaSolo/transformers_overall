import json
from pathlib import Path

import pandas as pd
import pytest

from src.compare import load_baseline_from_day4
from src.data_loading import make_split, split_fingerprint


def test_split_fingerprint_is_stable():
    texts, labels = [], []
    for i in range(20):
        texts.extend([f"good {i}", f"bad {i}"])
        labels.extend([1, 0])
    df = pd.DataFrame({"text": texts, "label": labels})
    _, test_a = make_split(df, random_state=42)
    _, test_b = make_split(df, random_state=42)
    assert split_fingerprint(test_a) == split_fingerprint(test_b)
    assert list(test_a["text"]) == list(test_b["text"])


def test_compare_rejects_mismatched_baseline_manifest(tmp_path, monkeypatch):
    import src.compare as compare_mod

    pkl = tmp_path / "baseline_model.pkl"
    man = tmp_path / "baseline_model.json"
    pkl.write_bytes(b"not-a-real-model")
    man.write_text(
        json.dumps({"n_used": 50000, "split_fingerprint": "deadbeefdeadbeef"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(compare_mod, "BASELINE_PKL", pkl)
    monkeypatch.setattr(compare_mod, "BASELINE_MANIFEST", man)

    current = {"n_used": 12500, "split_fingerprint": "cafebabecafebabe", "fraction": 0.25}
    with pytest.raises(ValueError, match="different data split"):
        load_baseline_from_day4(current)


def test_compare_rejects_missing_manifest(tmp_path, monkeypatch):
    import src.compare as compare_mod

    pkl = tmp_path / "baseline_model.pkl"
    pkl.write_bytes(b"x")
    monkeypatch.setattr(compare_mod, "BASELINE_PKL", pkl)
    monkeypatch.setattr(compare_mod, "BASELINE_MANIFEST", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="predates split"):
        load_baseline_from_day4({"split_fingerprint": "abc", "n_used": 1, "fraction": 0.25})
