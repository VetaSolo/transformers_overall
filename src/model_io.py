"""Helpers to validate the fine-tuned model directory."""

from __future__ import annotations

from pathlib import Path

WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",  # sharded
)


def model_weight_files(model_dir: Path) -> list[Path]:
    found: list[Path] = []
    for name in WEIGHT_NAMES:
        p = model_dir / name
        if p.exists():
            found.append(p)
    # sharded safetensors: model-00001-of-00002.safetensors
    found.extend(sorted(model_dir.glob("model-*.safetensors")))
    found.extend(sorted(model_dir.glob("pytorch_model-*.bin")))
    # unique
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def assert_finetuned_model_ready(model_dir: Path) -> Path:
    """Raise FileNotFoundError if directory or weight files are missing."""
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model directory missing: {model_dir}. Run: python -m src.finetune"
        )
    weights = model_weight_files(model_dir)
    if not weights:
        raise FileNotFoundError(
            f"No model weights in {model_dir} "
            f"(expected model.safetensors or pytorch_model.bin). "
            f"Found only: {[p.name for p in model_dir.iterdir()]}. "
            f"Re-run: python -m src.finetune"
        )
    return weights[0]
