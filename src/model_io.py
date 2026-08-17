"""Validation helpers for the fine-tuned model directory.

Checking that a file *name* exists is not enough: a Git LFS checkout without
``git lfs pull`` leaves a small text pointer in place of the real weights, and
``from_pretrained`` then fails with an opaque SafetensorError. These helpers
inspect file contents so callers fail early with an actionable message.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
)

LFS_POINTER_PREFIX = b"version https://git-lfs"
# A real checkpoint is megabytes; pointers are a few hundred bytes.
MIN_WEIGHT_BYTES = 1024 * 1024


class InvalidModelWeights(RuntimeError):
    """Weight file exists but does not contain usable model parameters."""


def model_weight_files(model_dir: Path) -> list[Path]:
    found: list[Path] = []
    for name in WEIGHT_NAMES:
        p = model_dir / name
        if p.exists():
            found.append(p)
    found.extend(sorted(model_dir.glob("model-*.safetensors")))
    found.extend(sorted(model_dir.glob("pytorch_model-*.bin")))

    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def _validate_safetensors(path: Path) -> None:
    """Parse the safetensors header the same way the loader does."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) < 8:
            raise InvalidModelWeights(f"{path.name} is truncated ({size} bytes).")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len <= 0 or header_len + 8 > size:
            raise InvalidModelWeights(
                f"{path.name} has an invalid safetensors header "
                f"(header={header_len} bytes, file={size} bytes)."
            )
        try:
            header = json.loads(fh.read(header_len).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidModelWeights(f"{path.name} header is not valid JSON: {exc}") from exc

    tensors = [k for k in header if k != "__metadata__"]
    if not tensors:
        raise InvalidModelWeights(f"{path.name} contains no tensors.")


def validate_weight_file(path: Path) -> None:
    """Raise InvalidModelWeights if the file is a pointer, stub or corrupt."""
    if is_lfs_pointer(path):
        raise InvalidModelWeights(
            f"{path} is a Git LFS pointer, not real weights. "
            "Run: python -m src.finetune  (or git lfs pull if you use LFS)"
        )

    if path.suffix == ".safetensors" or path.name.endswith(".safetensors"):
        _validate_safetensors(path)
        return

    size = path.stat().st_size
    if size < MIN_WEIGHT_BYTES:
        raise InvalidModelWeights(
            f"{path} is only {size} bytes — too small to be a checkpoint."
        )


def assert_finetuned_model_ready(model_dir: Path) -> Path:
    """Return the primary weight file, or raise with a fix-it message."""
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model directory missing: {model_dir}. Run: python -m src.finetune"
        )

    weights = model_weight_files(model_dir)
    if not weights:
        raise FileNotFoundError(
            f"No model weights in {model_dir} "
            f"(expected model.safetensors or pytorch_model.bin). "
            f"Found only: {sorted(p.name for p in model_dir.iterdir())}. "
            f"Re-run: python -m src.finetune"
        )

    for weight in weights:
        validate_weight_file(weight)
    return weights[0]
