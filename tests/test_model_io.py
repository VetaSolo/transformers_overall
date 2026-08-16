from pathlib import Path

import pytest

from src.model_io import InvalidModelWeights, assert_finetuned_model_ready, is_lfs_pointer


LFS_POINTER = """version https://git-lfs.github.com/spec/v1
oid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
size 267832560
"""


def test_lfs_pointer_is_detected(tmp_path: Path):
    fake = tmp_path / "model.safetensors"
    fake.write_text(LFS_POINTER, encoding="utf-8")
    assert is_lfs_pointer(fake)
    with pytest.raises(InvalidModelWeights, match="Git LFS pointer"):
        from src.model_io import validate_weight_file

        validate_weight_file(fake)


def test_assert_rejects_dir_with_only_pointer(tmp_path: Path):
    model_dir = tmp_path / "fine_tuned_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text(LFS_POINTER, encoding="utf-8")
    with pytest.raises(InvalidModelWeights, match="Git LFS pointer"):
        assert_finetuned_model_ready(model_dir)


def test_assert_rejects_missing_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="directory missing"):
        assert_finetuned_model_ready(tmp_path / "nope")
