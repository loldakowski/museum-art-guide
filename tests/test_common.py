import json

import numpy as np
import pytest
import torch

from museum_guide.common import check_checkpoint, dataset_fingerprint, provenance, save_json
from museum_guide.features import compute_features


def test_provenance_records_code_versions_and_dataset(fake_data, tmp_path):
    p = provenance(fake_data)
    assert p["git_sha"] is None or len(p["git_sha"]) == 40
    assert set(p["versions"]) == {"torch", "torchvision", "numpy", "sklearn"}
    assert p["subset_info"]["synthetic"] is True        # subset_info.json of the dataset is embedded
    save_json({"provenance": p}, tmp_path / "x.json")    # must be JSON-serialisable
    assert json.loads((tmp_path / "x.json").read_text())["provenance"]["python"] == p["python"]


def test_provenance_without_dataset():
    assert provenance()["subset_info"] is None


def test_checkpoint_from_another_dataset_is_refused(fake_data, other_fake_data):
    ckpt = {"state_dict": {}, "provenance": provenance(fake_data)}
    check_checkpoint(ckpt, fake_data, "same.pt")                      # same dataset: accepted
    with pytest.raises(ValueError, match="does not match"):
        check_checkpoint(ckpt, other_fake_data, "other.pt")
    with pytest.raises(ValueError, match="does not match"):
        check_checkpoint({"state_dict": {}}, fake_data, "old.pt")     # saved before provenance existed


def test_dataset_fingerprint_depends_on_image_content(fake_data, other_fake_data, tmp_path):
    assert (other_fake_data / "metadata.csv").read_bytes() == (fake_data / "metadata.csv").read_bytes()
    assert dataset_fingerprint(other_fake_data) != dataset_fingerprint(fake_data)  # same metadata, other pixels
    assert dataset_fingerprint(tmp_path / "missing") is None


def test_provenance_survives_a_safe_torch_load(fake_data, tmp_path):
    torch.save({"provenance": provenance(fake_data)}, tmp_path / "c.pt")
    assert torch.load(tmp_path / "c.pt")["provenance"]["versions"]["torch"]  # weights_only=True by default


def test_non_finite_floats_are_written_as_json_null(tmp_path):
    save_json({"best_val_loss": float("inf"), "acc": [np.float64("nan"), 0.5]}, tmp_path / "x.json")
    assert json.loads((tmp_path / "x.json").read_text()) == {"best_val_loss": None, "acc": [None, 0.5]}


def test_feature_cache_key_sees_every_pixel(tmp_path):
    X = np.zeros((4, 16, 16, 3), np.uint8)
    a = compute_features("pixels32", X, tmp_path)
    X[0, 1, 1, 0] = 255  # a pixel the old key (every 8th row and column) did not read
    b = compute_features("pixels32", X, tmp_path)
    assert len(list(tmp_path.glob("feat_pixels32_*.npy"))) == 2 and not np.array_equal(a, b)
