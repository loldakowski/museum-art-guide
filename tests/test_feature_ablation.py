import importlib.util
import json
import sys
from pathlib import Path

from museum_guide.data import load_metadata, make_split


def _step9():
    spec = importlib.util.spec_from_file_location("step9", Path(__file__).parents[1] / "scripts" / "09_feature_ablation.py")
    step9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step9)
    return step9


def test_ablation_never_loads_a_test_image(fake_data, tmp_path, monkeypatch):
    step9 = _step9()
    loaded = []
    real_load = step9.load_image_array

    def recording_load(meta, *args, **kwargs):
        loaded.extend(meta["path"])
        return real_load(meta, *args, **kwargs)

    monkeypatch.setattr(step9, "load_image_array", recording_load)
    monkeypatch.setattr(sys, "argv", ["09_feature_ablation.py", "--data-dir", str(fake_data), "--size", "32",
                                      "--out", str(tmp_path), "--reference", "", "--timing-images", "2",
                                      "--timing-repeats", "1"])
    step9.main()

    meta = load_metadata(fake_data)
    split = make_split(meta, "artist", seed=0)
    assert set(loaded) == set(meta.loc[split != "test", "path"])     # train + val, nothing from test
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["n_val"] == (split == "val").sum() and len(s["val_labels"]) == s["n_val"]
    assert [v["name"] for v in s["variants"]][:1] == ["all blocks (default)"] and len(s["variants"]) == 12
    assert s["variants"][0]["n_features"] == 1975  # same dimension as features.handcrafted
