import json

from museum_guide.common import provenance, save_json


def test_provenance_records_code_versions_and_dataset(fake_data, tmp_path):
    p = provenance(fake_data)
    assert p["git_sha"] is None or len(p["git_sha"]) == 40
    assert set(p["versions"]) == {"torch", "torchvision", "numpy", "sklearn"}
    assert p["subset_info"]["synthetic"] is True        # subset_info.json of the dataset is embedded
    save_json({"provenance": p}, tmp_path / "x.json")    # must be JSON-serialisable
    assert json.loads((tmp_path / "x.json").read_text())["provenance"]["python"] == p["python"]


def test_provenance_without_dataset():
    assert provenance()["subset_info"] is None
