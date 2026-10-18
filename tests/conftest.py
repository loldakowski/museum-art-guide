import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _make_fake(out, seed=0):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_dataset.py"), "--out", str(out),
                    "--per-class", "35", "--size", "32", "--seed", str(seed)], check=True, capture_output=True)
    return out


@pytest.fixture(scope="session")
def fake_data(tmp_path_factory):
    """A small synthetic dataset with the same layout as the WikiArt subset."""
    return _make_fake(tmp_path_factory.mktemp("data") / "fake")


@pytest.fixture(scope="session")
def other_fake_data(tmp_path_factory):
    """Same metadata.csv as fake_data, different pixels (another seed)."""
    return _make_fake(tmp_path_factory.mktemp("data") / "other", seed=1)
