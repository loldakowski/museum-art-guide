import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fake_data(tmp_path_factory):
    """A small synthetic dataset with the same layout as the WikiArt subset."""
    out = tmp_path_factory.mktemp("data") / "fake"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fake_dataset.py"), "--out", str(out),
                    "--per-class", "35", "--size", "32"], check=True, capture_output=True)
    return out
