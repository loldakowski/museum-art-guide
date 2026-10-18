"""Small shared helpers: seeding, device, JSON I/O, timing."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG we use so that a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(obj), indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parent, capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(data_dir=None) -> dict:
    """Which code, library versions and dataset produced a result file (saved next to the numbers).

    git_dirty=True means tracked files had uncommitted changes: the SHA alone does not describe the code."""
    import sklearn
    import torchvision
    status = _git("status", "--porcelain", "--untracked-files=no")
    info = data_dir and Path(data_dir) / "subset_info.json"
    return {
        "git_sha": _git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status),
        "dataset_fingerprint": dataset_fingerprint(data_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),  # plain str:
                     "numpy": np.__version__, "sklearn": sklearn.__version__},  # torch.load(weights_only) needs it
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "subset_info": load_json(info) if info and info.exists() else None,
    }


def dataset_fingerprint(data_dir) -> str | None:
    """Short hash of metadata.csv, of every image file it lists (in order) and of the dataset's duplicate verdicts:
    changes if images, labels, artists or verdicts change, including a different image size or seed with identical
    metadata. None if there is no dataset."""
    meta = data_dir and Path(data_dir) / "metadata.csv"
    if not meta or not meta.exists():
        return None
    h = hashlib.sha256(meta.read_bytes())
    with open(meta, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h.update((Path(data_dir) / row["path"]).read_bytes())
    from .data import duplicates_file  # the duplicate verdicts change which images are used and how they split
    dup = duplicates_file(data_dir)
    if dup:
        h.update(dup.read_bytes())
    return h.hexdigest()[:16]


def check_checkpoint(ckpt: dict, data_dir, path) -> None:
    """Refuse a checkpoint trained on another dataset (or saved before checkpoints recorded their dataset)."""
    saved = (ckpt.get("provenance") or {}).get("dataset_fingerprint")
    current = dataset_fingerprint(data_dir)
    if saved is None or current is None or saved != current:
        raise ValueError(f"{path}: checkpoint dataset {saved} does not match the current dataset {current} "
                         f"({data_dir}). Retrain it on this dataset, or delete it.")


@contextmanager
def timer():
    """with timer() as t: ...   then t() returns elapsed seconds."""
    t0 = time.perf_counter()
    end = [None]
    yield lambda: (end[0] if end[0] is not None else time.perf_counter()) - t0
    end[0] = time.perf_counter()
