"""Loading the WikiArt subset and splitting it WITHOUT artist leakage.

Why the split is grouped by artist
----------------------------------
A painter has a recognisable hand (palette, brushwork, favourite subjects). With a random split, the
same painter appears in train and test and a classifier can score well by recognising *Monet* rather
than *Impressionism*. The museum guide will meet paintings by artists it has never seen, so the test
set must contain unseen artists. We therefore use ``StratifiedGroupKFold`` with one group per named
artist. "Unknown Artist" is a catch-all label (about half the rows), so each of those images is its
own group. ``scripts/03_diagnose_mlp.py`` measures how much a random split inflates the accuracy.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

UNKNOWN_ARTIST = "Unknown Artist"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------------------------------
# Metadata and splits
# --------------------------------------------------------------------------------------------------
def load_metadata(data_dir) -> pd.DataFrame:
    """metadata.csv written by make_wikiart_subset.py (one row per image)."""
    path = Path(data_dir) / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build the dataset first: bash scripts/build_dataset.sh")
    meta = pd.read_csv(path)
    missing = {"path", "label", "label_idx", "id", "artist"} - set(meta.columns)
    if missing:
        raise ValueError(f"metadata.csv lacks columns {missing}")
    return meta.sort_values(["label_idx", "id"]).reset_index(drop=True)


def class_names(meta: pd.DataFrame) -> list[str]:
    """Class names ordered by label_idx (= torchvision ImageFolder order)."""
    return meta.drop_duplicates("label_idx").sort_values("label_idx")["label"].tolist()


def artist_groups(meta: pd.DataFrame) -> np.ndarray:
    """One group per named artist; every 'Unknown Artist' image is its own group."""
    artist = meta["artist"].astype(str).to_numpy()
    ids = meta["id"].astype(str).to_numpy()
    return np.where(artist == UNKNOWN_ARTIST, "unknown_" + ids, "artist_" + artist)


def make_split(meta: pd.DataFrame, mode: str = "artist", val_frac: float = 0.15,
               test_frac: float = 0.15, seed: int = 0) -> np.ndarray:
    """Return an array of 'train' / 'val' / 'test' (one entry per row of meta).

    mode='artist': stratified by class AND grouped by artist (no painter in two splits).
    mode='random': stratified random split (kept only to *measure* the leakage it causes).
    """
    y = meta["label_idx"].to_numpy()
    idx = np.arange(len(meta))
    if mode == "random":
        trval, test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
        train, val = train_test_split(trval, test_size=val_frac / (1 - test_frac), stratify=y[trval],
                                      random_state=seed)
    elif mode == "artist":
        groups = artist_groups(meta)
        outer = StratifiedGroupKFold(n_splits=max(2, round(1 / test_frac)), shuffle=True, random_state=seed)
        trval, test = next(outer.split(idx, y, groups))
        inner = StratifiedGroupKFold(n_splits=max(2, round((1 - test_frac) / val_frac)), shuffle=True,
                                     random_state=seed)
        tr, va = next(inner.split(trval, y[trval], groups[trval]))
        train, val = trval[tr], trval[va]
    else:
        raise ValueError(f"unknown split mode {mode!r}")
    split = np.empty(len(meta), dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"
    return split


def split_report(meta: pd.DataFrame, split: np.ndarray) -> dict:
    """Counts per split/class and the named artists shared between splits (must be 0 for mode='artist')."""
    named = meta["artist"] != UNKNOWN_ARTIST
    artists = {s: set(meta.loc[(split == s) & named, "artist"]) for s in SPLITS}
    counts = pd.crosstab(meta["label"], split).reindex(columns=list(SPLITS), fill_value=0)
    return {
        "n": {s: int((split == s).sum()) for s in SPLITS},
        "per_class": counts.to_dict(orient="index"),
        "named_artists": {s: len(a) for s, a in artists.items()},
        "shared_artists": {
            "train-val": sorted(artists["train"] & artists["val"]),
            "train-test": sorted(artists["train"] & artists["test"]),
            "val-test": sorted(artists["val"] & artists["test"]),
        },
    }


# --------------------------------------------------------------------------------------------------
# Images as a numpy array (for the classical baseline and the MLP)
# --------------------------------------------------------------------------------------------------
def _read_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = ImageOps.fit(im, (size, size), method=Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def load_image_array(meta: pd.DataFrame, data_dir, size: int = 128, cache: bool = True) -> np.ndarray:
    """All images of meta as a uint8 array (N, size, size, 3), cached in data_dir/.cache."""
    data_dir = Path(data_dir)
    key = hashlib.md5(("|".join(meta["path"]) + f"|{size}").encode()).hexdigest()[:10]
    cache_file = data_dir / ".cache" / f"images_{size}_{key}.npy"
    if cache and cache_file.exists():
        return np.load(cache_file)
    X = np.stack([_read_rgb(data_dir / p, size) for p in meta["path"]])
    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, X)
    return X


# --------------------------------------------------------------------------------------------------
# Image datasets and loaders (for the CNN)
# --------------------------------------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, data_dir, paths, labels, transform):
        self.data_dir = Path(data_dir)
        self.paths = list(paths)
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.data_dir / self.paths[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, self.labels[i]


def image_transform(size: int = 128, train: bool = False, augment: bool = True):
    """ImageNet normalisation (required by the pretrained ResNet).

    Augmentation is deliberately geometric only (crop, flip): colour is a key cue of style
    (Pop Art's flat saturated colours, Baroque's dark palette), so we do not jitter hue.
    """
    from torchvision import transforms as T
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train and augment:
        return T.Compose([T.RandomResizedCrop(size, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                          T.ToTensor(), norm])
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), norm])


def make_image_loaders(meta, split, data_dir, size=128, batch_size=32, augment=True, num_workers=2):
    loaders = {}
    for s in SPLITS:
        m = meta[split == s]
        ds = ImageDataset(data_dir, m["path"], m["label_idx"], image_transform(size, s == "train", augment))
        loaders[s] = DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"), num_workers=num_workers,
                                pin_memory=torch.cuda.is_available())
    return loaders
