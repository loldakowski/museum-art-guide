"""Loading the WikiArt subset and splitting it WITHOUT artist leakage.

Why the split is grouped by artist
----------------------------------
A painter has a recognisable hand (palette, brushwork, favourite subjects). With a random split, the
same painter appears in train and test and a classifier can score well by recognising *Monet* rather
than *Impressionism*. The museum guide will meet paintings by artists it has never seen, so the test
set must contain unseen artists. We therefore split by groups (``grouped_stratified_split``), one group
per named artist. "Unknown Artist" is a catch-all label (about half the rows), so each of those images is its
own group. ``scripts/03_diagnose_mlp.py`` measures how much a random split inflates the accuracy.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

UNKNOWN_ARTIST = "Unknown Artist"
ANNOTATIONS_DIR = Path(__file__).resolve().parents[2] / "annotations"  # repo root (package installed with -e)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------------------------------
# Metadata and splits
# --------------------------------------------------------------------------------------------------
def duplicates_file(data_dir) -> Path | None:
    """The committed duplicate verdicts of this dataset: annotations/<data dir name>_duplicates.csv, if any."""
    f = ANNOTATIONS_DIR / f"{Path(data_dir).name}_duplicates.csv"
    return f if f.exists() else None


def _components(pairs) -> dict:
    """Connected components of an undirected graph given as (a, b) pairs: {node: representative}."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    return {x: find(x) for x in parent}


def apply_duplicates(meta: pd.DataFrame, verdicts: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Use human duplicate verdicts (annotations/README.md): returns (meta with a 'dup_group' column, removed ids).

    Every verdict except 'different' links two images; a dup_group is a connected component of links, and the
    split keeps it whole. Images whose 'same_image' copies carry different labels are removed (label noise)."""
    v = verdicts[verdicts["id_a"].isin(meta["id"]) & verdicts["id_b"].isin(meta["id"])]
    label = dict(zip(meta["id"], meta["label"]))
    same = _components(v.loc[v["verdict"] == "same_image", ["id_a", "id_b"]].itertuples(index=False))
    copies = pd.Series(same).groupby(lambda i: same[i]).groups
    removed = sorted(i for members in copies.values() if len({label[m] for m in members}) > 1 for i in members)
    linked = _components(v.loc[v["verdict"] != "different", ["id_a", "id_b"]].itertuples(index=False))
    meta = meta.assign(dup_group=meta["id"].map(lambda i: f"dup_{linked[i]}" if i in linked else None))
    return meta[~meta["id"].isin(removed)].reset_index(drop=True), removed


def load_metadata(data_dir, duplicates="auto") -> pd.DataFrame:
    """metadata.csv written by make_wikiart_subset.py (one row per image), with the duplicate verdicts applied.

    duplicates: 'auto' = duplicates_file(data_dir), a path to a verdict CSV, or None (raw metadata)."""
    path = Path(data_dir) / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build the dataset first: bash scripts/build_dataset.sh")
    meta = pd.read_csv(path)
    missing = {"path", "label", "label_idx", "id", "artist"} - set(meta.columns)
    if missing:
        raise ValueError(f"metadata.csv lacks columns {missing}")
    meta = meta.sort_values(["label_idx", "id"]).reset_index(drop=True)
    f = duplicates_file(data_dir) if duplicates == "auto" else duplicates
    return apply_duplicates(meta, pd.read_csv(f))[0] if f else meta


def class_names(meta: pd.DataFrame) -> list[str]:
    """Class names ordered by label_idx (= torchvision ImageFolder order)."""
    return meta.drop_duplicates("label_idx").sort_values("label_idx")["label"].tolist()


def artist_groups(meta: pd.DataFrame) -> np.ndarray:
    """One group per named artist; every 'Unknown Artist' image is its own group. Groups that share a duplicate
    group (column 'dup_group', from apply_duplicates) are merged, so copies of an image never cross splits."""
    artist = meta["artist"].astype(str).to_numpy()
    ids = meta["id"].astype(str).to_numpy()
    groups = np.where(artist == UNKNOWN_ARTIST, "unknown_" + ids, "artist_" + artist)
    if "dup_group" not in meta:
        return groups
    links = [(g, d) for g, d in zip(groups, meta["dup_group"]) if isinstance(d, str)]
    merged = _components(links)
    return np.array([merged.get(g, g) for g in groups], dtype=object)


def grouped_stratified_split(strata, groups, fractions=(0.70, 0.15, 0.15), seed: int = 0) -> np.ndarray:
    """Assign whole groups to splits so that every split gets its share of every stratum (class...).

    Greedy and transparent: groups are visited largest first (random order among equal sizes); each
    group goes to the split with the largest relative class deficit, weighted by the group's class
    counts. Small groups (single 'Unknown Artist' images) come last and fill the remaining gaps, so they
    are spread over all splits. (sklearn's StratifiedGroupKFold put all singletons in one fold here.)
    Returns the split index (0, 1, 2...) of every sample.
    """
    rng = np.random.default_rng(seed)
    classes, y_idx = np.unique(strata, return_inverse=True)
    uniq, g_idx = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(uniq), len(classes)))
    np.add.at(counts, (g_idx, y_idx), 1)
    target = np.outer(fractions, counts.sum(0))
    current = np.zeros_like(target)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts.sum(1)[order], kind="stable")]
    assign = np.empty(len(uniq), dtype=int)
    for g in order:
        deficit = ((target - current) / np.maximum(target, 1)) @ counts[g]
        best = int(np.argmax(deficit + rng.uniform(0, 1e-9, len(deficit))))
        assign[g] = best
        current[best] += counts[g]
    return assign[g_idx]


def make_split(meta: pd.DataFrame, mode: str = "artist", val_frac: float = 0.15,
               test_frac: float = 0.15, seed: int = 0) -> np.ndarray:
    """Return an array of 'train' / 'val' / 'test' (one entry per row of meta).

    mode='artist': stratified by class AND grouped by artist (no painter in two splits).
    mode='random': stratified random split (kept only to *measure* the leakage it causes).
    """
    y = meta["label_idx"].to_numpy()
    if mode == "random":
        idx = np.arange(len(meta))
        trval, test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
        train, val = train_test_split(trval, test_size=val_frac / (1 - test_frac), stratify=y[trval],
                                      random_state=seed)
        split = np.empty(len(meta), dtype=object)
        split[train], split[val], split[test] = "train", "val", "test"
        return split
    if mode == "artist":
        # strata = class x (named / unknown artist): every split gets its share of both kinds of images
        strata = 2 * y + (meta["artist"].to_numpy() == UNKNOWN_ARTIST)
        a = grouped_stratified_split(strata, artist_groups(meta), (1 - val_frac - test_frac, val_frac, test_frac),
                                     seed)
        return np.array(SPLITS, dtype=object)[a]
    raise ValueError(f"unknown split mode {mode!r}")


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
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)
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
