"""Glue shared by the MLP scripts (training and diagnosis): data as tensors, one training run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .common import set_seed
from .data import class_names, load_image_array, load_metadata, make_split
from .evaluation import classification_metrics, predict
from .features import compute_features
from .mlp import MLP, standardization_stats
from .training import TrainConfig, train


@dataclass
class FeatureData:
    X: dict          # split -> float32 tensor (N, D)
    y: dict          # split -> int64 tensor (N,)
    classes: list
    features: str
    split_mode: str

    @property
    def dim(self):
        return self.X["train"].shape[1]


def load_feature_data(data_dir, features="pixels32", split_mode="artist", split_seed=0, size=128) -> FeatureData:
    meta = load_metadata(data_dir)
    split = make_split(meta, split_mode, seed=split_seed)
    X = compute_features(features, load_image_array(meta, data_dir, size), Path(data_dir) / ".cache")
    y = meta["label_idx"].to_numpy()
    Xs = {s: torch.from_numpy(X[split == s]).float() for s in ("train", "val", "test")}
    ys = {s: torch.from_numpy(y[split == s]).long() for s in ("train", "val", "test")}
    return FeatureData(Xs, ys, class_names(meta), features, split_mode)


def tensor_loaders(d: FeatureData, batch_size=64, seed=0, shuffle_labels=False):
    g = torch.Generator().manual_seed(seed)
    ytr = d.y["train"]
    if shuffle_labels:  # sanity check: with random labels nothing can generalise
        ytr = ytr[torch.randperm(len(ytr), generator=g)]
    return {
        "train": DataLoader(TensorDataset(d.X["train"], ytr), batch_size=batch_size, shuffle=True, generator=g),
        "val": DataLoader(TensorDataset(d.X["val"], d.y["val"]), batch_size=512),
        "test": DataLoader(TensorDataset(d.X["test"], d.y["test"]), batch_size=512),
    }


def run_mlp(d: FeatureData, mlp_kwargs: dict, cfg: TrainConfig, device="cpu", batch_size=64,
            normalize=True, shuffle_labels=False, log=print, log_every=10):
    """Build, train and evaluate one MLP. Returns (model, history, test_metrics)."""
    set_seed(cfg.seed)
    model = MLP(d.dim, len(d.classes), **mlp_kwargs)
    if normalize:
        model.set_normalization(*standardization_stats(d.X["train"]))
    loaders = tensor_loaders(d, batch_size, cfg.seed, shuffle_labels)
    hist = train(model, loaders["train"], loaders["val"], cfg, device, log=log, log_every=log_every)
    probs, y = predict(model, loaders["test"], device)
    metrics = classification_metrics(y, probs.argmax(1), d.classes, probs)
    hist["n_parameters"] = model.n_parameters()
    return model, hist, metrics


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}
