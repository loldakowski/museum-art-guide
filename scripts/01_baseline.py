#!/usr/bin/env python
"""Step 1 - Classical baseline: logistic regression on raw pixels and on hand-crafted features.

This is the "simpler classical model" the director asks about: every neural network must beat it.
    python scripts/01_baseline.py --data-dir data/wikiart_style5
Outputs: results/01_baseline/<features>.json, confusion matrices; checkpoints/baseline_<features>.joblib
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np

from museum_guide.baseline import fit_logreg
from museum_guide.common import provenance, save_json
from museum_guide.data import class_names, load_image_array, load_metadata, make_split, split_report
from museum_guide.evaluation import classification_metrics, fmt, most_confused
from museum_guide.features import FEATURE_SETS, compute_features
from museum_guide.plots import confusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_style5")
    p.add_argument("--out", default="results/01_baseline")
    p.add_argument("--split", choices=["artist", "random"], default="artist")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--fold", type=int, default=None, help="cross-validation fold (0-4) instead of the 70/15/15 split")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--features", nargs="+", default=["pixels32", "handcrafted"], choices=list(FEATURE_SETS))
    p.add_argument("--C", type=float, default=None, help="fixed regularisation (default: choose it on validation)")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    a = p.parse_args()
    out = Path(a.out)

    meta = load_metadata(a.data_dir)
    names = class_names(meta)
    split = make_split(meta, a.split, seed=a.split_seed, fold=a.fold)
    y = meta["label_idx"].to_numpy()
    X = load_image_array(meta, a.data_dir, a.size)
    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    for fs in a.features:
        t0 = time.perf_counter()
        F = compute_features(fs, X, Path(a.data_dir) / ".cache")
        t_feat = time.perf_counter() - t0
        t0 = time.perf_counter()
        model, val_scores = fit_logreg(F[tr], y[tr], F[va], y[va], **({"Cs": (a.C,)} if a.C else {}))
        t_fit = time.perf_counter() - t0
        probs = model.predict_proba(F[te])
        m = classification_metrics(y[te], probs.argmax(1), names, probs)
        m.update({"model": f"logreg_{fs}", "features": fs, "feature_dim": int(F.shape[1]),
                  "val_acc_per_C": val_scores, "best_C": float(model[-1].C),
                  "train_acc": float(np.mean(model.predict(F[tr]) == y[tr])),
                  "val_acc": float(max(val_scores.values())),
                  "feature_time_s": t_feat, "fit_time_s": t_fit, "split": a.split,
                  "split_info": split_report(meta, split)["n"],
                  "most_confused": most_confused(m["confusion_matrix"], names), "provenance": provenance(a.data_dir)})
        print(f"[{fs:12s}] dim={F.shape[1]:5d} C={m['best_C']:g}  train {m['train_acc']:.3f}  "
              f"val {m['val_acc']:.3f}  test {fmt(m)}")
        tag = fs if a.split == "artist" else f"{fs}_randomsplit"
        save_json(m, out / f"{tag}.json")
        confusion(m["confusion_matrix"], names, out / f"confusion_{tag}.png",
                  f"Logistic regression on {fs} - test")
        Path(a.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": fs, "classes": names, "provenance": provenance(a.data_dir)},
                    Path(a.checkpoint_dir) / f"baseline_{tag}.joblib")


if __name__ == "__main__":
    main()
