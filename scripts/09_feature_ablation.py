#!/usr/bin/env python
"""Step 9 - Ablation of the hand-crafted features, on the VALIDATION set only (the test set is never read).

Same pipeline as step 1 (StandardScaler + logistic regression, C chosen on validation), on:
  * each block alone: HSV histogram, colour moments, HOG, LBP;
  * all blocks minus one;
  * one parameter changed inside the full set: HOG 16x16 cells, HSV bins (8,3,3), LBP P=16 R=2.
Per variant: validation accuracy + bootstrap 95 % CI, paired comparison with the full default set on the same
validation images (paired bootstrap + McNemar, Holm over the variants), number of features, CPU extraction time per
image. Exploratory: the default features and every reported result stay as they are.
    python scripts/09_feature_ablation.py --data-dir data/wikiart_style5
Outputs: results/09_feature_ablation/summary.{json,md}
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np

from museum_guide.baseline import C_GRID, fit_logreg
from museum_guide.common import load_json, provenance, save_json
from museum_guide.data import load_image_array, load_metadata, make_split
from museum_guide.evaluation import bootstrap_ci, holm, paired_comparison
from museum_guide.features import color_moments, hog_features, hsv_histogram, lbp_histogram

BLOCKS = ("hsv", "moments", "hog", "lbp")  # order of the concatenation in features.handcrafted
DEFAULTS = {"hsv": {"bins": (12, 4, 4)}, "moments": {}, "hog": {"pixels_per_cell": 8}, "lbp": {"P": 8, "R": 1.0}}
EXTRACT = {"hsv": hsv_histogram, "moments": color_moments, "hog": hog_features, "lbp": lbp_histogram}


def variants() -> list[dict]:
    """name, group, blocks used and their parameters (anything not listed = default)."""
    v = [{"name": "all blocks (default)", "group": "reference", "blocks": BLOCKS, "params": {}}]
    v += [{"name": f"{b} alone", "group": "alone", "blocks": (b,), "params": {}} for b in BLOCKS]
    v += [{"name": f"all minus {b}", "group": "minus_one", "blocks": tuple(x for x in BLOCKS if x != b),
           "params": {}} for b in BLOCKS]
    v += [{"name": "HOG 16x16 cells", "group": "sensitivity", "blocks": BLOCKS, "params": {"hog": {"pixels_per_cell": 16}}},
          {"name": "HSV bins (8,3,3)", "group": "sensitivity", "blocks": BLOCKS, "params": {"hsv": {"bins": (8, 3, 3)}}},
          {"name": "LBP P=16 R=2", "group": "sensitivity", "blocks": BLOCKS, "params": {"lbp": {"P": 16, "R": 2.0}}}]
    return v


def block_params(v: dict, b: str) -> dict:
    return {**DEFAULTS[b], **v["params"].get(b, {})}


def extract(v: dict, X: np.ndarray) -> np.ndarray:
    return np.concatenate([EXTRACT[b](X, **block_params(v, b)) for b in v["blocks"]], axis=1)


def time_per_image(v: dict, X: np.ndarray, repeats: int) -> dict:
    """Extraction of this variant's blocks for ONE image at a time (as on the kiosk), over images x repeats."""
    extract(v, X[:1])  # warm-up
    times = []
    for _ in range(repeats):
        for i in range(len(X)):
            t0 = time.perf_counter()
            extract(v, X[i:i + 1])
            times.append(time.perf_counter() - t0)
    t = np.asarray(times) * 1000
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.quantile(t, 0.95)), "n": len(t)}


def train_val_rows(split: np.ndarray) -> np.ndarray:
    """Indices of the train and validation images: the only rows this step ever loads."""
    return np.flatnonzero((split == "train") | (split == "val"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_style5")
    p.add_argument("--out", default="results/09_feature_ablation")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--reference", default="results/01_baseline/handcrafted.json",
                   help="step 1 result: the default variant must reproduce its validation accuracy")
    p.add_argument("--timing-images", type=int, default=100, help="train images timed one at a time")
    p.add_argument("--timing-repeats", type=int, default=3)
    a = p.parse_args()

    meta = load_metadata(a.data_dir)
    split = make_split(meta, "artist", seed=a.split_seed)
    rows = train_val_rows(split)
    meta, split = meta.iloc[rows].reset_index(drop=True), split[rows]
    X = load_image_array(meta, a.data_dir, a.size, cache=False)  # train + val images only
    y = meta["label_idx"].to_numpy()
    tr, va = split == "train", split == "val"
    X_time = X[np.flatnonzero(tr)[:a.timing_images]]
    print(f"train {tr.sum()}  val {va.sum()}  (test images not loaded)  timing on {len(X_time)} train images")

    results = []
    for v in variants():
        t0 = time.perf_counter()
        F = extract(v, X)
        t_feat = time.perf_counter() - t0
        t0 = time.perf_counter()
        model, val_scores = fit_logreg(F[tr], y[tr], F[va], y[va])
        t_fit = time.perf_counter() - t0
        pred = model.predict(F[va])
        r = {"name": v["name"], "group": v["group"], "blocks": list(v["blocks"]),
             "params": {b: block_params(v, b) for b in v["blocks"]}, "n_features": int(F.shape[1]),
             "val_acc": float(np.mean(pred == y[va])), "val_acc_ci95": bootstrap_ci(y[va], pred),
             "best_C": float(model[-1].C), "val_acc_per_C": val_scores,
             "best_C_at_grid_edge": float(model[-1].C) in (min(C_GRID), max(C_GRID)),
             "train_acc": float(np.mean(model.predict(F[tr]) == y[tr])),
             "extraction_per_image": time_per_image(v, X_time, a.timing_repeats),
             "extraction_all_s": t_feat, "fit_time_s": t_fit, "val_predictions": pred.tolist()}
        results.append(r)
        print(f"[{v['name']:22s}] dim {r['n_features']:5d}  C={r['best_C']:g}  val {r['val_acc']:.3f} "
              f"[{r['val_acc_ci95'][0]:.3f}, {r['val_acc_ci95'][1]:.3f}]  "
              f"{r['extraction_per_image']['median_ms']:.2f} ms/image")

    ref = results[0]
    for r in results[1:]:
        c = paired_comparison(y[va], r["val_predictions"], ref["val_predictions"])
        r["vs_default"] = {k: c[k] for k in ("diff", "diff_ci95", "a_right_b_wrong", "a_wrong_b_right", "mcnemar_p")}
    for r, ph in zip(results[1:], holm([r["vs_default"]["mcnemar_p"] for r in results[1:]])):
        r["vs_default"]["mcnemar_p_holm"] = ph
        # the two decision rules agreed with the user before the run
        r["vs_default"]["above_default_ci"] = r["val_acc"] > ref["val_acc_ci95"][1]
        r["vs_default"]["paired_ci_above_0"] = r["vs_default"]["diff_ci95"][0] > 0

    check = None
    if a.reference and Path(a.reference).exists():
        step1 = load_json(a.reference)
        check = {"file": a.reference, "step1_val_acc": step1["val_acc"], "step1_best_C": step1["best_C"],
                 "reproduced": bool(np.isclose(step1["val_acc"], ref["val_acc"]) and step1["best_C"] == ref["best_C"])}
        print(f"default vs step 1: val {ref['val_acc']:.4f} vs {step1['val_acc']:.4f}, C {ref['best_C']:g} vs "
              f"{step1['best_C']:g} -> {'reproduced' if check['reproduced'] else 'NOT REPRODUCED'}")

    better = [r["name"] for r in results[1:]
              if r["vs_default"]["above_default_ci"] or r["vs_default"]["paired_ci_above_0"]]
    out = Path(a.out)
    save_json({"data_dir": a.data_dir, "split": "artist", "split_seed": a.split_seed, "n_train": int(tr.sum()),
               "n_val": int(va.sum()), "test_set_used": False, "C_grid": list(C_GRID),
               "note": "C is chosen on the same validation set the accuracy is reported on: every value is "
                       "optimistic by the same selection bias; compare variants, not with test accuracies.",
               "timing": {"images": len(X_time), "repeats": a.timing_repeats, "cpu_count": os.cpu_count(),
                          "machine": platform.machine(), "processor": platform.processor()},
               "reference_check": check, "variants_beating_default": better, "variants": results,
               "val_labels": y[va].tolist(), "provenance": provenance(a.data_dir)}, out / "summary.json")

    lines = ["# Hand-crafted feature ablation (validation set only)", "",
             f"Train {tr.sum()}, validation {va.sum()} images; the test set is not used. C chosen on validation, "
             "so every accuracy is optimistic by the same selection bias.",
             f"Extraction time: one {a.size} px image at a time, median over {len(X_time)} train images x "
             f"{a.timing_repeats} repeats ({platform.machine()}, {os.cpu_count()} CPUs).", "",
             "| variant | features | best C | val acc [95% CI] | diff vs default [paired 95% CI] | McNemar p (Holm) "
             "| extraction ms/image (median, p95) |", "|---|---|---|---|---|---|---|"]
    for r in results:
        lo, hi = r["val_acc_ci95"]
        if "vs_default" in r:
            d = r["vs_default"]
            cmp = f"{d['diff']:+.3f} [{d['diff_ci95'][0]:+.3f}, {d['diff_ci95'][1]:+.3f}]"
            pv = f"{d['mcnemar_p']:.3g} ({d['mcnemar_p_holm']:.3g})"
        else:
            cmp, pv = "-", "-"
        e = r["extraction_per_image"]
        lines.append(f"| {r['name']} | {r['n_features']} | {r['best_C']:g} | {r['val_acc']:.3f} [{lo:.3f}, {hi:.3f}] "
                     f"| {cmp} | {pv} | {e['median_ms']:.2f}, {e['p95_ms']:.2f} |")
    lines += ["", f"Variants above the default (accuracy above the default's CI, or paired CI above 0): "
                  f"{', '.join(better) if better else 'none'}."]
    if check:
        lines.append(f"Default reproduces step 1 ({check['file']}): {check['reproduced']}.")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
