#!/usr/bin/env python
"""Step 10 - Can ResNet-18 frozen + logistic regression do better than in step 4? (exploratory)

Three things are tried on the same split as step 4, without changing any default of the other scripts:
  - the C grid of the logistic regression: the repo grid (1e-5 ... 10), a finer grid around 0.01, and both
  - the input size: 128 px (step 4) or 224 px, the size ResNet-18 saw on ImageNet (the 224 px copy of the
    same images that step 7 uses for CLIP)
  - flip averaging: the embedding of a painting is averaged with the embedding of its mirror image

The setting is chosen on validation accuracy only. The test accuracy is printed for every row to see how
much it moves, but the honest number to report is the test accuracy of the row chosen on validation.

    python scripts/10_frozen_tuning.py
Outputs: results/10_frozen_tuning/summary.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from museum_guide.baseline import fit_logreg
from museum_guide.common import get_device, provenance, save_json, set_seed
from museum_guide.data import class_names, load_metadata, make_image_loaders, make_split
from museum_guide.evaluation import classification_metrics, paired_comparison
from museum_guide.resnet import ResNetClassifier

REPO_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
FINE_GRID = (0.002, 0.005, 0.01, 0.02, 0.05)
GRIDS = {"repo grid (1e-5...10)": REPO_GRID, "fine grid (0.002...0.05)": FINE_GRID,
         "both grids": tuple(sorted(set(REPO_GRID) | set(FINE_GRID)))}


@torch.no_grad()
def embeddings(model, loader, device, flip):
    model.eval().to(device)
    E, Y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        e = model.embed(xb)
        if flip:  # average with the mirror image
            e = (e + model.embed(torch.flip(xb, dims=[3]))) / 2
        E.append(e.cpu())
        Y.append(yb)
    return torch.cat(E).numpy(), torch.cat(Y).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_style5")
    p.add_argument("--image-dir-224", default="data/wikiart_style5_224", help="224 px copy of the same images")
    p.add_argument("--out", default="results/10_frozen_tuning")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true", help="random weights (smoke test only)")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    set_seed(0)
    device = get_device(a.device)
    meta = load_metadata(a.data_dir)  # duplicate verdicts applied, exactly as in step 4
    names = class_names(meta)
    split = make_split(meta, "artist", seed=0)
    model = ResNetClassifier(len(names), "frozen", pretrained=not a.no_pretrained)
    n = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    print(f"device {device} | {len(meta)} images | {n}")

    inputs = [("128 px", a.data_dir, meta, 128)]
    if (Path(a.image_dir_224) / "metadata.csv").exists():
        big = load_metadata(a.image_dir_224, duplicates=None).set_index("id")
        missing = set(meta["id"]) - set(big.index)
        if missing:
            raise SystemExit(f"{len(missing)} images are missing from {a.image_dir_224}")
        inputs.append(("224 px", a.image_dir_224, meta.assign(path=big.loc[meta["id"], "path"].to_numpy()), 224))
    else:
        print(f"{a.image_dir_224} not found: 224 px rows skipped")

    rows, preds = [], {}
    for size_name, image_dir, m_meta, size in inputs:
        L = make_image_loaders(m_meta, split, image_dir, size, a.batch_size, augment=False, num_workers=a.workers)
        for flip in (False, True):
            E = {s: embeddings(model, L[s], device, flip) for s in ("train", "val", "test")}
            for grid_name, Cs in GRIDS.items():
                clf, val_scores = fit_logreg(E["train"][0], E["train"][1], E["val"][0], E["val"][1], Cs=Cs)
                C = float(clf[-1].C)
                probs = clf.predict_proba(E["test"][0])
                met = classification_metrics(E["test"][1], probs.argmax(1), names, probs)
                key = f"{size_name} | flip {'yes' if flip else 'no'} | {grid_name}"
                preds[key] = (E["test"][1], probs.argmax(1))
                rows.append({"setting": key, "input": size_name, "flip": flip, "grid": grid_name, "best_C": C,
                             "val_acc": max(val_scores.values()),
                             "val_acc_per_C": {f"{k:g}": v for k, v in val_scores.items()},
                             "test_acc": met["accuracy"], "test_ci95": met["accuracy_ci95"]})
                edge = "  (edge of grid)" if C in (min(Cs), max(Cs)) else ""
                print(f"{key:<46} | C {C:<6g}{edge:<16} | val {rows[-1]['val_acc']:.1%} | "
                      f"test {met['accuracy']:.1%} [{met['accuracy_ci95'][0]:.1%}, {met['accuracy_ci95'][1]:.1%}]")

    ref = rows[0]  # 128 px, no flip, repo grid = step 4
    best = max(rows, key=lambda r: r["val_acc"])  # first one wins ties, so the simplest setting is preferred
    y, p_ref = preds[ref["setting"]]
    comparison = paired_comparison(y, preds[best["setting"]][1], p_ref)
    print(f"\nStep 4 setting   : {ref['setting']} -> val {ref['val_acc']:.1%}, test {ref['test_acc']:.1%}")
    print(f"Best on val      : {best['setting']}, C = {best['best_C']:g} -> val {best['val_acc']:.1%}, "
          f"test {best['test_acc']:.1%} [{best['test_ci95'][0]:.1%}, {best['test_ci95'][1]:.1%}]")
    print(f"Paired test, best vs step 4: {comparison['diff']:+.1%} "
          f"[{comparison['diff_ci95'][0]:+.1%}, {comparison['diff_ci95'][1]:+.1%}], "
          f"McNemar p = {comparison['mcnemar_p']:.3g} (only best right: {comparison['a_right_b_wrong']}, "
          f"only step 4 right: {comparison['a_wrong_b_right']})")
    print(f"1 validation image = {100 / n['val']:.2f} points, 1 test image = {100 / n['test']:.2f} points")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    save_json({"rows": rows, "step4_setting": ref, "best_on_val": best, "best_vs_step4": comparison,
               "provenance": provenance(a.data_dir)}, Path(a.out) / "summary.json")


if __name__ == "__main__":
    main()
