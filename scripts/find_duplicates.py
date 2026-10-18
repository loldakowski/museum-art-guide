#!/usr/bin/env python
"""Near-duplicate images in a subset: the same painting can appear twice in WikiArt (another scan, a crop,
another colour print). If the copies land in train and test, the test measures memory, not style recognition.

Finds candidate pairs with two signals:
  * cosine similarity of ImageNet ResNet-18 embeddings (same painting re-scanned or re-cropped);
  * Hamming distance of a 64-bit perceptual hash (dHash) of the grey image (near-exact copies).
Calibration: among images with a NAMED artist, the similarity of same-painter vs different-painter pairs.
The candidates are then judged by eye; the verdicts (annotations/<dataset>_duplicates.csv) are the ground truth
used by the split, and this script reports how a fixed cosine rule agrees with them.

Writes <out>/duplicates_report.json (ids and numbers only) and a grid of the most similar pairs to
<data-dir>/.cache/duplicate_pairs.png (shows paintings: never committed).
    python scripts/find_duplicates.py --data-dir data/wikiart_style5 --out results/00_explore
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from museum_guide.common import get_device, provenance, save_json
from museum_guide.data import (UNKNOWN_ARTIST, ImageDataset, duplicates_file, image_transform, load_image_array,
                               load_metadata, make_split)
from museum_guide.plots import image_grid
from museum_guide.resnet import ResNetClassifier, extract_embeddings

DUPLICATE_VERDICTS = ("same_image", "same_composition")


def dhash(path: Path, size: int = 8) -> np.ndarray:
    """64-bit difference hash: is each pixel brighter than its right neighbour, on a 9x8 grey thumbnail."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).ravel()


def quantiles(v):
    return {k: float(np.quantile(v, x)) for k, x in (("p50", .5), ("p90", .9), ("p99", .99), ("max", 1.0))} \
        if len(v) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_style5")
    p.add_argument("--out", default="results/00_explore")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--top", type=int, default=24, help="most similar pairs listed and shown")
    p.add_argument("--rule", type=float, default=0.86, help="cosine threshold compared with the human verdicts")
    p.add_argument("--no-pretrained", action="store_true", help="random weights (smoke test only)")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    data_dir, out = Path(a.data_dir), Path(a.out)
    meta = load_metadata(data_dir, duplicates=None)  # every image, before the verdicts are applied
    n, ids = len(meta), meta["id"].to_numpy()

    model = ResNetClassifier(meta["label_idx"].nunique(), "frozen", pretrained=not a.no_pretrained)
    ds = ImageDataset(data_dir, meta["path"], meta["label_idx"], image_transform(a.img_size, train=False))
    E, _ = extract_embeddings(model, DataLoader(ds, batch_size=64), get_device(a.device))
    E = torch.nn.functional.normalize(E, dim=1).numpy()
    S = E @ E.T
    np.fill_diagonal(S, -1.0)
    H = np.stack([dhash(data_dir / pth) for pth in meta["path"]])
    ham = (H[:, None, :] != H[None, :, :]).sum(-1)

    iu = np.triu_indices(n, 1)
    sim, hd = S[iu], ham[iu]
    artist = meta["artist"].to_numpy()
    named = (artist[iu[0]] != UNKNOWN_ARTIST) & (artist[iu[1]] != UNKNOWN_ARTIST)
    same_artist = named & (artist[iu[0]] == artist[iu[1]])

    # split before the verdicts (as in the committed results) and after them (current code)
    old = make_split(meta, "artist", seed=0)
    kept = load_metadata(data_dir)
    new = dict(zip(kept["id"], make_split(kept, "artist", seed=0)))
    new = np.array([new.get(i, "removed") for i in ids], dtype=object)

    def leaks(split, t):
        te, tr = split == "test", split == "train"
        return int((S[np.ix_(te, tr)].max(1) >= t).sum())

    order = np.argsort(-sim)[:a.top]
    pairs = [{"a": ids[i], "b": ids[j], "cosine": float(S[i, j]), "dhash_hamming": int(ham[i, j]),
              "label_a": meta["label"][i], "label_b": meta["label"][j], "artist_a": artist[i], "artist_b": artist[j],
              "split_before": [old[i], old[j]], "split_after": [new[i], new[j]]}
             for i, j in zip(iu[0][order], iu[1][order])]
    report = {
        "n_images": n, "img_size": a.img_size,
        "cosine_all_pairs": quantiles(sim), "cosine_nearest_neighbour": quantiles(S.max(1)),
        "calibration_named_artists": {"same_artist_pairs": int(same_artist.sum()),
                                      "cosine_same_artist": quantiles(sim[same_artist]),
                                      "cosine_different_artist": quantiles(sim[named & ~same_artist])},
        "pairs_cosine_ge": {str(t): int((sim >= t).sum()) for t in (0.95, 0.9, 0.86, 0.8)},
        "pairs_dhash_le": {str(t): int((hd <= t).sum()) for t in (3, 5, 10)},
        "test_images_with_train_neighbour_cosine_ge": {
            "before_verdicts": {str(t): leaks(old, t) for t in (0.9, 0.86, 0.8)},
            "after_verdicts": {str(t): leaks(new, t) for t in (0.9, 0.86, 0.8)}},
        "top_pairs": pairs,
    }

    f = duplicates_file(data_dir)
    if f:
        v = pd.read_csv(f)
        pos = {x: k for k, x in enumerate(ids)}
        v["cosine"] = [float(S[pos[x], pos[y]]) for x, y in zip(v["id_a"], v["id_b"])]
        v["dhash_hamming"] = [int(ham[pos[x], pos[y]]) for x, y in zip(v["id_a"], v["id_b"])]
        human, rule = v["verdict"].isin(DUPLICATE_VERDICTS), v["cosine"] >= a.rule
        removed = sorted(set(ids) - set(kept["id"]))
        report["verdicts"] = {
            "file": f.name, "pairs_judged": len(v), "counts": v["verdict"].value_counts().to_dict(),
            f"rule_cosine_ge_{a.rule}": {
                "agree_duplicate": int((human & rule).sum()), "agree_not_duplicate": int((~human & ~rule).sum()),
                "rule_only": v.loc[rule & ~human, ["id_a", "id_b", "verdict", "cosine"]].to_dict("records"),
                "human_only": v.loc[human & ~rule, ["id_a", "id_b", "verdict", "cosine"]].to_dict("records"),
                "unjudged_pairs_above_rule": int((sim >= a.rule).sum()) - int(rule.sum())},
            "pairs": v.to_dict("records"),
            "removed_label_conflicts": [{"id": i, "label": meta.set_index("id").at[i, "label"],
                                         "split_before": old[pos[i]]} for i in removed],
            "duplicate_groups_after": kept.dropna(subset=["dup_group"]).groupby("dup_group")["id"].apply(list).to_dict(),
        }
    report["provenance"] = provenance(data_dir)
    save_json(report, out / "duplicates_report.json")

    X = load_image_array(meta, data_dir, size=a.img_size)
    pos = {x: k for k, x in enumerate(ids)}
    imgs, titles = [], []
    for k, pr in enumerate(pairs):
        imgs += [X[pos[pr["a"]]], X[pos[pr["b"]]]]
        titles += [f"#{k} {pr['cosine']:.3f} h{pr['dhash_hamming']}\n{pr['a']} {pr['label_a'][:9]}",
                   f"#{k}\n{pr['b']} {pr['label_b'][:9]}"]
    image_grid(imgs, titles, data_dir / ".cache" / "duplicate_pairs.png", ncols=8,
               suptitle="Most similar pairs (cosine, dHash Hamming distance)")
    print({k: report[k] for k in ("cosine_nearest_neighbour", "calibration_named_artists", "pairs_cosine_ge",
                                  "test_images_with_train_neighbour_cosine_ge")})
    if f:
        print({k: report["verdicts"][k] for k in (f"rule_cosine_ge_{a.rule}", "removed_label_conflicts",
                                                    "duplicate_groups_after")})


if __name__ == "__main__":
    main()
