#!/usr/bin/env python
"""Step 0 - Exploratory analysis of the subset and the artist-grouped split.

Outputs (results/00_explore/): eda.json, class/artist figures, a sample grid per class.
    python scripts/00_explore.py --data-dir data/wikiart_style5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from museum_guide.common import provenance, save_json
from museum_guide.data import (UNKNOWN_ARTIST, class_names, duplicates_file, load_image_array, load_metadata,
                               make_split, split_report)
from museum_guide.plots import barh, image_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_style5")
    p.add_argument("--out", default="results/00_explore")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--size", type=int, default=128, help="image size on disk")
    a = p.parse_args()
    out = Path(a.out)

    meta = load_metadata(a.data_dir)
    names = class_names(meta)
    raw = load_metadata(a.data_dir, duplicates=None)
    dup = {"verdicts": getattr(duplicates_file(a.data_dir), "name", None),
           "removed_label_conflicts": sorted(set(raw["id"]) - set(meta["id"])),
           "duplicate_groups": int(meta["dup_group"].nunique()) if "dup_group" in meta else 0,
           "images_in_duplicate_groups": int(meta["dup_group"].notna().sum()) if "dup_group" in meta else 0}
    print(f"{len(meta)} images, {len(names)} classes: {names} | duplicates: {dup}")

    # Who painted the images? (the classifier must learn styles, not painters)
    named = meta[meta["artist"] != UNKNOWN_ARTIST]
    per_class = {}
    for c in names:
        m = meta[meta["label"] == c]
        top = m[m["artist"] != UNKNOWN_ARTIST]["artist"].value_counts()
        per_class[c] = {"n": len(m), "unknown_artist_share": float((m["artist"] == UNKNOWN_ARTIST).mean()),
                        "named_artists": int(top.size), "top_artists": top.head(5).to_dict()}
    top_art = named["artist"].value_counts().head(15)
    barh(top_art.index.tolist(), top_art.values.tolist(), out / "top_artists.png",
         "Most represented named artists", "images")

    # Split without artist leakage
    split = make_split(meta, "artist", seed=a.split_seed)
    rep = split_report(meta, split)
    leak = sum(len(v) for v in rep["shared_artists"].values())
    print(f"split sizes {rep['n']}, artists shared between splits: {leak}")

    # Image statistics (colour is an important style cue)
    X = load_image_array(meta, a.data_dir, size=a.size)
    mean_rgb = {c: (X[meta["label"].to_numpy() == c].reshape(-1, 3).mean(0) / 255).round(3).tolist() for c in names}

    # Sample grid: 6 images per class
    rng = np.random.default_rng(0)
    imgs, titles = [], []
    for c in names:
        for i in rng.choice(np.flatnonzero(meta["label"].to_numpy() == c), 6, replace=False):
            imgs.append(X[i])
            titles.append(f"{c}\n{str(meta['artist'].iloc[i])[:22]}")
    image_grid(imgs, titles, out / "samples.png", ncols=6, suptitle="Random samples per class")

    save_json({"n_images": len(meta), "classes": names, "per_class": per_class, "mean_rgb": mean_rgb,
               "split": rep, "duplicates": dup, "image_shape": list(X.shape[1:]), "provenance": provenance(a.data_dir)},
              out / "eda.json")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
