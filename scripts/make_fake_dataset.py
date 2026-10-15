#!/usr/bin/env python
"""Synthetic stand-in for the WikiArt subset, same file layout (ImageFolder + metadata.csv).

Used by the tests and for a smoke run of the whole pipeline without downloading 4 GB.
Each class has a characteristic colour and texture, each artist a small personal colour shift, so
that grouped vs random splits behave differently. Numbers obtained on it mean NOTHING about art.

    python scripts/make_fake_dataset.py --out data/fake --per-class 60
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["Baroque", "Cubism", "Impressionism", "Pop_Art", "Ukiyo_e"]


def make_image(rng, cls: int, artist_shift: np.ndarray, size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = 0.5 + 0.05 * np.array([[-1, -1, -1], [0, 0, 0], [0, 0.5, 1], [1, -1, 0.5], [0.8, 0.6, 0]])[cls]
    freq = [2, 6, 12, 3, 4][cls]
    pattern = np.sin(2 * np.pi * freq * (xx + yy * (cls % 2))) * 0.10
    img = base[None, None, :] + pattern[..., None] + artist_shift + rng.normal(0, 0.35, (size, size, 3))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/fake"))
    p.add_argument("--per-class", type=int, default=60)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--artists-per-class", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for c, name in enumerate(CLASSES):
        (a.out / name).mkdir(parents=True, exist_ok=True)
        shifts = rng.normal(0, 0.12, (a.artists_per_class, 3))
        for i in range(a.per_class):
            k = i % (a.artists_per_class + 1)  # the extra slot plays "Unknown Artist"
            artist = "Unknown Artist" if k == a.artists_per_class else f"{name} painter {k}"
            shift = np.zeros(3) if k == a.artists_per_class else shifts[k]
            rid = f"s{c:02d}_r{i:04d}"
            Image.fromarray(make_image(rng, c, shift, a.size)).save(a.out / name / f"{rid}.jpg", quality=90)
            rows.append({"path": f"{name}/{rid}.jpg", "label": name, "label_idx": c, "id": rid, "style": name,
                         "genre": "unknown", "artist": artist, "shard": c, "row": i,
                         "orig_width": a.size, "orig_height": a.size})
    with open(a.out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out / "subset_info.json").write_text(json.dumps({"synthetic": True, "classes": CLASSES,
                                                        "total": len(rows)}, indent=2))
    print(f"{len(rows)} synthetic images in {a.out}")


if __name__ == "__main__":
    main()
