"""Figures for the report and the slides (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colour-blind-validated categorical order; text stays in neutral ink, grid is recessive.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "lines.linewidth": 2, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def image_grid(images, titles, path, ncols=6, suptitle=None):
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 2.1 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, img, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=7, color=INK)
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    return _save(fig, path)


def barh(labels, values, path, title, xlabel):
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(labels) + 1))
    ax.barh(labels, values, color=PALETTE[0], height=0.6)
    ax.invert_yaxis()
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:g}", va="center", fontsize=8, color=MUTED)
    return _save(fig, path)
