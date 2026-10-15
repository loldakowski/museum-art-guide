#!/usr/bin/env python
"""Step 3 - Failure diagnosis (deliverable 2): break the MLP on purpose and show that the monitoring
detects each failure, with a measured symptom.

Every scenario changes ONE thing with respect to the reference configuration (selected in step 2):

  scenario            change                                  expected symptom (what we measure)
  ------------------  --------------------------------------  -------------------------------------------
  reference           -                                       healthy curves, update ratio ~1e-3
  overfit_one_batch   32 training images, no regularisation   train acc -> 100 %: the code CAN learn
  lr_too_high         lr = 0.1                                loss explodes / stuck, dead ReLUs
  lr_too_low          lr = 1e-6                               loss flat at ln(K), update ratio << 1e-4
  init_zeros          all weights = 0                         symmetry: loss stuck at ln(K), grads = 0
  init_large          weights ~ N(0, 1)                       initial loss >> ln(K), exploding activations
  deep_sigmoid        6 sigmoid layers                        vanishing gradients in the first layers
  no_normalisation    no input standardisation                slower / worse convergence
  no_regularisation   dropout 0, wd 0, no early stopping      train acc ~100 %, val loss climbs (overfit)
  shuffled_labels     training labels permuted                train acc rises, val acc = chance (memorise)
  random_split        random split instead of artist split    test acc inflated (artist leakage)

Plus a learning-rate range test. Outputs: results/03_diagnosis/{summary.md, summary.json, *.png}
    python scripts/03_diagnose_mlp.py --data-dir data/wikiart_style5
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from museum_guide.common import get_device, load_json, save_json, set_seed
from museum_guide.experiments import FeatureData, load_feature_data, run_mlp, tensor_loaders
from museum_guide.mlp import MLP, standardization_stats
from museum_guide.plots import MUTED, PALETTE, _save, plt, training_history
from museum_guide.training import TrainConfig, train

EXPECTED = {
    "reference": "healthy: val loss decreases, then early stopping",
    "overfit_one_batch": "train acc reaches ~100 % on 32 images (the code can learn)",
    "lr_too_high": "loss explodes or stays at chance, many dead ReLUs",
    "lr_too_low": "loss stays near ln(K), update/weight ratio << 1e-4",
    "init_zeros": "symmetry: loss stuck at ln(K), zero gradients in hidden layers",
    "init_large": "initial loss >> ln(K), unstable training",
    "deep_sigmoid": "vanishing gradients: first-layer grad norm << last-layer grad norm",
    "no_normalisation": "slower or worse convergence than the reference",
    "no_regularisation": "train acc ~100 %, val loss rises after its minimum (overfitting)",
    "shuffled_labels": "train acc rises, val/test acc ~ chance (memorisation, no leakage)",
    "random_split": "test acc higher than with the artist split (leakage through painters)",
}


def subset_train(d: FeatureData, n: int, seed: int = 0) -> FeatureData:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(d.y["train"]), generator=g)[:n]
    X, y = dict(d.X), dict(d.y)
    X["train"], y["train"] = d.X["train"][idx], d.y["train"][idx]
    X["val"], y["val"] = X["train"], y["train"]  # we only care about fitting these 32 images
    return FeatureData(X, y, d.classes, d.features, d.split_mode)


def first_last_grad_ratio(h, n_epochs=5):
    """Mean grad norm of the first hidden layer / output layer over the first epochs."""
    g = h["grad_norm"]
    keys = [k for k in g if k.startswith("hidden")]
    if not keys or "out" not in g or not g["out"]:
        return float("nan")
    first = np.mean(g[keys[0]][:n_epochs])
    last = np.mean(g["out"][:n_epochs])
    return float(first / last) if last > 0 else float("nan")


def summarise(name, h, m):
    ep = len(h["epoch"])
    hid = h.get("hidden_stats") or {}
    dead_end = float(np.mean([v[-1]["dead_or_saturated"] for v in hid.values() if v])) if hid and ep else float("nan")
    ur = [np.nanmean(v) for v in h["update_ratio"].values() if len(v)]
    min_val_ep = int(np.argmin(h["val_loss"])) if ep else 0
    return {
        "scenario": name, "expected": EXPECTED[name], "status": h["status"], "epochs_run": ep,
        "initial_loss": h["initial_val_loss"], "ln_K": h["expected_initial_loss"],
        "final_train_loss": h["train_loss"][-1] if ep else float("nan"),
        "final_train_acc": h["train_acc"][-1] if ep else float("nan"),
        "best_val_acc": h.get("best_val_acc", float("nan")),
        "val_loss_rise": (h["val_loss"][-1] - h["val_loss"][min_val_ep]) if ep else float("nan"),
        "test_acc": m["accuracy"] if m else float("nan"),
        "grad_first_over_out": first_last_grad_ratio(h),
        "mean_update_ratio": float(np.mean(ur)) if ur else float("nan"),
        "dead_or_saturated_end": dead_end,
    }


def lr_range_test(d, mlp_kwargs, lrs, epochs, device, batch_size):
    out = []
    for lr in lrs:
        set_seed(0)
        model = MLP(d.dim, len(d.classes), **mlp_kwargs).set_normalization(*standardization_stats(d.X["train"]))
        L = tensor_loaders(d, batch_size)
        h = train(model, L["train"], L["val"], TrainConfig(epochs=epochs, lr=lr, scheduler="none", patience=None,
                                                          track_updates=False), device, log=lambda *_: None)
        out.append({"lr": lr, "train_loss": h["train_loss"][-1] if h["epoch"] else float("nan"),
                    "val_loss": h["val_loss"][-1] if h["epoch"] else float("nan"), "status": h["status"]})
        print(f"  lr {lr:.0e}: train loss {out[-1]['train_loss']:.3f} ({h['status']})")
    return out


def overview_figure(H, rows, lr_test, path, n_classes):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    ax = axes[0, 0]
    for i, k in enumerate(["lr_too_low", "reference", "lr_too_high"]):
        if H[k]["epoch"]:
            ax.plot(H[k]["epoch"], H[k]["train_loss"], label=k, color=PALETTE[i])
    ax.axhline(H["reference"]["expected_initial_loss"], color=MUTED, ls=":", lw=1, label="ln(K)")
    ax.set(title="(a) Learning rate: train loss", xlabel="epoch", yscale="log")
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    for i, k in enumerate(["init_zeros", "reference", "init_large"]):
        if H[k]["epoch"]:
            ax.plot([0] + H[k]["epoch"], [H[k]["initial_val_loss"]] + H[k]["val_loss"], label=k, color=PALETTE[i])
    ax.axhline(H["reference"]["expected_initial_loss"], color=MUTED, ls=":", lw=1, label="ln(K)")
    ax.set(title="(b) Initialisation: val loss (epoch 0 = init)", xlabel="epoch", yscale="log")
    ax.legend(fontsize=7)
    ax = axes[0, 2]
    for i, k in enumerate(["reference", "no_regularisation"]):
        ax.plot(H[k]["epoch"], H[k]["train_loss"], color=PALETTE[i], ls="--", lw=1.5, label=f"{k} train")
        ax.plot(H[k]["epoch"], H[k]["val_loss"], color=PALETTE[i], label=f"{k} val")
    ax.set(title="(c) Overfitting: train vs val loss", xlabel="epoch")
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    g = H["deep_sigmoid"]["grad_norm"]
    names = list(g)
    vals = [np.mean(g[n][:5]) if g[n] else np.nan for n in names]
    gr = H["reference"]["grad_norm"]
    ax.bar(range(len(names)), vals, color=PALETTE[1], width=0.6, label="deep sigmoid")
    ax.set_xticks(range(len(names)), names, rotation=30, fontsize=7)
    ax.set(title="(d) Vanishing gradients (deep sigmoid)\nmean grad norm, first 5 epochs", yscale="log")
    ax.grid(axis="x", visible=False)
    ax.text(0.02, 0.95, f"reference first/out ratio: {first_last_grad_ratio(H['reference']):.2f}\n"
            f"deep sigmoid first/out ratio: {first_last_grad_ratio(H['deep_sigmoid']):.1e}",
            transform=ax.transAxes, va="top", fontsize=7, color=MUTED)
    ax = axes[1, 1]
    k = "shuffled_labels"
    ax.plot(H[k]["epoch"], H[k]["train_acc"], label="train (shuffled labels)", color=PALETTE[0])
    ax.plot(H[k]["epoch"], H[k]["val_acc"], label="val (true labels)", color=PALETTE[1])
    ax.axhline(1 / n_classes, color=MUTED, ls=":", lw=1, label="chance")
    ax.set(title="(e) Shuffled labels: memorisation", xlabel="epoch", ylim=(0, 1.02))
    ax.legend(fontsize=7)
    ax = axes[1, 2]
    r = {x["scenario"]: x for x in rows}
    bars = [("artist split\n(honest)", r["reference"]["test_acc"]), ("random split\n(leaky)", r["random_split"]["test_acc"])]
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[PALETTE[0], PALETTE[7]], width=0.5)
    for i, (_, v) in enumerate(bars):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set(title="(f) Artist leakage: same MLP, test accuracy", ylim=(0, 1.12))
    ax.grid(axis="x", visible=False)
    fig.suptitle("MLP failure diagnosis - each panel changes one thing", fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, path)

    if lr_test:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        lrs = [x["lr"] for x in lr_test]
        ax.plot(lrs, [min(x["train_loss"], 10) if math.isfinite(x["train_loss"]) else 10 for x in lr_test],
                marker="o", color=PALETTE[0], label="train loss after a few epochs")
        ax.axhline(H["reference"]["expected_initial_loss"], color=MUTED, ls=":", lw=1, label="ln(K)")
        ax.set(xscale="log", yscale="log", xlabel="learning rate (AdamW)", title="Learning-rate range test")
        ax.legend(fontsize=7)
        _save(fig, Path(path).with_name("lr_range_test.png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_style5")
    p.add_argument("--out", default="results/03_diagnosis")
    p.add_argument("--features", default="pixels32")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--reference", default="results/02_mlp/pixels32/summary.json",
                   help="summary of step 2 (selected config); defaults are used if it does not exist")
    p.add_argument("--only", nargs="*", help="run only these scenarios")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    device, out = get_device(a.device), Path(a.out)

    ref = {"hidden": (512, 256), "dropout": 0.5, "weight_decay": 1e-2}
    if Path(a.reference).exists():
        ref.update(load_json(a.reference)["selected"])
        ref["hidden"] = tuple(ref["hidden"])
    print(f"reference configuration: {ref}")
    d = load_feature_data(a.data_dir, a.features, "artist", size=a.size)
    E = a.epochs
    base_cfg = dict(epochs=E, lr=1e-3, weight_decay=ref["weight_decay"])
    mk = {"hidden": ref["hidden"], "dropout": ref["dropout"]}

    S = {  # name: (data, mlp kwargs, train config, run options)
        "reference": (d, mk, TrainConfig(**base_cfg), {}),
        "overfit_one_batch": (subset_train(d, 32), {**mk, "dropout": 0.0},
                              TrainConfig(epochs=200, lr=1e-3, weight_decay=0.0, patience=None, scheduler="none"), {}),
        "lr_too_high": (d, mk, TrainConfig(**{**base_cfg, "lr": 1e-1}), {}),
        "lr_too_low": (d, mk, TrainConfig(**{**base_cfg, "lr": 1e-6}), {}),
        "init_zeros": (d, {**mk, "init": "zeros"}, TrainConfig(**base_cfg), {}),
        "init_large": (d, {**mk, "init": "large"}, TrainConfig(**base_cfg), {}),
        "deep_sigmoid": (d, {"hidden": (256,) * 6, "dropout": 0.0, "activation": "sigmoid", "init": "xavier"},
                         TrainConfig(**{**base_cfg, "optimizer": "sgd", "lr": 0.05, "scheduler": "none"}), {}),
        "no_normalisation": (d, mk, TrainConfig(**base_cfg), {"normalize": False}),
        "no_regularisation": (d, {**mk, "dropout": 0.0},
                              TrainConfig(**{**base_cfg, "weight_decay": 0.0, "patience": None, "restore_best": False,
                                             "scheduler": "none"}), {}),
        "shuffled_labels": (d, {**mk, "dropout": 0.0},
                            TrainConfig(**{**base_cfg, "weight_decay": 0.0, "patience": None, "restore_best": False,
                                           "epochs": 2 * E}), {"shuffle_labels": True}),
        "random_split": (load_feature_data(a.data_dir, a.features, "random", size=a.size), mk,
                         TrainConfig(**base_cfg), {}),
    }
    H, rows = {}, []
    for name, (dd, kw, cfg, opts) in S.items():
        if a.only and name not in a.only and name != "reference":
            continue
        print(f"\n=== {name} ===")
        _, h, m = run_mlp(dd, kw, cfg, device, a.batch_size, log=lambda *_: None, **opts)
        H[name] = h
        rows.append(summarise(name, h, m))
        r = rows[-1]
        print(f"status {r['status']} | init loss {r['initial_loss']:.2f} (lnK {r['ln_K']:.2f}) | final train "
              f"loss {r['final_train_loss']:.3f} acc {r['final_train_acc']:.3f} | best val {r['best_val_acc']:.3f} "
              f"| test {r['test_acc']:.3f} | grad first/out {r['grad_first_over_out']:.2e} | "
              f"update ratio {r['mean_update_ratio']:.1e} | dead {r['dead_or_saturated_end']:.2f}")
        save_json(h, out / "histories" / f"{name}.json")
        training_history(h, out / "histories" / f"{name}.png", f"Diagnosis - {name}")

    print("\n=== learning-rate range test ===")
    lr_test = lr_range_test(d, mk, [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], 5, device, a.batch_size)

    if not a.only:
        overview_figure(H, rows, lr_test, out / "diagnosis_overview.png", len(d.classes))
    save_json({"reference_config": ref, "scenarios": rows, "lr_range_test": lr_test}, out / "summary.json")
    cols = ["scenario", "status", "initial_loss", "final_train_acc", "best_val_acc", "test_acc",
            "grad_first_over_out", "mean_update_ratio", "dead_or_saturated_end"]
    lines = ["| " + " | ".join(cols + ["expected symptom"]) + " |", "|" + "---|" * (len(cols) + 1)]
    for r in rows:
        cells = [r[c] if isinstance(r[c], str) else (f"{r[c]:.2e}" if c in ("grad_first_over_out", "mean_update_ratio")
                                                      else f"{r[c]:.3f}") for c in cols]
        lines.append("| " + " | ".join(cells + [r["expected"]]) + " |")
    (out / "summary.md").write_text("# MLP failure diagnosis\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n-> {out}/summary.md")


if __name__ == "__main__":
    main()
