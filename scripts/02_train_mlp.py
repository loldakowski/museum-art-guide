#!/usr/bin/env python
"""Step 2 - Train the from-scratch MLP (deliverable 1) with full monitoring (deliverable 2).

1. Small hyper-parameter sweep (architecture x dropout x weight decay), selected on VALIDATION accuracy.
2. The selected configuration is retrained with several seeds -> test accuracy mean +/- std.
3. Saves the monitoring history, figures, metrics and the checkpoint of the first seed.

    python scripts/02_train_mlp.py --data-dir data/wikiart_style5 --features pixels32
    python scripts/02_train_mlp.py --features handcrafted       # same MLP on hand-crafted features
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch

from museum_guide.common import get_device, provenance, save_json
from museum_guide.evaluation import fmt, most_confused
from museum_guide.experiments import load_feature_data, mean_std, run_mlp
from museum_guide.features import FEATURE_SETS
from museum_guide.plots import confusion, training_history
from museum_guide.training import TrainConfig

SWEEP = {
    "hidden": [(256,), (512, 256), (1024, 512, 256)],
    "dropout": [0.2, 0.5],
    "weight_decay": [1e-4, 1e-2],
}


def generalisation_gap(h):
    """Train acc - val acc at the best epoch (epochs count from 1; 0 = no epoch improved, e.g. divergence)."""
    e = h["best_epoch"]
    return h["train_acc"][e - 1] - h["val_acc"][e - 1] if e else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_style5")
    p.add_argument("--out", default="results/02_mlp")
    p.add_argument("--features", default="pixels32", choices=list(FEATURE_SETS))
    p.add_argument("--split", choices=["artist", "random"], default="artist")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--fold", type=int, default=None, help="cross-validation fold (0-4) instead of the 70/15/15 split")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--no-sweep", action="store_true", help="skip the sweep, use --hidden/--dropout/--wd")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    device = get_device(a.device)
    tag = a.features if a.split == "artist" else f"{a.features}_randomsplit"
    out = Path(a.out) / tag

    d = load_feature_data(a.data_dir, a.features, a.split, split_seed=a.split_seed, size=a.size,
                          fold=a.fold)
    print(f"features={a.features} dim={d.dim} | train {len(d.y['train'])} val {len(d.y['val'])} "
          f"test {len(d.y['test'])} | device {device}")

    # ---- 1. sweep on validation -------------------------------------------------------------------
    sweep_rows = []
    if a.no_sweep:
        best = {"hidden": tuple(a.hidden), "dropout": a.dropout, "weight_decay": a.wd}
    else:
        for hidden, dropout, wd in itertools.product(*SWEEP.values()):
            cfg = TrainConfig(epochs=a.epochs, lr=a.lr, weight_decay=wd, seed=0, track_updates=False)
            _, h, _ = run_mlp(d, {"hidden": hidden, "dropout": dropout}, cfg, device, a.batch_size,
                              log=lambda *_: None)
            row = {"hidden": hidden, "dropout": dropout, "weight_decay": wd, "val_acc": h["best_val_acc"],
                   "val_loss": h["best_val_loss"], "best_epoch": h["best_epoch"],
                   "train_acc_at_end": h["train_acc"][-1], "params": h["n_parameters"]}
            sweep_rows.append(row)
            print(f"  sweep {str(hidden):18s} dropout={dropout} wd={wd:g} -> val acc {row['val_acc']:.3f} "
                  f"(epoch {row['best_epoch']})")
        top = max(sweep_rows, key=lambda r: (r["val_acc"], -r["val_loss"]))
        best = {k: top[k] for k in ("hidden", "dropout", "weight_decay")}
    print(f"selected on validation: {best}")

    # ---- 2. retrain with several seeds, evaluate on test ---------------------------------------------
    runs = []
    for seed in a.seeds:
        cfg = TrainConfig(epochs=a.epochs, lr=a.lr, weight_decay=best["weight_decay"], seed=seed)
        model, h, m = run_mlp(d, {"hidden": tuple(best["hidden"]), "dropout": best["dropout"]}, cfg, device,
                              a.batch_size)
        print(f"seed {seed}: best epoch {h['best_epoch']} | test {fmt(m)}")
        runs.append({"seed": seed, "history": h, "metrics": m})
        if seed == a.seeds[0]:
            training_history(h, out / "training_history.png",
                             f"MLP on {a.features} - hidden {best['hidden']}, dropout {best['dropout']}")
            confusion(m["confusion_matrix"], d.classes, out / "confusion.png", f"MLP on {a.features} - test")
            Path(a.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "in_features": d.dim, "n_classes": len(d.classes),
                        "mlp_kwargs": {"hidden": tuple(best["hidden"]), "dropout": best["dropout"]},
                        "features": a.features, "classes": d.classes, "provenance": provenance(a.data_dir)},
                       Path(a.checkpoint_dir) / f"mlp_{tag}.pt")

    m0 = runs[0]["metrics"]
    summary = {
        "model": f"mlp_{a.features}", "features": a.features, "split": a.split, "selected": best,
        # the exact training setting of the seed runs (step 3 reuses it for its reference scenario)
        "train_config": {**{k: v for k, v in cfg.to_dict().items() if k != "seed"}, "batch_size": a.batch_size,
                         "seed": a.seeds[0]},
        "n_parameters": runs[0]["history"]["n_parameters"], "sweep": sweep_rows,
        "test_accuracy": mean_std([r["metrics"]["accuracy"] for r in runs]),
        "test_macro_f1": mean_std([r["metrics"]["macro_f1"] for r in runs]),
        "best_epoch": mean_std([r["history"]["best_epoch"] for r in runs]),
        "train_time_s": mean_std([r["history"]["total_time_s"] for r in runs]),
        "initial_val_loss": runs[0]["history"]["initial_val_loss"],
        "expected_initial_loss": runs[0]["history"]["expected_initial_loss"],
        "generalisation_gap_at_best": generalisation_gap(runs[0]["history"]),
        "metrics_seed0": m0, "most_confused_seed0": most_confused(m0["confusion_matrix"], d.classes),
        "device": str(device), "provenance": provenance(a.data_dir),
    }
    save_json(summary, out / "summary.json")
    save_json(runs[0]["history"], out / "history_seed0.json")
    t = summary["test_accuracy"]
    print(f"\nMLP {a.features}: test accuracy {t['mean']:.3f} +/- {t['std']:.3f} over seeds {a.seeds} -> {out}")


if __name__ == "__main__":
    main()
