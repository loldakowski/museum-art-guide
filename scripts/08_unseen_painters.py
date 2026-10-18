#!/usr/bin/env python
"""Step 8 - What does an unseen painter cost? Every classifier, random split vs painter-disjoint split.

The main subset cannot answer this (93 % 'Unknown Artist', 1 named Pop Art painter in all of WikiArt), so it
runs on the named-artist subset (scripts/build_named_dataset.sh): 5 styles, every image with a known painter.
Both split types use 5-fold cross-validation (test = fold k, val = fold k+1, train = the rest): with
--split artist every painter is tested exactly once, never seen in training; with --split random the test
painters also appear in training. Hyper-parameters are NOT re-tuned: they are read from the main results
(steps 1, 2 and 4), so this measures the same models. The accuracy difference is the cost of unseen painters.

    python scripts/08_unseen_painters.py --data-dir data/wikiart_named5 --results results
Outputs: <out>/summary.json, <out>/summary.md (committed); per-run outputs in <out>/runs/ (git-ignored).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from museum_guide.common import load_json, provenance, save_json

SCRIPTS = Path(__file__).resolve().parent
MODELS = ["logreg_pixels32", "logreg_handcrafted", "mlp_pixels32", "mlp_handcrafted",
          "resnet18_frozen_linear_head", "resnet18_frozen_logreg", "resnet18_finetune"]


def selected_hyperparameters(results: Path) -> dict:
    """The configuration each model got in the main experiment (chosen on its validation set)."""
    hp = {f"logreg_{fs}": {"C": load_json(results / "01_baseline" / f"{fs}.json")["best_C"]}
          for fs in ("pixels32", "handcrafted")}
    for fs in ("pixels32", "handcrafted"):
        hp[f"mlp_{fs}"] = load_json(results / "02_mlp" / fs / "summary.json")["selected"]
    hp["resnet18_frozen_logreg"] = {
        "C": load_json(results / "04_resnet" / "frozen" / "summary.json")["logreg_on_embeddings"]["best_C"]}
    return hp


def run(cmd, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        subprocess.run([sys.executable, *map(str, cmd)], check=True, stdout=f, stderr=subprocess.STDOUT)


def one_split(a, hp, split: str, fold: int) -> dict:
    """Train and test every model on one fold; returns {model: test metrics}."""
    d = Path(a.out) / "runs" / f"{split}_fold{fold}"
    common = ["--data-dir", a.data_dir, "--size", a.size, "--split", split, "--fold", fold,
              "--split-seed", a.split_seed, "--checkpoint-dir", Path(a.checkpoint_dir) / f"{split}_fold{fold}"]
    log = d / "log.txt"
    sfx = "" if split == "artist" else "_randomsplit"
    quick_mlp = ["--epochs", "3"] if a.quick else []
    quick_res = ["--no-pretrained", "--epochs-frozen", "3", "--epochs-finetune", "1", "--img-size", a.size] \
        if a.quick else []
    out = {}
    for fs in ("pixels32", "handcrafted"):
        run([SCRIPTS / "01_baseline.py", *common, "--out", d / "01_baseline", "--features", fs,
             "--C", hp[f"logreg_{fs}"]["C"]], log)
        out[f"logreg_{fs}"] = load_json(d / "01_baseline" / f"{fs}{sfx}.json")
        m = hp[f"mlp_{fs}"]
        run([SCRIPTS / "02_train_mlp.py", *common, "--out", d / "02_mlp", "--features", fs, "--no-sweep",
             "--hidden", *m["hidden"], "--dropout", m["dropout"], "--wd", m["weight_decay"], "--seeds", 0,
             "--device", a.device, *quick_mlp], log)
        out[f"mlp_{fs}"] = load_json(d / "02_mlp" / f"{fs}{sfx}" / "summary.json")["metrics_seed0"]
    run([SCRIPTS / "04_train_resnet.py", *common, "--out", d / "04_resnet", "--mode", "both",
         "--logreg-C", hp["resnet18_frozen_logreg"]["C"], "--device", a.device, *quick_res], log)
    frozen = load_json(d / "04_resnet" / f"frozen{sfx}" / "summary.json")
    out["resnet18_frozen_linear_head"] = frozen["metrics"]
    out["resnet18_frozen_logreg"] = frozen["logreg_on_embeddings"]["metrics"]
    out["resnet18_finetune"] = load_json(d / "04_resnet" / f"finetune{sfx}" / "summary.json")["metrics"]
    return out


def aggregate(folds: list[dict]) -> dict:
    """Per model: accuracy of each fold, mean +/- std over folds, and pooled accuracy over all test images."""
    res = {}
    for model in MODELS:
        acc = np.array([f[model]["accuracy"] for f in folds])
        n = np.array([f[model]["n"] for f in folds])
        correct = np.array([np.trace(np.asarray(f[model]["confusion_matrix"])) for f in folds])
        res[model] = {"fold_accuracy": acc.tolist(), "fold_n": n.tolist(), "mean": float(acc.mean()),
                      "std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                      "pooled_accuracy": float(correct.sum() / n.sum()), "pooled_n": int(n.sum()),
                      "fold_macro_f1": [f[model]["macro_f1"] for f in folds]}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_named5")
    p.add_argument("--results", default="results", help="main results: source of the selected hyper-parameters")
    p.add_argument("--out", default="results/08_unseen_painters")
    p.add_argument("--checkpoint-dir", default="checkpoints/unseen_painters")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--folds", type=int, default=5, help="only 5 is supported by make_split's default")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true", help="smoke test: few epochs, random ResNet weights")
    a = p.parse_args()
    hp = selected_hyperparameters(Path(a.results))
    print(f"hyper-parameters from {a.results}: {hp}")

    per_split = {}
    for split in ("random", "artist"):
        folds = []
        for k in range(a.folds):
            folds.append(one_split(a, hp, split, k))
            print(f"{split:6s} fold {k}: " + "  ".join(f"{m} {folds[-1][m]['accuracy']:.3f}" for m in MODELS))
        per_split[split] = {"aggregate": aggregate(folds),
                            "folds": [{m: {k: f[m][k] for k in ("accuracy", "n", "macro_f1", "confusion_matrix")}
                                       for m in MODELS} for f in folds]}

    R, A = per_split["random"]["aggregate"], per_split["artist"]["aggregate"]
    cost = {m: {"pooled_points": 100 * (R[m]["pooled_accuracy"] - A[m]["pooled_accuracy"]),
                "mean_points": 100 * (R[m]["mean"] - A[m]["mean"])} for m in MODELS}
    out = Path(a.out)
    save_json({"data_dir": a.data_dir, "folds": a.folds, "split_seed": a.split_seed, "quick": a.quick,
               "hyperparameters": hp, "random": per_split["random"], "artist": per_split["artist"],
               "cost_of_unseen_painters": cost, "provenance": provenance(a.data_dir)}, out / "summary.json")

    lines = ["| model | random split: pooled acc | folds mean ± std | painter-disjoint: pooled acc | "
             "folds mean ± std | cost of unseen painters (points, pooled) |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        lines.append(f"| {m} | {R[m]['pooled_accuracy']:.3f} | {R[m]['mean']:.3f} ± {R[m]['std']:.3f} | "
                     f"{A[m]['pooled_accuracy']:.3f} | {A[m]['mean']:.3f} ± {A[m]['std']:.3f} | "
                     f"{cost[m]['pooled_points']:+.1f} |")
    per_fold = ["| model | split | " + " | ".join(f"fold {k}" for k in range(a.folds)) + " |",
                "|---|---|" + "---|" * a.folds]
    for m in MODELS:
        for split, agg in (("random", R), ("artist", A)):
            per_fold.append(f"| {m} | {split} | " + " | ".join(f"{x:.3f}" for x in agg[m]["fold_accuracy"]) + " |")
    (out / "summary.md").write_text(
        f"# Cost of unseen painters ({a.data_dir}, {a.folds}-fold cross-validation)\n\n"
        + ("**QUICK smoke run: numbers are meaningless.**\n\n" if a.quick else "")
        + "Same models and hyper-parameters as the main experiment. Random split: test painters also appear in "
          "training. Painter-disjoint split: every painter is tested once, never seen in training. Pooled = all "
          f"test images of the {a.folds} folds (each image tested once).\n\n" + "\n".join(lines)
        + "\n\n## Accuracy per fold\n\n" + "\n".join(per_fold) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}/summary.md")


if __name__ == "__main__":
    main()
