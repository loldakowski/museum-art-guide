"""Metrics with bootstrap confidence intervals (small test set)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


@torch.no_grad()
def predict(model, loader, device="cpu"):
    """Class probabilities and true labels over a DataLoader."""
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        probs.append(torch.softmax(model(xb.to(device)), dim=1).cpu())
        ys.append(yb)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def bootstrap_ci(y, pred, metric="accuracy", n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval of a metric over test images."""
    fn = {"accuracy": accuracy_score, "macro_f1": lambda a, b: f1_score(a, b, average="macro")}[metric]
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    stats = [fn(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def classification_metrics(y, pred, class_names, probs=None, ci=True) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "chance_level": 1.0 / len(class_names),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(class_names)},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "class_names": list(class_names),
    }
    if probs is not None and len(class_names) > 2:
        top2 = np.argsort(-probs, axis=1)[:, :2]
        out["top2_accuracy"] = float(np.mean([yy in t for yy, t in zip(y, top2)]))
    if ci:
        out["accuracy_ci95"] = bootstrap_ci(y, pred, "accuracy")
        out["macro_f1_ci95"] = bootstrap_ci(y, pred, "macro_f1")
    return out


def most_confused(cm, class_names, k=5):
    """The k largest off-diagonal cells: (true, predicted, count, share of the true class)."""
    cm = np.asarray(cm)
    pairs = [(class_names[i], class_names[j], int(cm[i, j]), float(cm[i, j] / max(cm[i].sum(), 1)))
             for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda t: -t[2])[:k]


def fmt(m: dict) -> str:
    lo, hi = m.get("accuracy_ci95", (np.nan, np.nan))
    return (f"acc {m['accuracy']:.3f} [95% CI {lo:.3f}-{hi:.3f}] | macro-F1 {m['macro_f1']:.3f} | "
            f"balanced {m['balanced_accuracy']:.3f} | chance {m['chance_level']:.3f}")
