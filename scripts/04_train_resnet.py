#!/usr/bin/env python
"""Step 4 - Pretrained ResNet-18 (deliverable 3).

  --mode frozen   : ImageNet embeddings computed once, then (a) our linear head trained with the same
                    monitored loop as the MLP and (b) a logistic regression on the same embeddings
                    ("classical model on deep features" - a serious production candidate).
  --mode finetune : layer4 + head trained end-to-end with geometric augmentation (GPU recommended).
  --mode both     : both of the above.

    python scripts/04_train_resnet.py --data-dir data/wikiart_style5 --mode both
Outputs: results/04_resnet/<mode>/{summary.json, training_history.png, confusion.png, errors.png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from museum_guide.baseline import fit_logreg
from museum_guide.common import get_device, save_json, set_seed
from museum_guide.data import class_names, load_image_array, load_metadata, make_image_loaders, make_split
from museum_guide.evaluation import classification_metrics, fmt, most_confused, predict
from museum_guide.plots import confusion, image_grid, training_history
from museum_guide.resnet import LinearHead, ResNetClassifier, extract_embeddings
from museum_guide.training import TrainConfig, train


def error_grid(meta, split, data_dir, probs, names, path, size, k=12):
    """The most confident mistakes on the test set: what does the network get wrong, and why?"""
    test_meta = meta[split == "test"].reset_index(drop=True)
    y, pred = test_meta["label_idx"].to_numpy(), probs.argmax(1)
    wrong = np.flatnonzero(pred != y)
    wrong = wrong[np.argsort(-probs[wrong, pred[wrong]])][:k]
    if len(wrong) == 0:
        return
    X = load_image_array(test_meta.iloc[wrong].reset_index(drop=True), data_dir, size, cache=False)
    titles = [f"true {names[y[i]]}\npred {names[pred[i]]} ({probs[i, pred[i]]:.0%})\n"
              f"{str(test_meta['artist'].iloc[i])[:20]}" for i in wrong]
    image_grid(list(X), titles, path, ncols=6, suptitle="Most confident test errors")


def run_frozen(a, meta, split, names, device, out):
    set_seed(a.seed)
    model = ResNetClassifier(len(names), "frozen", pretrained=not a.no_pretrained)
    L = make_image_loaders(meta, split, a.data_dir, a.img_size, a.batch_size, augment=False, num_workers=a.workers)
    E = {s: extract_embeddings(model, L[s], device) for s in ("train", "val", "test")}
    print(f"embeddings: {', '.join(f'{s} {tuple(e[0].shape)}' for s, e in E.items())}")

    # (a) linear head with the monitored training loop
    head = LinearHead(model.embed_dim, len(names))
    TL = {s: DataLoader(TensorDataset(*E[s]), batch_size=64 if s == "train" else 512, shuffle=(s == "train"))
          for s in E}
    cfg = TrainConfig(epochs=a.epochs_frozen, lr=1e-3, weight_decay=1e-2, patience=15, seed=a.seed)
    h = train(head, TL["train"], TL["val"], cfg, device, log_every=10)
    probs, y = predict(head, TL["test"], device)
    m = classification_metrics(y, probs.argmax(1), names, probs)
    print(f"[frozen + linear head]   test {fmt(m)}")
    model.net.fc.load_state_dict(head.fc.state_dict())  # assemble the deployable model

    # (b) logistic regression on the same embeddings
    lr_model, val_scores = fit_logreg(E["train"][0].numpy(), E["train"][1].numpy(),
                                      E["val"][0].numpy(), E["val"][1].numpy())
    p2 = lr_model.predict_proba(E["test"][0].numpy())
    m2 = classification_metrics(E["test"][1].numpy(), p2.argmax(1), names, p2)
    print(f"[frozen + logistic reg.] test {fmt(m2)}")

    training_history(h, out / "training_history.png", "ResNet-18 frozen backbone + linear head")
    confusion(m["confusion_matrix"], names, out / "confusion.png", "ResNet-18 frozen + linear head - test")
    error_grid(meta, split, a.data_dir, probs, names, out / "errors.png", a.size)
    save_json({"model": "resnet18_frozen", "metrics": m, "history_summary": _hsum(h),
               "most_confused": most_confused(m["confusion_matrix"], names),
               "logreg_on_embeddings": {"metrics": m2, "val_acc_per_C": val_scores,
                                        "best_C": float(lr_model[-1].C)},
               "n_parameters": model.n_parameters(), "trainable_parameters": head.fc.weight.numel() + len(names),
               "img_size": a.img_size, "pretrained": not a.no_pretrained}, out / "summary.json")
    save_json(h, out / "history.json")
    return model


def run_finetune(a, meta, split, names, device, out):
    set_seed(a.seed)
    model = ResNetClassifier(len(names), "finetune", pretrained=not a.no_pretrained)
    L = make_image_loaders(meta, split, a.data_dir, a.img_size, a.batch_size, augment=True, num_workers=a.workers)
    print(f"trainable parameters: {model.n_parameters(True):,} / {model.n_parameters():,}")
    cfg = TrainConfig(epochs=a.epochs_finetune, lr=1e-3, weight_decay=1e-2, patience=6, seed=a.seed,
                      label_smoothing=0.1)
    h = train(model, L["train"], L["val"], cfg, device)
    probs, y = predict(model, L["test"], device)
    m = classification_metrics(y, probs.argmax(1), names, probs)
    print(f"[fine-tuned layer4]      test {fmt(m)}")
    training_history(h, out / "training_history.png", "ResNet-18 fine-tuning (layer4 + head)")
    confusion(m["confusion_matrix"], names, out / "confusion.png", "ResNet-18 fine-tuned - test")
    error_grid(meta, split, a.data_dir, probs, names, out / "errors.png", a.size)
    save_json({"model": "resnet18_finetune", "metrics": m, "history_summary": _hsum(h),
               "most_confused": most_confused(m["confusion_matrix"], names),
               "n_parameters": model.n_parameters(), "trainable_parameters": model.n_parameters(True),
               "img_size": a.img_size, "pretrained": not a.no_pretrained}, out / "summary.json")
    save_json(h, out / "history.json")
    return model


def _hsum(h):
    return {k: h[k] for k in ("initial_val_loss", "expected_initial_loss", "best_epoch", "best_val_loss",
                              "best_val_acc", "total_time_s", "status")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_style5")
    p.add_argument("--out", default="results/04_resnet")
    p.add_argument("--mode", choices=["frozen", "finetune", "both"], default="both")
    p.add_argument("--split", choices=["artist", "random"], default="artist")
    p.add_argument("--size", type=int, default=128, help="image size on disk")
    p.add_argument("--img-size", type=int, default=128, help="network input size (224 = ImageNet native)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-frozen", type=int, default=100)
    p.add_argument("--epochs-finetune", type=int, default=20)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true", help="random weights (smoke test / ablation only)")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    device = get_device(a.device)
    meta = load_metadata(a.data_dir)
    names = class_names(meta)
    split = make_split(meta, a.split, seed=0)
    suffix = "" if a.split == "artist" else "_randomsplit"
    print(f"device {device} | {len(meta)} images | classes {names}")
    Path("checkpoints").mkdir(exist_ok=True)
    for mode in (["frozen", "finetune"] if a.mode == "both" else [a.mode]):
        out = Path(a.out) / f"{mode}{suffix}"
        print(f"\n=== ResNet-18 {mode} ===")
        model = (run_frozen if mode == "frozen" else run_finetune)(a, meta, split, names, device, out)
        torch.save({"state_dict": model.state_dict(), "mode": mode, "classes": names, "img_size": a.img_size},
                   f"checkpoints/resnet18_{mode}{suffix}.pt")


if __name__ == "__main__":
    main()
