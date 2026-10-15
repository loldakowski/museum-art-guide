"""Hand-written training loop with monitoring (deliverable 2).

Recorded at every epoch (history dict, saved as JSON and plotted):
  * train / val loss and accuracy            -> under/over-fitting, generalisation gap
  * initial val loss (before any step)       -> must be ~ln(K) for a well initialised classifier
  * gradient L2 norm per layer               -> vanishing / exploding gradients
  * update/weight ratio per layer (|dW|/|W|) -> learning rate too small (<1e-4) or too large (>1e-1)
  * dead / saturated hidden units (MLP)      -> dying ReLUs, saturated sigmoids
  * learning rate, wall time, status (ok / early_stopped / diverged)
Early stopping on the validation loss; the best weights are restored at the end.
"""
from __future__ import annotations

import copy
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 1e-3
    optimizer: str = "adamw"        # sgd | adam | adamw
    momentum: float = 0.9           # sgd only
    weight_decay: float = 1e-4
    scheduler: str = "cosine"       # none | cosine
    patience: int | None = 10       # early stopping on val loss (None = off)
    grad_clip: float | None = None
    label_smoothing: float = 0.0
    track_updates: bool = True
    restore_best: bool = True
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def make_optimizer(params, cfg: TrainConfig):
    params = [p for p in params if p.requires_grad]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def _group(name: str) -> str:
    """Parameter name -> monitored layer: 'hidden.0.weight' -> 'hidden.0', 'net.layer4.1.conv1.weight' -> 'layer4.1'."""
    parts = [p for p in name.split(".")[:-1] if p != "net"]
    return ".".join(parts[:2]) if parts else name


@torch.no_grad()
def evaluate_loss(model, loader, device, label_smoothing=0.0):
    model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        tot_loss += F.cross_entropy(logits, yb, label_smoothing=label_smoothing, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return tot_loss / n, correct / n


def train(model, train_loader, val_loader, cfg: TrainConfig, device="cpu", log=print, log_every=1):
    """Train `model` in place. Returns the history dict."""
    model.to(device)
    opt = make_optimizer(model.parameters(), cfg)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs) if cfg.scheduler == "cosine"
             else None)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = sorted({_group(n) for n, _ in named})
    h = {"config": cfg.to_dict(), "epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
         "lr": [], "time_s": [], "grad_norm": {g: [] for g in groups}, "update_ratio": {g: [] for g in groups},
         "hidden_stats": defaultdict(list), "step_loss": [], "status": "ok"}
    h["initial_val_loss"], h["initial_val_acc"] = evaluate_loss(model, val_loader, device)
    h["expected_initial_loss"] = math.log(_n_classes(model, val_loader, device))
    probe = next(iter(val_loader))[0][:256].to(device)

    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        gsum, usum, nsteps = defaultdict(float), defaultdict(float), 0
        tot_loss, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=cfg.label_smoothing)
            if not torch.isfinite(loss):
                h["status"] = "diverged"
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # --- monitoring: gradient norm per layer (before clipping) ---
            gsq = defaultdict(float)
            for name, p in named:
                if p.grad is not None:
                    gsq[_group(name)] += p.grad.detach().pow(2).sum().item()
            for g, v in gsq.items():
                gsum[g] += math.sqrt(v)
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_([p for _, p in named], cfg.grad_clip)
            before = ({n_: p.detach().clone() for n_, p in named if n_.endswith("weight")}
                      if cfg.track_updates else None)
            opt.step()
            # --- monitoring: size of the update relative to the weights ---
            if before is not None:
                dsq, wsq = defaultdict(float), defaultdict(float)
                for n_, p in named:
                    if n_ in before:
                        dsq[_group(n_)] += (p.detach() - before[n_]).pow(2).sum().item()
                        wsq[_group(n_)] += before[n_].pow(2).sum().item()
                for g in dsq:
                    usum[g] += math.sqrt(dsq[g]) / (math.sqrt(wsq[g]) + 1e-12)
            nsteps += 1
            tot_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
            h["step_loss"].append(loss.item())
        if h["status"] == "diverged":
            log(f"epoch {epoch}: loss is NaN/inf -> diverged")
            break

        val_loss, val_acc = evaluate_loss(model, val_loader, device)
        if not math.isfinite(val_loss):
            h["status"] = "diverged"
            log(f"epoch {epoch}: val loss is NaN/inf -> diverged")
            break
        h["epoch"].append(epoch)
        h["train_loss"].append(tot_loss / n)
        h["train_acc"].append(correct / n)
        h["val_loss"].append(val_loss)
        h["val_acc"].append(val_acc)
        h["lr"].append(opt.param_groups[0]["lr"])
        h["time_s"].append(time.perf_counter() - t_start)
        for g in groups:
            h["grad_norm"][g].append(gsum[g] / max(nsteps, 1))
            h["update_ratio"][g].append(usum[g] / max(nsteps, 1) if cfg.track_updates else float("nan"))
        if hasattr(model, "activation_stats"):
            for layer, st in model.activation_stats(probe).items():
                h["hidden_stats"][layer].append(st)
        if sched:
            sched.step()
        if epoch % log_every == 0 or epoch == 1:
            log(f"epoch {epoch:3d} | train {h['train_loss'][-1]:.3f} / {h['train_acc'][-1]:.3f} | "
                f"val {val_loss:.3f} / {val_acc:.3f} | lr {h['lr'][-1]:.2e}")

        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            if cfg.restore_best:
                best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if cfg.patience is not None and bad >= cfg.patience:
                h["status"] = "early_stopped"
                log(f"early stopping at epoch {epoch} (best epoch {best_epoch}, val loss {best_loss:.3f})")
                break

    if cfg.restore_best and best_state is not None:
        model.load_state_dict(best_state)
    h["best_epoch"], h["best_val_loss"] = best_epoch, best_loss
    h["best_val_acc"] = h["val_acc"][best_epoch - 1] if best_epoch else float("nan")
    h["total_time_s"] = time.perf_counter() - t_start
    h["hidden_stats"] = dict(h["hidden_stats"])
    return h


@torch.no_grad()
def _n_classes(model, loader, device):
    model.eval()
    return model(next(iter(loader))[0][:2].to(device)).shape[1]
