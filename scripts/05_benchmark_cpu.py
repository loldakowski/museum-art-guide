#!/usr/bin/env python
"""Step 5 - What does each model cost on a kiosk CPU? End-to-end latency for ONE visitor photo.

Measured pipeline per candidate: decode the JPEG -> preprocess -> (features) -> predict. Feature
extraction is part of the cost of the classical models, so it is included. Latency does not depend
on the values of the weights, so untrained models of the same architecture are used when a checkpoint
is missing (reported in the output). Run it ON CPU, ideally on hardware close to the kiosks.

    python scripts/05_benchmark_cpu.py --data-dir data/wikiart_style5 --threads 1 2 4
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps

from museum_guide.baseline import make_logreg
from museum_guide.common import check_checkpoint, provenance, save_json
from museum_guide.data import image_transform
from museum_guide.features import handcrafted, pixels
from museum_guide.mlp import MLP
from museum_guide.resnet import ResNetClassifier


def load_rgb(path_or_bytes, size):
    with Image.open(path_or_bytes) as im:
        return np.asarray(ImageOps.fit(im.convert("RGB"), (size, size)), dtype=np.uint8)


def time_fn(fn, n_warmup=10, n_runs=100):
    for _ in range(n_warmup):
        fn()
    t = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(t)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.quantile(t, 0.95)), "mean_ms": float(t.mean())}


def nbytes(obj, kind):
    buf = io.BytesIO()
    torch.save(obj, buf) if kind == "torch" else joblib.dump(obj, buf)
    return buf.tell()


def load_checkpoint(path, data_dir):
    """A checkpoint trained on the dataset in data_dir, or None if the file does not exist.
    A checkpoint from another dataset is refused (ValueError), never silently benchmarked."""
    path = Path(path)
    if not path.exists():
        return None
    ckpt = joblib.load(path) if path.suffix == ".joblib" else torch.load(path, map_location="cpu")
    check_checkpoint(ckpt, data_dir, path)
    return ckpt


def sklearn_candidate(name, dim, n_classes, data_dir):
    ck = load_checkpoint(f"checkpoints/baseline_{name}.joblib", data_dir)
    if ck is not None:
        return ck["model"], True
    rng = np.random.default_rng(0)  # same architecture, random data: same latency
    return make_logreg(1.0).fit(rng.normal(size=(50, dim)), np.arange(50) % n_classes), False


def build_candidates(img_bytes, n_classes, img_size, data_dir):
    C = {}
    # 1-2. classical: logistic regression on pixels / hand-crafted features
    for name, feat in (("pixels32", lambda x: pixels(x, 32)), ("handcrafted", handcrafted)):
        dim = feat(np.zeros((1, 128, 128, 3), np.uint8)).shape[1]
        model, trained = sklearn_candidate(name, dim, n_classes, data_dir)
        C[f"logreg_{name}"] = {
            "fn": (lambda m=model, f=feat: m.predict_proba(f(load_rgb(io.BytesIO(img_bytes), 128)[None]))),
            "size_bytes": nbytes(model, "joblib"), "params": int(model[-1].coef_.size + model[-1].intercept_.size),
            "trained_weights": trained}
    # 3. from-scratch MLP on 32x32 pixels
    c = load_checkpoint("checkpoints/mlp_pixels32.pt", data_dir)
    if c is not None:
        mlp = MLP(c["in_features"], c["n_classes"], **c["mlp_kwargs"])
        mlp.load_state_dict(c["state_dict"])
    else:
        mlp = MLP(3072, n_classes, hidden=(512, 256))
    mlp.eval()

    @torch.no_grad()
    def run_mlp():
        return mlp(torch.from_numpy(pixels(load_rgb(io.BytesIO(img_bytes), 128)[None], 32)))
    C["mlp_pixels32"] = {"fn": run_mlp, "size_bytes": nbytes(mlp.state_dict(), "torch"),
                         "params": mlp.n_parameters(), "trained_weights": c is not None}
    # 4. ResNet-18 (frozen and fine-tuned share the architecture, hence the latency)
    for s in sorted({img_size, 224}):
        net = ResNetClassifier(n_classes, "frozen", pretrained=False)
        ck = load_checkpoint("checkpoints/resnet18_finetune.pt", data_dir)
        trained = ck is not None
        if trained:
            net.load_state_dict(ck["state_dict"])
        net.eval()
        tf = image_transform(s, train=False)

        @torch.no_grad()
        def run_resnet(net=net, tf=tf):
            with Image.open(io.BytesIO(img_bytes)) as im:
                return net(tf(im.convert("RGB"))[None])
        C[f"resnet18_{s}px"] = {"fn": run_resnet, "size_bytes": nbytes(net.state_dict(), "torch"),
                                "params": net.n_parameters(), "trained_weights": trained}
    return C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_style5")
    p.add_argument("--out", default="results/05_benchmark")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--img-size", type=int, default=128, help="ResNet input size used in step 4")
    p.add_argument("--n-classes", type=int, default=5)
    p.add_argument("--runs", type=int, default=100)
    a = p.parse_args()

    # a real visitor-like photo: the first image of the dataset (or a synthetic one)
    imgs = sorted(Path(a.data_dir).glob("*/*.jpg"))
    if imgs:
        img_bytes = imgs[0].read_bytes()
    else:
        buf = io.BytesIO()
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(buf, "JPEG")
        img_bytes = buf.getvalue()
    C = build_candidates(img_bytes, a.n_classes, a.img_size, a.data_dir)
    env = {"platform": platform.platform(), "processor": platform.processor() or "n/a",
           "cpu_count": os.cpu_count(), "torch": torch.__version__}
    print(env)
    rows = []
    for t in a.threads:
        torch.set_num_threads(t)
        for name, c in C.items():
            r = time_fn(c["fn"], n_runs=a.runs)
            rows.append({"model": name, "threads": t, **r, "size_mb": c["size_bytes"] / 1e6, "params": c["params"],
                         "trained_weights": c["trained_weights"]})
            print(f"threads={t} {name:22s} median {r['median_ms']:7.2f} ms  p95 {r['p95_ms']:7.2f} ms  "
                  f"size {c['size_bytes'] / 1e6:6.2f} MB  params {c['params']:>10,}")
    save_json({"environment": env, "results": rows, "provenance": provenance(a.data_dir)},
              Path(a.out) / "benchmark.json")
    print(f"-> {a.out}/benchmark.json")


if __name__ == "__main__":
    main()
