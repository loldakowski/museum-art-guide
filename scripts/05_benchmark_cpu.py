#!/usr/bin/env python
"""Step 5 - What does each model cost on a kiosk CPU? End-to-end latency for ONE visitor photo.

Measured pipeline per candidate: decode the JPEG -> preprocess -> (features) -> predict. Feature
extraction is part of the cost of the classical models, so it is included. Two inputs:
  photo : a visitor's phone photo, --photo (a real one, git-ignored; its resolution, file size and EXIF
          orientation are recorded). Without it, a synthetic 12-MP JPEG (a dataset image upscaled, with
          sensor-like noise so that it compresses like a real photo).
  small : the dataset image itself (128 px), the input of the earlier benchmark, for comparison.
Two ways to decode the photo:
  full  : decode every pixel, then resize (ImageOps.fit for the 128 px models, Resize for ResNet).
  draft : im.draft() first (the JPEG decoder skips detail and decodes at 1/2, 1/4 or 1/8 scale, never below
          the requested size), then ImageOps.fit, bicubic. This is exactly how make_wikiart_subset.save_image
          built every training image, so it matches the training preprocessing.
The recommended model is timed as deployed: the file written by step 4 (resnet.FrozenLogReg: backbone +
scaler + logistic regression). Free-text search is timed per query (CLIP text encoder + dot product with the
image index; TF-IDF). Latency does not depend on the values of the weights, so untrained models of the same
architecture are used when a checkpoint is missing (trained_weights=false in the output). Thread counts above
the number of CPU cores are skipped (they would oversubscribe the CPU) and listed as skipped.

Training cost on CPU (all cores): frozen backbone = embed the training images + fit the logistic regression
(what a kiosk operator would re-run to add a style); fine-tuning = ONE epoch of step 4's fine-tuning, as run.

    python scripts/05_benchmark_cpu.py --data-dir data/wikiart_style5 --threads 1 2 4
    python scripts/05_benchmark_cpu.py --machine "laptop: Apple M5" --out results/05_benchmark_laptop
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
from museum_guide.data import class_names, image_transform, load_metadata, make_image_loaders, make_split
from museum_guide.features import handcrafted, pixels
from museum_guide.mlp import MLP
from museum_guide.resnet import FrozenLogReg, ResNetClassifier, extract_embeddings
from museum_guide.training import TrainConfig, train

PHOTO_SIZE = (4032, 3024)  # 12 MP, a common phone camera resolution


def decode_fit(img_bytes: bytes, size: int, draft: bool) -> Image.Image:
    """JPEG bytes -> size x size RGB image (centre crop). draft=True = the training-set preprocessing."""
    with Image.open(io.BytesIO(img_bytes)) as im:
        if draft:
            im.draft("RGB", (size, size))
        return ImageOps.fit(im.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)


def load_rgb(img_bytes: bytes, size: int, draft: bool = False) -> np.ndarray:
    return np.asarray(decode_fit(img_bytes, size, draft), dtype=np.uint8)


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


def _jpeg(im: Image.Image, quality=90) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def test_inputs(data_dir, photo=None, seed=0) -> tuple[dict, dict]:
    """({'photo': phone-photo JPEG bytes, 'small': the first dataset image}, description of each input).
    Without --photo (or a dataset), synthetic images stand in."""
    rng = np.random.default_rng(seed)
    imgs = sorted(Path(data_dir).glob("*/*.jpg"))
    small = imgs[0].read_bytes() if imgs else _jpeg(Image.fromarray(rng.integers(0, 255, (128, 128, 3), np.uint8)))
    if photo and Path(photo).exists():
        big, source = Path(photo).read_bytes(), f"real phone photo ({Path(photo).name})"
    else:
        with Image.open(io.BytesIO(small)) as im:
            px = np.asarray(im.convert("RGB").resize(PHOTO_SIZE, Image.BICUBIC), dtype=np.float32)
        px += rng.normal(0.0, 4.0, px.shape).astype(np.float32)  # sensor-like noise: real photos are not smooth
        big, source = _jpeg(Image.fromarray(np.clip(px, 0, 255).astype(np.uint8))), "synthetic (upscaled dataset image)"
    inputs, info = {"photo": big, "small": small}, {}
    for k, v in inputs.items():
        with Image.open(io.BytesIO(v)) as im:
            info[k] = {"bytes": len(v), "megabytes": len(v) / 1e6, "size": list(im.size), "format": im.format,
                       "exif_orientation": im.getexif().get(274)}
    info["photo"]["source"], info["small"]["source"] = source, "first dataset image"
    return inputs, info


def load_checkpoint(path, data_dir):
    """A checkpoint trained on the dataset in data_dir, or None if the file does not exist.
    A checkpoint from another dataset is refused (ValueError), never silently benchmarked."""
    path = Path(path)
    if not path.exists():
        return None
    ckpt = joblib.load(path) if path.suffix == ".joblib" else torch.load(path, map_location="cpu")
    check_checkpoint(ckpt, data_dir, path)
    return ckpt


def sklearn_candidate(path, dim, n_classes, data_dir):
    ck = load_checkpoint(path, data_dir)
    if ck is not None:
        return ck["model"], True
    rng = np.random.default_rng(0)  # same architecture, random data: same latency
    return make_logreg(1.0).fit(rng.normal(size=(50, dim)), np.arange(50) % n_classes), False


def resnet_input(img_bytes: bytes, size: int, draft: bool) -> Image.Image:
    """full: the whole decoded photo (image_transform resizes it); draft: as the training images were made."""
    if draft:
        return decode_fit(img_bytes, size, True)
    with Image.open(io.BytesIO(img_bytes)) as im:
        return im.convert("RGB")


def frozen_logreg_candidate(ckpt_dir, n_classes, img_size, data_dir):
    """The deployed file of step 4, or the same pipeline with random weights (same latency)."""
    f = Path(ckpt_dir) / "resnet18_frozen_logreg.joblib"
    if f.exists():
        return FrozenLogReg.load(f, data_dir), f.stat().st_size, True
    rng = np.random.default_rng(0)
    m = FrozenLogReg(ResNetClassifier(n_classes, "frozen", pretrained=False),
                     make_logreg(1.0).fit(rng.normal(size=(50, 512)), np.arange(50) % n_classes),
                     range(n_classes), img_size)
    size = nbytes({"backbone": m.model.state_dict(), "logreg": m.logreg}, "joblib")
    return m, size, False


def build_candidates(img_bytes, n_classes, img_size, data_dir, ckpt_dir, draft=False):
    """Every classifier as a function of the photo bytes. draft: decoding as in make_wikiart_subset.save_image."""
    ck_dir = Path(ckpt_dir)
    C = {}
    feats = {"pixels32": lambda x: pixels(x, 32), "handcrafted": handcrafted}
    # 1-2. classical: logistic regression on pixels / hand-crafted features
    for name, feat in feats.items():
        dim = feat(np.zeros((1, 128, 128, 3), np.uint8)).shape[1]
        model, trained = sklearn_candidate(ck_dir / f"baseline_{name}.joblib", dim, n_classes, data_dir)
        C[f"logreg_{name}"] = {
            "fn": (lambda m=model, f=feat: m.predict_proba(f(load_rgb(img_bytes, 128, draft)[None]))),
            "size_bytes": nbytes(model, "joblib"), "params": int(model[-1].coef_.size + model[-1].intercept_.size),
            "trained_weights": trained}
    # 3-4. from-scratch MLP on 32x32 pixels / hand-crafted features
    for name, feat in feats.items():
        c = load_checkpoint(ck_dir / f"mlp_{name}.pt", data_dir)
        if c is not None:
            mlp = MLP(c["in_features"], c["n_classes"], **c["mlp_kwargs"])
            mlp.load_state_dict(c["state_dict"])
        else:
            mlp = MLP(feat(np.zeros((1, 128, 128, 3), np.uint8)).shape[1], n_classes, hidden=(512, 256))
        mlp.eval()

        @torch.no_grad()
        def run_mlp(mlp=mlp, feat=feat):
            x = feat(load_rgb(img_bytes, 128, draft)[None])
            return mlp(torch.as_tensor(x, dtype=torch.float32))
        C[f"mlp_{name}"] = {"fn": run_mlp, "size_bytes": nbytes(mlp.state_dict(), "torch"),
                            "params": mlp.n_parameters(), "trained_weights": c is not None}
    # 5. the recommended model as deployed: frozen ResNet-18 embedding + scaler + logistic regression
    fl, size, trained = frozen_logreg_candidate(ck_dir, n_classes, img_size, data_dir)
    tf = image_transform(fl.img_size, train=False)

    def run_frozen_logreg(fl=fl, tf=tf):
        return fl.predict_proba(tf(resnet_input(img_bytes, fl.img_size, draft))[None])
    C["resnet18_frozen_logreg"] = {"fn": run_frozen_logreg, "size_bytes": size,
                                   "params": fl.model.n_parameters() - sum(p.numel() for p in fl.model.net.fc.parameters())
                                   + int(fl.logreg[-1].coef_.size + fl.logreg[-1].intercept_.size),
                                   "trained_weights": trained}
    # 6. ResNet-18 with its own head (frozen + linear head and fine-tuned share the architecture, hence the latency)
    for s in sorted({img_size, 224}):
        net = ResNetClassifier(n_classes, "frozen", pretrained=False)
        ck = load_checkpoint(ck_dir / "resnet18_finetune.pt", data_dir)
        trained = ck is not None
        if trained:
            net.load_state_dict(ck["state_dict"])
        net.eval()
        tf = image_transform(s, train=False)

        @torch.no_grad()
        def run_resnet(net=net, tf=tf, s=s):
            return net(tf(resnet_input(img_bytes, s, draft))[None])
        C[f"resnet18_{s}px"] = {"fn": run_resnet, "size_bytes": nbytes(net.state_dict(), "torch"),
                                "params": net.n_parameters(), "trained_weights": trained}
    return C


def search_candidates(data_dir, n_images: int, clip: bool) -> dict:
    """Cost of answering one free-text question (the photo plays no part): TF-IDF and, unless --no-clip, CLIP."""
    from museum_guide.search import ClipSearcher, TfidfSearcher, metadata_documents
    query = "Which painting shows a stormy sea?"
    C = {}
    if (Path(data_dir) / "metadata.csv").exists():
        meta = load_metadata(data_dir)
        t = TfidfSearcher(list(meta["id"]), metadata_documents(meta))
        C["tfidf_search_query"] = {"fn": lambda: t.search(query, 10),
                                   "size_bytes": nbytes(t, "joblib"), "params": 0, "trained_weights": True}
    if clip:
        s = ClipSearcher("cpu")
        E = np.random.default_rng(0).normal(size=(n_images, 512)).astype(np.float32)  # same shape as the index
        s.index(list(range(n_images)), E / np.linalg.norm(E, axis=1, keepdims=True))
        C["clip_search_query"] = {"fn": lambda: s.search(query, 10),
                                  "size_bytes": nbytes(s.model.state_dict(), "torch") + E.nbytes,
                                  "params": sum(p.numel() for n, p in s.model.named_parameters()
                                                if not n.startswith("visual")),
                                  "trained_weights": True}
    return C


def training_cost(a, threads: int) -> dict:
    """Wall-clock training cost on this CPU (all cores), on the training set of the artist-grouped split."""
    torch.set_num_threads(threads)
    meta = load_metadata(a.data_dir)
    names = class_names(meta)
    split = make_split(meta, "artist")
    out = {"threads": threads, "n_train": int((split == "train").sum()), "img_size": a.img_size,
           "pretrained": not a.no_pretrained}
    model = ResNetClassifier(len(names), "frozen", pretrained=not a.no_pretrained)
    L = make_image_loaders(meta, split, a.data_dir, a.img_size, 32, augment=False, num_workers=0)
    t0 = time.perf_counter()
    E, y = extract_embeddings(model, L["train"], "cpu")
    out["frozen_embed_train_s"] = time.perf_counter() - t0
    ck = Path(a.checkpoint_dir) / "resnet18_frozen_logreg.joblib"
    C = float(joblib.load(ck)["logreg"][-1].C) if ck.exists() else 1e-3
    t0 = time.perf_counter()
    make_logreg(C).fit(E.numpy(), y.numpy())
    out["frozen_logreg_fit_s"], out["logreg_C"] = time.perf_counter() - t0, C
    out["frozen_total_s"] = out["frozen_embed_train_s"] + out["frozen_logreg_fit_s"]

    # one epoch of step 4's fine-tuning, same loop and settings (train + validation pass)
    ft = ResNetClassifier(len(names), "finetune", pretrained=not a.no_pretrained)
    L = make_image_loaders(meta, split, a.data_dir, a.img_size, 32, augment=True, num_workers=0)
    cfg = TrainConfig(epochs=1, lr=1e-3, weight_decay=1e-2, patience=6, seed=0, label_smoothing=0.1)
    t0 = time.perf_counter()
    train(ft, L["train"], L["val"], cfg, "cpu", log=lambda *_: None)
    out["finetune_one_epoch_s"] = time.perf_counter() - t0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_style5")
    p.add_argument("--out", default="results/05_benchmark")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--img-size", type=int, default=128, help="ResNet input size used in step 4")
    p.add_argument("--n-classes", type=int, default=5)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--photo", default="data/kiosk/phone_photo.jpg", help="a real phone photo (JPEG, git-ignored)")
    p.add_argument("--machine", default="", help="what this machine stands for, e.g. 'laptop: Apple M5'")
    p.add_argument("--no-clip", action="store_true", help="skip CLIP (605 MB download), e.g. in the smoke test")
    p.add_argument("--no-training-cost", action="store_true")
    p.add_argument("--no-pretrained", action="store_true", help="training cost with random weights (smoke test)")
    a = p.parse_args()

    cores = os.cpu_count() or 1
    threads = [t for t in a.threads if t <= cores]
    skipped = [t for t in a.threads if t > cores]
    inputs, info = test_inputs(a.data_dir, a.photo)
    env = {"machine": a.machine, "platform": platform.platform(), "processor": platform.processor() or "n/a",
           "cpu_count": cores, "torch": torch.__version__, "threads_skipped_above_cpu_count": skipped,
           "inputs": info}
    print(env)
    n_images = len(load_metadata(a.data_dir)) if (Path(a.data_dir) / "metadata.csv").exists() else 1500
    # (input, decoding) -> candidates; draft decoding changes nothing on an image already at the target size
    candidates = {(inp, dec): build_candidates(inputs[inp], a.n_classes, a.img_size, a.data_dir, a.checkpoint_dir,
                                               draft=(dec == "draft"))
                  for inp, dec in (("photo", "full"), ("photo", "draft"), ("small", "full"))}
    candidates[("text query", "-")] = search_candidates(a.data_dir, n_images, clip=not a.no_clip)
    rows = []
    for t in threads:
        torch.set_num_threads(t)
        for (inp, dec), C in candidates.items():
            for name, c in C.items():
                r = time_fn(c["fn"], n_runs=a.runs)
                rows.append({"model": name, "input": inp, "decode": dec, "threads": t, **r,
                             "size_mb": c["size_bytes"] / 1e6, "params": c["params"],
                             "trained_weights": c["trained_weights"]})
                print(f"threads={t} {inp:10s} {dec:5s} {name:22s} median {r['median_ms']:8.2f} ms  "
                      f"p95 {r['p95_ms']:8.2f} ms  size {c['size_bytes'] / 1e6:7.2f} MB  params {c['params']:>11,}")
    res = {"environment": env, "results": rows}
    if not a.no_training_cost and (Path(a.data_dir) / "metadata.csv").exists():
        res["training_cost"] = training_cost(a, cores)
        print("training cost on CPU:", res["training_cost"])
    save_json({**res, "provenance": provenance(a.data_dir)}, Path(a.out) / "benchmark.json")
    print(f"-> {a.out}/benchmark.json")


if __name__ == "__main__":
    main()
