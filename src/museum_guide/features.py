"""Hand-crafted image features for the classical baseline (and raw pixels for the MLP).

Every function takes a uint8 array (N, H, W, 3) and returns float32 (N, D). They are written so that
a single image (N=1) can be timed in the CPU benchmark: in production, feature extraction is part of
the latency of the classical model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize


def pixels(X: np.ndarray, size: int = 32) -> np.ndarray:
    """Down-sampled raw pixels in [0, 1], flattened: (N, 3*size*size). Input of the MLP."""
    t = torch.from_numpy(X if X.flags.writeable else X.copy()).permute(0, 3, 1, 2).float() / 255.0
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return t.reshape(len(X), -1).numpy().astype(np.float32)


def hsv_histogram(X: np.ndarray, bins=(12, 4, 4)) -> np.ndarray:
    """Joint HSV colour histogram (palette): 12 hues x 4 saturations x 4 values = 192 dims."""
    out = []
    for img in X:
        hsv = rgb2hsv(img).reshape(-1, 3)
        h, _ = np.histogramdd(hsv, bins=bins, range=((0, 1), (0, 1), (0, 1)))
        out.append(h.ravel() / hsv.shape[0])
    return np.asarray(out, dtype=np.float32)


def color_moments(X: np.ndarray) -> np.ndarray:
    """Mean, std and skewness of each RGB channel: 9 dims."""
    x = X.reshape(len(X), -1, 3).astype(np.float32) / 255.0
    mu, sd = x.mean(1), x.std(1) + 1e-6
    skew = (((x - mu[:, None]) / sd[:, None]) ** 3).mean(1)
    return np.concatenate([mu, sd, skew], 1).astype(np.float32)


def hog_features(X: np.ndarray, size: int = 64) -> np.ndarray:
    """Histogram of oriented gradients on a 64x64 grey image (shapes, contours): 1764 dims."""
    out = []
    for img in X:
        g = rgb2gray(img)
        if g.shape[0] != size:
            g = resize(g, (size, size), anti_aliasing=True)
        out.append(hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True))
    return np.asarray(out, dtype=np.float32)


def lbp_histogram(X: np.ndarray, P: int = 8, R: float = 1.0) -> np.ndarray:
    """Uniform local binary patterns (micro-texture, brushwork): P+2 = 10 dims."""
    out = []
    for img in X:
        g = (rgb2gray(img) * 255).astype(np.uint8)
        lbp = local_binary_pattern(g, P, R, method="uniform")
        h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2))
        out.append(h / h.sum())
    return np.asarray(out, dtype=np.float32)


def handcrafted(X: np.ndarray) -> np.ndarray:
    """Colour (HSV histogram + moments) + shape (HOG) + texture (LBP): 1975 dims."""
    return np.concatenate([hsv_histogram(X), color_moments(X), hog_features(X), lbp_histogram(X)], axis=1)


FEATURE_SETS = {
    "pixels32": lambda X: pixels(X, 32),
    "pixels64": lambda X: pixels(X, 64),
    "handcrafted": handcrafted,
}


def compute_features(name: str, X: np.ndarray, cache_dir=None) -> np.ndarray:
    """FEATURE_SETS[name](X), cached on disk (hand-crafted features take ~1 min for 1 500 images)."""
    import hashlib
    from pathlib import Path
    if cache_dir is None:
        return FEATURE_SETS[name](X)
    # key: every pixel and the code of this module (a change of a feature parameter gives a new file)
    key = hashlib.md5(np.ascontiguousarray(X).tobytes() + str(X.shape).encode()
                      + Path(__file__).read_bytes()).hexdigest()[:10]
    f = Path(cache_dir) / f"feat_{name}_{key}.npy"
    if f.exists():
        return np.load(f)
    F_ = FEATURE_SETS[name](X)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, F_)
    return F_
