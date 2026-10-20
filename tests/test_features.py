import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize

from museum_guide.features import color_moments, handcrafted, hog_features, hsv_histogram, lbp_histogram


def _images(n=4, size=128, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, size, size, 3), dtype=np.uint8)


def test_hog_default_is_the_original_8x8_cells():
    """The pixels_per_cell keyword (feature ablation) must not change the default features."""
    X = _images()
    original = np.asarray([hog(resize(rgb2gray(img), (64, 64), anti_aliasing=True), orientations=9,
                               pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True) for img in X],
                          dtype=np.float32)
    assert np.array_equal(hog_features(X), original)
    assert np.array_equal(hog_features(X, pixels_per_cell=8), original)
    assert np.array_equal(handcrafted(X), np.concatenate([hsv_histogram(X, (12, 4, 4)), color_moments(X),
                                                          original, lbp_histogram(X, 8, 1.0)], axis=1))
    assert handcrafted(X).shape == (len(X), 1975)


def test_feature_parameters_change_the_dimension():
    X = _images(2)
    assert hog_features(X, pixels_per_cell=16).shape[1] < hog_features(X).shape[1]
    assert hsv_histogram(X, (8, 3, 3)).shape[1] == 8 * 3 * 3
    assert lbp_histogram(X, 16, 2.0).shape[1] == 16 + 2
