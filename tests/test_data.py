import numpy as np

from museum_guide.data import (UNKNOWN_ARTIST, class_names, load_image_array, load_metadata, make_split,
                               split_report)


def test_metadata_and_classes(fake_data):
    meta = load_metadata(fake_data)
    assert len(meta) == 5 * 35
    assert class_names(meta) == sorted(class_names(meta))  # ImageFolder order


def test_artist_split_has_no_shared_named_artist(fake_data):
    meta = load_metadata(fake_data)
    split = make_split(meta, "artist", seed=0)
    rep = split_report(meta, split)
    assert all(len(v) == 0 for v in rep["shared_artists"].values())
    assert sum(rep["n"].values()) == len(meta)
    for s in ("train", "val", "test"):
        assert rep["n"][s] > 0


def test_unknown_artist_images_are_spread(fake_data):
    meta = load_metadata(fake_data)
    split = make_split(meta, "artist", seed=0)
    unk = meta["artist"].to_numpy() == UNKNOWN_ARTIST
    assert len(set(split[unk])) == 3  # each unknown-artist image is its own group
    for sp in ("train", "val", "test"):  # and their share is similar in every split
        assert abs(unk[split == sp].mean() - unk.mean()) < 0.15


def test_split_proportions_per_class(fake_data):
    meta = load_metadata(fake_data)
    split = make_split(meta, "artist", seed=0)
    for c in meta["label"].unique():
        share = (split[meta["label"].to_numpy() == c] == "train").mean()
        assert 0.55 < share < 0.85


def test_split_is_deterministic(fake_data):
    meta = load_metadata(fake_data)
    assert (make_split(meta, "artist", seed=3) == make_split(meta, "artist", seed=3)).all()
    assert (make_split(meta, "random", seed=3) == make_split(meta, "random", seed=3)).all()


def test_image_array(fake_data):
    meta = load_metadata(fake_data)
    X = load_image_array(meta.head(4), fake_data, size=32, cache=False)
    assert X.shape == (4, 32, 32, 3) and X.dtype == np.uint8
