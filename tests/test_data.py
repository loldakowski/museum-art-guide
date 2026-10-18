import numpy as np
import pandas as pd

from museum_guide.data import (UNKNOWN_ARTIST, apply_duplicates, artist_groups, class_names, duplicates_file,
                               load_image_array, load_metadata, make_split,
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


def _verdicts(rows):
    return pd.DataFrame(rows, columns=["id_a", "id_b", "verdict"])


def test_duplicates_stay_in_one_split_and_conflicting_copies_are_removed(fake_data):
    meta = load_metadata(fake_data)
    unk = meta[meta["artist"] == UNKNOWN_ARTIST]
    a, b, c, d = unk["id"].iloc[:4]                                     # all 'Baroque', unknown artist
    other = unk[unk["label"] != unk["label"].iloc[0]]["id"].iloc[0]     # another class
    named = meta[meta["artist"] != UNKNOWN_ARTIST]["id"].iloc[0]
    v = _verdicts([(a, b, "same_image"), (b, c, "same_hidden_painter"), (d, other, "same_image"),
                   (a, named, "different")])
    kept, removed = apply_duplicates(meta, v)
    assert removed == sorted([d, other])                                # same image, two labels: both copies go
    assert len(kept) == len(meta) - 2
    g = dict(zip(kept["id"], kept["dup_group"]))
    assert g[a] == g[b] == g[c] and pd.isna(g[named])                   # "different" links nothing
    for seed in range(5):
        for mode in ("artist", "random"):
            split = dict(zip(kept["id"], make_split(kept, mode, seed=seed)))
            assert split[a] == split[b] == split[c], (mode, seed)


def test_duplicate_of_a_named_artist_joins_the_artist_group(fake_data):
    meta = load_metadata(fake_data)
    named = meta[meta["artist"] != UNKNOWN_ARTIST].iloc[0]
    unk = meta[(meta["artist"] == UNKNOWN_ARTIST) & (meta["label"] == named["label"])]["id"].iloc[0]
    kept, _ = apply_duplicates(meta, _verdicts([(named["id"], unk, "same_image")]))
    groups = dict(zip(kept["id"], artist_groups(kept)))
    same_artist = kept[kept["artist"] == named["artist"]]["id"]
    assert {groups[i] for i in same_artist} == {groups[unk]}           # the copy follows the painter's group


def test_verdict_file_is_found_by_dataset_name(fake_data, tmp_path):
    assert duplicates_file(fake_data) is None                          # no annotations for the synthetic data
    assert duplicates_file("data/wikiart_style5") is not None


def test_cross_validation_folds_test_every_image_once(fake_data):
    meta = load_metadata(fake_data)
    for mode in ("artist", "random"):
        splits = [make_split(meta, mode, seed=0, fold=k) for k in range(5)]
        tested = np.stack([s == "test" for s in splits])
        assert (tested.sum(0) == 1).all(), mode                          # each image in exactly one test fold
        for k, s in enumerate(splits):
            assert (s == "val").any() and not ((s == "val") & tested[k]).any()
            assert ((splits[(k + 1) % 5] == "test") == (s == "val")).all()  # val = next fold's test
        if mode == "artist":
            for s in splits:
                rep = split_report(meta, s)
                assert all(len(v) == 0 for v in rep["shared_artists"].values())
