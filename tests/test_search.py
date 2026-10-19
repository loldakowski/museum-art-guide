import numpy as np
import pandas as pd

from museum_guide.data import load_metadata
from museum_guide.search import (CLIP_MODEL, ClipSearcher, TfidfSearcher, _keywords, _rank, auto_relevance,
                                 metadata_documents, precision_at_k)


def _meta():
    return pd.DataFrame({"id": list("abcd"), "label": ["Cubism", "Baroque", "Cubism", "Pop_Art"],
                         "style": ["Cubism", "Baroque", "Cubism", "Pop_Art"],
                         "genre": ["still_life", "Unknown Genre", "landscape", "genre_painting"],
                         "artist": ["juan-gris", "Unknown Artist", "Unknown Artist", "Unknown Artist"]})


def test_metadata_documents_drop_unknown_values():
    assert metadata_documents(_meta()) == ["Cubism still life juan gris", "Baroque", "Cubism landscape",
                                           "Pop Art genre painting"]


def test_tfidf_matches_word_variants_and_ignores_generic_words():
    m = _meta()
    t = TfidfSearcher(list(m["id"]), metadata_documents(m))
    assert set(t.search("a cubist painting", 2)) == {"a", "c"}     # cubist ~ Cubism; 'painting' is ignored
    assert t.search("a baroque painting", 1) == ["b"]              # not the 'genre painting' record
    assert _keywords("A Baroque painting of the sea") == "baroque sea"


def test_ties_are_broken_by_a_seeded_permutation_not_by_id():
    ids = list(range(100))
    r0, r1 = _rank(np.zeros(100), ids, 100, seed=0), _rank(np.zeros(100), ids, 100, seed=1)
    assert sorted(r0) == ids and r0 != ids and r0 != r1
    assert _rank(np.zeros(100), ids, 100, seed=0) == r0              # reproducible


def test_genre_queries_are_scored_only_on_known_genres():
    m = _meta()
    cand, rel = auto_relevance(m, {"field": "genre", "value": "landscape"})
    assert cand.tolist() == [True, False, True, True] and rel.tolist() == [False, False, True, False]
    scores = np.array([0.0, 9.0, 1.0, 0.5])                        # 'b' scores best but has no known genre
    assert precision_at_k(scores, cand, rel, 1) == 1.0
    cand, rel = auto_relevance(m, {"field": "label", "value": "Cubism"})
    assert cand.all() and rel.sum() == 2


class StubClip(ClipSearcher):
    """ClipSearcher without the 605 MB model: text embeddings come from a fixed table."""
    def __init__(self, table):
        self.table, self.seed, self.name = table, 0, "stub"

    def embed_text(self, queries):
        return np.stack([self.table[q] for q in queries])


def test_clip_search_ranks_by_cosine(fake_data):
    meta = load_metadata(fake_data)
    E = np.zeros((len(meta), 8))
    E[:, 0] = np.linspace(0, 1, len(meta))                         # image i: similarity i / (N - 1) with query
    s = StubClip({"q": np.eye(8)[0]}).index(list(meta["id"]), E)
    assert s.search("q", 3) == list(meta["id"])[::-1][:3]


def test_clip_model_uses_the_activation_the_openai_weights_were_trained_with():
    import open_clip
    model = open_clip.create_model(CLIP_MODEL)  # architecture only, no download
    assert {type(m).__name__ for m in model.modules() if "GELU" in type(m).__name__} == {"QuickGELU"}
