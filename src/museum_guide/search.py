"""Free-text search over the collection (Part 2): "Which painting shows a stormy sea?" -> ranked artworks.

Two searchers with the same interface, search(query, k) -> list of image ids, best first:

* ClipSearcher (neural): pretrained CLIP ViT-B/32 (open_clip). Images and text are embedded in one space by
  contrastive pre-training on image-caption pairs. Image embeddings are computed ONCE, offline; on the kiosk
  only the text encoder runs, then a dot product with the stored image embeddings (cosine similarity).
* TfidfSearcher (classical baseline): TF-IDF match of character n-grams between the query and each artwork's
  metadata text (style, genre, artist). It can only find what the catalogue says, never what the painting shows.

Scores that tie (identical metadata texts, or no matching word at all) are ordered by a seeded random
permutation, not by id, so that tie-breaking neither helps nor hurts a system.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# The OpenAI weights were trained with QuickGELU: "ViT-B-32" would build plain GELU layers around them
# (open_clip warns "QuickGELU mismatch"), a slightly different network from the one OpenAI trained.
CLIP_MODEL, CLIP_PRETRAINED = "ViT-B-32-quickgelu", "openai"


def _rank(scores: np.ndarray, ids, k: int, seed: int = 0) -> list:
    tiebreak = np.random.default_rng(seed).permutation(len(scores))
    order = np.lexsort((tiebreak, -scores))  # by score (descending), then by the random permutation
    return [ids[i] for i in order[:k]]


def metadata_documents(meta: pd.DataFrame) -> list[str]:
    """What a catalogue knows about each artwork: style, genre, artist (unknown values left out)."""
    def clean(v):
        v = str(v)
        return "" if v.startswith("Unknown") else v.replace("_", " ").replace("-", " ")
    return [" ".join(filter(None, (clean(s), clean(g), clean(a))))
            for s, g, a in zip(meta["style"], meta["genre"], meta["artist"])]


# Words that every artwork of a museum matches: removed from queries and catalogue texts. Added after seeing
# "a baroque painting" retrieve 'genre painting' records instead of Baroque ones (the word 'painting' outweighed
# 'baroque'); fixed list, independent of the evaluation queries.
DOMAIN_STOP_WORDS = {"painting", "paintings", "picture", "pictures", "image", "images", "artwork", "artworks"}


def _keywords(text: str) -> str:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(w for w in words if w not in ENGLISH_STOP_WORDS and w not in DOMAIN_STOP_WORDS)


class TfidfSearcher:
    def __init__(self, ids, documents, seed: int = 0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.ids, self.seed = list(ids), seed
        # character 3-5-grams inside words: "cubist" matches "cubism", "impressionist" matches "impressionism"
        # (word tokens would not: a weaker baseline than a real catalogue search would be)
        self.vectorizer = TfidfVectorizer(preprocessor=_keywords, analyzer="char_wb", ngram_range=(3, 5),
                                          sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(documents)  # (N, vocabulary), rows L2-normalised

    def scores(self, query: str) -> np.ndarray:
        return (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()

    def search(self, query: str, k: int = 10) -> list:
        return _rank(self.scores(query), self.ids, k, self.seed)


class ClipSearcher:
    def __init__(self, device="cpu", model_name: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED, seed: int = 0):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device, self.name, self.seed = device, f"{model_name}/{pretrained}", seed
        self.ids, self.image_embeddings = [], None

    @torch.no_grad()
    def embed_images(self, paths, batch_size: int = 64) -> np.ndarray:
        """L2-normalised image embeddings (N, 512). Offline step: done once for the whole collection."""
        out = []
        for i in range(0, len(paths), batch_size):
            batch = []
            for p in paths[i:i + batch_size]:
                with Image.open(p) as im:
                    batch.append(self.preprocess(im.convert("RGB")))
            out.append(torch.nn.functional.normalize(self.model.encode_image(torch.stack(batch).to(self.device)),
                                                     dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def embed_text(self, queries) -> np.ndarray:
        t = self.model.encode_text(self.tokenizer(list(queries)).to(self.device))
        return torch.nn.functional.normalize(t, dim=-1).cpu().numpy()

    def index(self, ids, image_embeddings: np.ndarray) -> "ClipSearcher":
        self.ids, self.image_embeddings = list(ids), image_embeddings
        return self

    def scores(self, query: str) -> np.ndarray:
        return self.image_embeddings @ self.embed_text([query])[0]

    def search(self, query: str, k: int = 10) -> list:
        return _rank(self.scores(query), self.ids, k, self.seed)


def cached_image_embeddings(searcher: ClipSearcher, meta: pd.DataFrame, data_dir) -> np.ndarray:
    """searcher.embed_images over the collection, cached in <data_dir>/.cache (key: model, image paths, bytes)."""
    data_dir = Path(data_dir)
    paths = [data_dir / p for p in meta["path"]]
    h = hashlib.md5(searcher.name.encode())
    for p in paths:
        h.update(str(p.relative_to(data_dir)).encode() + p.read_bytes())
    f = data_dir / ".cache" / f"clip_{searcher.name.replace('/', '_')}_{h.hexdigest()[:10]}.npy"
    if f.exists():
        return np.load(f)
    E = searcher.embed_images(paths)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, E)
    return E


# --------------------------------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------------------------------
def auto_relevance(meta: pd.DataFrame, q: dict) -> tuple[np.ndarray, np.ndarray]:
    """For a metadata query {'field', 'value'}: (candidates, relevant) boolean masks over meta rows.
    Genre queries are judged only on images whose genre is known ('Unknown Genre' is not 'not a landscape')."""
    values = meta[q["field"]].astype(str).to_numpy()
    candidates = ~np.char.startswith(values.astype(str), "Unknown") if q["field"] == "genre" else np.ones(len(meta), bool)
    return candidates, candidates & (values == q["value"])


def precision_at_k(scores: np.ndarray, candidates: np.ndarray, relevant: np.ndarray, k: int, seed: int = 0) -> float:
    """Share of relevant images among the k best-scored candidates (ties broken as in search())."""
    idx = np.flatnonzero(candidates)
    top = _rank(scores[idx], idx, k, seed)
    return float(relevant[top].mean())
