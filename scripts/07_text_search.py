#!/usr/bin/env python
"""Step 7 - Free-text search over the collection (Part 2): CLIP vs a TF-IDF catalogue search vs random.

Queries (search_eval/queries.json):
  * auto: style and genre queries, relevance read from metadata.csv. Genre queries are scored only on images whose
    genre is known (82 % of the collection is 'Unknown Genre'). TF-IDF reads the very field that defines relevance
    here, so its score on these queries is an upper bound, not a fair win.
  * content: free-text queries ("a stormy sea"), relevance judged by hand in search_eval/judgments.csv. Queries
    with 'paraphrase_of' are the brief's own wording of another query: reported next to it (phrasing comparison)
    and left out of the content mean.

Two modes:
  --make-pool   for each content query, pool the top-k of CLIP, the top-k of TF-IDF and k random images, shuffle
                them (the judge cannot tell which system found an image) -> search_eval/to_judge.csv, plus a local
                viewer <image-dir>/.cache/judge.html (shows paintings: never committed) that exports judgments.csv.
                judgments.csv is filled in by a human only.
  (default)     precision@5/@10 of every system on the auto queries and, if judgments.csv exists, on the content
                queries; CPU latency per query (1 thread) and model / index sizes -> <out>/summary.{json,md}.

The CLIP index uses --image-dir (a 224 px copy of the subset, CLIP's native input size; same ids as --data-dir,
whose metadata and duplicate verdicts define the collection).
    python scripts/07_text_search.py --make-pool
    python scripts/07_text_search.py
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from museum_guide.common import load_json, provenance, save_json
from museum_guide.data import load_metadata
from museum_guide.search import (ClipSearcher, TfidfSearcher, _rank, auto_relevance, cached_image_embeddings,
                                 metadata_documents, precision_at_k)

KS = (5, 10)


def collection(a):
    """Metadata of the collection (verdicts applied) with image paths resolved in --image-dir."""
    meta = load_metadata(a.data_dir)
    images = load_metadata(a.image_dir, duplicates=None).set_index("id")
    missing = set(meta["id"]) - set(images.index)
    if missing:
        raise SystemExit(f"{len(missing)} images of {a.data_dir} are missing from {a.image_dir}")
    return meta.assign(path=images.loc[meta["id"], "path"].to_numpy())


def build_searchers(a, meta):
    clip = ClipSearcher("cpu", seed=a.seed)
    clip.index(list(meta["id"]), cached_image_embeddings(clip, meta, a.image_dir))
    return {"clip": clip, "tfidf": TfidfSearcher(list(meta["id"]), metadata_documents(meta), seed=a.seed)}


def random_ranking(ids, query_id: str, seed: int) -> list:
    """A fixed random order per query (same draw in the pool and in the evaluation)."""
    rng = np.random.default_rng([seed, *query_id.encode()])
    return [ids[i] for i in rng.permutation(len(ids))]


def make_pool(a, meta, S, queries):
    ids = list(meta["id"])
    rows = []
    for q in queries["content"]:
        pool = set(S["clip"].search(q["query"], a.k)) | set(S["tfidf"].search(q["query"], a.k)) \
            | set(random_ranking(ids, q["id"], a.seed)[:a.k])
        order = np.random.default_rng([a.seed + 1, *q["id"].encode()]).permutation(sorted(pool))
        rows += [{"query_id": q["id"], "query": q["query"], "image_id": i} for i in order]
    out = Path(a.eval_dir)
    pd.DataFrame(rows).to_csv(out / "to_judge.csv", index=False)
    save_json({"k": a.k, "systems": ["clip", "tfidf", "random"], "n_queries": len(queries["content"]),
               "n_to_judge": len(rows), "clip_model": S["clip"].name, "image_dir": a.image_dir,
               "image_provenance": provenance(a.image_dir)}, out / "pool_info.json")

    path_of = dict(zip(meta["id"], meta["path"]))
    viewer = Path(a.image_dir) / ".cache" / "judge.html"
    viewer.parent.mkdir(parents=True, exist_ok=True)
    viewer.write_text(viewer_html(rows, path_of), encoding="utf-8")
    print(f"{len(rows)} (query, image) pairs to judge -> {out / 'to_judge.csv'}\nviewer: {viewer.resolve()}")


def viewer_html(rows, path_of) -> str:
    """Self-contained page: one section per query, each image with its id and a 'relevant' checkbox.
    Ticks are kept in the browser (localStorage); 'Download judgments.csv' writes query_id,image_id,relevant."""
    sections, current = [], None
    for r in rows:
        if r["query_id"] != current:
            if current is not None:
                sections.append("</div></section>")
            current = r["query_id"]
            sections.append(f'<section><h2>{html.escape(r["query_id"])}: “{html.escape(r["query"])}”</h2><div class="grid">')
        key = f'{r["query_id"]}|{r["image_id"]}'
        sections.append(f'<label class="cell"><img src="../{html.escape(path_of[r["image_id"]])}" loading="lazy">'
                        f'<span><input type="checkbox" data-key="{html.escape(key)}"> {html.escape(r["image_id"])}</span>'
                        f'</label>')
    sections.append("</div></section>")
    return """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Relevance judgments</title><style>
body{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#111}
header{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ccc;z-index:1}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.cell{display:flex;flex-direction:column;gap:4px;cursor:pointer}
.cell img{width:100%;aspect-ratio:1;object-fit:contain;background:#eee}
.cell:has(input:checked) img{outline:4px solid #1baf7a}
</style></head><body><header><b>Tick every image that is relevant to the query.</b> Unticked = not relevant.
Progress is saved in this browser. <button id="dl">Download judgments.csv</button> <span id="n"></span></header>
""" + "\n".join(sections) + """
<script>
const boxes=[...document.querySelectorAll('input[data-key]')];
let saved={};try{saved=JSON.parse(localStorage.getItem('judgments')||'{}')}catch(e){}
const count=()=>{document.getElementById('n').textContent=boxes.filter(b=>b.checked).length+' ticked / '+boxes.length};
boxes.forEach(b=>{b.checked=!!saved[b.dataset.key];b.onchange=()=>{saved[b.dataset.key]=b.checked;
 try{localStorage.setItem('judgments',JSON.stringify(saved))}catch(e){};count()}});count();
document.getElementById('dl').onclick=()=>{const lines=['query_id,image_id,relevant'].concat(
 boxes.map(b=>{const[q,i]=b.dataset.key.split('|');return q+','+i+','+(b.checked?1:0)}));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')+'\\n'],{type:'text/csv'}));
 a.download='judgments.csv';a.click()};
</script></body></html>"""


def latency(fn, queries, repeats: int) -> dict:
    for q in queries[:3]:
        fn(q)  # warm-up
    t = []
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            fn(q)
            t.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(t)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.quantile(t, 0.95)), "n": len(t)}


def nbytes(obj) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.tell()


def evaluate(a, meta, S, queries) -> dict:
    ids = list(meta["id"])
    pos = {i: k for k, i in enumerate(ids)}
    res = {"auto": [], "content": None}
    for q in queries["auto"]:
        cand, rel = auto_relevance(meta, q)
        row = {"id": q["id"], "query": q["query"], "n_relevant": int(rel.sum()), "n_candidates": int(cand.sum()),
               "random_expected": float(rel.sum() / cand.sum())}  # expected precision@k of a random ranking
        for name, s in S.items():
            sc = s.scores(q["query"])
            for k in KS:
                row[f"{name}_P@{k}"] = precision_at_k(sc, cand, rel, k, a.seed)
        res["auto"].append(row)

    jf = Path(a.eval_dir) / "judgments.csv"
    if jf.exists():
        j = pd.read_csv(jf)
        judged = {(r.query_id, r.image_id): int(r.relevant) for r in j.itertuples()}
        rows, unjudged = [], []
        for q in queries["content"]:
            ranked = {name: s.search(q["query"], max(KS)) for name, s in S.items()}
            ranked["random"] = random_ranking(ids, q["id"], a.seed)[:max(KS)]
            row = {"id": q["id"], "query": q["query"], "paraphrase_of": q.get("paraphrase_of")}
            for name, r in ranked.items():
                unjudged += [(q["id"], i) for i in r if (q["id"], i) not in judged]
                for k in KS:
                    row[f"{name}_P@{k}"] = float(np.mean([judged.get((q["id"], i), np.nan) for i in r[:k]]))
            rows.append(row)
        if unjudged:
            raise SystemExit(f"judgments.csv does not cover {len(set(unjudged))} pooled images, e.g. "
                             f"{sorted(set(unjudged))[:5]}: re-run --make-pool and judge again")
        res["content"] = rows
    res["phrasing"] = phrasing(S, queries)

    torch.set_num_threads(1)  # kiosk-like: one core
    texts = [q["query"] for q in queries["content"]] + [q["query"] for q in queries["auto"]]
    clip, tfidf = S["clip"], S["tfidf"]
    res["cost"] = {
        "threads": 1,
        "clip_latency": latency(lambda t: clip.search(t, 10), texts, a.repeats),
        "clip_text_encoder_latency": latency(lambda t: clip.embed_text([t]), texts, a.repeats),
        "tfidf_latency": latency(lambda t: tfidf.search(t, 10), texts, a.repeats),
        "clip_model_mb": nbytes(clip.model.state_dict()) / 1e6,
        "clip_text_tower_params": sum(p.numel() for n, p in clip.model.named_parameters() if not n.startswith("visual")),
        "clip_image_index_mb": clip.image_embeddings.nbytes / 1e6,
        "clip_index_build_s_per_image": None,
        "tfidf_index_mb": (tfidf.matrix.data.nbytes + tfidf.matrix.indices.nbytes + tfidf.matrix.indptr.nbytes) / 1e6,
        "tfidf_vocabulary": len(tfidf.vectorizer.vocabulary_),
        "collection_size": len(ids),
    }
    t0 = time.perf_counter()
    clip.embed_images([Path(a.image_dir) / p for p in meta["path"][:32]], batch_size=32)
    res["cost"]["clip_index_build_s_per_image"] = (time.perf_counter() - t0) / 32
    return res


def phrasing(S, queries) -> list[dict]:
    """For each query worded as in the brief ('paraphrase_of'): how many of its top-10 images each system shares
    with the original wording (10 = the wording does not matter). Needs no judgments."""
    by_id = {q["id"]: q for q in queries["content"]}
    out = []
    for q in queries["content"]:
        if q.get("paraphrase_of"):
            o = by_id[q["paraphrase_of"]]
            out.append({"id": q["id"], "query": q["query"], "original_id": o["id"], "original": o["query"],
                        **{f"{name}_top10_shared": len(set(s.search(q["query"], 10)) & set(s.search(o["query"], 10)))
                           for name, s in S.items()}})
    return out


def phrasing_table(res) -> list[str]:
    """Each brief wording next to the original query: precision (if judged) and shared top-10 images."""
    content = {r["id"]: r for r in res["content"] or []}
    cols = [f"{s}_P@{k}" for s in ("clip", "tfidf", "random") for k in KS]
    lines = ["| query | " + " | ".join(cols) + " | CLIP top-10 shared | TF-IDF top-10 shared |",
             "|---|" + "---|" * (len(cols) + 2)]
    for p in res["phrasing"]:
        for qid, text, shared in ((p["original_id"], p["original"], ("", "")),
                                  (p["id"], p["query"], (p["clip_top10_shared"], p["tfidf_top10_shared"]))):
            vals = [f"{content[qid][c]:.2f}" if qid in content else "-" for c in cols]
            lines.append(f"| {qid}: {text} | " + " | ".join(vals) + f" | {shared[0]} | {shared[1]} |")
    return lines


def mean_table(rows, systems) -> list[str]:
    cols = [f"{s}_P@{k}" for s in systems for k in KS]
    df = pd.DataFrame(rows)
    lines = ["| query | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += [f"| {r['query']} | " + " | ".join(f"{r[c]:.2f}" for c in cols) + " |" for r in rows]
    lines.append("| **mean** | " + " | ".join(f"**{df[c].mean():.3f}**" for c in cols) + " |")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/wikiart_style5", help="collection: metadata + duplicate verdicts")
    p.add_argument("--image-dir", default="data/wikiart_style5_224", help="images indexed by CLIP (same ids)")
    p.add_argument("--eval-dir", default="search_eval")
    p.add_argument("--out", default="results/07_search")
    p.add_argument("--k", type=int, default=10, help="pool depth per system (--make-pool)")
    p.add_argument("--repeats", type=int, default=5, help="latency: passes over all queries")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--make-pool", action="store_true")
    a = p.parse_args()
    queries = load_json(Path(a.eval_dir) / "queries.json")
    meta = collection(a)
    S = build_searchers(a, meta)
    if a.make_pool:
        make_pool(a, meta, S, queries)
        return

    res = evaluate(a, meta, S, queries)
    out = Path(a.out)
    save_json({**res, "clip_model": S["clip"].name, "seed": a.seed, "provenance": provenance(a.data_dir),
               "image_provenance": provenance(a.image_dir)}, out / "summary.json")
    auto = pd.DataFrame(res["auto"])
    lines = ["# Free-text search (Part 2)", "",
             f"Collection: {len(meta)} artworks. CLIP {S['clip'].name} on {a.image_dir}; TF-IDF on the metadata "
             "(style, genre, artist).", "",
             "## Metadata queries (relevance from metadata.csv)", "",
             "TF-IDF reads the field that defines relevance: an upper bound, not a fair comparison. Random = "
             "expected precision of a random ranking (share of relevant candidates). TF-IDF ignores English stop "
             "words and the words every artwork matches (painting, picture, image, artwork): this list was added "
             "after 'a baroque painting' retrieved 'genre painting' records (precision 0.00 before, 1.00 after).",
             ""]
    lines += mean_table(res["auto"], ["clip", "tfidf"])
    lines += ["", f"Random ranking, mean expected precision: {auto['random_expected'].mean():.3f}", ""]
    lines += ["## Content queries (relevance judged by hand, search_eval/judgments.csv)", ""]
    lines += mean_table([r for r in res["content"] if not r["paraphrase_of"]], ["clip", "tfidf", "random"]) \
        if res["content"] else ["Not evaluated yet: judgments.csv does not exist."]
    lines += ["", "## Phrasing: the brief's wording vs the short query", "",
              "The brief's questions, verbatim, next to the short query on the same topic (left out of the mean "
              "above). Shared = images that appear in the top 10 of both wordings.", ""]
    lines += phrasing_table(res)
    c = res["cost"]
    lines += ["", "## CPU cost per query (1 thread)", "",
              f"- CLIP: median {c['clip_latency']['median_ms']:.1f} ms (text encoder alone "
              f"{c['clip_text_encoder_latency']['median_ms']:.1f} ms); model {c['clip_model_mb']:.0f} MB "
              f"(text tower {c['clip_text_tower_params']:,} parameters); image index {c['clip_image_index_mb']:.1f} MB, "
              f"built offline at {c['clip_index_build_s_per_image'] * 1000:.0f} ms per image",
              f"- TF-IDF: median {c['tfidf_latency']['median_ms']:.2f} ms; index {c['tfidf_index_mb']:.2f} MB, "
              f"{c['tfidf_vocabulary']:,} character n-grams", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
