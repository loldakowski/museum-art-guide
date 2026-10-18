#!/usr/bin/env python
"""Step 6 - One table and one figure for the director: test quality vs CPU cost of every candidate.

Reads the JSON files written by steps 1-5 (whatever exists) and writes
results/summary.md and results/tradeoff.png.
    python scripts/06_summary.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from museum_guide.common import load_json, save_json
from museum_guide.plots import INK, MUTED, PALETTE, _save, plt


INPUTS = {"photo": "phone photo", "small": "128 px image"}


def acc_row(name, family, m, ci=None, std=None, latency_key=None, train_time=None, note=""):
    lo, hi = ci if ci else m.get("accuracy_ci95", (np.nan, np.nan))
    return {"model": name, "family": family, "test_acc": m["accuracy"], "ci_lo": lo, "ci_hi": hi,
            "seed_std": std, "macro_f1": m["macro_f1"], "latency_key": latency_key, "train_time_s": train_time,
            "note": note}


def search_lines(f: Path) -> str:
    """Free-text search (step 7): mean precision@10 per system, CPU cost per query."""
    if not f.exists():
        return ""
    s = load_json(f)
    c = s["cost"]

    def mean(rows, key):  # the brief's rewordings of a query ('paraphrase_of') are not counted twice
        rows = [r for r in rows or [] if not r.get("paraphrase_of")]
        return f"{np.mean([r[key] for r in rows]):.3f}" if rows else "not judged yet"
    rows = [("CLIP " + s["clip_model"], "clip", f"{c['clip_latency']['median_ms']:.1f}",
             f"{c['clip_model_mb']:.0f} + index {c['clip_image_index_mb']:.1f}"),
            ("TF-IDF on metadata", "tfidf", f"{c['tfidf_latency']['median_ms']:.2f}", f"{c['tfidf_index_mb']:.2f}")]
    out = ["", "## Free-text search (see 07_search/summary.md)", "",
           "| system | P@10 metadata queries (TF-IDF: upper bound) | P@10 content queries (judged) | "
           f"latency per query (ms, {c['threads']} thread) | size (MB) |", "|---|---|---|---|---|"]
    out += [f"| {name} | {mean(s['auto'], key + '_P@10')} | {mean(s['content'], key + '_P@10')} | {lat} | {size} |"
            for name, key, lat, size in rows]
    if s["content"]:
        out.append(f"| random | - | {mean(s['content'], 'random_P@10')} | - | - |")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--threads", type=int, default=1, help="benchmark thread count shown in the table")
    p.add_argument("--input", default="photo", choices=["photo", "small"], help="benchmark input shown in the table")
    a = p.parse_args()
    R = Path(a.results)
    rows = []
    for fs in ("pixels32", "handcrafted"):
        f = R / "01_baseline" / f"{fs}.json"
        if f.exists():
            m = load_json(f)
            rows.append(acc_row(f"LogReg - {fs}", "classical", m, latency_key=f"logreg_{fs}",
                                train_time=m["fit_time_s"]))
    for fs in ("pixels32", "pixels64", "handcrafted"):
        f = R / "02_mlp" / fs / "summary.json"
        if f.exists():
            s = load_json(f)
            rows.append(acc_row(f"MLP - {fs}", "MLP (from scratch)", s["metrics_seed0"],
                                std=s["test_accuracy"]["std"], latency_key=f"mlp_{fs}" if fs != "pixels64" else None,
                                train_time=s["train_time_s"]["mean"],
                                note=f"{len(s['test_accuracy']['values'])} seeds: {s['test_accuracy']['mean']:.3f} +/- {s['test_accuracy']['std']:.3f}"))
    f = R / "04_resnet" / "frozen" / "summary.json"
    if f.exists():
        s = load_json(f)
        rows.append(acc_row("ResNet-18 frozen + linear head", "CNN (pretrained)", s["metrics"],
                            latency_key="resnet18_128px", train_time=s["history_summary"]["total_time_s"]))
        rows.append(acc_row("ResNet-18 frozen + LogReg", "CNN features + classical",
                            s["logreg_on_embeddings"]["metrics"], latency_key="resnet18_frozen_logreg"))
    f = R / "04_resnet" / "finetune" / "summary.json"
    if f.exists():
        s = load_json(f)
        rows.append(acc_row("ResNet-18 fine-tuned", "CNN (pretrained)", s["metrics"], latency_key="resnet18_128px",
                            train_time=s["history_summary"]["total_time_s"]))

    bench = {}
    fb = R / "05_benchmark" / "benchmark.json"
    if fb.exists():
        for b in load_json(fb)["results"]:
            if b["threads"] == a.threads and b.get("input", "small") == a.input and \
                    b.get("exif_transpose", a.input == "photo") == (a.input == "photo"):  # photos are turned upright
                bench[(b["model"], b.get("decode", "full"))] = b
    for r in rows:
        b = bench.get((r["latency_key"], "full"), {})
        r["latency_ms"], r["size_mb"], r["params"] = b.get("median_ms"), b.get("size_mb"), b.get("params")
        r["latency_draft_ms"] = bench.get((r["latency_key"], "draft"), {}).get("median_ms")

    def f_(v, spec):
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)
    lines = [f"| model | test acc [95% CI] | macro-F1 | latency (ms, {a.threads} thread) | latency, draft decoding (ms) "
             "| size (MB) | params | train time (s) | note |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['test_acc']:.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}] | "
                     f"{r['macro_f1']:.3f} | {f_(r['latency_ms'], '.1f')} | {f_(r['latency_draft_ms'], '.1f')} | "
                     f"{f_(r['size_mb'], '.1f')} | "
                     f"{f_(r['params'], ',')} | {f_(r['train_time_s'], '.0f')} | {r['note']} |")
    # Which commits produced the inputs (a table mixing several commits must be re-run before it is reported)
    shas = {}
    for f in sorted(R.glob("0[0-5]_*/**/*.json")):
        d = load_json(f)
        prov = d.get("provenance") if isinstance(d, dict) else None
        if prov:
            key = f"{prov['git_sha'] or 'unknown'}{' (dirty)' if prov['git_dirty'] else ''}"
            shas.setdefault(key, []).append(str(f.relative_to(R)))
    sources = ("\n".join(f"- `{k}`: {', '.join(v)}" for k, v in shas.items()) if shas
               else "- no provenance recorded in the input files")
    search = search_lines(R / "07_search" / "summary.json")
    (R / "summary.md").write_text("# Results summary (test set, artist-grouped split)\n\n" + "\n".join(lines) +
                                  f"\n\nLatency = median end-to-end time for one {INPUTS[a.input]} "
                                  "on CPU, decoding included (see 05_benchmark). Draft decoding = im.draft() + "
                                  "ImageOps.fit, the preprocessing that built the training images.\n"
                                  + search +
                                  "\nCode that produced these numbers (git commit):\n" + sources + "\n",
                                  encoding="utf-8")
    save_json(rows, R / "summary.json")
    print("\n".join(lines))

    pts = [r for r in rows if r["latency_ms"]]
    if pts:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        # labels: placed to the right of the points, pushed apart vertically when points are close
        pts = sorted(pts, key=lambda r: r["test_acc"])
        placed = []
        for r in pts:
            y = r["test_acc"]
            for (x0, y0) in placed:
                if 0.5 < r["latency_ms"] / x0 < 2.0 and abs(y - y0) < 0.045:
                    y = y0 + 0.045
            placed.append((r["latency_ms"], y))
            ax.errorbar(r["latency_ms"], r["test_acc"], yerr=[[r["test_acc"] - r["ci_lo"]], [r["ci_hi"] - r["test_acc"]]],
                        fmt="o", color=PALETTE[0], ms=7, capsize=3, lw=1.5)
            # a neighbour just to the right -> write this label on the left instead
            left = any(1.0 < o["latency_ms"] / r["latency_ms"] < 2.5 for o in pts)
            ax.annotate(f"{r['model']}  {r['test_acc']:.0%}", (r["latency_ms"], r["test_acc"]),
                        xytext=(r["latency_ms"] / 1.12 if left else r["latency_ms"] * 1.12, y), textcoords="data",
                        fontsize=8, color=INK, va="center", ha="right" if left else "left",
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
        ax.set_xlim(min(r["latency_ms"] for r in pts) / 8, max(r["latency_ms"] for r in pts) * 12)
        k = len(load_json(R / "00_explore" / "eda.json")["classes"]) if (R / "00_explore" / "eda.json").exists() else 5
        ax.axhline(1 / k, color=MUTED, ls=":", lw=1)
        ax.text(min(r["latency_ms"] for r in pts), 1 / k + 0.01, "chance", fontsize=7,
                color=MUTED)
        ax.set(xscale="log", xlabel=f"CPU latency per photo, ms ({a.threads} thread, log scale)",
               ylabel="test accuracy (95% CI)", ylim=(0, 1), title="Quality vs cost on a kiosk CPU")
        _save(fig, R / "tradeoff.png")
        print(f"-> {R}/summary.md, {R}/tradeoff.png")


if __name__ == "__main__":
    main()
