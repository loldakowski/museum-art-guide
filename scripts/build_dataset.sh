#!/usr/bin/env bash
# Build the WikiArt subset used in every experiment of this repository.
# Run from the repository root (Colab or local). ~4 GB streamed, ~8 MB written, 5-40 min.
# The images land in data/ which is git-ignored: never commit them (licence unknown).
set -euo pipefail

CLASSES="Impressionism Baroque Cubism Ukiyo_e Pop_Art"   # 5 visually contrasted styles
PER_CLASS=${PER_CLASS:-300}
OUT=${OUT:-data/wikiart_style5}

# 1) Plan only (how many shards / GB) - cheap, always do it first
python scripts/make_wikiart_subset.py --label style --classes $CLASSES \
    --per-class "$PER_CLASS" --size 128 --out "$OUT" --dry-run

# 2) Download (re-running resumes)
python scripts/make_wikiart_subset.py --label style --classes $CLASSES \
    --per-class "$PER_CLASS" --size 128 --out "$OUT"

echo "Done. Exact command and counts recorded in $OUT/subset_info.json"
