#!/usr/bin/env bash
# Build the WikiArt subset used in every experiment of this repository.
# Run from the repository root (Colab or local). ~8.4 GB streamed (two image sizes), ~40 MB written, 5-40 min.
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

# 3) The same images at 224 px for the free-text search only (CLIP's native input size; same plan, same ids)
python scripts/make_wikiart_subset.py --label style --classes $CLASSES \
    --per-class "$PER_CLASS" --size 224 --out "${OUT}_224"

echo "Done. Exact command and counts recorded in $OUT/subset_info.json and ${OUT}_224/subset_info.json"
