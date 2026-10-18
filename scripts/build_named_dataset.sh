#!/usr/bin/env bash
# Build the NAMED-ARTIST subset used only to measure the cost of unseen painters (scripts/run_unseen_painters.sh).
# Every image has a known painter (no "Unknown Artist"), at most 50 per painter, so an artist-disjoint split is
# real. The 5 styles are the WikiArt styles with the most named painters; Pop Art and Ukiyo-e have only 1 and 2.
# ~3.7 GB streamed (8 shards), ~8 MB written. The images land in data/ which is git-ignored: never commit them.
set -euo pipefail

CLASSES="Realism Impressionism Post_Impressionism Romanticism Symbolism"
PER_CLASS=${PER_CLASS:-300}
OUT=${OUT:-data/wikiart_named5}

python scripts/make_wikiart_subset.py --label style --classes $CLASSES --named-only --max-per-artist 50 \
    --per-class "$PER_CLASS" --size 128 --out "$OUT" --dry-run
python scripts/make_wikiart_subset.py --label style --classes $CLASSES --named-only --max-per-artist 50 \
    --per-class "$PER_CLASS" --size 128 --out "$OUT"

echo "Done. Exact command and counts recorded in $OUT/subset_info.json"
