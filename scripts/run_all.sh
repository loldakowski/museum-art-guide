#!/usr/bin/env bash
# Run every experiment in order. Usage:
#   bash scripts/run_all.sh                                  # real data, results/ (report numbers)
#   DATA=data/fake SIZE=64 RESULTS=/tmp/r QUICK=1 bash scripts/run_all.sh   # smoke test, minutes
set -euo pipefail
DATA=${DATA:-data/wikiart_style5}
RESULTS=${RESULTS:-results}
SIZE=${SIZE:-128}
if [ "${QUICK:-0}" = "1" ]; then
  MLP_ARGS="--epochs 15 --seeds 0 1 --no-sweep"; DIAG_ARGS="--epochs 15"
  RES_ARGS="--epochs-frozen 20 --epochs-finetune 2 --no-pretrained --img-size $SIZE"; BENCH_ARGS="--runs 20 --threads 1 --no-clip --no-pretrained"
  DUP_ARGS="--no-pretrained --img-size $SIZE"
else
  MLP_ARGS=""; DIAG_ARGS=""; RES_ARGS=""; BENCH_ARGS=""; DUP_ARGS=""
fi
python scripts/00_explore.py     --data-dir "$DATA" --size "$SIZE" --out "$RESULTS/00_explore"
python scripts/find_duplicates.py --data-dir "$DATA" --out "$RESULTS/00_explore" $DUP_ARGS
python scripts/01_baseline.py    --data-dir "$DATA" --size "$SIZE" --out "$RESULTS/01_baseline"
python scripts/01_baseline.py    --data-dir "$DATA" --size "$SIZE" --out "$RESULTS/01_baseline" --split random
python scripts/02_train_mlp.py   --data-dir "$DATA" --size "$SIZE" --out "$RESULTS/02_mlp" $MLP_ARGS
python scripts/02_train_mlp.py   --data-dir "$DATA" --size "$SIZE" --out "$RESULTS/02_mlp" --features handcrafted $MLP_ARGS
python scripts/03_diagnose_mlp.py --data-dir "$DATA" --size "$SIZE" --out "$RESULTS/03_diagnosis" \
       --reference "$RESULTS/02_mlp/pixels32/summary.json" $DIAG_ARGS
python scripts/04_train_resnet.py --data-dir "$DATA" --size "$SIZE" --out "$RESULTS/04_resnet" $RES_ARGS
python scripts/05_benchmark_cpu.py --data-dir "$DATA" --out "$RESULTS/05_benchmark" $BENCH_ARGS
# Step 7 (free-text search) indexes the 224 px copy built by build_dataset.sh. Skipped in QUICK mode: CLIP
# weights are a 605 MB download.
SEARCH_IMAGES=${SEARCH_IMAGES:-${DATA}_224}
if [ "${QUICK:-0}" != "1" ] && [ -d "$SEARCH_IMAGES" ]; then
  python scripts/07_text_search.py --data-dir "$DATA" --image-dir "$SEARCH_IMAGES" --out "$RESULTS/07_search"
else
  echo "skipping step 7 (QUICK mode or no $SEARCH_IMAGES)"
fi
python scripts/06_summary.py --results "$RESULTS"
# Step 8 (cost of unseen painters) needs the named-artist subset (bash scripts/build_named_dataset.sh).
# Skipped in QUICK mode: 10 trainings of every model, about 7 min even on synthetic data.
NAMED_DATA=${NAMED_DATA:-data/wikiart_named5}
if [ "${QUICK:-0}" != "1" ] && [ -d "$NAMED_DATA" ]; then
  python scripts/08_unseen_painters.py --data-dir "$NAMED_DATA" --results "$RESULTS" --out "$RESULTS/08_unseen_painters"
else
  echo "skipping step 8 (QUICK mode or no $NAMED_DATA)"
fi
