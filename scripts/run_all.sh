#!/usr/bin/env bash
# Run every experiment (RQ1 + RQ2) for all seeds, then aggregate the results.
# Usage:  bash scripts/run_all.sh
# Optional environment variables:
#   DATA_ROOT  folder with one sub-folder per class
#   CKPT_DIR   where to save model weights (default: checkpoints)
#   STEPS      optimisation steps per run (default: 500)
#   SEEDS      seeds to run (default: "0 1 2")
set -e
DATA_ROOT="${DATA_ROOT:-data/Garbage classification/Garbage classification}"
CKPT_DIR="${CKPT_DIR:-checkpoints}"
STEPS="${STEPS:-500}"
SEEDS="${SEEDS:-0 1 2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

run() {
  python src/train.py --data_root "$DATA_ROOT" --ckpt_dir "$CKPT_DIR" \
    --steps "$STEPS" $EXTRA_ARGS "$@"
}

# Fixed test set (25 images per class) + training pool
if [ ! -f splits/test.csv ]; then
  python src/make_splits.py --data_root "$DATA_ROOT"
fi

for SEED in $SEEDS; do
  # RQ1: training-set size (balanced classes)
  for N in 5 10 30 100; do
    run --run_name "rq1_n${N}_s${SEED}" --per_class "$N" --seed "$SEED"
  done
  # RQ2: class imbalance (100 images for other classes, fewer trash images)
  for T in 30 10 3; do
    run --run_name "rq2_t${T}_ce_s${SEED}"  --per_class 100 --trash "$T" --seed "$SEED"
    run --run_name "rq2_t${T}_wce_s${SEED}" --per_class 100 --trash "$T" --seed "$SEED" --weighted_loss
  done
done

python src/aggregate.py --runs_dir runs --out_dir results
