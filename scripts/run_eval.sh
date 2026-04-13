#!/usr/bin/env bash
# Phase 8: Full evaluation (A/B/C/D conditions)
# Run from project root: bash scripts/run_eval.sh
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)

ENV="gaze_telme"
LOG_DIR="$ROOT/logs/eval"
mkdir -p "$LOG_DIR"

echo "=== Phase 8: Evaluate all conditions ==="
conda run -n $ENV --no-capture-output python -u \
  eval/eval_all.py --out results/ \
  2>&1 | tee "$LOG_DIR/eval_all.log"

echo ""
echo "=== Results ==="
cat results/table_main.md
