#!/usr/bin/env bash
# Phase 5-7: Train VideoGazeStudent + ASF Fusion
# Run from project root: bash scripts/run_phase5.sh [fusion_type] [lambda]
# Examples:
#   bash scripts/run_phase5.sh gated 0.3     (default)
#   bash scripts/run_phase5.sh add 0.5
#   bash scripts/run_phase5.sh concat_proj 0.0
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)

ENV="gaze_telme"
FUSION="${1:-gated}"
LAMBDA="${2:-0.3}"

LOG_P5="$ROOT/logs/phase5"
LOG_P7="$ROOT/logs/phase7"
mkdir -p "$LOG_P5" "$LOG_P7"

echo "=== Phase 6: Train VideoGazeStudent (fusion=$FUSION  λ=$LAMBDA) ==="
conda run -n $ENV --no-capture-output python -u \
  train/train_student_gaze.py \
  --config configs/phase5_video_gaze.yaml \
  --fusion-type "$FUSION" \
  --fusion-lambda "$LAMBDA" \
  --no-wandb \
  2>&1 | tee "$LOG_P5/train_student_gaze_${FUSION}_${LAMBDA}.log"

echo "=== Phase 7: Train ASF Fusion (seeds 42, 7, 2024) ==="
conda run -n $ENV --no-capture-output python -u \
  train/train_fusion.py \
  --config configs/phase7_fusion.yaml \
  --no-wandb \
  2>&1 | tee "$LOG_P7/train_fusion_gaze_${FUSION}.log"

echo "=== Phase 5-7 Done. Checkpoints in models/checkpoints/ ==="
ls -lh models/checkpoints/
