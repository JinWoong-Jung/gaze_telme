#!/usr/bin/env bash
# Phase 2-3: Video → Gaze pipeline
# Run from project root: bash scripts/run_phase2.sh [train|dev|test|all]
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)

SPLIT="${1:-all}"
ENV="gaze_telme"
LOG_DIR="$ROOT/logs/phase2"
mkdir -p "$LOG_DIR"

run_split() {
  local S="$1"
  echo "====== [$S] Phase 2.1: extract clips ======"
  conda run -n $ENV --no-capture-output python -u pipeline/extract_clips.py \
    --split "$S" --symlink 2>&1 | tee "$LOG_DIR/extract_clips_${S}.log"

  echo "====== [$S] Phase 2.2: sample frames ======"
  conda run -n $ENV --no-capture-output python -u pipeline/sample_frames.py \
    --split "$S" --fps 6 --max-frames 32 --num-workers 4 \
    2>&1 | tee "$LOG_DIR/sample_frames_${S}.log"

  echo "====== [$S] Phase 2.3: detect faces ======"
  conda run -n $ENV --no-capture-output python -u pipeline/detect_faces.py \
    --split "$S" --resume 2>&1 | tee "$LOG_DIR/detect_faces_${S}.log"

  echo "====== [$S] Phase 2.4: gaze inference ======"
  conda run -n $ENV --no-capture-output python -u pipeline/gaze_infer.py \
    --split "$S" --resume --batch-size 16 \
    2>&1 | tee "$LOG_DIR/gaze_infer_${S}.log"

  echo "====== [$S] Phase 3: build gaze features ======"
  conda run -n $ENV --no-capture-output python -u pipeline/build_features.py \
    --split "$S" 2>&1 | tee "$LOG_DIR/build_features_${S}.log"
}

if [ "$SPLIT" = "all" ]; then
  run_split train
  run_split dev
  run_split test
else
  run_split "$SPLIT"
fi

echo "=== Phase 2-3 Done. Features in features/gaze/ ==="
ls -lh features/gaze/
