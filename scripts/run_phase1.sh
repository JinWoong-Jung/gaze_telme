#!/usr/bin/env bash
# Phase 1: TelME Baseline Reproduction
# Run from project root: bash scripts/run_phase1_baseline.sh

set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)

conda_env="gaze_telme"
LOG_DIR="$ROOT/logs/phase1"
mkdir -p "$LOG_DIR"

echo "=== Phase 1: TelME Baseline ==="
echo "Device: $(conda run -n $conda_env python -c 'import torch; d = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"; print(d)')"

export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "[1/3] Training Teacher (RoBERTa-large)..."
conda run -n $conda_env --no-capture-output python -u TelME/MELD/teacher.py 2>&1 | tee "$LOG_DIR/teacher.log"

echo "[2/3] Training Students (Data2Vec-Audio + TimeSformer)..."
conda run -n $conda_env --no-capture-output python -u TelME/MELD/student.py 2>&1 | tee "$LOG_DIR/student.log"

echo "[3/3] Training Fusion (ASF)..."
conda run -n $conda_env --no-capture-output python -u TelME/MELD/fusion.py 2>&1 | tee "$LOG_DIR/fusion.log"

echo "=== Phase 1 Done. Checkpoints in models/checkpoints/ ==="
ls -lh models/checkpoints/
