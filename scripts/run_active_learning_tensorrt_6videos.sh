#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
DATASET_NAME="active_learning(dataset=3000frame)7-17"

cd "${PROJECT_DIR}"

# Reuses CPU-prepared mixed-FP16 ONNX files, builds exact-GPU engines, and
# enforces PyTorch/TensorRT agreement at every min/opt/max profile point.
"${PYTHON}" scripts/build_active_learning_tensorrt.py

# All six sources share one flat frames/ directory. selection_manifest.csv
# beside it preserves the extraction reason for every JPEG.
"${PYTHON}" scripts/run_strongest_active_learning.py \
  --config configs/active_learning_tensorrt_6videos.yaml \
  --dataset-name "${DATASET_NAME}" \
  --stage all \
  "$@"
