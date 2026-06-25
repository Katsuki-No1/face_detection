#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/dino_phase1_feature_analysis.yaml}"
MAX_IMAGES="${MAX_IMAGES:-50}"

.venv/bin/python -m src.phase1_dino.run_phase1 \
  --config "${CONFIG_PATH}" \
  --max-images "${MAX_IMAGES}"
