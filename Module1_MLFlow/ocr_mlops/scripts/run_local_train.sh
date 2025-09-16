#!/usr/bin/env bash
set -euo pipefail

# Local MLflow folder-based tracking store and artifacts
export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
export MLFLOW_ARTIFACT_URI="$(pwd)/mlartifacts"  # optional

# ensure directories exist
mkdir -p mlruns mlartifacts

# Run trainer (params can be passed)
python src/ocr/trainer.py --alpha 0.5 --max_iter 200
