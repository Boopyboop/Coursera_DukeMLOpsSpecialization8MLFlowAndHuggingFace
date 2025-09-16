#!/usr/bin/env bash
set -euo pipefail

# Usage:
# MLFLOW_TRACKING_URI=file://$PWD/mlruns ./scripts/register_model.sh <run_id> <model_name>
# e.g. ./scripts/register_model.sh 12345-... ocr_model_demo

MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file://$(pwd)/mlruns}
RUN_ID=${1:-}
MODEL_NAME=${2:-ocr_model_demo}

if [ -z "$RUN_ID" ]; then
  echo "Usage: $0 <run_id> <model_name>"
  exit 1
fi

# Model artifact path in run (we logged name=ocr_model)
MODEL_ARTIFACT_URI="runs:/${RUN_ID}/ocr_model"

echo "Registering model $MODEL_NAME from $MODEL_ARTIFACT_URI"
mlflow models prepare-docker -m "$MODEL_ARTIFACT_URI" -n "$MODEL_NAME"
# Alternatively register model in registry (if a tracking server with registry is used)
# mlflow.register_model -m "$MODEL_ARTIFACT_URI" -n "$MODEL_NAME"
