#!/usr/bin/env bash
set -euo pipefail

# ===============================
# MLflow folders
# ===============================
export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
export MLFLOW_ARTIFACT_URI="$(pwd)/mlartifacts"

mkdir -p mlruns mlartifacts

# ===============================
# Step 1: Train the model
# ===============================
echo "==> Training OCR model..."
TRAIN_RUN_ID=$(python src/ocr/trainer.py --alpha 0.5 --max_iter 200)
echo "Training complete. Run ID: $TRAIN_RUN_ID"

# ===============================
# Step 2: Register the model
# ===============================
MODEL_NAME="ocr_model"
echo "==> Registering model as $MODEL_NAME..."
./scripts/register_model.sh "$TRAIN_RUN_ID" "$MODEL_NAME"

# ===============================
# Step 3: Serve the model for inference
# ===============================
echo "==> Starting Flask server for inference..."
python src/ocr/predictor.py
