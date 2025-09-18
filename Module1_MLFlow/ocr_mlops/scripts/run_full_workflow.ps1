# run_full_workflow.ps1
param(
    [double]$Alpha = 0.5,
    [int]$MaxIter = 200
)

# Activate venv
. .\.venv\Scripts\Activate.ps1

# Set MLflow environment variables
$env:MLFLOW_TRACKING_URI = "file://$PWD/mlruns"
$env:MLFLOW_ARTIFACT_URI = "$PWD/mlartifacts"

# Ensure directories exist
New-Item -ItemType Directory -Force -Path mlruns
New-Item -ItemType Directory -Force -Path mlartifacts

# Step 1: Train model
Write-Host "==> Training OCR model..."
$run_id = python src\ocr\trainer.py --alpha $Alpha --max_iter $MaxIter

Write-Host "Training complete. Run ID: $run_id"

# Step 2: Register model
$model_name = "ocr_model"
Write-Host "==> Registering model as $model_name..."
python src\ocr\register_model.py --run_id $run_id --model_name $model_name

# Step 3: Serve model
Write-Host "==> Starting Flask server..."
python src\ocr\predictor.py
