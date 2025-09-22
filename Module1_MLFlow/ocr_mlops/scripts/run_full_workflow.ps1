# run_full_workflow.ps1
param(
    [double]$Alpha = 0.5,
    [int]$MaxIter = 200
)

# Activate virtual environment
. .\.venv\Scripts\Activate.ps1

# Convert $PWD to MLflow-friendly URI
$mlruns_uri = "file:///$((Get-Location).Path -replace '\\','/')/mlruns"
$mlartifacts_uri = "file:///$((Get-Location).Path -replace '\\','/')/mlartifacts"

# Set MLflow environment variables
$env:MLFLOW_TRACKING_URI = $mlruns_uri
$env:MLFLOW_ARTIFACT_URI = $mlartifacts_uri

# Ensure tracking and artifact directories exist
New-Item -ItemType Directory -Force -Path mlruns | Out-Null
New-Item -ItemType Directory -Force -Path mlartifacts | Out-Null

# Step 1: Train model
Write-Host "==> Training OCR model..."
$trainer_output = & python src\ocr\trainer.py --alpha $Alpha --max_iter $MaxIter
$run_id = $trainer_output.Split("`n")[-1].Trim()
$parts = $run_id.Split(":")
$exp_id = $parts[0]
$run_id = $parts[1]
$model_uri = $parts[2]

Write-Host "Training complete. Run ID: $run_id"

# Step 2: Register model
$model_name = "ocr_model"
Write-Host "==> Registering model as $model_name..."
& python src\ocr\register_model.py --run_id $run_id --model_name $model_name

# Set OCR_MODEL_URI directly to artifacts (avoids registry issues on Windows)
$env:OCR_MODEL_URI = "$mlruns_uri/$exp_id/models/$model_uri/artifacts/"
Write-Host "Model registered. Model URI: $env:OCR_MODEL_URI"


# Step 3: Serve model
Write-Host "==> Starting Flask server..."
& python src\ocr\predictor.py

