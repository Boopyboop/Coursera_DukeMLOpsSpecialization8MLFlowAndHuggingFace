# README.md (professional draft)

````markdown
# OCR MLOps Project with MLflow

This project demonstrates an end-to-end MLOps workflow for an OCR (Optical Character Recognition) model using **MLflow**.  
It includes training, experiment tracking, model registration, and serving via a Flask API.

---

## Prerequisites

- Python 3.10+
- Virtual environment (recommended)
- PowerShell (Windows) or Bash (Linux/Mac)

---

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
````

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Running the Full Workflow

To train the model, register it, and start the API server in one command:

```powershell
.\scripts\run_full_workflow.ps1 -Alpha 0.5 -MaxIter 200
```

### What this does:

1. Trains the OCR model with MLflow experiment tracking
2. Registers the trained model in the MLflow Model Registry
3. Starts a Flask API server to serve real-time predictions

---

## Project Structure

```
ocr_mlops/
├── scripts/
│   ├── run_full_workflow.ps1   # PowerShell script for Windows
│   └── run_full_workflow.sh    # Bash script for Linux/Mac
├── src/
│   └── ocr/
│       ├── trainer.py          # Model training
│       ├── predictor.py        # Flask serving app
│       ├── register_model.py   # Model registration
│       └── utils.py            # Utilities for MLflow
├── requirements.txt
├── Dockerfile.train
├── Dockerfile.serve
└── README.md
```

---

## Running Tests

Run unit tests with:

```powershell
pytest tests/ -v
```

---

## API Usage

Once the Flask server is running, you can send requests:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -Body @{ text="Hello OCR" }
```

---

## ⚠️ Notes for Windows Users

* MLflow’s `file://` tracking URIs can be problematic on Windows.
  This project defaults to **local directories (`mlruns/`, `mlartifacts/`)** for simplicity.
* Ensure `OCR_MODEL_URI` is set if you want to serve a specific model version:

```powershell
$env:OCR_MODEL_URI = "models:/ocr_model/1"
```

```
