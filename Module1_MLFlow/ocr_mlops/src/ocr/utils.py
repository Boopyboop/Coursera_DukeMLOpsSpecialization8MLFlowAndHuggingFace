"""Utilities for OCR project: image preprocessing and feature extraction."""

from PIL import Image
import numpy as np
import os
import shutil
from pathlib import Path
import mlflow


def image_to_feature_vector(image_path: str, size=(28, 28)) -> np.ndarray:
    """
    Convert an image to a flattened grayscale feature vector.
    This is intentionally simple for demo/test purposes.
    """
    img = Image.open(image_path).convert("L")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()

def clean_mlruns():
    """Delete the local mlruns directory if it exists."""
    if os.path.exists("mlruns"):
        shutil.rmtree("mlruns", ignore_errors=True)
        print("Removed existing mlruns/ directory")

def ensure_mlruns_dirs():
    """
    Ensure mlruns and mlruns/.trash exist for MLflow.
    """
    mlruns_path = Path(__file__).resolve().parents[2] / "mlruns"
    mlruns_path = Path("mlruns")
    mlruns_path.mkdir(exist_ok=True)
    trash_path = mlruns_path / ".trash"
    trash_path.mkdir(exist_ok=True)

def ensure_experiment(experiment_name: str):
    """
    Ensure the MLflow experiment exists and mlruns folder structure is valid.
    Returns the experiment ID.
    """
    # Ensure mlruns/.trash exists
    mlruns_path = Path("mlruns")
    mlruns_path.mkdir(exist_ok=True)
    (mlruns_path / ".trash").mkdir(exist_ok=True)

    # Check if experiment exists
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment '{experiment_name}' with ID {exp_id}")
    else:
        exp_id = exp.experiment_id
        print(f"Using existing experiment '{experiment_name}' with ID {exp_id}")
    return exp_id