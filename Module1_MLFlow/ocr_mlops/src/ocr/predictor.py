"""Load a logged MLflow model and provide a simple Flask API for inference."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import numpy as np
from flask import Flask, request, jsonify
from utils import image_to_feature_vector

MODEL_URI_ENV = "OCR_MODEL_URI"  # e.g., "models:/ocr_model/1" or path to local model folder


def load_model(model_uri: str | None = None) -> Any:
    """
    Load a model from MLflow using either:
    - An explicit model_uri argument
    - The OCR_MODEL_URI environment variable
    
    Supports both registry URIs (models:/...) and file URIs (file:///...).
    """
    if model_uri is None:
        model_uri = os.environ.get(MODEL_URI_ENV)

    if not model_uri:
        raise RuntimeError(
            f"No model URI provided. Set {MODEL_URI_ENV} env var or pass model_uri."
        )

    # Normalize model path for MLflow
    if model_uri.startswith("file:///"):
        # Keep it as file:///, MLflow understands this
        pass
    elif Path(model_uri).exists():
        # Convert plain Windows path -> proper URI
        model_uri = Path(model_uri).absolute().as_uri()

    try:
        # Works for both registry and file URIs
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from URI: {model_uri}\n"
            f"Error: {str(e)}"
        ) from e


def predict_from_image_path(model, image_path: str):
    feat = image_to_feature_vector(image_path)
    arr = np.expand_dims(feat, axis=0)
    preds = model.predict(arr)
    return preds.tolist()


def create_app(model_uri: str | None = None):
    app = Flask(__name__)
    model = load_model(model_uri)

    @app.route("/health", methods=["GET"])
    def health():
        return "ok"

    @app.route("/predict", methods=["POST"])
    def predict():
        # Expect JSON with {"image_path": "/path/to/image.png"} for simplicity
        payload = request.get_json()
        if not payload or "image_path" not in payload:
            return jsonify({"error": "image_path required in json body"}), 400
        image_path = payload["image_path"]
        try:
            preds = predict_from_image_path(model, image_path)
            return jsonify({"predictions": preds})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
