"""
Trainer module for OCR demo.
- Produces a tiny demo model (scikit-learn) trained on synthetic data or images
- Logs parameters, metrics, and the model to MLflow
- Uses mlflow.sklearn.log_model with name and input_example to avoid warnings
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

from utils import image_to_feature_vector, clean_mlruns, ensure_mlruns_dirs, ensure_experiment


def build_synthetic_dataset(n_samples: int = 500, n_features: int = 28 * 28, seed: int = 42):
    """
    Build a synthetic dataset to simulate OCR classification (digits/characters).
    Replace this with real labeled image dataset for production.
    """
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features)
    # simple 3-class synthetic labels for demo
    y = rng.randint(0, 3, size=(n_samples,))
    return X, y


def train_and_log(
    experiment_name: str = "ocr_demo",
    alpha: float = 1.0,
    max_iter: int = 200,
    use_images_dir: str | None = None,
):
    """
    Train a small logistic regression model and log run to MLflow.
    If use_images_dir is provided, attempt to read images (experimental).
    """
    # Ensure mlruns and mlruns/.trash exist
    ensure_mlruns_dirs()

    # Ensure experiment exists before MLflow logging
    exp_id = ensure_experiment(experiment_name)
    #mlflow.set_experiment(experiment_name = exp_id)

    # prepare dataset
    if use_images_dir:
        # convert images in directory to features and dummy labels
        paths = list(Path(use_images_dir).glob("*.*"))
        if len(paths) == 0:
            raise RuntimeError("No images found in given directory")
        X = [image_to_feature_vector(str(p)) for p in paths]
        # for demo: labels derived from filename prefix (0,1,2) if present else random
        y = []
        for p in paths:
            name = p.stem
            try:
                y.append(int(name.split("_")[0]) % 3)
            except Exception:
                y.append(0)
        X = np.stack(X)
        y = np.array(y)
    else:
        X, y = build_synthetic_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(experiment_id=exp_id) as run:
        # log parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("max_iter", max_iter)

        model = LogisticRegression(C=1.0 / alpha, max_iter=max_iter, solver="lbfgs", multi_class="auto")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", float(acc))

        # Save a small input example (first rows of X_train) to help MLflow infer signature
        input_example = pd.DataFrame(X_train[:5])

        # Log the sklearn model using 'name' to avoid artifact_path deprecation warning
        mlflow.sklearn.log_model(
            sk_model=model,
            name="ocr_model",
            input_example=input_example
        )

        # Optional: save a tiny CSV of test results as an artifact
        tmp = tempfile.mkdtemp()
        results_df = pd.DataFrame({"y_true": y_test, "y_pred": preds})
        results_path = os.path.join(tmp, "results.csv")
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path, artifact_path="results")

        print(f"Logged run with accuracy: {acc:.4f}")
        # return run info for tests or scripts
        return run.info.run_id
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="ocr_demo")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--images_dir", type=str, default=None)
    args = parser.parse_args()

    try:
        run_id = train_and_log(args.experiment, args.alpha, args.max_iter, args.images_dir)
    except Exception as e:
        print(f"Error occurred: {e}. Cleaning mlruns/ and retrying once...")
        clean_mlruns()
        run_id = train_and_log(args.experiment, args.alpha, args.max_iter, args.images_dir)

    print(f"Final run_id: {run_id}")
