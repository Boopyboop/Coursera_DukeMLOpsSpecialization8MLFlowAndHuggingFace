import tempfile
import os
import mlflow
from src.ocr.trainer import train_and_log
from src.ocr.predictor import load_model, predict_from_image_path
from PIL import Image
import numpy as np


def test_predictor_load_and_predict(tmp_path):
    mlruns_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")

    # train and get run id
    run_id = train_and_log(experiment_name="predictor_test", alpha=0.5, max_iter=50)
    assert run_id is not None

    # Find the saved model path using mlflow artifacts pattern
    # The trainer logged model under run artifacts: 'ocr_model'
    # Build local model path
    run_folder = next((mlruns_dir / "0").glob("*/"))  # crude but stable for this test
    model_path = f"file://{mlruns_dir}/0/{run_id}/artifacts/ocr_model"

    model = load_model(model_path)

    # create a small fake image to test prediction
    img_path = tmp_path / "img.png"
    arr = (np.random.rand(28, 28) * 255).astype("uint8")
    Image.fromarray(arr).save(img_path)

    preds = predict_from_image_path(model, str(img_path))
    assert isinstance(preds, list)
