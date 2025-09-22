import os
import tempfile
import mlflow
from src.ocr.trainer import train_and_log


def test_train_and_log_creates_run(tmp_path):
    # set tracking uri to a temp directory so test is isolated
    tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file://{tracking_dir}")

    run_id = train_and_log(experiment_name="test_experiment", alpha=0.1, max_iter=50)
    assert run_id is not None
    # ensure mlruns directory now exists and stores run
    assert tracking_dir.exists()
