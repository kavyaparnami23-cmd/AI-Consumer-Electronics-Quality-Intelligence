"""
mlflow_tracker.py
──────────────────
Reusable MLflow experiment tracker.

Usage (context-manager style):
    with MLflowTracker("Classic-ML-Sensor-Failure", run_name="XGBoost-v1") as tracker:
        tracker.log_params({"n_estimators": 300, "max_depth": 6})
        tracker.log_metrics({"f1": 0.87, "roc_auc": 0.94})
        tracker.log_artifact("/path/to/report.json")
        tracker.log_model_sklearn(model, "xgboost_model")

    # Or direct calls (no context manager):
    tracker = MLflowTracker("experiment_name")
    tracker.start("run_name")
    ...
    tracker.end()
"""

import os
import json
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from typing import Any, Dict, Optional

from src.mlflow_utils.mlflow_config import MLFLOW_TRACKING_URI, COMMON_TAGS


class MLflowTracker:
    """
    Thin wrapper around mlflow that:
      - Sets up the tracking URI & experiment on init
      - Adds common project tags to every run
      - Provides typed log_* helpers
      - Supports both context-manager and manual start/end usage
    """

    def __init__(self, experiment_name: str, tags: Optional[Dict[str, str]] = None):
        self.experiment_name = experiment_name
        self.extra_tags = tags or {}
        self._active = False

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self, run_name: Optional[str] = None):
        """Start a new MLflow run. Returns self for chaining."""
        all_tags = {**COMMON_TAGS, **self.extra_tags}
        mlflow.start_run(run_name=run_name, tags=all_tags)
        self._active = True
        print(f"  [MLflow] Experiment: '{self.experiment_name}' | "
              f"Run ID: {mlflow.active_run().info.run_id}")
        return self

    def __exit__(self, *args):
        mlflow.end_run()
        self._active = False
        print(f"  [MLflow] Run ended → {MLFLOW_TRACKING_URI}")

    # ── Manual start/end ───────────────────────────────────────────────────────

    def start(self, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        all_tags = {**COMMON_TAGS, **self.extra_tags, **(tags or {})}
        mlflow.start_run(run_name=run_name, tags=all_tags)
        self._active = True
        run_id = mlflow.active_run().info.run_id
        print(f"  [MLflow] Started run '{run_name}' | ID: {run_id}")
        return run_id

    def end(self):
        if self._active:
            mlflow.end_run()
            self._active = False

    # ── Logging helpers ────────────────────────────────────────────────────────

    def log_params(self, params: Dict[str, Any]):
        """Log a dict of hyperparameters."""
        mlflow.log_params(params)

    def log_param(self, key: str, value: Any):
        mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log a dict of scalar metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact (JSON report, CSV, plots, etc.)."""
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)
        else:
            print(f"  [MLflow] WARNING: artifact not found: {local_path}")

    def log_artifacts_dir(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log an entire directory of artifacts."""
        if os.path.isdir(local_dir):
            mlflow.log_artifacts(local_dir, artifact_path)

    def log_tags(self, tags: Dict[str, str]):
        mlflow.set_tags(tags)

    # ── Model logging ──────────────────────────────────────────────────────────

    def log_model_sklearn(self, model, artifact_path: str, registered_name: Optional[str] = None):
        """Log and optionally register a scikit-learn / XGBoost / LightGBM model."""
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_name,
        )
        print(f"  [MLflow] Logged sklearn model → '{artifact_path}'" +
              (f" | Registered as '{registered_name}'" if registered_name else ""))

    def log_model_pytorch(self, model, artifact_path: str, registered_name: Optional[str] = None):
        """Log and optionally register a PyTorch model."""
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_name,
            serialization_format="pickle",
        )
        print(f"  [MLflow] Logged PyTorch model → '{artifact_path}'" +
              (f" | Registered as '{registered_name}'" if registered_name else ""))

    def log_json_dict(self, data: dict, filename: str):
        """Save a Python dict as a JSON artifact."""
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix=filename.replace(".json", "_")
        ) as f:
            json.dump(data, f, indent=4)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path=None)
        os.unlink(tmp_path)

    # ── Epoch-level logging (for DL training loops) ────────────────────────────

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, val_f1: float):
        """Log per-epoch metrics with step=epoch for MLflow chart view."""
        mlflow.log_metrics(
            {"train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1},
            step=epoch,
        )

    @staticmethod
    def get_run_url(run_id: str) -> str:
        """Return URL to view this run in MLflow UI."""
        return f"{MLFLOW_TRACKING_URI}/#/experiments/*/runs/{run_id}"
