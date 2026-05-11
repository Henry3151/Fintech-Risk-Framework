"""
MLflowTracker — thin wrapper around mlflow for consistent experiment logging.
All training scripts use this instead of calling mlflow directly.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional

import mlflow


class MLflowTracker:
    """
    Centralised MLflow logging helper.

    Usage
    -----
    tracker = MLflowTracker(experiment_name="fraud-detection")
    with tracker.start_run(run_name="autoencoder-v3"):
        tracker.log_params({"lr": 1e-3, "epochs": 50})
        tracker.log_metrics({"pr_auc": 0.86})
        tracker.log_artifact("models/autoencoder.pt")
    """

    def __init__(
        self,
        experiment_name: str = "fraud-detection",
        tracking_uri: Optional[str] = None,
    ) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = None

    @contextmanager
    def start_run(self, run_name: Optional[str] = None):
        with mlflow.start_run(run_name=run_name) as run:
            self._run = run
            try:
                yield self
            finally:
                self._run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_figure(self, fig, artifact_file: str) -> None:
        """Log a matplotlib figure directly."""
        mlflow.log_figure(fig, artifact_file)

    @property
    def run_id(self) -> Optional[str]:
        return self._run.info.run_id if self._run else None
