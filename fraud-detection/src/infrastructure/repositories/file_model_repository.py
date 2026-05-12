"""
FileModelRepository — loads model artifacts from disk.
Concrete implementation of IModelRepository.

Artifacts expected at MODELS_DIR (configurable via env var):
  preprocessor.joblib
  autoencoder.pt
  autoencoder_metadata.json   ← {"threshold_p95": 0.8249, "pr_auc": 0.4603}
  classifier.joblib
  classifier_metadata.json    ← {"best_threshold": 0.9810, "pr_auc": 0.8665}
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import torch

from domain.exceptions import RepositoryError
from domain.interfaces.model_interfaces import IModelRepository
from infrastructure.models.autoencoder import FraudAutoencoder


class FileModelRepository(IModelRepository):
    """
    Loads pre-trained artifacts from a local directory.

    Parameters
    ----------
    models_dir : path to the models/ folder
                 defaults to MODELS_DIR env var or '../../models' relative to src/
    """

    def __init__(self, models_dir: str | Path | None = None) -> None:
        if models_dir is None:
            models_dir = os.getenv(
                "MODELS_DIR",
                str(Path(__file__).resolve().parent.parent.parent.parent / "models"),
            )
        self._dir = Path(models_dir)

    # ------------------------------------------------------------------
    # IModelRepository protocol
    # ------------------------------------------------------------------

    def load_preprocessor(self) -> Any:
        return self._joblib_load("preprocessor.joblib")

    def load_autoencoder(self) -> FraudAutoencoder:
        path = self._path("autoencoder.pt")
        try:
            model = FraudAutoencoder(input_dim=30)
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            return model
        except Exception as exc:
            raise RepositoryError(f"Failed to load autoencoder: {exc}") from exc

    def load_classifier(self) -> Any:
        return self._joblib_load("classifier.joblib")

    def get_autoencoder_threshold(self) -> float:
        meta = self._load_json("autoencoder_metadata.json")
        return float(meta["threshold_p95"])

    def get_classifier_threshold(self) -> float:
        meta = self._load_json("classifier_metadata.json")
        return float(meta["best_threshold"])

    def get_metadata(self) -> dict:
        ae_meta = self._load_json("autoencoder_metadata.json")
        clf_meta = self._load_json("classifier_metadata.json")
        return {
            "autoencoder": ae_meta,
            "classifier": clf_meta,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _path(self, filename: str) -> Path:
        return self._dir / filename

    def _joblib_load(self, filename: str) -> Any:
        path = self._path(filename)
        try:
            return joblib.load(path)
        except Exception as exc:
            raise RepositoryError(f"Failed to load {filename}: {exc}") from exc

    def _load_json(self, filename: str) -> dict:
        path = self._path(filename)
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            raise RepositoryError(f"Failed to read {filename}: {exc}") from exc
