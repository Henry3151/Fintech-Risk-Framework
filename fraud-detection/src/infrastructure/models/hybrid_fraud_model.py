"""
HybridFraudModel — concrete IFraudModel implementation.

Pipeline (mirrors the original train_classifier.py logic):
  1. StandardScaler on Amount and Time
  2. Autoencoder → reconstruction_error (used as extra feature)
  3. XGBoost on [scaled_features + reconstruction_error]
  4. Threshold decision from classifier_metadata.json
"""
from __future__ import annotations

import time
from typing import List

import numpy as np
import torch

from domain.entities.transaction import Transaction, FraudPrediction
from domain.exceptions import InferenceError, ModelNotLoadedError
from domain.interfaces.model_interfaces import IFraudModel, IModelRepository


class HybridFraudModel(IFraudModel):
    """
    Wraps the fitted autoencoder + XGBoost pipeline.
    Loaded lazily on first call; thread-safe via the GIL for read-only inference.
    """

    def __init__(self, repository: IModelRepository) -> None:
        self._repository = repository
        self._preprocessor = None
        self._autoencoder = None
        self._classifier = None
        self._ae_threshold: float | None = None
        self._clf_threshold: float | None = None
        self._device = torch.device("cpu")

    # ------------------------------------------------------------------
    # IFraudModel protocol
    # ------------------------------------------------------------------

    def predict(self, transaction: Transaction) -> FraudPrediction:
        self._ensure_loaded()
        start = time.perf_counter()
        try:
            features_2d = transaction.to_numpy().reshape(1, -1)
            scaled, rec_error = self._preprocess_and_encode(features_2d)
            enriched = np.hstack([scaled, [[rec_error]]])
            proba = float(self._classifier.predict_proba(enriched)[0, 1])
            is_fraud = proba >= self._clf_threshold
        except Exception as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000
        return FraudPrediction(
            transaction_id=transaction.transaction_id,
            fraud_probability=proba,
            is_fraud=is_fraud,
            reconstruction_error=rec_error,
            latency_ms=round(latency_ms, 2),
        )

    def predict_batch(self, transactions: List[Transaction]) -> List[FraudPrediction]:
        self._ensure_loaded()
        start = time.perf_counter()
        try:
            matrix = np.vstack([t.to_numpy() for t in transactions])
            scaled, rec_errors = self._preprocess_and_encode_batch(matrix)
            enriched = np.hstack([scaled, rec_errors.reshape(-1, 1)])
            probas = self._classifier.predict_proba(enriched)[:, 1]
        except Exception as exc:
            raise InferenceError(f"Batch inference failed: {exc}") from exc

        total_ms = (time.perf_counter() - start) * 1000
        per_ms = total_ms / len(transactions)

        return [
            FraudPrediction(
                transaction_id=t.transaction_id,
                fraud_probability=float(p),
                is_fraud=float(p) >= self._clf_threshold,
                reconstruction_error=float(r),
                latency_ms=round(per_ms, 2),
            )
            for t, p, r in zip(transactions, probas, rec_errors)
        ]

    def get_reconstruction_error(self, features: np.ndarray) -> float:
        self._ensure_loaded()
        scaled = self._preprocessor.transform(features.reshape(1, -1))
        tensor = torch.FloatTensor(scaled).to(self._device)
        with torch.no_grad():
            reconstructed = self._autoencoder(tensor).cpu().numpy()
        return float(np.mean((scaled - reconstructed) ** 2))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._preprocessor is None:
            self._load()

    def _load(self) -> None:
        try:
            self._preprocessor = self._repository.load_preprocessor()
            self._autoencoder = self._repository.load_autoencoder()
            self._classifier = self._repository.load_classifier()
            self._ae_threshold = self._repository.get_autoencoder_threshold()
            self._clf_threshold = self._repository.get_classifier_threshold()
        except Exception as exc:
            raise ModelNotLoadedError(f"Could not load model artifacts: {exc}") from exc

    def _preprocess_and_encode(self, features_2d: np.ndarray):
        """Scale + autoencoder for a single row. Returns (scaled, rec_error)."""
        scaled = self._preprocessor.transform(features_2d)
        tensor = torch.FloatTensor(scaled).to(self._device)
        with torch.no_grad():
            reconstructed = self._autoencoder(tensor).cpu().numpy()
        rec_error = float(np.mean((scaled - reconstructed) ** 2))
        return scaled, rec_error

    def _preprocess_and_encode_batch(self, matrix: np.ndarray):
        """Scale + autoencoder for a batch. Returns (scaled_matrix, rec_errors)."""
        scaled = self._preprocessor.transform(matrix)
        tensor = torch.FloatTensor(scaled).to(self._device)
        with torch.no_grad():
            reconstructed = self._autoencoder(tensor).cpu().numpy()
        rec_errors = np.mean((scaled - reconstructed) ** 2, axis=1)
        return scaled, rec_errors
