"""
Preprocessor — wraps the StandardScaler used during training.

During TRAINING: fit on Amount and Time columns, persist with joblib.
During INFERENCE: loaded by FileModelRepository, called by HybridFraudModel.

This mirrors the original data/make_dataset.py logic but keeps it
isolated from the domain and API layers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


AMOUNT_IDX = 29   # index in [Time, V1..V28, Amount]
TIME_IDX = 0


class FraudPreprocessor:
    """
    Scales Amount (index 29) and Time (index 0) columns.
    All other features (V1–V28) are already PCA-transformed and need no scaling.
    """

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FraudPreprocessor":
        """Fit scaler on Amount and Time columns of a training matrix."""
        cols = X[:, [TIME_IDX, AMOUNT_IDX]]
        self._scaler.fit(cols)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return a scaled copy of X."""
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted.")
        X_out = X.copy().astype(float)
        X_out[:, [TIME_IDX, AMOUNT_IDX]] = self._scaler.transform(
            X_out[:, [TIME_IDX, AMOUNT_IDX]]
        )
        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "FraudPreprocessor":
        return joblib.load(path)
