"""
Transaction entity — core business object.
No external dependencies; pure Python dataclass.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Transaction:
    """
    Represents a single credit-card transaction.

    All 28 PCA features (V1–V28) plus Amount and Time are kept as a
    numpy array to avoid coupling to any ML framework.
    """

    features: np.ndarray          # shape (30,): [Time, V1..V28, Amount]
    transaction_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.features.shape != (30,):
            raise ValueError(
                f"Expected 30 features (Time + V1-V28 + Amount), "
                f"got {self.features.shape}"
            )

    @classmethod
    def from_dict(cls, data: dict, transaction_id: Optional[str] = None) -> "Transaction":
        """Build a Transaction from a raw dict (e.g. from an API request)."""
        feature_keys = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        features = np.array([float(data[k]) for k in feature_keys], dtype=np.float64)
        return cls(features=features, transaction_id=transaction_id)

    def to_numpy(self) -> np.ndarray:
        return self.features.copy()


@dataclass(frozen=True)
class FraudPrediction:
    """
    Result of a fraud-detection inference.

    fraud_probability: raw model score in [0, 1]
    is_fraud:          thresholded decision (uses best_threshold from metadata)
    reconstruction_error: autoencoder signal used as an extra feature
    """

    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    reconstruction_error: float
    latency_ms: float

    @property
    def risk_label(self) -> str:
        if self.fraud_probability >= 0.90:
            return "HIGH"
        if self.fraud_probability >= 0.50:
            return "MEDIUM"
        return "LOW"
