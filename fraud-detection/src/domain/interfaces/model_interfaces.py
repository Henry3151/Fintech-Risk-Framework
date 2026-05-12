"""
Abstract contracts that every concrete implementation must honour.
Only standard-library imports allowed here.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from domain.entities.transaction import Transaction, FraudPrediction


class IFraudModel(ABC):
    """Contract for any fraud-detection model (hybrid or otherwise)."""

    @abstractmethod
    def predict(self, transaction: Transaction) -> FraudPrediction:
        """Run inference on a single transaction."""
        ...

    @abstractmethod
    def predict_batch(self, transactions: List[Transaction]) -> List[FraudPrediction]:
        """Run inference on a list of transactions."""
        ...

    @abstractmethod
    def get_reconstruction_error(self, features: np.ndarray) -> float:
        """Return the autoencoder reconstruction error for raw features."""
        ...


class IModelRepository(ABC):
    """Contract for loading / persisting model artifacts."""

    @abstractmethod
    def load_preprocessor(self) -> object:
        """Load the fitted StandardScaler."""
        ...

    @abstractmethod
    def load_autoencoder(self) -> object:
        """Load the PyTorch autoencoder and its threshold."""
        ...

    @abstractmethod
    def load_classifier(self) -> object:
        """Load the XGBoost classifier and its decision threshold."""
        ...

    @abstractmethod
    def get_autoencoder_threshold(self) -> float:
        ...

    @abstractmethod
    def get_classifier_threshold(self) -> float:
        ...

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return a dict with pr_auc, thresholds, and training metadata."""
        ...
