"""
Use case: PredictFraud
Orchestrates a single-transaction fraud prediction.

Input : Transaction entity
Output: FraudPrediction entity

This class has NO knowledge of FastAPI, PyTorch, or XGBoost.
It only speaks in domain types and calls the IFraudModel contract.
"""
from __future__ import annotations

from domain.entities.transaction import Transaction, FraudPrediction
from domain.exceptions import ValidationError
from domain.interfaces.model_interfaces import IFraudModel


class PredictFraud:
    """
    Single-transaction fraud-detection use case.

    Usage
    -----
    use_case = PredictFraud(model=HybridFraudModel(...))
    prediction = use_case.execute(transaction)
    """

    def __init__(self, model: IFraudModel) -> None:
        self._model = model

    def execute(self, transaction: Transaction) -> FraudPrediction:
        self._validate(transaction)
        return self._model.predict(transaction)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(transaction: Transaction) -> None:
        import numpy as np

        features = transaction.features
        if np.any(np.isnan(features)):
            raise ValidationError("Transaction contains NaN values.")
        if np.any(np.isinf(features)):
            raise ValidationError("Transaction contains infinite values.")
