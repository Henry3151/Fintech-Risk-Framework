"""
Use case: PredictFraudBatch
Orchestrates batch fraud prediction over a list of transactions.
"""
from __future__ import annotations
from typing import List

from domain.entities.transaction import Transaction, FraudPrediction
from domain.exceptions import ValidationError
from domain.interfaces.model_interfaces import IFraudModel


class PredictFraudBatch:
    """
    Batch fraud-detection use case.

    Usage
    -----
    use_case = PredictFraudBatch(model=HybridFraudModel(...))
    predictions = use_case.execute(transactions)
    """

    MAX_BATCH_SIZE = 1000

    def __init__(self, model: IFraudModel) -> None:
        self._model = model

    def execute(self, transactions: List[Transaction]) -> List[FraudPrediction]:
        self._validate_batch(transactions)
        return self._model.predict_batch(transactions)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_batch(self, transactions: List[Transaction]) -> None:
        if not transactions:
            raise ValidationError("Batch must contain at least one transaction.")
        if len(transactions) > self.MAX_BATCH_SIZE:
            raise ValidationError(
                f"Batch size {len(transactions)} exceeds maximum of {self.MAX_BATCH_SIZE}."
            )
