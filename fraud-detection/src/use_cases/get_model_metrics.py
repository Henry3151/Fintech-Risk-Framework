"""
Use case: GetModelMetrics
Returns training metadata (PR-AUC, thresholds, dataset stats).
"""
from __future__ import annotations
from dataclasses import dataclass

from domain.interfaces.model_interfaces import IModelRepository


@dataclass(frozen=True)
class ModelMetrics:
    pr_auc: float
    autoencoder_threshold: float
    classifier_threshold: float
    autoencoder_pr_auc: float
    metadata: dict


class GetModelMetrics:
    """
    Returns metrics from the last training run.

    Usage
    -----
    use_case = GetModelMetrics(repository=FileModelRepository(...))
    metrics  = use_case.execute()
    """

    def __init__(self, repository: IModelRepository) -> None:
        self._repository = repository

    def execute(self) -> ModelMetrics:
        meta = self._repository.get_metadata()
        return ModelMetrics(
            pr_auc=meta["classifier"]["pr_auc"],
            autoencoder_threshold=self._repository.get_autoencoder_threshold(),
            classifier_threshold=self._repository.get_classifier_threshold(),
            autoencoder_pr_auc=meta["autoencoder"]["pr_auc"],
            metadata=meta,
        )
