"""
Dependency injection — wires infrastructure → use cases.
FastAPI's Depends() system calls these functions once per request
(or once at startup for cached singletons).
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from infrastructure.models.hybrid_fraud_model import HybridFraudModel
from infrastructure.repositories.file_model_repository import FileModelRepository
from use_cases.predict_fraud import PredictFraud
from use_cases.predict_fraud_batch import PredictFraudBatch
from use_cases.get_model_metrics import GetModelMetrics


# ---------------------------------------------------------------------------
# Singletons (loaded once at startup)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_repository() -> FileModelRepository:
    return FileModelRepository()


@lru_cache(maxsize=1)
def get_model(
    repository: FileModelRepository = Depends(get_repository),
) -> HybridFraudModel:
    return HybridFraudModel(repository=repository)


# ---------------------------------------------------------------------------
# Use-case factories (cheap; one instance per request is fine)
# ---------------------------------------------------------------------------

def get_predict_fraud_use_case(
    model: HybridFraudModel = Depends(get_model),
) -> PredictFraud:
    return PredictFraud(model=model)


def get_predict_fraud_batch_use_case(
    model: HybridFraudModel = Depends(get_model),
) -> PredictFraudBatch:
    return PredictFraudBatch(model=model)


def get_model_metrics_use_case(
    repository: FileModelRepository = Depends(get_repository),
) -> GetModelMetrics:
    return GetModelMetrics(repository=repository)
