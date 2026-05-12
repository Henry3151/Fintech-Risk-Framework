# -*- coding: utf-8 -*-
from __future__ import annotations
from functools import lru_cache
from fastapi import Depends

from infrastructure.models.cox_default_model import CoxDefaultModel
from infrastructure.repositories.file_default_repository import FileDefaultRepository
from use_cases.predict_default import PredictDefault, PredictDefaultBatch, GetDefaultMetrics


# Retorna o repositorio como singleton
@lru_cache(maxsize=1)
def get_repository() -> FileDefaultRepository:
    return FileDefaultRepository()


# Retorna o modelo Cox como singleton
@lru_cache(maxsize=1)
def get_model(
    repository: FileDefaultRepository = Depends(get_repository),
) -> CoxDefaultModel:
    return CoxDefaultModel(repository=repository)


# Fabrica o use case de predicao unica
def get_predict_use_case(
    model: CoxDefaultModel = Depends(get_model),
) -> PredictDefault:
    return PredictDefault(model=model)


# Fabrica o use case de predicao em lote
def get_predict_batch_use_case(
    model: CoxDefaultModel = Depends(get_model),
) -> PredictDefaultBatch:
    return PredictDefaultBatch(model=model)


# Fabrica o use case de metricas
def get_metrics_use_case(
    repository: FileDefaultRepository = Depends(get_repository),
) -> GetDefaultMetrics:
    return GetDefaultMetrics(repository=repository)
