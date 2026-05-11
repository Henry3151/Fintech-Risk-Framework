# -*- coding: utf-8 -*-
from __future__ import annotations
from functools import lru_cache
from fastapi import Depends

from infrastructure.models.lightgbm_scoring_model import LightGBMScoringModel
from infrastructure.repositories.file_scoring_repository import FileScoringRepository
from use_cases.score_applicant import ScoreApplicant, ScoreApplicantBatch
from use_cases.get_scoring_metrics import GetScoringMetrics


# Retorna o repositorio de artefatos como singleton via lru_cache
@lru_cache(maxsize=1)
def get_repository() -> FileScoringRepository:
    return FileScoringRepository()


# Retorna o modelo de scoring como singleton carregado via repositorio
@lru_cache(maxsize=1)
def get_model(
    repository: FileScoringRepository = Depends(get_repository),
) -> LightGBMScoringModel:
    return LightGBMScoringModel(repository=repository)


# Fabrica o use case de scoring unico injetando o modelo
def get_score_use_case(
    model: LightGBMScoringModel = Depends(get_model),
) -> ScoreApplicant:
    return ScoreApplicant(model=model)


# Fabrica o use case de scoring em lote injetando o modelo
def get_score_batch_use_case(
    model: LightGBMScoringModel = Depends(get_model),
) -> ScoreApplicantBatch:
    return ScoreApplicantBatch(model=model)


# Fabrica o use case de metricas injetando o repositorio
def get_metrics_use_case(
    repository: FileScoringRepository = Depends(get_repository),
) -> GetScoringMetrics:
    return GetScoringMetrics(repository=repository)
