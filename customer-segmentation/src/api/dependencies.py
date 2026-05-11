# -*- coding: utf-8 -*-
from __future__ import annotations
from functools import lru_cache
from fastapi import Depends

from infrastructure.models.kmeans_umap_model import KMeansUMAPModel
from infrastructure.repositories.file_segmentation_repository import FileSegmentationRepository
from use_cases.segment_customers import SegmentCustomers
from use_cases.get_segmentation_metrics import GetSegmentationMetrics


# Retorna o repositorio de artefatos como singleton via lru_cache
@lru_cache(maxsize=1)
def get_repository() -> FileSegmentationRepository:
    return FileSegmentationRepository()


# Retorna o modelo de segmentacao como singleton carregado via repositorio
@lru_cache(maxsize=1)
def get_model(
    repository: FileSegmentationRepository = Depends(get_repository),
) -> KMeansUMAPModel:
    return KMeansUMAPModel(repository=repository)


# Fabrica o use case de segmentacao injetando o modelo
def get_segment_use_case(
    model: KMeansUMAPModel = Depends(get_model),
) -> SegmentCustomers:
    return SegmentCustomers(model=model)


# Fabrica o use case de metricas injetando o repositorio
def get_metrics_use_case(
    repository: FileSegmentationRepository = Depends(get_repository),
) -> GetSegmentationMetrics:
    return GetSegmentationMetrics(repository=repository)
