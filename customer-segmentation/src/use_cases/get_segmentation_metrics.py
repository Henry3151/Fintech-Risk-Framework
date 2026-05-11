# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass

from domain.interfaces.segmentation_interfaces import ISegmentationRepository


# Dataclass tipado com as metricas principais do modelo de segmentacao
@dataclass(frozen=True)
class SegmentationMetrics:
    n_clusters: int
    silhouette_score: float
    inertia: float
    n_customers: int
    metadata: dict


# Carrega e retorna as metricas do ultimo treinamento via repositorio
class GetSegmentationMetrics:
    """
    Retorna metricas do treinamento: silhouette score, inertia,
    numero de clusters e total de clientes segmentados.
    """

    # Recebe o repositorio de artefatos via injecao de dependencia
    def __init__(self, repository: ISegmentationRepository) -> None:
        self._repository = repository

    # Le os metadados do repositorio e retorna um SegmentationMetrics tipado
    def execute(self) -> SegmentationMetrics:
        meta = self._repository.get_metadata()
        return SegmentationMetrics(
            n_clusters=meta["n_clusters"],
            silhouette_score=meta["silhouette_score"],
            inertia=meta["inertia"],
            n_customers=meta["n_customers"],
            metadata=meta,
        )
