# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass

from domain.interfaces.scoring_interfaces import ICreditScoringRepository


# Dataclass tipado com as metricas principais do modelo de credit scoring
@dataclass(frozen=True)
class ScoringMetrics:
    auc_roc: float
    pr_auc: float
    ks_statistic: float
    brier_score: float
    n_train: int
    metadata: dict


# Carrega e retorna as metricas do ultimo treinamento via repositorio
class GetScoringMetrics:
    """
    Retorna metricas do treinamento: AUC-ROC, PR-AUC, KS statistic
    e Brier score (calibracao de probabilidade).
    """

    # Recebe o repositorio de artefatos via injecao de dependencia
    def __init__(self, repository: ICreditScoringRepository) -> None:
        self._repository = repository

    # Le os metadados do repositorio e retorna um ScoringMetrics tipado
    def execute(self) -> ScoringMetrics:
        meta = self._repository.get_metadata()
        return ScoringMetrics(
            auc_roc=meta["auc_roc"],
            pr_auc=meta["pr_auc"],
            ks_statistic=meta["ks_statistic"],
            brier_score=meta["brier_score"],
            n_train=meta["n_train"],
            metadata=meta,
        )
