# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List
from dataclasses import dataclass

from domain.entities.loan import Loan, DefaultPrediction
from domain.exceptions import ValidationError
from domain.interfaces.default_interfaces import IDefaultModel, IDefaultRepository


# Orquestra a predicao de default para um unico emprestimo
class PredictDefault:
    """
    Use case principal: recebe um Loan e retorna DefaultPrediction
    com curva de sobrevivencia, hazard ratio e risk tier.
    """

    # Recebe o modelo de predicao via injecao de dependencia
    def __init__(self, model: IDefaultModel) -> None:
        self._model = model

    # Valida o emprestimo e executa a predicao retornando DefaultPrediction
    def execute(self, loan: Loan) -> DefaultPrediction:
        self._validate(loan)
        return self._model.predict(loan)

    # Valida que os campos criticos estao dentro dos limites aceitaveis
    @staticmethod
    def _validate(loan: Loan) -> None:
        if loan.dti < 0 or loan.dti > 100:
            raise ValidationError(f"dti must be in [0, 100], got {loan.dti}")
        if loan.fico_range_low < 300 or loan.fico_range_low > 850:
            raise ValidationError(f"fico_range_low must be in [300, 850], got {loan.fico_range_low}")


# Orquestra a predicao de default em lote para multiplos emprestimos
class PredictDefaultBatch:
    """
    Use case de batch: recebe lista de Loan e retorna lista de DefaultPrediction.
    """

    MAX_BATCH = 2000

    # Recebe o modelo de predicao via injecao de dependencia
    def __init__(self, model: IDefaultModel) -> None:
        self._model = model

    # Valida o lote e executa a predicao retornando lista de DefaultPrediction
    def execute(self, loans: List[Loan]) -> List[DefaultPrediction]:
        if not loans:
            raise ValidationError("Batch must contain at least one loan.")
        if len(loans) > self.MAX_BATCH:
            raise ValidationError(f"Batch size {len(loans)} exceeds maximum of {self.MAX_BATCH}.")
        return self._model.predict_batch(loans)


# Dataclass tipado com as metricas principais do modelo Cox
@dataclass(frozen=True)
class DefaultMetrics:
    c_index: float
    brier_score_12m: float
    n_events: int
    n_censored: int
    n_train: int
    metadata: dict


# Carrega e retorna as metricas do ultimo treinamento via repositorio
class GetDefaultMetrics:
    """
    Retorna metricas do treinamento: C-index (concordance index),
    Brier score e estatisticas do dataset de treino.
    """

    # Recebe o repositorio de artefatos via injecao de dependencia
    def __init__(self, repository: IDefaultRepository) -> None:
        self._repository = repository

    # Le os metadados do repositorio e retorna um DefaultMetrics tipado
    def execute(self) -> DefaultMetrics:
        meta = self._repository.get_metadata()
        return DefaultMetrics(
            c_index=meta["c_index"],
            brier_score_12m=meta["brier_score_12m"],
            n_events=meta["n_events"],
            n_censored=meta["n_censored"],
            n_train=meta["n_train"],
            metadata=meta,
        )
