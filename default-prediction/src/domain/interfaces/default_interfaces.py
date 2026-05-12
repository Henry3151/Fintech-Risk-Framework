# -*- coding: utf-8 -*-
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from domain.entities.loan import Loan, DefaultPrediction


# Contrato abstrato para qualquer modelo de predicao de default
class IDefaultModel(ABC):

    # Prediz a curva de sobrevivencia e risco de default para um emprestimo
    @abstractmethod
    def predict(self, loan: Loan) -> DefaultPrediction: ...

    # Prediz em lote para uma lista de emprestimos
    @abstractmethod
    def predict_batch(self, loans: List[Loan]) -> List[DefaultPrediction]: ...

    # Retorna as metricas de performance do modelo (C-index, Brier score)
    @abstractmethod
    def get_metrics(self) -> dict: ...


# Contrato abstrato para carregamento e persistencia de artefatos do modelo
class IDefaultRepository(ABC):

    # Carrega o modelo Cox treinado
    @abstractmethod
    def load_model(self) -> object: ...

    # Carrega o scaler das features
    @abstractmethod
    def load_scaler(self) -> object: ...

    # Retorna o dicionario de metadados do treinamento
    @abstractmethod
    def get_metadata(self) -> dict: ...

    # Salva todos os artefatos do modelo em disco
    @abstractmethod
    def save_artifacts(self, model, scaler, metadata: dict) -> None: ...
