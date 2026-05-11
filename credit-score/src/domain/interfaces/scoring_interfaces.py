# -*- coding: utf-8 -*-
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from domain.entities.credit_applicant import CreditApplicant, CreditScore


# Contrato abstrato para qualquer modelo de credit scoring
class ICreditScoringModel(ABC):

    # Calcula o score de credito para um unico solicitante
    @abstractmethod
    def predict(self, applicant: CreditApplicant) -> CreditScore: ...

    # Calcula o score de credito para uma lista de solicitantes em lote
    @abstractmethod
    def predict_batch(self, applicants: List[CreditApplicant]) -> List[CreditScore]: ...

    # Retorna as metricas de performance do modelo (AUC, KS, Brier score)
    @abstractmethod
    def get_metrics(self) -> dict: ...


# Contrato abstrato para carregamento e persistencia de artefatos do modelo
class ICreditScoringRepository(ABC):

    # Carrega o modelo LightGBM treinado
    @abstractmethod
    def load_model(self) -> object: ...

    # Carrega o calibrador de probabilidade (Platt scaling)
    @abstractmethod
    def load_calibrator(self) -> object: ...

    # Carrega o explainer SHAP para o modelo
    @abstractmethod
    def load_explainer(self) -> object: ...

    # Retorna o dicionario de metadados do treinamento
    @abstractmethod
    def get_metadata(self) -> dict: ...

    # Salva todos os artefatos do modelo em disco
    @abstractmethod
    def save_artifacts(self, model, calibrator, explainer, metadata: dict) -> None: ...
