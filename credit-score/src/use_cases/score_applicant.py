# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List

from domain.entities.credit_applicant import CreditApplicant, CreditScore
from domain.exceptions import ValidationError
from domain.interfaces.scoring_interfaces import ICreditScoringModel


# Orquestra o scoring de credito de um unico solicitante
class ScoreApplicant:
    """
    Use case principal: recebe um CreditApplicant e retorna CreditScore
    com score 0-1000, risk grade, recomendacao e top fatores SHAP.
    """

    # Recebe o modelo de scoring via injecao de dependencia
    def __init__(self, model: ICreditScoringModel) -> None:
        self._model = model

    # Valida o solicitante e executa o scoring retornando CreditScore
    def execute(self, applicant: CreditApplicant) -> CreditScore:
        self._validate(applicant)
        return self._model.predict(applicant)

    # Valida que os campos criticos estao dentro dos limites aceitaveis
    @staticmethod
    def _validate(applicant: CreditApplicant) -> None:
        if applicant.age < 18:
            raise ValidationError(f"Applicant must be at least 18 years old, got {applicant.age}")
        if applicant.age > 100:
            raise ValidationError(f"Invalid age: {applicant.age}")
        if applicant.late_90_days < 0 or applicant.late_30_59_days < 0:
            raise ValidationError("Late payment counts cannot be negative")


# Orquestra o scoring de credito em lote para multiplos solicitantes
class ScoreApplicantBatch:
    """
    Use case de batch scoring: recebe lista de CreditApplicant
    e retorna lista de CreditScore com validacao de tamanho.
    """

    MAX_BATCH = 5000

    # Recebe o modelo de scoring via injecao de dependencia
    def __init__(self, model: ICreditScoringModel) -> None:
        self._model = model

    # Valida o lote e executa o scoring retornando lista de CreditScore
    def execute(self, applicants: List[CreditApplicant]) -> List[CreditScore]:
        if not applicants:
            raise ValidationError("Batch must contain at least one applicant.")
        if len(applicants) > self.MAX_BATCH:
            raise ValidationError(f"Batch size {len(applicants)} exceeds maximum of {self.MAX_BATCH}.")
        return self._model.predict_batch(applicants)
