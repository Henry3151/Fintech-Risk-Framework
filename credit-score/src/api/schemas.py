# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# Schema de entrada para scoring de um unico solicitante via API
class ApplicantRequest(BaseModel):
    applicant_id: Optional[str] = "unknown"
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0.0, le=1.0)
    age: int                                     = Field(..., ge=18, le=100)
    NumberOfTime30_59DaysPastDueNotWorse: int    = Field(..., ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float                             = Field(..., ge=0.0)
    MonthlyIncome: float                         = Field(..., ge=0.0)
    NumberOfOpenCreditLinesAndLoans: int         = Field(..., ge=0)
    NumberOfTimes90DaysLate: int                 = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int            = Field(..., ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int    = Field(..., ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: int                      = Field(..., ge=0)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {"example": {
            "applicant_id": "A001",
            "RevolvingUtilizationOfUnsecuredLines": 0.15,
            "age": 45,
            "NumberOfTime30-59DaysPastDueNotWorse": 0,
            "DebtRatio": 0.35,
            "MonthlyIncome": 6500.0,
            "NumberOfOpenCreditLinesAndLoans": 8,
            "NumberOfTimes90DaysLate": 0,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 0,
            "NumberOfDependents": 2,
        }}
    }


# Schema de entrada para scoring em lote de multiplos solicitantes
class BatchApplicantRequest(BaseModel):
    applicants: List[ApplicantRequest] = Field(..., min_length=1, max_length=5000)


# Schema de um fator SHAP retornado na resposta
class ShapFactor(BaseModel):
    feature: str
    shap_value: float
    impact: str   # increases_risk | decreases_risk


# Schema de resposta com o resultado do scoring de um solicitante
class CreditScoreResponse(BaseModel):
    applicant_id: str
    score: int              = Field(..., ge=0, le=1000)
    pd: float               = Field(..., ge=0.0, le=1.0)
    risk_grade: str         # A, B, C, D, E
    recommendation: str     # APPROVE, REVIEW, DENY
    top_factors: List[ShapFactor]
    latency_ms: float


# Schema de resposta para scoring em lote com resumo por recomendacao
class BatchScoreResponse(BaseModel):
    scores: List[CreditScoreResponse]
    total_applicants: int
    recommendation_summary: dict


# Schema de resposta para o endpoint de health
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    auc_roc: float
    pr_auc: float
    ks_statistic: float
    brier_score: float
    n_train: int
