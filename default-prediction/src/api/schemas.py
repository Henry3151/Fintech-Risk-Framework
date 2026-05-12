# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# Schema de entrada para predicao de default de um unico emprestimo
class LoanRequest(BaseModel):
    loan_id: Optional[str] = "unknown"
    loan_amnt: float        = Field(..., gt=0)
    int_rate: float         = Field(..., gt=0, lt=100)
    grade: str              = Field(..., pattern="^[A-G]$")
    emp_length_years: float = Field(..., ge=0, le=50)
    annual_inc: float       = Field(..., ge=0)
    dti: float              = Field(..., ge=0, le=100)
    fico_range_low: float   = Field(..., ge=300, le=850)
    open_acc: int           = Field(..., ge=0)
    revol_util: float       = Field(..., ge=0, le=100)
    total_acc: int          = Field(..., ge=0)
    inq_last_6mths: int     = Field(..., ge=0)
    pub_rec: int            = Field(..., ge=0)
    term_months: int        = Field(..., ge=36, le=60)

    model_config = {"json_schema_extra": {"example": {
        "loan_id": "L001",
        "loan_amnt": 15000.0,
        "int_rate": 13.99,
        "grade": "C",
        "emp_length_years": 5.0,
        "annual_inc": 65000.0,
        "dti": 18.5,
        "fico_range_low": 690.0,
        "open_acc": 10,
        "revol_util": 45.0,
        "total_acc": 25,
        "inq_last_6mths": 1,
        "pub_rec": 0,
        "term_months": 36,
    }}}


# Schema de entrada para predicao em lote
class BatchLoanRequest(BaseModel):
    loans: List[LoanRequest] = Field(..., min_length=1, max_length=2000)


# Schema de resposta com a curva de sobrevivencia e risco
class DefaultPredictionResponse(BaseModel):
    loan_id: str
    survival_at_12m: float
    survival_at_24m: float
    survival_at_36m: float
    median_survival_months: Optional[float]
    hazard_ratio: float
    risk_tier: str
    pd_12m: float
    latency_ms: float


# Schema de resposta para predicao em lote
class BatchDefaultResponse(BaseModel):
    predictions: List[DefaultPredictionResponse]
    total_loans: int
    risk_summary: dict


# Schema de resposta para o endpoint de health
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    c_index: float
    brier_score_12m: float
    n_events: int
    n_train: int
