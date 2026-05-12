# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import Counter
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_predict_use_case, get_predict_batch_use_case, get_metrics_use_case
from api.schemas import (
    LoanRequest, BatchLoanRequest,
    DefaultPredictionResponse, BatchDefaultResponse, HealthResponse,
)
from domain.entities.loan import Loan
from domain.exceptions import ValidationError, PredictionError, ModelNotLoadedError
from use_cases.predict_default import PredictDefault, PredictDefaultBatch, GetDefaultMetrics

app = FastAPI(
    title="Default Prediction API",
    description="Cox Proportional Hazards survival analysis para predicao de default.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Converte LoanRequest Pydantic em entidade Loan do dominio
def _to_loan(req: LoanRequest) -> Loan:
    return Loan(
        loan_id=req.loan_id or "unknown",
        loan_amnt=req.loan_amnt,
        int_rate=req.int_rate,
        grade=req.grade,
        emp_length_years=req.emp_length_years,
        annual_inc=req.annual_inc,
        dti=req.dti,
        fico_range_low=req.fico_range_low,
        open_acc=req.open_acc,
        revol_util=req.revol_util,
        total_acc=req.total_acc,
        inq_last_6mths=req.inq_last_6mths,
        pub_rec=req.pub_rec,
        term_months=req.term_months,
    )


# Converte DefaultPrediction do dominio em DefaultPredictionResponse Pydantic
def _to_response(pred) -> DefaultPredictionResponse:
    return DefaultPredictionResponse(
        loan_id=pred.loan_id,
        survival_at_12m=pred.survival_at_12m,
        survival_at_24m=pred.survival_at_24m,
        survival_at_36m=pred.survival_at_36m,
        median_survival_months=pred.median_survival_months,
        hazard_ratio=pred.hazard_ratio,
        risk_tier=pred.risk_tier,
        pd_12m=pred.pd_12m,
        latency_ms=pred.latency_ms,
    )


# Mapeia excecoes de dominio para HTTP status codes
def _handle(exc: Exception) -> HTTPException:
    if isinstance(exc, ValidationError):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if isinstance(exc, ModelNotLoadedError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# Prediz a curva de sobrevivencia e risco de default para um emprestimo
@app.post("/predict", response_model=DefaultPredictionResponse, tags=["Prediction"])
async def predict(
    request: LoanRequest,
    use_case: PredictDefault = Depends(get_predict_use_case),
) -> DefaultPredictionResponse:
    try:
        result = use_case.execute(_to_loan(request))
        return _to_response(result)
    except (ValidationError, PredictionError, ModelNotLoadedError) as exc:
        raise _handle(exc)


# Prediz em lote retornando curvas de sobrevivencia e resumo de risco
@app.post("/predict/batch", response_model=BatchDefaultResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchLoanRequest,
    use_case: PredictDefaultBatch = Depends(get_predict_batch_use_case),
) -> BatchDefaultResponse:
    try:
        loans   = [_to_loan(r) for r in request.loans]
        results = use_case.execute(loans)
    except (ValidationError, PredictionError, ModelNotLoadedError) as exc:
        raise _handle(exc)

    responses = [_to_response(r) for r in results]
    summary   = dict(Counter(r.risk_tier for r in responses))
    return BatchDefaultResponse(
        predictions=responses,
        total_loans=len(responses),
        risk_summary=summary,
    )


# Retorna o status do modelo com C-index e Brier score
@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health(
    use_case: GetDefaultMetrics = Depends(get_metrics_use_case),
) -> HealthResponse:
    try:
        metrics = use_case.execute()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            c_index=metrics.c_index,
            brier_score_12m=metrics.brier_score_12m,
            n_events=metrics.n_events,
            n_train=metrics.n_train,
        )
    except Exception:
        return HealthResponse(
            status="degraded", model_loaded=False,
            c_index=0.0, brier_score_12m=0.0,
            n_events=0, n_train=0,
        )
