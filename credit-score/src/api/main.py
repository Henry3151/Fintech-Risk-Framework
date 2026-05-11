# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import Counter
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import (
    get_score_use_case, get_score_batch_use_case, get_metrics_use_case
)
from api.schemas import (
    ApplicantRequest, BatchApplicantRequest,
    CreditScoreResponse, BatchScoreResponse,
    ShapFactor, HealthResponse,
)
from domain.entities.credit_applicant import CreditApplicant
from domain.exceptions import ValidationError, ScoringError, ModelNotLoadedError
from use_cases.score_applicant import ScoreApplicant, ScoreApplicantBatch
from use_cases.get_scoring_metrics import GetScoringMetrics

app = FastAPI(
    title="Credit Score API",
    description="LightGBM + SHAP credit scoring. AUC-ROC otimizado com Platt calibration.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Converte um ApplicantRequest Pydantic em uma entidade CreditApplicant do dominio
def _to_applicant(req: ApplicantRequest) -> CreditApplicant:
    return CreditApplicant(
        applicant_id=req.applicant_id or "unknown",
        revolving_utilization=req.RevolvingUtilizationOfUnsecuredLines,
        age=req.age,
        late_30_59_days=req.NumberOfTime30_59DaysPastDueNotWorse,
        debt_ratio=req.DebtRatio,
        monthly_income=req.MonthlyIncome,
        open_credit_lines=req.NumberOfOpenCreditLinesAndLoans,
        late_90_days=req.NumberOfTimes90DaysLate,
        real_estate_loans=req.NumberRealEstateLoansOrLines,
        late_60_89_days=req.NumberOfTime60_89DaysPastDueNotWorse,
        dependents=req.NumberOfDependents,
    )


# Converte um CreditScore de dominio em um CreditScoreResponse Pydantic
def _to_response(cs) -> CreditScoreResponse:
    return CreditScoreResponse(
        applicant_id=cs.applicant_id,
        score=cs.score,
        pd=cs.pd,
        risk_grade=cs.risk_grade,
        recommendation=cs.recommendation,
        top_factors=[ShapFactor(**f) for f in cs.top_factors],
        latency_ms=cs.latency_ms,
    )


# Mapeia excecoes de dominio para os HTTP status codes corretos
def _handle(exc: Exception) -> HTTPException:
    if isinstance(exc, ValidationError):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if isinstance(exc, ModelNotLoadedError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# Calcula o credit score de um unico solicitante com SHAP e recomendacao
@app.post("/score", response_model=CreditScoreResponse, tags=["Scoring"])
async def score(
    request: ApplicantRequest,
    use_case: ScoreApplicant = Depends(get_score_use_case),
) -> CreditScoreResponse:
    try:
        result = use_case.execute(_to_applicant(request))
        return _to_response(result)
    except (ValidationError, ScoringError, ModelNotLoadedError) as exc:
        raise _handle(exc)


# Calcula o credit score em lote retornando scores e resumo por recomendacao
@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
async def score_batch(
    request: BatchApplicantRequest,
    use_case: ScoreApplicantBatch = Depends(get_score_batch_use_case),
) -> BatchScoreResponse:
    try:
        applicants = [_to_applicant(r) for r in request.applicants]
        results    = use_case.execute(applicants)
    except (ValidationError, ScoringError, ModelNotLoadedError) as exc:
        raise _handle(exc)

    responses = [_to_response(r) for r in results]
    summary   = dict(Counter(r.recommendation for r in responses))
    return BatchScoreResponse(
        scores=responses,
        total_applicants=len(responses),
        recommendation_summary=summary,
    )


# Retorna o status do modelo com metricas de performance e calibracao
@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health(
    use_case: GetScoringMetrics = Depends(get_metrics_use_case),
) -> HealthResponse:
    try:
        metrics = use_case.execute()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            auc_roc=metrics.auc_roc,
            pr_auc=metrics.pr_auc,
            ks_statistic=metrics.ks_statistic,
            brier_score=metrics.brier_score,
            n_train=metrics.n_train,
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            auc_roc=0.0, pr_auc=0.0,
            ks_statistic=0.0, brier_score=0.0,
            n_train=0,
        )
