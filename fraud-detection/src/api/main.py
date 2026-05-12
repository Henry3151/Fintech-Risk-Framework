"""
FastAPI application — /predict  /predict/batch  /health

All business logic lives in use cases; this file only handles
HTTP wiring, serialization, and error mapping.
"""
from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import (
    get_predict_fraud_use_case,
    get_predict_fraud_batch_use_case,
    get_model_metrics_use_case,
)
from api.schemas import (
    TransactionRequest,
    BatchTransactionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)
from domain.entities.transaction import Transaction
from domain.exceptions import ValidationError, InferenceError, ModelNotLoadedError
from use_cases.predict_fraud import PredictFraud
from use_cases.predict_fraud_batch import PredictFraudBatch
from use_cases.get_model_metrics import GetModelMetrics


app = FastAPI(
    title="Fraud Detection API",
    description="Hybrid Autoencoder + XGBoost fraud detection. PR-AUC 0.8665.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_transaction(req: TransactionRequest) -> Transaction:
    return Transaction.from_dict(req.model_dump(), transaction_id=req.transaction_id)


def _to_response(prediction) -> PredictionResponse:
    return PredictionResponse(
        transaction_id=prediction.transaction_id,
        fraud_probability=prediction.fraud_probability,
        is_fraud=prediction.is_fraud,
        risk_label=prediction.risk_label,
        reconstruction_error=prediction.reconstruction_error,
        latency_ms=prediction.latency_ms,
    )


def _handle_domain_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, ValidationError):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if isinstance(exc, ModelNotLoadedError):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    if isinstance(exc, InferenceError):
        return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single transaction fraud prediction",
    tags=["Inference"],
)
async def predict(
    request: TransactionRequest,
    use_case: PredictFraud = Depends(get_predict_fraud_use_case),
) -> PredictionResponse:
    try:
        transaction = _to_transaction(request)
        prediction = use_case.execute(transaction)
        return _to_response(prediction)
    except (ValidationError, InferenceError, ModelNotLoadedError) as exc:
        raise _handle_domain_errors(exc)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch transaction fraud prediction (max 1 000)",
    tags=["Inference"],
)
async def predict_batch(
    request: BatchTransactionRequest,
    use_case: PredictFraudBatch = Depends(get_predict_fraud_batch_use_case),
) -> BatchPredictionResponse:
    try:
        transactions = [_to_transaction(r) for r in request.transactions]
        predictions = use_case.execute(transactions)
    except (ValidationError, InferenceError, ModelNotLoadedError) as exc:
        raise _handle_domain_errors(exc)

    responses = [_to_response(p) for p in predictions]
    return BatchPredictionResponse(
        predictions=responses,
        total_transactions=len(responses),
        fraud_count=sum(1 for p in responses if p.is_fraud),
        total_latency_ms=sum(p.latency_ms for p in responses),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Model health and metadata",
    tags=["Operations"],
)
async def health(
    use_case: GetModelMetrics = Depends(get_model_metrics_use_case),
) -> HealthResponse:
    try:
        metrics = use_case.execute()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            pr_auc=metrics.pr_auc,
            classifier_threshold=metrics.classifier_threshold,
            autoencoder_threshold=metrics.autoencoder_threshold,
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            pr_auc=0.0,
            classifier_threshold=0.0,
            autoencoder_threshold=0.0,
        )
