# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import Counter
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_segment_use_case, get_metrics_use_case
from api.schemas import (
    CustomerRequest, BatchCustomerRequest,
    SegmentResponse, BatchSegmentResponse, HealthResponse,
)
from domain.entities.customer import Customer
from domain.exceptions import ValidationError, SegmentationError, ModelNotLoadedError
from use_cases.segment_customers import SegmentCustomers
from use_cases.get_segmentation_metrics import GetSegmentationMetrics

app = FastAPI(
    title="Customer Segmentation API",
    description="K-Means + UMAP segmentation over RFM features. Silhouette-optimized.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Converte um CustomerRequest Pydantic em uma entidade Customer do dominio
def _to_customer(req: CustomerRequest) -> Customer:
    return Customer(
        customer_id=req.customer_id,
        recency=req.recency,
        frequency=req.frequency,
        monetary=req.monetary,
    )


# Converte um CustomerSegment de dominio em um SegmentResponse Pydantic
def _to_response(seg) -> SegmentResponse:
    return SegmentResponse(
        customer_id=seg.customer_id,
        cluster_id=seg.cluster_id,
        segment_label=seg.segment_label,
        umap_x=seg.umap_x,
        umap_y=seg.umap_y,
        rfm_score=seg.rfm_score,
        is_high_value=seg.is_high_value,
    )


# Mapeia excecoes de dominio para os HTTP status codes corretos
def _handle(exc: Exception) -> HTTPException:
    if isinstance(exc, ValidationError):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if isinstance(exc, ModelNotLoadedError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


# Segmenta um unico cliente e retorna cluster, label, coordenadas UMAP e rfm_score
@app.post("/segment", response_model=SegmentResponse, tags=["Segmentation"])
async def segment(
    request: CustomerRequest,
    use_case: SegmentCustomers = Depends(get_segment_use_case),
) -> SegmentResponse:
    try:
        results = use_case.execute([_to_customer(request)])
        return _to_response(results[0])
    except (ValidationError, SegmentationError, ModelNotLoadedError) as exc:
        raise _handle(exc)


# Segmenta um lote de clientes retornando segmentos e resumo por label
@app.post("/segment/batch", response_model=BatchSegmentResponse, tags=["Segmentation"])
async def segment_batch(
    request: BatchCustomerRequest,
    use_case: SegmentCustomers = Depends(get_segment_use_case),
) -> BatchSegmentResponse:
    try:
        customers = [_to_customer(r) for r in request.customers]
        results   = use_case.execute(customers)
    except (ValidationError, SegmentationError, ModelNotLoadedError) as exc:
        raise _handle(exc)

    responses = [_to_response(r) for r in results]
    summary   = dict(Counter(r.segment_label for r in responses))
    return BatchSegmentResponse(
        segments=responses,
        total_customers=len(responses),
        segment_summary=summary,
    )


# Retorna o status do modelo com metricas de silhouette e numero de clusters
@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health(
    use_case: GetSegmentationMetrics = Depends(get_metrics_use_case),
) -> HealthResponse:
    try:
        metrics = use_case.execute()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            n_clusters=metrics.n_clusters,
            silhouette_score=metrics.silhouette_score,
            n_customers_trained=metrics.n_customers,
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            n_clusters=0,
            silhouette_score=0.0,
            n_customers_trained=0,
        )
