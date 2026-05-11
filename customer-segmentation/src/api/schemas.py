# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# Schema de entrada para segmentacao de um unico cliente via API
class CustomerRequest(BaseModel):
    customer_id: str
    recency: float   = Field(..., ge=0, description="Days since last purchase")
    frequency: float = Field(..., ge=0, description="Number of unique orders")
    monetary: float  = Field(..., ge=0, description="Total amount spent (GBP)")

    model_config = {"json_schema_extra": {"example": {
        "customer_id": "12345",
        "recency": 30.0,
        "frequency": 12.0,
        "monetary": 850.0,
    }}}


# Schema de entrada para segmentacao em lote de multiplos clientes
class BatchCustomerRequest(BaseModel):
    customers: List[CustomerRequest] = Field(..., min_length=1, max_length=5000)


# Schema de resposta com o resultado da segmentacao de um cliente
class SegmentResponse(BaseModel):
    customer_id: str
    cluster_id: int
    segment_label: str
    umap_x: float
    umap_y: float
    rfm_score: float
    is_high_value: bool


# Schema de resposta para segmentacao em lote com resumo por segmento
class BatchSegmentResponse(BaseModel):
    segments: List[SegmentResponse]
    total_customers: int
    segment_summary: dict


# Schema de resposta para o endpoint de health com metricas do modelo
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_clusters: int
    silhouette_score: float
    n_customers_trained: int
