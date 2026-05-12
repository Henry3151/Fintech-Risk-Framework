"""
API Pydantic schemas — request and response models.
No domain logic here; pure data validation and serialization.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    """
    Single transaction payload.
    Mirrors the ULB/Kaggle Credit Card Fraud dataset schema.
    """

    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")
    transaction_id: Optional[str] = Field(None, description="Client-supplied ID")

    model_config = {"json_schema_extra": {
        "example": {
            "Time": 0.0, "V1": -1.36, "V2": -0.07,
            "V3": 2.54, "V4": 1.38, "V5": -0.34,
            "V6": 0.46, "V7": 0.24, "V8": 0.10,
            "V9": 0.36, "V10": 0.09, "V11": -0.55,
            "V12": -0.62, "V13": -0.99, "V14": -0.31,
            "V15": 1.47, "V16": -0.47, "V17": 0.21,
            "V18": 0.03, "V19": 0.40, "V20": 0.25,
            "V21": -0.02, "V22": 0.28, "V23": -0.11,
            "V24": 0.07, "V25": 0.13, "V26": -0.19,
            "V27": 0.13, "V28": -0.02,
            "Amount": 149.62,
            "transaction_id": "txn-001",
        }
    }}


class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(
        ..., min_length=1, max_length=1000
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool
    risk_label: str = Field(..., description="LOW | MEDIUM | HIGH")
    reconstruction_error: float
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_count: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pr_auc: float
    classifier_threshold: float
    autoencoder_threshold: float
