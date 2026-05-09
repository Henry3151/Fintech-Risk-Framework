"""
main.py — API FastAPI de deteccao de fraude em tempo real.
Uso: uvicorn src.api.main:app --reload --port 8000
"""
import json, time, joblib, numpy as np, torch, sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))
from models.train_autoencoder import FraudAutoencoder

class ModelStore:
    autoencoder: Optional[FraudAutoencoder] = None
    classifier = None
    scaler = None
    ae_metadata: dict = {}
    clf_metadata: dict = {}

store = ModelStore()

def load_models():
    print("[API] Carregando modelos ...")
    store.scaler = joblib.load(MODELS_DIR / "preprocessor.joblib")
    store.ae_metadata = json.load(open(MODELS_DIR / "autoencoder_metadata.json"))
    store.autoencoder = FraudAutoencoder(input_dim=store.ae_metadata["input_dim"])
    store.autoencoder.load_state_dict(torch.load(MODELS_DIR / "autoencoder.pt", map_location="cpu"))
    store.autoencoder.eval()
    store.classifier = joblib.load(MODELS_DIR / "classifier.joblib")
    store.clf_metadata = json.load(open(MODELS_DIR / "classifier_metadata.json"))
    print(f"[API] Pronto. Versao: {store.clf_metadata.get('model_version')}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Pipeline hibrido: Autoencoder (anomaly detection) + XGBoost (supervisionado).",
    version="1.0.0", lifespan=lifespan)

class TransactionInput(BaseModel):
    Time: float; Amount: float = Field(..., ge=0)
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

    @field_validator("Amount")
    @classmethod
    def amount_positive(cls, v):
        if v < 0: raise ValueError("Amount deve ser >= 0")
        return v

    def to_array(self):
        return np.array([getattr(self, f"V{i}") for i in range(1, 29)] +
                        [self.Amount, self.Time], dtype=np.float32)

class FraudResponse(BaseModel):
    fraud_probability: float
    fraud_prediction: bool
    risk_score: str
    reconstruction_error: float
    model_version: str
    latency_ms: float

class BatchResponse(BaseModel):
    results: list[FraudResponse]
    total: int
    fraud_count: int
    latency_ms: float

def classify_risk(prob):
    if prob < 0.3: return "low"
    if prob < 0.6: return "medium"
    if prob < 0.85: return "high"
    return "critical"

def predict_single(transaction):
    t0 = time.perf_counter()
    raw = transaction.to_array().reshape(1, -1)
    scaled = raw.copy()
    scaled[:, -2:] = store.scaler.transform(raw[:, -2:])
    with torch.no_grad():
        recon_error = float(store.autoencoder.reconstruction_error(torch.tensor(scaled)).item())
    X_enriched = np.column_stack([scaled, [[recon_error]]])
    prob = float(store.classifier.predict_proba(X_enriched)[0, 1])
    is_fraud = prob >= store.clf_metadata["best_threshold"]
    return FraudResponse(
        fraud_probability=round(prob, 4), fraud_prediction=is_fraud,
        risk_score=classify_risk(prob), reconstruction_error=round(recon_error, 6),
        model_version=store.clf_metadata.get("model_version", "v1"),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2))

@app.post("/predict", response_model=FraudResponse)
async def predict(transaction: TransactionInput):
    try:
        return predict_single(transaction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(transactions: list[TransactionInput]):
    if len(transactions) > 1000:
        raise HTTPException(status_code=400, detail="Maximo de 1000 transacoes por lote.")
    t0 = time.perf_counter()
    results = [predict_single(t) for t in transactions]
    return BatchResponse(results=results, total=len(results),
                         fraud_count=sum(r.fraud_prediction for r in results),
                         latency_ms=round((time.perf_counter() - t0) * 1000, 2))

@app.get("/health")
async def health():
    ok = all([store.autoencoder, store.classifier, store.scaler])
    return {"status": "healthy" if ok else "degraded", "models_loaded": ok,
            "model_version": store.clf_metadata.get("model_version", "unknown"),
            "autoencoder_pr_auc": store.ae_metadata.get("pr_auc"),
            "classifier_pr_auc": store.clf_metadata.get("pr_auc")}
