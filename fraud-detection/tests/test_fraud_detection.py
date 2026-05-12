"""
pytest test suite — 14 tests covering:
  - Schema validation (Pydantic)
  - Domain entity behaviour
  - Use case business rules
  - API endpoints (FastAPI TestClient with mocked dependencies)

Run: pytest tests/ -v
"""
from __future__ import annotations

import time
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from domain.entities.transaction import Transaction, FraudPrediction
from domain.exceptions import ValidationError
from use_cases.predict_fraud import PredictFraud
from use_cases.predict_fraud_batch import PredictFraudBatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_COUNT = 30

def _make_features(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(FEATURE_COUNT)


def _make_transaction(seed: int = 0, tx_id: str = "txn-test") -> Transaction:
    return Transaction(features=_make_features(seed), transaction_id=tx_id)


def _make_prediction(is_fraud: bool = False, proba: float = 0.05) -> FraudPrediction:
    return FraudPrediction(
        transaction_id="txn-test",
        fraud_probability=proba,
        is_fraud=is_fraud,
        reconstruction_error=0.01,
        latency_ms=10.0,
    )


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = _make_prediction(is_fraud=False, proba=0.05)
    model.predict_batch.return_value = [_make_prediction(is_fraud=False, proba=0.05)]
    return model


@pytest.fixture
def client():
    from api.main import app
    from api.dependencies import get_predict_fraud_use_case, get_predict_fraud_batch_use_case, get_model_metrics_use_case
    from use_cases.get_model_metrics import ModelMetrics

    mock_use_case = MagicMock()
    mock_use_case.execute.return_value = _make_prediction(is_fraud=False, proba=0.05)

    mock_batch = MagicMock()
    mock_batch.execute.return_value = [_make_prediction(is_fraud=False, proba=0.05)]

    mock_metrics_uc = MagicMock()
    mock_metrics_uc.execute.return_value = ModelMetrics(
        pr_auc=0.8665,
        autoencoder_threshold=0.8249,
        classifier_threshold=0.9810,
        autoencoder_pr_auc=0.4603,
        metadata={},
    )

    app.dependency_overrides[get_predict_fraud_use_case] = lambda: mock_use_case
    app.dependency_overrides[get_predict_fraud_batch_use_case] = lambda: mock_batch
    app.dependency_overrides[get_model_metrics_use_case] = lambda: mock_metrics_uc

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


VALID_PAYLOAD = {
    "Time": 0.0, "Amount": 149.62,
    **{f"V{i}": float(i) * 0.01 for i in range(1, 29)},
}


# ---------------------------------------------------------------------------
# 1. Domain entity tests
# ---------------------------------------------------------------------------

def test_transaction_correct_shape():
    t = _make_transaction()
    assert t.features.shape == (FEATURE_COUNT,)


def test_transaction_wrong_shape_raises():
    with pytest.raises(ValueError, match="Expected 30 features"):
        Transaction(features=np.zeros(10))


def test_transaction_from_dict():
    data = {"Time": 1.0, "Amount": 50.0, **{f"V{i}": float(i) for i in range(1, 29)}}
    t = Transaction.from_dict(data, transaction_id="abc")
    assert t.transaction_id == "abc"
    assert t.features.shape == (30,)


def test_fraud_prediction_risk_label_high():
    p = FraudPrediction("x", 0.95, True, 0.5, 10.0)
    assert p.risk_label == "HIGH"


def test_fraud_prediction_risk_label_medium():
    p = FraudPrediction("x", 0.75, True, 0.5, 10.0)
    assert p.risk_label == "MEDIUM"


def test_fraud_prediction_risk_label_low():
    p = FraudPrediction("x", 0.10, False, 0.01, 10.0)
    assert p.risk_label == "LOW"


# ---------------------------------------------------------------------------
# 2. Use case tests
# ---------------------------------------------------------------------------

def test_predict_fraud_calls_model(mock_model):
    use_case = PredictFraud(model=mock_model)
    tx = _make_transaction()
    result = use_case.execute(tx)
    mock_model.predict.assert_called_once_with(tx)
    assert isinstance(result, FraudPrediction)


def test_predict_fraud_rejects_nan(mock_model):
    features = _make_features()
    features[0] = float("nan")
    tx = Transaction(features=features)
    use_case = PredictFraud(model=mock_model)
    with pytest.raises(ValidationError, match="NaN"):
        use_case.execute(tx)


def test_predict_fraud_batch_rejects_empty(mock_model):
    use_case = PredictFraudBatch(model=mock_model)
    with pytest.raises(ValidationError, match="at least one"):
        use_case.execute([])


def test_predict_fraud_batch_rejects_oversized(mock_model):
    use_case = PredictFraudBatch(model=mock_model)
    batch = [_make_transaction(seed=i) for i in range(1001)]
    with pytest.raises(ValidationError, match="maximum"):
        use_case.execute(batch)


# ---------------------------------------------------------------------------
# 3. API endpoint tests
# ---------------------------------------------------------------------------

def test_predict_endpoint_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "risk_label" in data


def test_predict_endpoint_missing_field_returns_422(client):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "Amount"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_batch_endpoint_returns_200(client):
    payload = {"transactions": [VALID_PAYLOAD]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 1


def test_health_endpoint_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["pr_auc"] == pytest.approx(0.8665)
