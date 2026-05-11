# -*- coding: utf-8 -*-
"""
pytest test suite — Customer Segmentation
Uso: pytest tests/ -v
"""
from __future__ import annotations
from unittest.mock import MagicMock
import numpy as np
import pytest
from fastapi.testclient import TestClient

from domain.entities.customer import Customer, CustomerSegment
from domain.exceptions import ValidationError
from use_cases.segment_customers import SegmentCustomers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Cria um Customer valido com valores RFM tipicos para testes
def _make_customer(cid: str = "C001", r=30.0, f=5.0, m=200.0) -> Customer:
    return Customer(customer_id=cid, recency=r, frequency=f, monetary=m)


# Cria um CustomerSegment mock para testes de API
def _make_segment(cid: str = "C001") -> CustomerSegment:
    return CustomerSegment(
        customer_id=cid, cluster_id=0, segment_label="Champions",
        umap_x=1.0, umap_y=2.0, rfm_score=0.9,
    )


# Fixture do FastAPI TestClient com dependencias mockadas
@pytest.fixture
def client():
    from api.main import app
    from api.dependencies import get_segment_use_case, get_metrics_use_case
    from use_cases.get_segmentation_metrics import SegmentationMetrics

    mock_seg = MagicMock()
    mock_seg.execute.return_value = [_make_segment()]

    mock_metrics = MagicMock()
    mock_metrics.execute.return_value = SegmentationMetrics(
        n_clusters=5, silhouette_score=0.42,
        inertia=1200.0, n_customers=4000, metadata={},
    )

    app.dependency_overrides[get_segment_use_case]  = lambda: mock_seg
    app.dependency_overrides[get_metrics_use_case]  = lambda: mock_metrics

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


VALID_PAYLOAD = {"customer_id": "C001", "recency": 30.0, "frequency": 5.0, "monetary": 200.0}


# ---------------------------------------------------------------------------
# 1. Domain entity tests
# ---------------------------------------------------------------------------

# Verifica que Customer aceita valores RFM validos e retorna array de shape (3,)
def test_customer_valid():
    c = _make_customer()
    assert c.to_numpy().shape == (3,)


# Verifica que Customer rejeita recency negativa com ValueError
def test_customer_negative_recency():
    with pytest.raises(ValueError):
        Customer(customer_id="x", recency=-1.0, frequency=5.0, monetary=100.0)


# Verifica que Customer rejeita valores NaN com ValueError
def test_customer_nan_raises():
    with pytest.raises(ValueError):
        Customer(customer_id="x", recency=float("nan"), frequency=5.0, monetary=100.0)


# Verifica que is_high_value retorna True para Champions
def test_segment_high_value_champions():
    s = _make_segment()
    assert s.is_high_value is True


# Verifica que is_high_value retorna False para Lost
def test_segment_not_high_value_lost():
    s = CustomerSegment("x", 4, "Lost", 0.0, 0.0, 0.1)
    assert s.is_high_value is False


# ---------------------------------------------------------------------------
# 2. Use case tests
# ---------------------------------------------------------------------------

# Verifica que SegmentCustomers chama model.predict com a lista correta
def test_segment_use_case_calls_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = [_make_segment()]
    uc = SegmentCustomers(model=mock_model)
    customers = [_make_customer()]
    result = uc.execute(customers)
    mock_model.predict.assert_called_once_with(customers)
    assert len(result) == 1


# Verifica que SegmentCustomers rejeita lista vazia com ValidationError
def test_segment_use_case_rejects_empty():
    mock_model = MagicMock()
    uc = SegmentCustomers(model=mock_model)
    with pytest.raises(ValidationError, match="empty"):
        uc.execute([])


# Verifica que SegmentCustomers rejeita IDs duplicados com ValidationError
def test_segment_use_case_rejects_duplicate_ids():
    mock_model = MagicMock()
    uc = SegmentCustomers(model=mock_model)
    customers = [_make_customer("C001"), _make_customer("C001")]
    with pytest.raises(ValidationError, match="Duplicate"):
        uc.execute(customers)


# ---------------------------------------------------------------------------
# 3. API endpoint tests
# ---------------------------------------------------------------------------

# Verifica que /segment retorna 200 com payload valido
def test_segment_endpoint_200(client):
    r = client.post("/segment", json=VALID_PAYLOAD)
    assert r.status_code == 200


# Verifica que /segment retorna os campos esperados na resposta
def test_segment_endpoint_fields(client):
    data = client.post("/segment", json=VALID_PAYLOAD).json()
    assert {"cluster_id", "segment_label", "rfm_score", "umap_x", "umap_y"}.issubset(data)


# Verifica que /segment retorna 422 quando monetary e negativo
def test_segment_endpoint_negative_monetary(client):
    r = client.post("/segment", json={**VALID_PAYLOAD, "monetary": -1.0})
    assert r.status_code == 422


# Verifica que /segment/batch retorna total_customers correto
def test_batch_endpoint_count(client):
    payload = {"customers": [VALID_PAYLOAD]}
    r = client.post("/segment/batch", json=payload)
    assert r.status_code == 200
    assert r.json()["total_customers"] == 1


# Verifica que /health retorna status ok e silhouette_score correto
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["silhouette_score"] == pytest.approx(0.42)
