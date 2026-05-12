# -*- coding: utf-8 -*-
"""
pytest test suite — Default Prediction
Uso: pytest tests/ -v
"""
from __future__ import annotations
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from domain.entities.loan import Loan, DefaultPrediction
from domain.exceptions import ValidationError
from use_cases.predict_default import PredictDefault, PredictDefaultBatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Cria um Loan valido com perfil de baixo risco para testes
def _make_loan(lid="L001", grade="C", int_rate=13.99, dti=18.5) -> Loan:
    return Loan(
        loan_id=lid, loan_amnt=15000.0, int_rate=int_rate,
        grade=grade, emp_length_years=5.0, annual_inc=65000.0,
        dti=dti, fico_range_low=690.0, open_acc=10,
        revol_util=45.0, total_acc=25, inq_last_6mths=1,
        pub_rec=0, term_months=36,
    )


# Cria um DefaultPrediction mock para testes
def _make_prediction(lid="L001", risk_tier="LOW") -> DefaultPrediction:
    return DefaultPrediction(
        loan_id=lid, survival_at_12m=0.95, survival_at_24m=0.88,
        survival_at_36m=0.82, median_survival_months=48.0,
        hazard_ratio=0.85, risk_tier=risk_tier,
        pd_12m=0.05, latency_ms=10.0,
    )


# Fixture do FastAPI TestClient com dependencias mockadas
@pytest.fixture
def client():
    from api.main import app
    from api.dependencies import get_predict_use_case, get_predict_batch_use_case, get_metrics_use_case
    from use_cases.predict_default import DefaultMetrics

    mock_pred = MagicMock()
    mock_pred.execute.return_value = _make_prediction()

    mock_batch = MagicMock()
    mock_batch.execute.return_value = [_make_prediction()]

    mock_metrics = MagicMock()
    mock_metrics.execute.return_value = DefaultMetrics(
        c_index=0.72, brier_score_12m=0.08,
        n_events=25000, n_censored=135000,
        n_train=160000, metadata={},
    )

    app.dependency_overrides[get_predict_use_case]       = lambda: mock_pred
    app.dependency_overrides[get_predict_batch_use_case] = lambda: mock_batch
    app.dependency_overrides[get_metrics_use_case]       = lambda: mock_metrics

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


VALID_PAYLOAD = {
    "loan_id": "L001", "loan_amnt": 15000.0, "int_rate": 13.99,
    "grade": "C", "emp_length_years": 5.0, "annual_inc": 65000.0,
    "dti": 18.5, "fico_range_low": 690.0, "open_acc": 10,
    "revol_util": 45.0, "total_acc": 25, "inq_last_6mths": 1,
    "pub_rec": 0, "term_months": 36,
}


# ---------------------------------------------------------------------------
# 1. Domain entity tests
# ---------------------------------------------------------------------------

# Verifica que Loan aceita valores validos e retorna array de shape (13,)
def test_loan_valid():
    l = _make_loan()
    assert l.to_numpy().shape == (13,)


# Verifica que Loan rejeita grade invalida com ValueError
def test_loan_invalid_grade():
    with pytest.raises(ValueError):
        Loan(loan_id="x", loan_amnt=1000, int_rate=10, grade="Z",
             emp_length_years=5, annual_inc=50000, dti=15,
             fico_range_low=700, open_acc=5, revol_util=30,
             total_acc=10, inq_last_6mths=0, pub_rec=0, term_months=36)


# Verifica que Loan rejeita term_months invalido com ValueError
def test_loan_invalid_term():
    with pytest.raises(ValueError):
        Loan(loan_id="x", loan_amnt=1000, int_rate=10, grade="A",
             emp_length_years=5, annual_inc=50000, dti=15,
             fico_range_low=700, open_acc=5, revol_util=30,
             total_acc=10, inq_last_6mths=0, pub_rec=0, term_months=48)


# Verifica que classify_risk mapeia corretamente as faixas de risco
def test_classify_risk():
    assert DefaultPrediction.classify_risk(0.03) == "LOW"
    assert DefaultPrediction.classify_risk(0.10) == "MEDIUM"
    assert DefaultPrediction.classify_risk(0.25) == "HIGH"
    assert DefaultPrediction.classify_risk(0.40) == "VERY_HIGH"


# Verifica que is_low_risk retorna True apenas para LOW
def test_is_low_risk():
    p = _make_prediction(risk_tier="LOW")
    assert p.is_low_risk is True
    p2 = _make_prediction(risk_tier="HIGH")
    assert p2.is_low_risk is False


# Verifica que from_dict cria Loan corretamente a partir de dicionario
def test_loan_from_dict():
    data = {
        "loan_amnt": 15000, "int_rate": 13.99, "grade": "C",
        "emp_length_years": 5, "annual_inc": 65000, "dti": 18.5,
        "fico_range_low": 690, "open_acc": 10, "revol_util": 45,
        "total_acc": 25, "inq_last_6mths": 1, "pub_rec": 0, "term_months": 36,
    }
    l = Loan.from_dict(data, loan_id="L999")
    assert l.loan_id == "L999"
    assert l.grade == "C"


# ---------------------------------------------------------------------------
# 2. Use case tests
# ---------------------------------------------------------------------------

# Verifica que PredictDefault chama model.predict com o loan correto
def test_predict_use_case_calls_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_prediction()
    uc = PredictDefault(model=mock_model)
    l = _make_loan()
    result = uc.execute(l)
    mock_model.predict.assert_called_once_with(l)
    assert isinstance(result, DefaultPrediction)


# Verifica que PredictDefault rejeita dti invalido com ValidationError
def test_predict_use_case_rejects_invalid_dti():
    mock_model = MagicMock()
    uc = PredictDefault(model=mock_model)
    with pytest.raises(ValidationError):
        uc.execute(_make_loan(dti=150))


# Verifica que PredictDefaultBatch rejeita lista vazia com ValidationError
def test_batch_use_case_rejects_empty():
    mock_model = MagicMock()
    uc = PredictDefaultBatch(model=mock_model)
    with pytest.raises(ValidationError, match="at least one"):
        uc.execute([])


# ---------------------------------------------------------------------------
# 3. API endpoint tests
# ---------------------------------------------------------------------------

# Verifica que /predict retorna 200 com payload valido
def test_predict_endpoint_200(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200


# Verifica que /predict retorna os campos da curva de sobrevivencia
def test_predict_endpoint_fields(client):
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    assert {"survival_at_12m", "survival_at_24m", "survival_at_36m",
            "risk_tier", "pd_12m", "hazard_ratio"}.issubset(data)


# Verifica que /predict retorna 422 para grade invalida
def test_predict_endpoint_invalid_grade(client):
    r = client.post("/predict", json={**VALID_PAYLOAD, "grade": "Z"})
    assert r.status_code == 422


# Verifica que /predict/batch retorna total_loans correto
def test_batch_endpoint_count(client):
    r = client.post("/predict/batch", json={"loans": [VALID_PAYLOAD]})
    assert r.status_code == 200
    assert r.json()["total_loans"] == 1


# Verifica que /health retorna status ok e c_index correto
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["c_index"] == pytest.approx(0.72)
