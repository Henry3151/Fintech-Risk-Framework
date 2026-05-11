# -*- coding: utf-8 -*-
"""
pytest test suite — Credit Score
Uso: pytest tests/ -v
"""
from __future__ import annotations
from unittest.mock import MagicMock
import numpy as np
import pytest
from fastapi.testclient import TestClient

from domain.entities.credit_applicant import CreditApplicant, CreditScore
from domain.exceptions import ValidationError
from use_cases.score_applicant import ScoreApplicant, ScoreApplicantBatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Cria um CreditApplicant valido com perfil de bom pagador para testes
def _make_applicant(aid="A001", age=45, util=0.15, late90=0) -> CreditApplicant:
    return CreditApplicant(
        applicant_id=aid, revolving_utilization=util, age=age,
        late_30_59_days=0, debt_ratio=0.35, monthly_income=6500.0,
        open_credit_lines=8, late_90_days=late90,
        real_estate_loans=1, late_60_89_days=0, dependents=2,
    )


# Cria um CreditScore mock para testes de API
def _make_score(aid="A001", score=750) -> CreditScore:
    return CreditScore(
        applicant_id=aid, score=score, pd=round(1 - score/1000, 4),
        risk_grade=CreditScore.score_to_grade(score),
        recommendation=CreditScore.score_to_recommendation(score),
        top_factors=[{"feature": "RevolvingUtilizationOfUnsecuredLines",
                      "shap_value": -0.12, "impact": "decreases_risk"}],
        latency_ms=15.0,
    )


# Fixture do FastAPI TestClient com dependencias mockadas
@pytest.fixture
def client():
    from api.main import app
    from api.dependencies import get_score_use_case, get_score_batch_use_case, get_metrics_use_case
    from use_cases.get_scoring_metrics import ScoringMetrics

    mock_score = MagicMock()
    mock_score.execute.return_value = _make_score()

    mock_batch = MagicMock()
    mock_batch.execute.return_value = [_make_score()]

    mock_metrics = MagicMock()
    mock_metrics.execute.return_value = ScoringMetrics(
        auc_roc=0.8654, pr_auc=0.4231, ks_statistic=0.5123,
        brier_score=0.0412, n_train=120000, metadata={},
    )

    app.dependency_overrides[get_score_use_case]       = lambda: mock_score
    app.dependency_overrides[get_score_batch_use_case] = lambda: mock_batch
    app.dependency_overrides[get_metrics_use_case]     = lambda: mock_metrics

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


VALID_PAYLOAD = {
    "applicant_id": "A001",
    "RevolvingUtilizationOfUnsecuredLines": 0.15,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 6500.0,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2,
}


# ---------------------------------------------------------------------------
# 1. Domain entity tests
# ---------------------------------------------------------------------------

# Verifica que CreditApplicant aceita valores validos e retorna array de shape (13,)
def test_applicant_valid():
    a = _make_applicant()
    assert a.to_numpy().shape == (13,)


# Verifica que CreditApplicant rejeita age <= 0 com ValueError
def test_applicant_invalid_age():
    with pytest.raises(ValueError):
        CreditApplicant(applicant_id="x", revolving_utilization=0.1,
                        age=0, late_30_59_days=0, debt_ratio=0.3,
                        monthly_income=5000, open_credit_lines=5,
                        late_90_days=0, real_estate_loans=1,
                        late_60_89_days=0, dependents=1)


# Verifica que CreditApplicant rejeita revolving_utilization > 1
def test_applicant_invalid_utilization():
    with pytest.raises(ValueError):
        CreditApplicant(applicant_id="x", revolving_utilization=1.5,
                        age=30, late_30_59_days=0, debt_ratio=0.3,
                        monthly_income=5000, open_credit_lines=5,
                        late_90_days=0, real_estate_loans=1,
                        late_60_89_days=0, dependents=1)


# Verifica que pd_to_score converte corretamente PD em score 0-1000
def test_score_pd_to_score():
    assert CreditScore.pd_to_score(0.0) == 1000
    assert CreditScore.pd_to_score(1.0) == 0
    assert CreditScore.pd_to_score(0.25) == 750


# Verifica que score_to_grade mapeia corretamente os ranges
def test_score_to_grade():
    assert CreditScore.score_to_grade(850) == "A"
    assert CreditScore.score_to_grade(700) == "B"
    assert CreditScore.score_to_grade(550) == "C"
    assert CreditScore.score_to_grade(400) == "D"
    assert CreditScore.score_to_grade(200) == "E"


# Verifica que is_approved retorna True apenas para APPROVE
def test_score_is_approved():
    s = _make_score(score=750)
    assert s.is_approved is True
    s2 = _make_score(score=300)
    assert s2.is_approved is False


# ---------------------------------------------------------------------------
# 2. Use case tests
# ---------------------------------------------------------------------------

# Verifica que ScoreApplicant chama model.predict com o applicant correto
def test_score_use_case_calls_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = _make_score()
    uc = ScoreApplicant(model=mock_model)
    a  = _make_applicant()
    result = uc.execute(a)
    mock_model.predict.assert_called_once_with(a)
    assert isinstance(result, CreditScore)


# Verifica que ScoreApplicant rejeita menores de 18 anos com ValidationError
def test_score_use_case_rejects_minor():
    mock_model = MagicMock()
    uc = ScoreApplicant(model=mock_model)
    with pytest.raises(ValidationError, match="18"):
        uc.execute(_make_applicant(age=17))


# Verifica que ScoreApplicantBatch rejeita lista vazia com ValidationError
def test_batch_use_case_rejects_empty():
    mock_model = MagicMock()
    uc = ScoreApplicantBatch(model=mock_model)
    with pytest.raises(ValidationError, match="at least one"):
        uc.execute([])


# ---------------------------------------------------------------------------
# 3. API endpoint tests
# ---------------------------------------------------------------------------

# Verifica que /score retorna 200 com payload valido
def test_score_endpoint_200(client):
    r = client.post("/score", json=VALID_PAYLOAD)
    assert r.status_code == 200


# Verifica que /score retorna os campos esperados na resposta
def test_score_endpoint_fields(client):
    data = client.post("/score", json=VALID_PAYLOAD).json()
    assert {"score", "pd", "risk_grade", "recommendation", "top_factors"}.issubset(data)


# Verifica que /score retorna 422 para age invalido
def test_score_endpoint_invalid_age(client):
    r = client.post("/score", json={**VALID_PAYLOAD, "age": 15})
    assert r.status_code == 422


# Verifica que /score/batch retorna total_applicants correto
def test_batch_endpoint_count(client):
    payload = {"applicants": [VALID_PAYLOAD]}
    r = client.post("/score/batch", json=payload)
    assert r.status_code == 200
    assert r.json()["total_applicants"] == 1


# Verifica que /health retorna status ok e auc_roc correto
def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["auc_roc"] == pytest.approx(0.8654)
