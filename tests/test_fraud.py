# -*- coding: utf-8 -*-
"""
test_fraud.py - Testes automatizados da API.
Uso: pytest tests/test_fraud.py -v
"""
import pytest
from fastapi.testclient import TestClient

LEGIT = {
    "Time": 1000.0, "Amount": 45.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": -0.48, "V7": 0.21, "V8": 0.10, "V9": 0.14, "V10": -0.33,
    "V11": -0.17, "V12": -0.17, "V13": -0.49, "V14": -0.44, "V15": -0.02,
    "V16": -0.16, "V17": -0.05, "V18": -0.19, "V19": -0.03, "V20": 0.06,
    "V21": 0.04, "V22": 0.45, "V23": 0.22, "V24": 0.06, "V25": 0.29,
    "V26": -0.03, "V27": -0.07, "V28": -0.06,
}
SUSPICIOUS = {
    # Fraude real do dataset ULB (linha 492, Class=1)
    "Time": 406.0, "Amount": 0.0,
    "V1": -2.3122, "V2": 1.9520, "V3": -1.6099, "V4": 3.9979, "V5": -0.5222,
    "V6": -1.4265, "V7": -2.5374, "V8": 1.3917, "V9": -2.7701, "V10": -2.7723,
    "V11": 3.2020, "V12": -2.8999, "V13": -0.5952, "V14": -4.2893, "V15": 0.3897,
    "V16": -1.1407, "V17": -2.8301, "V18": -0.0168, "V19": 0.4170, "V20": 0.1269,
    "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3202, "V25": 0.0445,
    "V26": 0.1778, "V27": 0.2611, "V28": -0.1433,
}

@pytest.fixture(scope="module")
def client():
    try:
        from src.api.main import app
        with TestClient(app) as c:
            yield c
    except Exception as e:
        pytest.skip(f"Modelos nao encontrados. Treine antes. Erro: {e}")

class TestHealth:
    def test_200(self, client): assert client.get("/health").status_code == 200
    def test_models_loaded(self, client): assert client.get("/health").json()["models_loaded"]
    def test_version(self, client): assert client.get("/health").json()["model_version"] != "unknown"

class TestSchema:
    def test_predict_200(self, client): assert client.post("/predict", json=LEGIT).status_code == 200
    def test_fields(self, client):
        data = client.post("/predict", json=LEGIT).json()
        assert {"fraud_probability","fraud_prediction","risk_score",
                "reconstruction_error","model_version","latency_ms"}.issubset(data)
    def test_prob_range(self, client):
        assert 0 <= client.post("/predict", json=LEGIT).json()["fraud_probability"] <= 1
    def test_risk_valid(self, client):
        assert client.post("/predict", json=LEGIT).json()["risk_score"] in {"low","medium","high","critical"}
    def test_latency(self, client):
        assert client.post("/predict", json=LEGIT).json()["latency_ms"] < 50

class TestValidation:
    def test_negative_amount(self, client): assert client.post("/predict", json={**LEGIT,"Amount":-1}).status_code == 422
    def test_missing_field(self, client):
        assert client.post("/predict", json={k:v for k,v in LEGIT.items() if k!="V1"}).status_code == 422

class TestBehavior:
    def test_higher_recon_error(self, client):
        assert (client.post("/predict", json=SUSPICIOUS).json()["reconstruction_error"] >
                client.post("/predict", json=LEGIT).json()["reconstruction_error"])
    def test_higher_prob(self, client):
        assert (client.post("/predict", json=SUSPICIOUS).json()["fraud_probability"] >
                client.post("/predict", json=LEGIT).json()["fraud_probability"])

class TestBatch:
    def test_count(self, client):
        assert client.post("/predict/batch", json=[LEGIT, SUSPICIOUS]).json()["total"] == 2
    def test_too_large(self, client):
        assert client.post("/predict/batch", json=[LEGIT]*1001).status_code == 400
