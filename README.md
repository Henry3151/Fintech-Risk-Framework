# 💳 Credit Card Fraud Detection

> Pipeline híbrido de detecção de fraude em tempo real combinando **Anomaly Detection** (Autoencoder) e **classificação supervisionada** (XGBoost + SMOTE).

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange?logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker&logoColor=white)

---

## 📌 Contexto do problema

Com apenas **0,17% de fraudes** no dataset, um modelo que classifica tudo como legítimo acerta 99,83% das vezes — e é completamente inútil. Esse projeto ataca exatamente esse problema:

| Desafio | Solução |
|---|---|
| Desbalanceamento extremo (0,17%) | SMOTE + `scale_pos_weight` no XGBoost |
| Fraudes sem padrão supervisionado | Autoencoder treinado só com legítimas |
| Threshold padrão 0.5 inadequado | Calibração via curva Precision-Recall |
| Latência < 50ms antes da aprovação | Pipeline otimizado em memória |

---

## 🏗️ Arquitetura

```
Transação → Preprocessor → Autoencoder → XGBoost → Resposta
                                ↓
                        reconstruction_error
                        (anomaly score como feature extra)
```

**Estágio 1 — Autoencoder (unsupervised)**
Treinado apenas com transações legítimas. Aprende o padrão normal. Fraudes geram alto erro de reconstrução.

**Estágio 2 — XGBoost (supervised)**
Recebe as 30 features originais + `reconstruction_error` como feature extra. Treinado com SMOTE para balancear o dataset.

---

## 📊 Dataset

**Credit Card Fraud Detection** — Université Libre de Bruxelles (ULB)

- 284.807 transações europeias de setembro de 2013
- 492 fraudes (**0,17%**)
- Features `V1`–`V28`: componentes PCA de dados reais anonimizados
- Features originais: `Time`, `Amount`, `Class`

📥 Download: [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📈 Métricas esperadas

| Modelo | PR-AUC | Recall | Precisão |
|---|---|---|---|
| Logistic Regression | 0.62 | 0.71 | 0.84 |
| Random Forest | 0.78 | 0.79 | 0.89 |
| XGBoost + SMOTE | 0.89 | 0.88 | 0.91 |
| **Autoencoder + XGBoost** | **0.94** | **0.92** | **0.93** |

> **Por que PR-AUC e não AUC-ROC?**
> Com 0,17% de fraudes, a AUC-ROC é otimista demais. A PR-AUC mede performance exatamente na classe minoritária — onde importa.

---

## 🚀 Como executar

### 1. Pré-requisitos

```bash
git clone https://github.com/Henry3151/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset

Baixe `creditcard.csv` do Kaggle e coloque em `data/raw/creditcard.csv`.

### 3. Pipeline completo

```bash
# Preparar dados
python src/data/make_dataset.py

# Treinar Autoencoder
python src/models/train_autoencoder.py

# Treinar XGBoost Classifier
python src/models/train_classifier.py

# Subir API
uvicorn src.api.main:app --reload --port 8000
```

### 4. Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

### 5. Testes

```bash
pytest tests/ -v
```

---

## 🔌 API

Documentação interativa disponível em `http://localhost:8000/docs` após subir a API.

### `POST /predict`

```json
{
  "Time": 406.0,
  "Amount": 2125.87,
  "V1": -3.04, "V2": -3.16, "V3": 1.09,
  "...": "..."
}
```

**Response:**

```json
{
  "fraud_probability": 0.9231,
  "fraud_prediction": true,
  "risk_score": "critical",
  "reconstruction_error": 0.847,
  "model_version": "autoencoder_v1",
  "latency_ms": 12.4
}
```

### `POST /predict/batch`

Processa até 1000 transações em lote.

### `GET /health`

Status dos modelos carregados.

### Risk Score

| Score | Probabilidade | Ação sugerida |
|---|---|---|
| `low` | < 30% | Aprovar |
| `medium` | 30–60% | Monitorar |
| `high` | 60–85% | Revisão manual |
| `critical` | > 85% | Bloquear |

---

## 📁 Estrutura do projeto

```
creditcard-fraud/
├── data/
│   ├── raw/                    # CSV original (não commitado)
│   └── processed/              # Arrays .npy gerados pelo pipeline
├── models/
│   ├── preprocessor.joblib
│   ├── autoencoder.pt
│   ├── autoencoder_metadata.json
│   ├── classifier.joblib
│   └── classifier_metadata.json
├── notebooks/
│   ├── 01_eda_fraud.ipynb
│   └── 02_model_evaluation.ipynb
├── src/
│   ├── api/main.py
│   ├── data/make_dataset.py
│   ├── features/build_features.py
│   └── models/
│       ├── train_autoencoder.py
│       └── train_classifier.py
├── tests/test_fraud.py
├── Dockerfile
├── ARCHITECTURE.md
└── requirements.txt
```

---

## 🧠 O que esse projeto demonstra

- Tratamento de **desbalanceamento extremo** (0,17%) com SMOTE e threshold calibrado
- **Anomaly detection** com Autoencoder — não só classificação supervisionada
- Diferença entre **AUC-ROC e PR-AUC** e quando usar cada uma
- Pensamento em **custo de negócio** (FN = fraude passa, FP = cliente bloqueado)
- Arquitetura de **modelo híbrido** (unsupervised + supervised)
- API de **inferência em tempo real** com latência < 50ms
- Pipeline reproducível com **Docker**

---

## 📚 Referências

- Dal Pozzolo, A. et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI.
- Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
