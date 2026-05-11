# Fintech Risk Framework

> Portfolio de Data Science aplicado ao setor financeiro — modelos de risco, segmentacao e deteccao de anomalias em producao, todos seguindo Clean Architecture.

![Python](https://img.shields.io/badge/Python-3.14+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange?logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-55%20passed-brightgreen)
![Architecture](https://img.shields.io/badge/architecture-Clean%20Architecture-purple)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker&logoColor=white)
![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

---

## Projetos

| # | Projeto | Tecnica | Diferencial | Metrica | Status |
|---|---------|---------|-------------|---------|--------|
| 1 | [Credit Card Fraud Detection](#1-credit-card-fraud-detection) | Autoencoder + XGBoost | Pipeline hibrido unsupervised + supervised | PR-AUC 0.8665 | ✅ Concluido |
| 2 | [Customer Segmentation](#2-customer-segmentation) | K-Means + UMAP + RFM | Visualizacao 2D de alta dimensao | Silhouette 0.4271 | ✅ Concluido |
| 3 | [Credit Score](#3-credit-score) | LightGBM + SHAP | Explicabilidade regulatoria + fairness | AUC-ROC 0.9651 | ✅ Concluido |
| 4 | Default Prediction | Survival Analysis (Cox) | Modelagem de tempo ate o evento | — | 🔜 Em breve |

---

## Arquitetura

Todos os projetos seguem Clean Architecture com a mesma regra de dependencia:

```
api → use_cases → domain ← infrastructure
```

```
src/
├── domain/          # Entidades e contratos — zero dependencias externas
├── use_cases/       # Regras de negocio — orquestra sem conhecer frameworks
├── infrastructure/  # PyTorch, XGBoost, LightGBM, SHAP — implementacoes concretas
└── api/             # FastAPI, Pydantic, injecao de dependencia
```

---

## Compatibilidade

| Sistema | Suporte | Ativacao do ambiente virtual |
|---------|---------|------------------------------|
| Windows | ✅ | `.venv\Scripts\Activate.ps1` |
| Linux | ✅ | `source .venv/bin/activate` |
| macOS | ✅ | `source .venv/bin/activate` |

Todo o codigo Python e cross-platform. O pipeline de ML e a API rodam identicamente em qualquer OS.

---

## 1. Credit Card Fraud Detection

Pipeline hibrido de deteccao de fraude em tempo real combinando **Anomaly Detection** (Autoencoder PyTorch) e **classificacao supervisionada** (XGBoost + SMOTE).

### Dashboard

![Dashboard](reports/figures/dashboard.png)

### O problema

Com apenas **0,17% de fraudes** no dataset, um modelo que classifica tudo como legitimo acerta 99,83% das vezes — e e completamente inutil.

| Desafio | Solucao |
|---------|---------|
| Desbalanceamento extremo (0,17%) | SMOTE + `scale_pos_weight` no XGBoost |
| Fraudes sem padrao supervisionado | Autoencoder treinado so com transacoes legitimas |
| Threshold padrao 0.5 inadequado | Calibracao via curva Precision-Recall |
| Latencia < 50ms | Pipeline otimizado em memoria |

### Pipeline

```
Transacao → StandardScaler → Autoencoder → reconstruction_error
                                                    ↓
                             XGBoost([30 features + reconstruction_error])
                                                    ↓
                                          FraudPrediction
```

### Resultados

| Modelo | PR-AUC |
|--------|--------|
| Baseline aleatorio | 0.0017 |
| Logistic Regression | ~0.62 |
| Random Forest | ~0.78 |
| XGBoost + SMOTE | ~0.84 |
| **Autoencoder + XGBoost (este projeto)** | **0.8665** |

> **Por que PR-AUC e nao AUC-ROC?** Com 0,17% de fraudes, a AUC-ROC e otimista demais. A PR-AUC mede performance exatamente na classe minoritaria.

### API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 406.0, "Amount": 2125.87, "V1": -3.04, ...}'

# Resposta
{
  "fraud_probability": 0.9231,
  "is_fraud": true,
  "risk_label": "HIGH",
  "reconstruction_error": 0.847,
  "latency_ms": 12.4
}
```

| risk_label | Probabilidade | Acao sugerida |
|------------|---------------|---------------|
| `LOW` | < 50% | Aprovar |
| `MEDIUM` | 50-90% | Monitorar |
| `HIGH` | >= 90% | Bloquear |

### Como executar

```bash
pip install -r requirements.txt

# Windows
$env:PYTHONPATH = "src"
# Linux / macOS
export PYTHONPATH=src

python scripts/train_autoencoder.py
python scripts/train_classifier.py
uvicorn api.main:app --reload --port 8000
pytest tests/test_fraud_detection.py -v   # 14 testes
```

---

## 2. Customer Segmentation

Segmentacao de clientes usando **K-Means + UMAP** sobre features **RFM** (Recency, Frequency, Monetary) extraidas do dataset Online Retail UCI (541k transacoes, 5.819 clientes).

### Dashboard

![Dashboard](customer-segmentation/reports/figures/dashboard.png)

### O problema

Nem todo cliente e igual. Um banco ou fintech precisa saber quem sao seus **Champions**, quem esta **At Risk** e quem e **Lost** — para agir diferente com cada grupo.

| Desafio | Solucao |
|---------|---------|
| Outliers extremos em valor monetario | RobustScaler (resistente a outliers vs StandardScaler) |
| Numero de clusters | KMeans com 5 clusters otimizados por Silhouette Score |
| Alta dimensao dificil de visualizar | UMAP reduz para 2D mantendo estrutura de vizinhanca |
| Labels sem significado de negocio | Mapeamento automatico por valor monetario medio |

### Resultados

| Metrica | Valor |
|---------|-------|
| Silhouette Score | **0.4271** |
| Clientes segmentados | 5.819 |
| Clusters | 5 |
| Dataset | Online Retail UCI (2009-2011) |

### Segmentos

| Segmento | Perfil RFM | Acao de negocio |
|----------|-----------|-----------------|
| Champions | Baixo recency, alta freq, alto monetary | Programa de fidelidade VIP |
| Loyal Customers | Frequencia alta, valor medio | Upsell / cross-sell |
| At Risk | Alto recency, bom historico | Campanha de reativacao urgente |
| New Customers | Baixo recency, baixa freq | Onboarding e educacao |
| Lost | Alto recency, baixo valor | Desconto agressivo ou abandono |

### Como executar

```bash
cd customer-segmentation
pip install -r requirements.txt

# Windows
$env:PYTHONPATH = "src"
# Linux / macOS
export PYTHONPATH=src

python scripts/train_segmentation.py --data data/raw/online_retail.csv
uvicorn api.main:app --reload --port 8001
pytest tests/ -v   # 13 testes
```

---

## 3. Credit Score

Modelo de credit scoring com **LightGBM + Platt calibration + SHAP** e analise de vies algoritmico por faixa etaria. Dataset Give Me Some Credit (150k clientes, Kaggle).

### Dashboard

![Dashboard](credit-score/reports/figures/dashboard.png)

### O problema

Credit scoring requer mais do que boa performance — reguladores exigem **probabilidades calibradas** e **explicabilidade por cliente**.

| Desafio | Solucao |
|---------|---------|
| 19.8% de nulls em MonthlyIncome | Imputacao por mediana da faixa etaria |
| Outliers extremos (DebtRatio ate 329k) | Clip por percentil 99 + feature derivada |
| Probabilidades descalibradas | Platt scaling (CalibratedClassifierCV) |
| Explicabilidade regulatoria | SHAP TreeExplainer — top 3 fatores por cliente |
| Vies algoritmico por idade | Fairness analysis por faixa etaria |

### Pipeline

```
CSV Give Me Some Credit → CreditDataProcessor → feature engineering
                                    ↓
                         LightGBM(scale_pos_weight)
                                    ↓
                         CalibratedClassifierCV (Platt scaling)
                                    ↓
                         SHAP TreeExplainer
                                    ↓
                  CreditScore(score 0-1000, grade A-E, top_factors)
```

### Resultados

| Metrica | Valor | Benchmark de mercado |
|---------|-------|---------------------|
| AUC-ROC | **0.9651** | > 0.75 bom |
| PR-AUC | **0.6641** | > 0.40 bom para 6.7% default |
| KS Statistic | **0.8266** | > 0.40 excelente |
| Brier Score | **0.0389** | < 0.10 bem calibrado |

### Fairness Analysis

| Faixa etaria | Aprovacao | Inadimplencia real |
|---|---|---|
| 18-30 | 90.1% | 11.9% |
| 31-40 | 91.9% | 9.6% |
| 41-50 | 93.2% | 8.5% |
| 51-60 | 95.5% | 5.9% |
| 61-70 | 98.1% | 3.5% |
| 71+ | 99.3% | 2.1% |

> Taxas de aprovacao consistentes com inadimplencia real — sem vies discriminatorio por idade.

### Score e Grades

| Grade | Score | Recomendacao |
|-------|-------|--------------|
| A | 800-1000 | APPROVE |
| B | 650-799 | APPROVE |
| C | 500-649 | REVIEW |
| D | 350-499 | REVIEW |
| E | 0-349 | DENY |

### API

```bash
curl -X POST http://localhost:8002/score \
  -H "Content-Type: application/json" \
  -d '{
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
    "NumberOfDependents": 2
  }'

# Resposta
{
  "score": 820,
  "pd": 0.0180,
  "risk_grade": "A",
  "recommendation": "APPROVE",
  "top_factors": [
    {"feature": "NumberOfTimes90DaysLate", "shap_value": -0.42, "impact": "decreases_risk"},
    {"feature": "RevolvingUtilizationOfUnsecuredLines", "shap_value": -0.18, "impact": "decreases_risk"},
    {"feature": "total_late_payments", "shap_value": -0.11, "impact": "decreases_risk"}
  ],
  "latency_ms": 18.3
}
```

### Como executar

```bash
cd credit-score
pip install -r requirements.txt

# Windows
$env:PYTHONPATH = "src"
# Linux / macOS
export PYTHONPATH=src

python scripts/train_credit_score.py --data data/raw/cs-training.csv
uvicorn api.main:app --reload --port 8002
pytest tests/ -v   # 14 testes
```

---

## Stack tecnologico

| Categoria | Tecnologias |
|-----------|-------------|
| ML / DL | PyTorch, XGBoost, LightGBM, scikit-learn, SHAP, UMAP |
| API | FastAPI, Pydantic, Uvicorn |
| Tracking | MLflow |
| Testes | pytest, 55 testes automatizados |
| Infra | Docker multi-stage, venv |
| Arquitetura | Clean Architecture (domain / use_cases / infrastructure / api) |
| OS | Windows, Linux, macOS |

---

## O que esse portfolio demonstra

- **Clean Architecture aplicada a ML** — separacao real entre dominio, casos de uso, infraestrutura e API
- **Tratamento de desbalanceamento extremo** — SMOTE, threshold calibrado por PR-AUC, `scale_pos_weight`
- **Anomaly detection** com Autoencoder como feature engineering para modelo supervisionado
- **RFM + clustering** — padrao da industria financeira para segmentacao de clientes
- **UMAP para visualizacao** — reducao de alta dimensao mantendo estrutura de vizinhanca
- **Calibracao de probabilidade** (Platt scaling) — requisito regulatorio em credit scoring
- **Explicabilidade por cliente** com SHAP TreeExplainer — top 3 fatores por decisao
- **Fairness analysis** — deteccao de vies algoritmico por faixa etaria
- **Score 0-1000 com grades A-E** — padrao de mercado financeiro
- **APIs de inferencia em tempo real** — latencia < 50ms, endpoints `/predict`, `/segment`, `/score`, `/health`
- **55 testes automatizados** cobrindo entidades, casos de uso e endpoints
- **Cross-platform** — Windows, Linux e macOS

---

## Referencias

- Dal Pozzolo, A. et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI.
- McInnes, L. et al. (2018). *UMAP: Uniform Manifold Approximation and Projection*. arXiv:1802.03426.
- Niculescu-Mizil, A. & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning*. ICML.
- Dataset 1: [ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Dataset 2: [Online Retail UCI](https://archive.ics.uci.edu/dataset/352/online+retail)
- Dataset 3: [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)