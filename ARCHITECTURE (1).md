# Architecture — Credit Card Fraud Detection

## Visão geral

Pipeline híbrido de dois estágios para detecção de fraude em tempo real:

```
Transação → Preprocessor → Autoencoder → XGBoost → Resposta
                                ↓
                        reconstruction_error
                        (anomaly score como feature extra)
```

---

## Por que pipeline híbrido?

Com apenas **0,17% de fraudes**, técnicas puramente supervisionadas sofrem com desbalanceamento extremo. A combinação unsupervised + supervised resolve isso em camadas:

| Estágio | Técnica | Objetivo |
|---|---|---|
| 1 | Autoencoder | Aprende o padrão "normal" sem precisar de labels de fraude |
| 2 | XGBoost + SMOTE | Decisão final com supervisão, usando o anomaly score como feature |

---

## Componentes

### 1. Preprocessor (`models/preprocessor.joblib`)

- `StandardScaler` aplicado em `Amount` e `Time`
- `V1`–`V28` já vêm normalizadas via PCA (dataset ULB)
- Fitted apenas no treino, aplicado em treino e teste para evitar data leakage

### 2. Autoencoder (`models/autoencoder.pt`)

```
Encoder: Linear(30→16) → ReLU → Dropout(0.2) → Linear(16→8) → ReLU
Decoder: Linear(8→16)  → ReLU → Dropout(0.2) → Linear(16→30)
```

**Decisões de design:**

| Decisão | Justificativa |
|---|---|
| Treinado só com legítimas | Não expõe o modelo a padrões de fraude — aprende apenas o "normal" |
| Gargalo de 8 dimensões | Compressão suficiente para capturar padrões sem memorizar ruído |
| Dropout 0.2 | Regularização — evita que o modelo reconstrua tudo perfeitamente |
| Arquitetura simples | Complexidade demais → generaliza para fraudes também |

**Threshold:** Percentil 95 dos erros de reconstrução das transações legítimas no conjunto de teste. Ou seja, 5% das legítimas são sinalizadas como suspeitas — aceitável para um primeiro filtro.

**Loss:** MSE por amostra → `reconstruction_error = mean((x - x̂)²)`

**Early stopping:** Patience de 7 épocas monitorando `val_loss` nas legítimas.

### 3. Classifier (`models/classifier.joblib`)

**XGBoost** treinado com as **31 features** (30 originais + `reconstruction_error`):

```python
XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg/pos,   # peso extra na classe fraude
    eval_metric="aucpr",        # PR-AUC como métrica de early stopping
    early_stopping_rounds=30,
)
```

**Por que XGBoost e não MLP?**
- Mais robusto com features de escalas diferentes
- Melhor interpretabilidade (feature importance nativa)
- Menos sensível a hiperparâmetros que redes neurais
- `scale_pos_weight` lida nativamente com desbalanceamento

---

## Tratamento de desbalanceamento

### SMOTE (Synthetic Minority Oversampling Technique)

Aplicado **após** a divisão treino/teste, **apenas no treino**:

```
sampling_strategy=0.1 → fraudes elevadas para 10% do treino
```

Não chegamos a 50/50 intencionalmente — preservamos alguma realidade na distribuição para evitar overfitting.

**Sequência correta para evitar data leakage:**

```
1. train_test_split (stratified)
2. SMOTE apenas em X_train, y_train
3. Avaliação sempre em X_test original (sem SMOTE)
```

### scale_pos_weight

Calculado como `n_negativos / n_positivos` após o SMOTE — camada extra de segurança além do oversampling.

### Threshold calibrado

Não usamos 0.5. O threshold é encontrado maximizando F1 na curva Precision-Recall:

```python
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1 = 2 * p * r / (p + r)
best_threshold = thresholds[argmax(f1)]
```

---

## Decisão de métricas

> "Com 0,17% de fraude, um modelo que chuta legítimo para tudo acerta 99,83% — mas é inútil."

| Métrica | Por quê usar |
|---|---|
| **PR-AUC** (principal) | AUC-ROC é enganoso com desbalanceamento extremo — não distingue bem modelos na classe minoritária |
| **Recall** | Minimizar fraudes não detectadas — FN tem custo financeiro real |
| **Precisão** | Evitar bloquear clientes legítimos — FP gera experiência ruim e custo operacional |
| **F1** | Equilíbrio para calibração do threshold |

### Por que PR-AUC e não AUC-ROC?

A AUC-ROC mede a área sob a curva TPR vs FPR. Com 99,83% de legítimas, mesmo um modelo ruim tem FPR baixíssimo, inflando artificialmente a AUC-ROC.

A PR-AUC mede precisão vs recall — focada exatamente na classe minoritária onde o problema existe.

---

## Fluxo de inferência

```
1. TransactionInput recebido via POST /predict

2. to_array() → [V1...V28, Amount, Time] como np.float32

3. StandardScaler.transform() em Amount e Time

4. FraudAutoencoder.reconstruction_error(tensor)
   → float: anomaly score

5. np.column_stack([scaled_features, recon_error])
   → shape (1, 31)

6. XGBClassifier.predict_proba(X_enriched)[:, 1]
   → fraud_probability

7. fraud_probability >= best_threshold
   → fraud_prediction (bool)

8. classify_risk(fraud_probability)
   → risk_score: "low" | "medium" | "high" | "critical"

9. FraudResponse retornada com latency_ms
```

**Latência alvo:** < 50ms (todos os modelos carregados em memória no startup da API)

---

## Estrutura de arquivos

```
creditcard-fraud/
├── data/
│   ├── raw/                        # CSV original do Kaggle (não commitado)
│   └── processed/                  # Arrays .npy gerados pelo pipeline
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       └── X_train_legit.npy       # Só legítimas — input do Autoencoder
├── models/
│   ├── preprocessor.joblib         # StandardScaler fitted
│   ├── autoencoder.pt              # Pesos do Autoencoder (melhor checkpoint)
│   ├── autoencoder_metadata.json   # threshold, pr_auc, input_dim
│   ├── classifier.joblib           # XGBoost fitted
│   └── classifier_metadata.json   # best_threshold, pr_auc, model_version
├── notebooks/
│   ├── 01_eda_fraud.ipynb          # Análise exploratória
│   └── 02_model_evaluation.ipynb  # Comparação de modelos e curvas
├── src/
│   ├── api/main.py                 # FastAPI — endpoints de inferência
│   ├── data/make_dataset.py        # Preprocessing + splits
│   ├── features/build_features.py # SMOTE
│   └── models/
│       ├── train_autoencoder.py   # Treino do Autoencoder
│       └── train_classifier.py    # Treino do XGBoost
├── tests/test_fraud.py            # Testes automatizados da API
├── Dockerfile                     # Multi-stage build
├── ARCHITECTURE.md
└── requirements.txt
```

---

## Reprodução completa

```bash
# 1. Clonar e configurar ambiente
git clone https://github.com/Henry3151/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
python -m venv .venv && source .venv/bin/activate  # ou .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Dataset
# Baixar creditcard.csv de https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Colocar em data/raw/creditcard.csv

# 3. Pipeline
python src/data/make_dataset.py
python src/models/train_autoencoder.py
python src/models/train_classifier.py

# 4. API
uvicorn src.api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs

# 5. Testes
pytest tests/ -v

# 6. Docker
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

---

## Referências

- Dal Pozzolo, A. et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI.
- Chawla, N. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR.
- Dataset: [ULB Machine Learning Group — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
