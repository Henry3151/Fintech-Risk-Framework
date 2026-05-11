from __future__ import annotations
import time
from typing import List
import numpy as np
import pandas as pd
import torch
from domain.entities.transaction import Transaction, FraudPrediction
from domain.exceptions import InferenceError, ModelNotLoadedError
from domain.interfaces.model_interfaces import IFraudModel, IModelRepository

FEATURE_COLS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

class HybridFraudModel(IFraudModel):
    """
    Implementacao concreta de IFraudModel.
    Pipeline: StandardScaler -> Autoencoder (reconstruction_error) -> XGBoost.
    Artefatos carregados de forma lazy na primeira chamada de inferencia.
    """

    # Recebe o repositorio de artefatos e inicializa os atributos como None ate o primeiro uso
    def __init__(self, repository: IModelRepository) -> None:
        self._repository   = repository
        self._preprocessor = None
        self._autoencoder  = None
        self._classifier   = None
        self._ae_threshold: float | None = None
        self._clf_threshold: float | None = None
        self._device = torch.device('cpu')

    # Executa inferencia em uma unica transacao e retorna FraudPrediction com probabilidade e latencia
    def predict(self, transaction: Transaction) -> FraudPrediction:
        self._ensure_loaded()
        start = time.perf_counter()
        try:
            features_2d       = transaction.to_numpy().reshape(1, -1)
            scaled, rec_error = self._preprocess_and_encode(features_2d)
            enriched = np.hstack([scaled, [[rec_error]]])
            proba    = float(self._classifier.predict_proba(enriched)[0, 1])
            is_fraud = proba >= self._clf_threshold
        except Exception as exc:
            raise InferenceError(f'Inference failed: {exc}') from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return FraudPrediction(
            transaction_id=transaction.transaction_id,
            fraud_probability=proba,
            is_fraud=is_fraud,
            reconstruction_error=rec_error,
            latency_ms=round(latency_ms, 2),
        )

    # Executa inferencia em lote com uma unica passagem pelo preprocessor e autoencoder
    def predict_batch(self, transactions: List[Transaction]) -> List[FraudPrediction]:
        self._ensure_loaded()
        start = time.perf_counter()
        try:
            matrix = np.vstack([t.to_numpy() for t in transactions])
            scaled, rec_errors = self._preprocess_and_encode_batch(matrix)
            enriched = np.hstack([scaled, rec_errors.reshape(-1, 1)])
            probas   = self._classifier.predict_proba(enriched)[:, 1]
        except Exception as exc:
            raise InferenceError(f'Batch inference failed: {exc}') from exc
        total_ms = (time.perf_counter() - start) * 1000
        per_ms   = total_ms / len(transactions)
        return [
            FraudPrediction(
                transaction_id=t.transaction_id,
                fraud_probability=float(p),
                is_fraud=float(p) >= self._clf_threshold,
                reconstruction_error=float(r),
                latency_ms=round(per_ms, 2),
            )
            for t, p, r in zip(transactions, probas, rec_errors)
        ]

    # Calcula o erro de reconstrucao do autoencoder para um vetor de features bruto
    def get_reconstruction_error(self, features: np.ndarray) -> float:
        self._ensure_loaded()
        full   = features.reshape(1, -1)
        df     = pd.DataFrame(full, columns=FEATURE_COLS)
        scaled = full.copy()
        scaled[:, [0, 29]] = self._preprocessor.transform(df[['Amount', 'Time']])
        tensor = torch.FloatTensor(scaled).to(self._device)
        with torch.no_grad():
            reconstructed = self._autoencoder(tensor).cpu().numpy()
        return float(np.mean((scaled - reconstructed) ** 2))

    # Carrega os artefatos na primeira chamada se ainda nao foram inicializados
    def _ensure_loaded(self) -> None:
        if self._preprocessor is None:
            self._load()

    # Carrega preprocessor, autoencoder, classifier e thresholds do repositorio
    def _load(self) -> None:
        try:
            self._preprocessor  = self._repository.load_preprocessor()
            self._autoencoder   = self._repository.load_autoencoder()
            self._classifier    = self._repository.load_classifier()
            self._ae_threshold  = self._repository.get_autoencoder_threshold()
            self._clf_threshold = self._repository.get_classifier_threshold()
        except Exception as exc:
            raise ModelNotLoadedError(f'Could not load model artifacts: {exc}') from exc

    # Escala Time e Amount via StandardScaler e passa pelo autoencoder retornando (features_completas, rec_error)
    def _preprocess_and_encode(self, features_2d: np.ndarray):
        df     = pd.DataFrame(features_2d, columns=FEATURE_COLS)
        scaled = self._preprocessor.transform(df[['Amount', 'Time']])
        full   = features_2d.copy()
        full[:, [0, 29]] = scaled
        tensor = torch.FloatTensor(full).to(self._device)
        with torch.no_grad():
            reconstructed = self._autoencoder(tensor).cpu().numpy()
        rec_error = float(np.mean((full - reconstructed) ** 2))
        return full, rec_error

    # Versao batch de _preprocess_and_encode operando sobre uma matriz inteira de uma vez
    def _preprocess_and_encode_batch(self, matrix: np.ndarray):
        df     = pd.DataFrame(matrix, columns=FEATURE_COLS)
        scaled = self._preprocessor.transform(df[['Amount', 'Time']])
        full   = matrix.copy()
        full[:, [0, 29]] = scaled
        tensor = torch.FloatTensor(full).to(self._device)
        with torch.no_grad():
            reconstructed = self._autoencoder(tensor).cpu().numpy()
        rec_errors = np.mean((full - reconstructed) ** 2, axis=1)
        return full, rec_errors
