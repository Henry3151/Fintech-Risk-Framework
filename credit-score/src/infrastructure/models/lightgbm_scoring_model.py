# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import List
import numpy as np
import lightgbm as lgb
import shap
from sklearn.calibration import CalibratedClassifierCV

from domain.entities.credit_applicant import CreditApplicant, CreditScore, FEATURE_NAMES
from domain.exceptions import ScoringError, ModelNotLoadedError
from domain.interfaces.scoring_interfaces import ICreditScoringModel, ICreditScoringRepository


class LightGBMScoringModel(ICreditScoringModel):
    """
    Pipeline: LightGBM -> CalibratedClassifierCV (Platt) -> SHAP TreeExplainer.
    Calibracao e essencial em credit scoring — reguladores exigem probabilidades
    calibradas, nao apenas rankings.
    """

    # Inicializa o modelo com hiperparametros e repositorio de artefatos
    def __init__(
        self,
        repository: ICreditScoringRepository,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self._repository     = repository
        self._n_estimators   = n_estimators
        self._learning_rate  = learning_rate
        self._num_leaves     = num_leaves
        self._random_state   = random_state
        self._model          = None
        self._calibrator     = None
        self._explainer      = None
        self._metrics        = {}
        self._loaded         = False

    # Executa inferencia em um unico solicitante retornando CreditScore com SHAP
    def predict(self, applicant: CreditApplicant) -> CreditScore:
        self._ensure_loaded()
        start = time.perf_counter()
        try:
            X        = applicant.to_numpy().reshape(1, -1)
            pd_value = float(self._calibrator.predict_proba(X)[0, 1])
            shap_vals = self._get_shap_values(X)
            top_factors = self._top_shap_factors(shap_vals[0])
        except Exception as exc:
            raise ScoringError(f"Scoring failed: {exc}") from exc

        score = CreditScore.pd_to_score(pd_value)
        latency_ms = (time.perf_counter() - start) * 1000
        return CreditScore(
            applicant_id=applicant.applicant_id,
            score=score,
            pd=round(pd_value, 4),
            risk_grade=CreditScore.score_to_grade(score),
            recommendation=CreditScore.score_to_recommendation(score),
            top_factors=top_factors,
            latency_ms=round(latency_ms, 2),
        )

    # Executa inferencia em lote com uma unica passagem pelo modelo e explainer
    def predict_batch(self, applicants: List[CreditApplicant]) -> List[CreditScore]:
        self._ensure_loaded()
        start = time.perf_counter()
        try:
            X         = np.vstack([a.to_numpy() for a in applicants])
            pds       = self._calibrator.predict_proba(X)[:, 1]
            shap_vals = self._get_shap_values(X)
        except Exception as exc:
            raise ScoringError(f"Batch scoring failed: {exc}") from exc

        total_ms = (time.perf_counter() - start) * 1000
        per_ms   = total_ms / len(applicants)

        results = []
        for i, (applicant, pd_value) in enumerate(zip(applicants, pds)):
            score = CreditScore.pd_to_score(float(pd_value))
            results.append(CreditScore(
                applicant_id=applicant.applicant_id,
                score=score,
                pd=round(float(pd_value), 4),
                risk_grade=CreditScore.score_to_grade(score),
                recommendation=CreditScore.score_to_recommendation(score),
                top_factors=self._top_shap_factors(shap_vals[i]),
                latency_ms=round(per_ms, 2),
            ))
        return results

    # Retorna as metricas de performance carregadas do repositorio
    def get_metrics(self) -> dict:
        self._ensure_loaded()
        return self._metrics

    # Carrega os artefatos do repositorio se ainda nao foram inicializados
    def _ensure_loaded(self) -> None:
        if not self._loaded:
            try:
                self._model      = self._repository.load_model()
                self._calibrator = self._repository.load_calibrator()
                self._explainer  = self._repository.load_explainer()
                self._metrics    = self._repository.get_metadata()
                self._loaded     = True
            except Exception as exc:
                raise ModelNotLoadedError(f"Could not load artifacts: {exc}") from exc

    # Calcula SHAP values usando TreeExplainer otimizado para LightGBM
    def _get_shap_values(self, X: np.ndarray) -> np.ndarray:
        vals = self._explainer.shap_values(X)
        # TreeExplainer retorna lista [class0, class1] para classificacao binaria
        if isinstance(vals, list):
            return vals[1]
        return vals

    # Retorna os top 3 fatores com maior impacto no score (nome + direcao)
    def _top_shap_factors(self, shap_row: np.ndarray) -> list:
        indices = np.argsort(np.abs(shap_row))[::-1][:3]
        factors = []
        for idx in indices:
            name   = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
            impact = "increases_risk" if shap_row[idx] > 0 else "decreases_risk"
            factors.append({"feature": name, "shap_value": round(float(shap_row[idx]), 4), "impact": impact})
        return factors
