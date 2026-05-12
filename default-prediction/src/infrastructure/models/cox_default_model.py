# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import List, Optional
import numpy as np
import pandas as pd

from domain.entities.loan import Loan, DefaultPrediction, FEATURE_NAMES
from domain.exceptions import PredictionError, ModelNotLoadedError
from domain.interfaces.default_interfaces import IDefaultModel, IDefaultRepository


class CoxDefaultModel(IDefaultModel):
    """
    Modelo Cox Proportional Hazards para predicao de default.
    Diferencial vs. classificacao binaria: modela o TEMPO ate o evento,
    nao apenas a probabilidade. Permite responder:
    'Qual a probabilidade de este emprestimo defaultar nos proximos 12 meses?'
    em vez de apenas 'vai defaultar ou nao?'

    Usa lifelines.CoxPHFitter — implementacao robusta com regularizacao L2.
    """

    # Inicializa o modelo com repositorio de artefatos
    def __init__(self, repository: IDefaultRepository) -> None:
        self._repository = repository
        self._model      = None
        self._scaler     = None
        self._metrics    = {}
        self._loaded     = False

    # Executa predicao para um unico emprestimo retornando curva de sobrevivencia
    def predict(self, loan: Loan) -> DefaultPrediction:
        self._ensure_loaded()
        start = time.perf_counter()
        try:
            X  = self._to_dataframe([loan])
            result = self._predict_single(X, loan.loan_id)
        except Exception as exc:
            raise PredictionError(f"Prediction failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        result = DefaultPrediction(
            loan_id=result.loan_id,
            survival_at_12m=result.survival_at_12m,
            survival_at_24m=result.survival_at_24m,
            survival_at_36m=result.survival_at_36m,
            median_survival_months=result.median_survival_months,
            hazard_ratio=result.hazard_ratio,
            risk_tier=result.risk_tier,
            pd_12m=result.pd_12m,
            latency_ms=round(latency_ms, 2),
        )
        return result

    # Executa predicao em lote para multiplos emprestimos
    def predict_batch(self, loans: List[Loan]) -> List[DefaultPrediction]:
        self._ensure_loaded()
        start = time.perf_counter()
        try:
            X = self._to_dataframe(loans)
            results = []
            for i, loan in enumerate(loans):
                row = X.iloc[[i]]
                pred = self._predict_single(row, loan.loan_id)
                results.append(pred)
        except Exception as exc:
            raise PredictionError(f"Batch prediction failed: {exc}") from exc
        total_ms = (time.perf_counter() - start) * 1000
        per_ms = total_ms / len(loans)
        # Atualiza latencia por item
        return [
            DefaultPrediction(
                loan_id=r.loan_id,
                survival_at_12m=r.survival_at_12m,
                survival_at_24m=r.survival_at_24m,
                survival_at_36m=r.survival_at_36m,
                median_survival_months=r.median_survival_months,
                hazard_ratio=r.hazard_ratio,
                risk_tier=r.risk_tier,
                pd_12m=r.pd_12m,
                latency_ms=round(per_ms, 2),
            )
            for r in results
        ]

    # Retorna as metricas de performance carregadas do repositorio
    def get_metrics(self) -> dict:
        self._ensure_loaded()
        return self._metrics

    # Carrega os artefatos do repositorio se ainda nao foram inicializados
    def _ensure_loaded(self) -> None:
        if not self._loaded:
            try:
                self._model   = self._repository.load_model()
                self._scaler  = self._repository.load_scaler()
                self._metrics = self._repository.get_metadata()
                self._loaded  = True
            except Exception as exc:
                raise ModelNotLoadedError(f"Could not load artifacts: {exc}") from exc

    # Converte lista de Loan em DataFrame com colunas nomeadas para o Cox
    def _to_dataframe(self, loans: List[Loan]) -> pd.DataFrame:
        X = np.vstack([l.to_numpy() for l in loans])
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        df_scaled = pd.DataFrame(
            self._scaler.transform(df),
            columns=FEATURE_NAMES
        )
        return df_scaled

    # Executa predicao Cox para uma linha do DataFrame e retorna DefaultPrediction
    def _predict_single(self, X_row: pd.DataFrame, loan_id: str) -> DefaultPrediction:
        # Curva de sobrevivencia S(t) para os tempos 12, 24, 36 meses
        sf = self._model.predict_survival_function(X_row)

        # Interpola S(t) nos tempos desejados
        s12 = self._interpolate_survival(sf, 12)
        s24 = self._interpolate_survival(sf, 24)
        s36 = self._interpolate_survival(sf, 36)

        # Tempo mediano de sobrevivencia (onde S(t) = 0.5)
        median_t = self._median_survival(sf)

        # Hazard ratio relativo ao caso base (exp(linear predictor))
        hr = float(self._model.predict_partial_hazard(X_row).iloc[0])

        pd_12m = 1.0 - s12
        risk_tier = DefaultPrediction.classify_risk(pd_12m)

        return DefaultPrediction(
            loan_id=loan_id,
            survival_at_12m=round(s12, 4),
            survival_at_24m=round(s24, 4),
            survival_at_36m=round(s36, 4),
            median_survival_months=median_t,
            hazard_ratio=round(hr, 4),
            risk_tier=risk_tier,
            pd_12m=round(pd_12m, 4),
            latency_ms=0.0,
        )

    # Interpola o valor da curva de sobrevivencia em um tempo especifico
    def _interpolate_survival(self, sf, t: int) -> float:
        times = sf.index.values
        values = sf.iloc[:, 0].values
        if t <= times[0]:
            return float(values[0])
        if t >= times[-1]:
            return float(values[-1])
        return float(np.interp(t, times, values))

    # Calcula o tempo mediano de sobrevivencia onde S(t) = 0.5
    def _median_survival(self, sf) -> Optional[float]:
        times  = sf.index.values
        values = sf.iloc[:, 0].values
        below = np.where(values <= 0.5)[0]
        if len(below) == 0:
            return None
        return float(times[below[0]])
