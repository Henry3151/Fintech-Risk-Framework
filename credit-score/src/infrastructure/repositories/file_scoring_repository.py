# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import joblib

from domain.exceptions import RepositoryError
from domain.interfaces.scoring_interfaces import ICreditScoringRepository


class FileScoringRepository(ICreditScoringRepository):
    """
    Carrega e salva artefatos de credit scoring do disco.
    Artefatos esperados em MODELS_DIR:
      lgbm_model.joblib
      calibrator.joblib
      shap_explainer.joblib
      scoring_metadata.json
    """

    # Resolve o diretorio de modelos via parametro, env var ou caminho relativo padrao
    def __init__(self, models_dir: str | Path | None = None) -> None:
        if models_dir is None:
            models_dir = os.getenv(
                "MODELS_DIR",
                str(Path(__file__).resolve().parent.parent.parent.parent / "models"),
            )
        self._dir = Path(models_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # Carrega o modelo LightGBM salvo com joblib
    def load_model(self) -> Any:
        return self._joblib_load("lgbm_model.joblib")

    # Carrega o calibrador de probabilidade salvo com joblib
    def load_calibrator(self) -> Any:
        return self._joblib_load("calibrator.joblib")

    # Carrega o SHAP TreeExplainer salvo com joblib
    def load_explainer(self) -> Any:
        return self._joblib_load("shap_explainer.joblib")

    # Retorna o dicionario de metadados do treinamento
    def get_metadata(self) -> dict:
        return self._load_json("scoring_metadata.json")

    # Salva modelo, calibrador, explainer e metadados em disco
    def save_artifacts(self, model, calibrator, explainer, metadata: dict) -> None:
        try:
            joblib.dump(model,      self._dir / "lgbm_model.joblib")
            joblib.dump(calibrator, self._dir / "calibrator.joblib")
            joblib.dump(explainer,  self._dir / "shap_explainer.joblib")
            with open(self._dir / "scoring_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as exc:
            raise RepositoryError(f"Failed to save artifacts: {exc}") from exc

    # Carrega um arquivo .joblib lancando RepositoryError em caso de falha
    def _joblib_load(self, filename: str) -> Any:
        path = self._dir / filename
        try:
            return joblib.load(path)
        except Exception as exc:
            raise RepositoryError(f"Failed to load {filename}: {exc}") from exc

    # Abre e retorna o conteudo de um arquivo JSON como dicionario
    def _load_json(self, filename: str) -> dict:
        path = self._dir / filename
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            raise RepositoryError(f"Failed to read {filename}: {exc}") from exc
