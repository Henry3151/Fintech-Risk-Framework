# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import joblib

from domain.exceptions import RepositoryError
from domain.interfaces.default_interfaces import IDefaultRepository


class FileDefaultRepository(IDefaultRepository):
    """
    Carrega e salva artefatos do modelo Cox do disco.
    Artefatos esperados em MODELS_DIR:
      cox_model.joblib
      scaler.joblib
      default_metadata.json
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

    # Carrega o modelo CoxPHFitter salvo com joblib
    def load_model(self) -> Any:
        return self._joblib_load("cox_model.joblib")

    # Carrega o StandardScaler salvo com joblib
    def load_scaler(self) -> Any:
        return self._joblib_load("scaler.joblib")

    # Retorna o dicionario de metadados do treinamento
    def get_metadata(self) -> dict:
        return self._load_json("default_metadata.json")

    # Salva modelo, scaler e metadados em disco
    def save_artifacts(self, model, scaler, metadata: dict) -> None:
        try:
            joblib.dump(model,  self._dir / "cox_model.joblib")
            joblib.dump(scaler, self._dir / "scaler.joblib")
            with open(self._dir / "default_metadata.json", "w") as f:
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
