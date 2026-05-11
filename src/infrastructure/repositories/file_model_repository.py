# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import joblib
import torch
from domain.exceptions import RepositoryError
from domain.interfaces.model_interfaces import IModelRepository
from infrastructure.models.autoencoder import FraudAutoencoder

class FileModelRepository(IModelRepository):
    """
    Carrega artefatos pre-treinados do disco.
    Todos os metadados vivem em classifier_metadata.json.
    """

    # Resolve o diretorio de modelos via parametro, env var MODELS_DIR ou caminho relativo padrao
    def __init__(self, models_dir: str | Path | None = None) -> None:
        if models_dir is None:
            models_dir = os.getenv(
                'MODELS_DIR',
                str(Path(__file__).resolve().parent.parent.parent.parent / 'models'),
            )
        self._dir = Path(models_dir)

    # Carrega o StandardScaler salvo com joblib
    def load_preprocessor(self) -> Any:
        return self._joblib_load('preprocessor.joblib')

    # Carrega o FraudAutoencoder do state_dict salvo em autoencoder.pt
    def load_autoencoder(self) -> FraudAutoencoder:
        path = self._path('autoencoder.pt')
        try:
            model = FraudAutoencoder(input_dim=30)
            state = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(state)
            model.eval()
            return model
        except Exception as exc:
            raise RepositoryError(f'Failed to load autoencoder: {exc}') from exc

    # Carrega o XGBoost classifier salvo com joblib
    def load_classifier(self) -> Any:
        return self._joblib_load('classifier.joblib')

    # Retorna o threshold do autoencoder salvo em classifier_metadata.json
    def get_autoencoder_threshold(self) -> float:
        return float(self._load_json('classifier_metadata.json')['autoencoder_threshold'])

    # Retorna o melhor threshold do classifier salvo em classifier_metadata.json
    def get_classifier_threshold(self) -> float:
        return float(self._load_json('classifier_metadata.json')['best_threshold'])

    # Retorna o dicionario completo de metadados do classifier_metadata.json
    def get_metadata(self) -> dict:
        meta = self._load_json('classifier_metadata.json')
        return {
            'autoencoder': {'pr_auc': meta.get('pr_auc', 0.0), 'threshold_p95': meta['autoencoder_threshold']},
            'classifier':  {'pr_auc': meta['pr_auc'], 'best_threshold': meta['best_threshold']},
        }

    # Carrega um arquivo .joblib lancando RepositoryError em caso de falha
    def _joblib_load(self, filename: str) -> Any:
        path = self._path(filename)
        try:
            return joblib.load(path)
        except Exception as exc:
            raise RepositoryError(f'Failed to load {filename}: {exc}') from exc

    # Abre e retorna o conteudo de um arquivo JSON como dicionario
    def _load_json(self, filename: str) -> dict:
        path = self._path(filename)
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            raise RepositoryError(f'Failed to read {filename}: {exc}') from exc

    # Retorna o Path completo para um arquivo dentro do diretorio de modelos
    def _path(self, filename: str) -> Path:
        return self._dir / filename
