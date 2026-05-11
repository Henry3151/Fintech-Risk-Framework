# -*- coding: utf-8 -*-
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np

from domain.entities.customer import Customer, CustomerSegment


# Contrato abstrato para qualquer modelo de segmentacao de clientes
class ISegmentationModel(ABC):

    # Treina o modelo com uma lista de clientes e persiste os artefatos
    @abstractmethod
    def fit(self, customers: List[Customer]) -> None: ...

    # Retorna o segmento de cada cliente com cluster_id, label e coordenadas UMAP
    @abstractmethod
    def predict(self, customers: List[Customer]) -> List[CustomerSegment]: ...

    # Retorna as metricas de qualidade do clustering (silhouette, inertia, etc)
    @abstractmethod
    def get_metrics(self) -> dict: ...


# Contrato abstrato para carregamento e persistencia de artefatos do modelo
class ISegmentationRepository(ABC):

    # Carrega o scaler RobustScaler salvo em disco
    @abstractmethod
    def load_scaler(self) -> object: ...

    # Carrega o modelo KMeans salvo em disco
    @abstractmethod
    def load_kmeans(self) -> object: ...

    # Carrega o modelo UMAP salvo em disco
    @abstractmethod
    def load_umap(self) -> object: ...

    # Retorna o dicionario de metadados do treinamento (silhouette, n_clusters, etc)
    @abstractmethod
    def get_metadata(self) -> dict: ...

    # Salva todos os artefatos do modelo em disco
    @abstractmethod
    def save_artifacts(self, scaler, kmeans, umap_model, metadata: dict) -> None: ...
