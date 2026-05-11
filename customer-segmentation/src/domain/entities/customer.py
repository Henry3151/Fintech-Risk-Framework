# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


# Representa um cliente com features RFM calculadas a partir do historico de transacoes
@dataclass(frozen=True)
class Customer:
    """
    Entidade central do dominio de segmentacao.
    RFM: Recency (dias desde ultima compra),
         Frequency (numero de pedidos),
         Monetary (valor total gasto).
    """
    customer_id: str
    recency: float       # dias desde a ultima compra
    frequency: float     # numero de pedidos unicos
    monetary: float      # valor total gasto (GBP)

    # Valida que os valores RFM sao numericos e nao negativos
    def __post_init__(self) -> None:
        for field, value in [("recency", self.recency),
                              ("frequency", self.frequency),
                              ("monetary", self.monetary)]:
            if not np.isfinite(value):
                raise ValueError(f"Customer.{field} must be finite, got {value}")
            if value < 0:
                raise ValueError(f"Customer.{field} must be >= 0, got {value}")

    # Retorna um array numpy com os valores RFM na ordem [recency, frequency, monetary]
    def to_numpy(self) -> np.ndarray:
        return np.array([self.recency, self.frequency, self.monetary], dtype=np.float64)


# Representa o resultado da segmentacao de um cliente com cluster e coordenadas UMAP
@dataclass(frozen=True)
class CustomerSegment:
    """
    Resultado da segmentacao para um cliente.
    cluster_id    : inteiro de 0 a N-1
    segment_label : nome legivel (ex: 'Champions', 'At Risk')
    umap_x, umap_y: coordenadas 2D para visualizacao
    rfm_score     : score composto normalizado [0, 1]
    """
    customer_id: str
    cluster_id: int
    segment_label: str
    umap_x: float
    umap_y: float
    rfm_score: float

    # Retorna True se o cliente e considerado de alto valor (Champions ou Loyal)
    @property
    def is_high_value(self) -> bool:
        return self.segment_label in {"Champions", "Loyal Customers"}
