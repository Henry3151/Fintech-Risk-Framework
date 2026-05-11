# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List

from domain.entities.customer import Customer, CustomerSegment
from domain.exceptions import ValidationError
from domain.interfaces.segmentation_interfaces import ISegmentationModel


# Orquestra a segmentacao de uma lista de clientes usando o modelo injetado
class SegmentCustomers:
    """
    Use case principal: recebe clientes com RFM calculado
    e retorna a segmentacao com cluster, label e coordenadas UMAP.
    """

    # Recebe o modelo de segmentacao via injecao de dependencia
    def __init__(self, model: ISegmentationModel) -> None:
        self._model = model

    # Valida os clientes e executa a segmentacao retornando CustomerSegment para cada um
    def execute(self, customers: List[Customer]) -> List[CustomerSegment]:
        self._validate(customers)
        return self._model.predict(customers)

    # Verifica que a lista nao esta vazia e que nao ha customer_ids duplicados
    @staticmethod
    def _validate(customers: List[Customer]) -> None:
        if not customers:
            raise ValidationError("Customer list must not be empty.")
        ids = [c.customer_id for c in customers]
        if len(ids) != len(set(ids)):
            raise ValidationError("Duplicate customer_ids found in input.")
