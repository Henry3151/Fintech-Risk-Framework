# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn

class FraudAutoencoder(nn.Module):
    """
    Autoencoder treinado apenas em transacoes legitimas.
    Alta reconstrucao indica anomalia - provavel fraude.
    Arquitetura: 30 -> 16 -> 8 (bottleneck) -> 16 -> 30
    state_dict: encoder.0=Linear(30,16), encoder.3=Linear(16,8)
                decoder.0=Linear(8,16),  decoder.3=Linear(16,30)
    indices 1 e 2 sao Dropout e ReLU (sem pesos, nao aparecem no state_dict)
    """

    # Inicializa encoder e decoder com Dropout(0.2) e ReLU entre as camadas lineares
    def __init__(self, input_dim: int = 30, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    # Passa o tensor pelo encoder e depois pelo decoder retornando a reconstrucao
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
