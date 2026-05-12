"""
Autoencoder architecture — 30→16→8→16→30.
Mirrors the original train_autoencoder.py definition so that
torch.load() resolves the class correctly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FraudAutoencoder(nn.Module):
    """
    Symmetric autoencoder trained only on legitimate transactions.
    High reconstruction error → anomaly → likely fraud.

    Architecture: 30 → 16 → 8 (bottleneck) → 16 → 30
    Activation  : ReLU on hidden layers, Sigmoid on output
    """

    def __init__(self, input_dim: int = 30) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
