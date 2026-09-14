"""A small temporal network over windows of feature vectors: Conv1d -> GRU -> multi-label head."""
from __future__ import annotations

import torch
from torch import nn


class HabitNet(nn.Module):
    def __init__(self, n_features: int, n_habits: int, hidden: int = 64, dropout: float = 0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_habits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, H)
        _, hn = self.gru(h)
        return self.head(hn[-1])  # logits (B, n_habits)
