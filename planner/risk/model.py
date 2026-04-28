"""Heatmap regressor model — small MLP from config vector to grid."""
from __future__ import annotations

import torch
from torch import nn


class HeatmapMLP(nn.Module):
    def __init__(self, in_dim: int = 10,
                 grid_shape: tuple[int, int] = (43, 70),
                 hidden: tuple[int, ...] = (256, 512, 1024),
                 dropout: float = 0.1):
        super().__init__()
        self.grid_shape = grid_shape
        out_dim = grid_shape[0] * grid_shape[1]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)  → out: (B, ny, nx)
        flat = self.net(x)
        return flat.view(-1, *self.grid_shape)
