"""Stage-0 baseline: concat-everything MLP with optional small image encoders.

Selects encoders from a :class:`ModalityConfig`. Modalities with windowed
inputs (T frames) are flattened before fusion — matches the plan's "MLP
flattens, Transformer/Diffusion attend" split.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from planner.risk.benchmark_dataset import ModalityConfig

STATE_DIM = 18    # qpos(7) + qvel(7) + ee_pos(3) + grip(1)
GOAL_DIM = 11     # qpos(7) + ee_pos(3) + grip(1)


class _SmallCNN(nn.Module):
    """Stride-2 conv stack → adaptive avg pool → linear to embed_dim.

    Input ``(B, C_in, H, W)``; output ``(B, embed_dim)``.
    """

    def __init__(self, c_in: int, embed_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, 16, 5, stride=2, padding=2), nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),   nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),   nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Linear(64 * 4 * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return self.head(h.flatten(1))


class BenchmarkMLP(nn.Module):
    """Concat-encoder MLP → dense pixel head.

    All windowed inputs are pooled to a single embedding per modality (state
    flatten; images per-frame CNN + temporal mean).
    """

    def __init__(self, *, modalities: ModalityConfig, grid_hw: tuple,
                 T: int = 8, K: int = 3,
                 hidden=(512, 1024, 1024), dropout: float = 0.1,
                 img_embed: int = 128, dino_dim: int = 384,
                 failure_mode_dim: int = 5):
        super().__init__()
        self.modalities = modalities
        self.grid_hw = tuple(grid_hw)
        self.T = T
        self.K = K

        fused_dim = 0
        if modalities.state:
            fused_dim += T * STATE_DIM
        if modalities.goal:
            fused_dim += K * GOAL_DIM
        if modalities.rgb:
            self.rgb_cnn = _SmallCNN(3, img_embed)
            fused_dim += img_embed
        if modalities.depth:
            self.depth_cnn = _SmallCNN(1, img_embed)
            fused_dim += img_embed
        if modalities.dino:
            fused_dim += dino_dim
        if modalities.failure_mode:
            fused_dim += failure_mode_dim
        if modalities.failure_joints:
            fused_dim += 7
        if fused_dim == 0:
            raise ValueError("at least one modality must be enabled")

        layers = []
        d = fused_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU(inplace=True), nn.Dropout(dropout)]
            d = h
        H, W = self.grid_hw
        layers.append(nn.Linear(d, H * W))
        self.mlp = nn.Sequential(*layers)
        self.fused_dim = fused_dim

    def _pool_window(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, ...) → (B, T*...) flatten for state, (B, embed) mean for images."""
        return x.flatten(start_dim=1)

    def forward(self, batch: dict) -> dict:
        feats = []
        if self.modalities.state:
            feats.append(batch["state_window"].flatten(start_dim=1))      # (B, T*18)
        if self.modalities.goal:
            feats.append(batch["goal"].flatten(start_dim=1))              # (B, K*11)
        if self.modalities.rgb:
            B, T, C, H, W = batch["rgb_window"].shape
            emb = self.rgb_cnn(batch["rgb_window"].reshape(B * T, C, H, W))
            feats.append(emb.view(B, T, -1).mean(dim=1))                  # (B, embed)
        if self.modalities.depth:
            B, T, C, H, W = batch["depth_window"].shape
            emb = self.depth_cnn(batch["depth_window"].reshape(B * T, C, H, W))
            feats.append(emb.view(B, T, -1).mean(dim=1))
        if self.modalities.dino:
            feats.append(batch["dino_window"].mean(dim=1))                # (B, 384)
        if self.modalities.failure_mode:
            feats.append(batch["failure_mode"])                           # (B, 5)
        if self.modalities.failure_joints:
            feats.append(batch["failure_joints"])                         # (B, 7)

        z = torch.cat(feats, dim=1)
        H, W = self.grid_hw
        return {"pred": self.mlp(z).view(-1, H, W)}
