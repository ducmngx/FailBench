"""Stage-1 baseline: per-modality MLP encoder → ConvTranspose-style decoder.

Same input contract as :class:`BenchmarkMLP` — what changes is the head: a
small ConvTranspose-flavoured upsampling decoder replaces the dense
``Linear(h, H*W)`` head. Fewer params, spatial prior, usually 10-20% better
val MSE than the dense version on heatmap prediction tasks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from planner.risk.benchmark_dataset import ModalityConfig
from planner.risk.models.mlp import STATE_DIM, GOAL_DIM, _SmallCNN


class _UpBlock(nn.Module):
    """Bilinear x2 upsample + double 3×3 conv with SiLU."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv2(F.silu(self.conv1(self.up(x)))))


class BenchmarkConvDec(nn.Module):
    """Per-modality MLP encoder fused into a small conv-decoder head.

    Decoder shape (default): ``(B, base_ch, base_h, base_w) = (B, 128, 15, 20)``
    → three :class:`_UpBlock` (×2 each) → ``(B, 16, 120, 160)`` →
    bilinear resize → ``(B, 1, H, W)`` → squeeze.
    """

    def __init__(self, *, modalities: ModalityConfig, grid_hw: tuple,
                 T: int = 8, K: int = 3,
                 hidden=(512, 512), dropout: float = 0.1,
                 base_hw: tuple = (15, 20),
                 base_ch: int = 128,
                 img_embed: int = 128, dino_dim: int = 384,
                 failure_mode_dim: int = 5):
        super().__init__()
        self.modalities = modalities
        self.grid_hw = tuple(grid_hw)
        self.T = T; self.K = K
        self.base_hw = tuple(base_hw); self.base_ch = base_ch

        fused_dim = 0
        if modalities.state:
            fused_dim += T * STATE_DIM
        if modalities.goal:
            fused_dim += K * GOAL_DIM
        if modalities.rgb:
            self.rgb_cnn = _SmallCNN(3, img_embed); fused_dim += img_embed
        if modalities.depth:
            self.depth_cnn = _SmallCNN(1, img_embed); fused_dim += img_embed
        if modalities.dino:
            fused_dim += dino_dim
        if modalities.failure_mode:
            fused_dim += failure_mode_dim
        if modalities.failure_joints:
            fused_dim += 7
        if fused_dim == 0:
            raise ValueError("at least one modality must be enabled")

        # Encoder MLP → base feature map.
        enc = []
        d = fused_dim
        for h in hidden:
            enc += [nn.Linear(d, h), nn.SiLU(inplace=True), nn.Dropout(dropout)]
            d = h
        enc.append(nn.Linear(d, base_ch * base_hw[0] * base_hw[1]))
        self.encoder = nn.Sequential(*enc)

        # Decoder: 3 upsamples (×8), then bilinear-fit to grid_hw, 1×1 conv to 1ch.
        self.up1 = _UpBlock(base_ch, base_ch // 2)
        self.up2 = _UpBlock(base_ch // 2, base_ch // 4)
        self.up3 = _UpBlock(base_ch // 4, base_ch // 8)
        self.head = nn.Conv2d(base_ch // 8, 1, 1)
        self.fused_dim = fused_dim

    def forward(self, batch: dict) -> dict:
        feats = []
        if self.modalities.state:
            feats.append(batch["state_window"].flatten(start_dim=1))
        if self.modalities.goal:
            feats.append(batch["goal"].flatten(start_dim=1))
        if self.modalities.rgb:
            B, T, C, H, W = batch["rgb_window"].shape
            emb = self.rgb_cnn(batch["rgb_window"].reshape(B * T, C, H, W))
            feats.append(emb.view(B, T, -1).mean(dim=1))
        if self.modalities.depth:
            B, T, C, H, W = batch["depth_window"].shape
            emb = self.depth_cnn(batch["depth_window"].reshape(B * T, C, H, W))
            feats.append(emb.view(B, T, -1).mean(dim=1))
        if self.modalities.dino:
            feats.append(batch["dino_window"].mean(dim=1))
        if self.modalities.failure_mode:
            feats.append(batch["failure_mode"])
        if self.modalities.failure_joints:
            feats.append(batch["failure_joints"])

        z = torch.cat(feats, dim=1)
        B = z.shape[0]
        base_h, base_w = self.base_hw
        x = self.encoder(z).view(B, self.base_ch, base_h, base_w)
        x = self.up3(self.up2(self.up1(x)))
        x = F.interpolate(x, size=self.grid_hw, mode="bilinear", align_corners=False)
        return {"pred": self.head(x).squeeze(1)}
