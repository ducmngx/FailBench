"""Heatmap regressor models — config vector → 2D heatmap.

Two heads sharing the same input contract:
  HeatmapMLP            dense Linear(... → ny·nx) head  (Stage 0 baseline)
  HeatmapConvDecoder    Linear encoder → ConvTranspose-style decoder (Stage 1)
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        return x


class _CNNEncoder(nn.Module):
    """Small from-scratch CNN. Maps (B, in_ch, H, W) → (B, emb_dim).

    Sized for 96×128 inputs: 4 stride-2 convs → 6×8 feature map → AvgPool 4×4.
    Tiny by deep-learning standards (~120k params); appropriate for ~3k images.
    `in_ch=3` for RGB (Stage 3) or `in_ch=4` for RGB+depth (Stage 4).
    """
    def __init__(self, emb_dim: int = 128, in_ch: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        self.fc = nn.Linear(64 * 4 * 4, emb_dim)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return F.silu(self.fc(self.conv(rgb)))


class HeatmapVisionConvDecoder(nn.Module):
    """Stage 3: state vector + RGB → heatmap.

    State path: same Linear encoder as HeatmapConvDecoder.
    Vision path: small CNN → fixed-dim embedding.
    Fuse by concatenation, then the same conv decoder.
    """
    def __init__(self, state_dim: int = 29,
                 grid_shape: tuple[int, int] = (43, 70),
                 hidden: tuple[int, ...] = (256, 512),
                 feat_ch: int = 64,
                 feat_hw: tuple[int, int] = (6, 9),
                 decoder_channels: tuple[int, ...] = (32, 16, 8),
                 rgb_emb_dim: int = 128,
                 rgb_in_ch: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.grid_shape = grid_shape
        self.feat_ch = feat_ch
        self.feat_hw = feat_hw

        self.cnn = _CNNEncoder(emb_dim=rgb_emb_dim, in_ch=rgb_in_ch)

        enc: list[nn.Module] = []
        prev = state_dim + rgb_emb_dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        enc.append(nn.Linear(prev, feat_ch * feat_hw[0] * feat_hw[1]))
        self.encoder = nn.Sequential(*enc)

        blocks: list[nn.Module] = []
        prev_ch = feat_ch
        for ch in decoder_channels:
            blocks.append(_UpBlock(prev_ch, ch))
            prev_ch = ch
        self.decoder = nn.Sequential(*blocks)
        self.head = nn.Conv2d(prev_ch, 1, 3, padding=1)

    def forward(self, state: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        v = self.cnn(rgb)
        z = self.encoder(torch.cat([state, v], dim=1))
        z = z.view(-1, self.feat_ch, *self.feat_hw)
        z = self.decoder(z)
        z = self.head(z)
        z = F.interpolate(z, size=self.grid_shape, mode="bilinear", align_corners=False)
        return z.squeeze(1)


class HeatmapDINOConvDecoder(nn.Module):
    """Stage 7: state vector + frozen DINOv2 CLS feature → heatmap.

    Vision path is a plain Linear projection of the cached DINOv2 (384,)
    embedding — the backbone is frozen and offline, so no CNN training,
    no overfitting risk on the vision side. State path matches
    `HeatmapVisionConvDecoder`.
    """
    def __init__(self, state_dim: int = 76,
                 grid_shape: tuple[int, int] = (93, 133),
                 hidden: tuple[int, ...] = (256, 512),
                 feat_ch: int = 64,
                 feat_hw: tuple[int, int] = (6, 9),
                 decoder_channels: tuple[int, ...] = (32, 16, 8),
                 dino_dim: int = 384,
                 vis_emb_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.grid_shape = grid_shape
        self.feat_ch = feat_ch
        self.feat_hw = feat_hw

        self.vis_proj = nn.Sequential(
            nn.Linear(dino_dim, vis_emb_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        enc: list[nn.Module] = []
        prev = state_dim + vis_emb_dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        enc.append(nn.Linear(prev, feat_ch * feat_hw[0] * feat_hw[1]))
        self.encoder = nn.Sequential(*enc)

        blocks: list[nn.Module] = []
        prev_ch = feat_ch
        for ch in decoder_channels:
            blocks.append(_UpBlock(prev_ch, ch))
            prev_ch = ch
        self.decoder = nn.Sequential(*blocks)
        self.head = nn.Conv2d(prev_ch, 1, 3, padding=1)

    def forward(self, state: torch.Tensor, dino: torch.Tensor) -> torch.Tensor:
        v = self.vis_proj(dino)
        z = self.encoder(torch.cat([state, v], dim=1))
        z = z.view(-1, self.feat_ch, *self.feat_hw)
        z = self.decoder(z)
        z = self.head(z)
        z = F.interpolate(z, size=self.grid_shape, mode="bilinear", align_corners=False)
        return z.squeeze(1)


class HeatmapConvDecoder(nn.Module):
    """MLP encoder → small feature map → conv-upsample → bilinear-resize to grid.

    The decoder gives neighbouring heatmap cells a shared spatial inductive
    bias that the dense Linear head in HeatmapMLP lacks.
    """
    def __init__(self, in_dim: int = 17,
                 grid_shape: tuple[int, int] = (43, 70),
                 hidden: tuple[int, ...] = (256, 512),
                 feat_ch: int = 64,
                 feat_hw: tuple[int, int] = (6, 9),
                 decoder_channels: tuple[int, ...] = (32, 16, 8),
                 dropout: float = 0.1):
        super().__init__()
        self.grid_shape = grid_shape
        self.feat_ch = feat_ch
        self.feat_hw = feat_hw

        enc: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.SiLU(), nn.Dropout(dropout)]
            prev = h
        enc.append(nn.Linear(prev, feat_ch * feat_hw[0] * feat_hw[1]))
        self.encoder = nn.Sequential(*enc)

        blocks: list[nn.Module] = []
        prev_ch = feat_ch
        for ch in decoder_channels:
            blocks.append(_UpBlock(prev_ch, ch))
            prev_ch = ch
        self.decoder = nn.Sequential(*blocks)
        self.head = nn.Conv2d(prev_ch, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = z.view(-1, self.feat_ch, *self.feat_hw)
        z = self.decoder(z)
        z = self.head(z)
        z = F.interpolate(z, size=self.grid_shape, mode="bilinear", align_corners=False)
        return z.squeeze(1)
