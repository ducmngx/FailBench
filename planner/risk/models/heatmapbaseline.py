"""HeatmapUNet variants used by the dual-model contact-prediction benchmark.

Mirror of the notebook-local copy at
``notebooks/model_playground/models/heatmapbaseline.py``. Keeps the four
ablation variants importable from a stable package path so downstream code
(``planner.risk.inference``, ``planner.policy.*``, rollout scripts) doesn't
need notebook-local ``sys.path`` shims.

Variants:

- ``HeatmapUNet`` — 4-stage U-Net, additive state injection at the bottleneck.
- ``FiLMHeatmapUNet`` — FiLM (scale+shift) state modulation at the bottleneck.
- ``CoordFiLMHeatmapUNet`` — adds a normalised [-1,1] coordinate stack to the
  RGB input (CoordConv).
- ``GatekeeperCoordFiLMUNet`` — adds a binary "did this failure produce a
  contact?" classification head at the bottleneck; forward returns
  ``(heatmap, contact_logit)``.

All variants accept ``rgb`` of shape ``(B, 3, H, W)`` (uint8 in [0,255] or
float in any range — the head divides by 255 only when dtype is uint8) and
``state`` of shape ``(B, state_dim)`` (default ``state_dim=156`` = 144 state
window + 5 failure_mode one-hot + 7 failure_joints multi-hot).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear upsample → concat skip → DoubleConv (checkerboard-free)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UpLegacy(nn.Module):
    """Pre-2026-06-16 Up block: ConvTranspose2d that halves the channel count
    before concat. Preserved only so older checkpoints (the very first
    ``unet_state_rgb/epoch5.pt`` U-Net) remain loadable for the ablation
    table. New training should use :class:`Up`.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Heatmap U-Net (additive state injection, baseline)
# ---------------------------------------------------------------------------

class HeatmapUNet(nn.Module):
    """State + RGB → (B, H, W) log1p contact heatmap."""

    def __init__(self, state_dim: int = 156, base_ch: int = 16,
                 H: int = 240, W: int = 320):
        super().__init__()
        self.H, self.W = H, W

        self.stem = DoubleConv(3, base_ch)
        self.down1 = Down(base_ch,      base_ch * 2)
        self.down2 = Down(base_ch * 2,  base_ch * 4)
        self.down3 = Down(base_ch * 4,  base_ch * 8)
        self.down4 = Down(base_ch * 8,  base_ch * 16)

        bottleneck_ch = base_ch * 16
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_ch, bottleneck_ch),
        )

        self.up4 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = Up(base_ch * 8,  base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4,  base_ch * 2, base_ch * 2)
        self.up1 = Up(base_ch * 2,  base_ch,     base_ch)

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, rgb, state):
        x = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb.float()

        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        s = self.state_proj(state)
        x4 = x4 + s[:, :, None, None]

        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        return self.head(x).squeeze(1)


# ---------------------------------------------------------------------------
# FiLM (scale + shift) state modulation
# ---------------------------------------------------------------------------

class FiLMHeatmapUNet(nn.Module):
    """Version 1: FiLM bottleneck. Same encoder/decoder as HeatmapUNet but the
    state projects to (scale, shift) tuples that modulate the bottleneck:
    ``x4 = x4 * (1 + scale) + shift``.
    """

    def __init__(self, state_dim: int = 156, base_ch: int = 16,
                 H: int = 240, W: int = 320):
        super().__init__()
        self.H, self.W = H, W

        self.stem = DoubleConv(3, base_ch)
        self.down1 = Down(base_ch,      base_ch * 2)
        self.down2 = Down(base_ch * 2,  base_ch * 4)
        self.down3 = Down(base_ch * 4,  base_ch * 8)
        self.down4 = Down(base_ch * 8,  base_ch * 16)

        bottleneck_ch = base_ch * 16
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, bottleneck_ch),
            nn.LayerNorm(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_ch, bottleneck_ch * 2),
        )

        self.up4 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = Up(base_ch * 8,  base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4,  base_ch * 2, base_ch * 2)
        self.up1 = Up(base_ch * 2,  base_ch,     base_ch)

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, rgb, state):
        x = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb.float()

        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        s = self.state_proj(state)[:, :, None, None]
        scale, shift = s.chunk(2, dim=1)
        x4 = x4 * (1.0 + scale) + shift

        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        return self.head(x).squeeze(1)


# ---------------------------------------------------------------------------
# CoordConv stem + FiLM bottleneck
# ---------------------------------------------------------------------------

class CoordFiLMHeatmapUNet(nn.Module):
    """Version 2: CoordConv stem appends [-1, 1] X/Y grids to the RGB input
    so spatial position is explicit. FiLM injection at the bottleneck.
    """

    def __init__(self, state_dim: int = 156, base_ch: int = 16,
                 H: int = 240, W: int = 320):
        super().__init__()
        self.H, self.W = H, W

        self.stem = DoubleConv(5, base_ch)  # 3 RGB + 2 coords
        self.down1 = Down(base_ch,      base_ch * 2)
        self.down2 = Down(base_ch * 2,  base_ch * 4)
        self.down3 = Down(base_ch * 4,  base_ch * 8)
        self.down4 = Down(base_ch * 8,  base_ch * 16)

        bottleneck_ch = base_ch * 16
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, bottleneck_ch),
            nn.LayerNorm(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_ch, bottleneck_ch * 2),
        )

        self.up4 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = Up(base_ch * 8,  base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4,  base_ch * 2, base_ch * 2)
        self.up1 = Up(base_ch * 2,  base_ch,     base_ch)

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, rgb, state):
        x = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb.float()
        B, _, H, W = x.shape

        y_coords = torch.linspace(-1, 1, steps=H, device=x.device)
        x_coords = torch.linspace(-1, 1, steps=W, device=x.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid_y = grid_y.expand(B, 1, H, W)
        grid_x = grid_x.expand(B, 1, H, W)
        x = torch.cat([x, grid_y, grid_x], dim=1)

        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        s = self.state_proj(state)[:, :, None, None]
        scale, shift = s.chunk(2, dim=1)
        x4 = x4 * (1.0 + scale) + shift

        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        return self.head(x).squeeze(1)


# ---------------------------------------------------------------------------
# Dual-head: gatekeeper "is there contact?" + spatial heatmap
# ---------------------------------------------------------------------------

class GatekeeperCoordFiLMUNet(nn.Module):
    """Version 3: CoordConv + FiLM bottleneck + dual-head gatekeeper.

    The gatekeeper branch reads the FiLM-conditioned bottleneck features
    and predicts a single binary logit ("did this failure produce any
    contact?"). The spatial branch decodes the heatmap as usual.

    Returns
    -------
    heatmap : (B, H, W) — spatial log1p contact heatmap
    gate    : (B,) float — raw logit (use sigmoid for probability)
    """

    def __init__(self, state_dim: int = 156, base_ch: int = 16,
                 H: int = 240, W: int = 320):
        super().__init__()
        self.H, self.W = H, W

        self.stem = DoubleConv(5, base_ch)
        self.down1 = Down(base_ch,      base_ch * 2)
        self.down2 = Down(base_ch * 2,  base_ch * 4)
        self.down3 = Down(base_ch * 4,  base_ch * 8)
        self.down4 = Down(base_ch * 8,  base_ch * 16)

        bottleneck_ch = base_ch * 16
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, bottleneck_ch),
            nn.LayerNorm(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_ch, bottleneck_ch * 2),
        )

        self.gatekeeper = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.up4 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = Up(base_ch * 8,  base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4,  base_ch * 2, base_ch * 2)
        self.up1 = Up(base_ch * 2,  base_ch,     base_ch)

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, rgb, state):
        x = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb.float()
        B, _, H, W = x.shape

        y_coords = torch.linspace(-1, 1, steps=H, device=x.device)
        x_coords = torch.linspace(-1, 1, steps=W, device=x.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid_y = grid_y.expand(B, 1, H, W)
        grid_x = grid_x.expand(B, 1, H, W)
        x = torch.cat([x, grid_y, grid_x], dim=1)

        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        s = self.state_proj(state)[:, :, None, None]
        scale, shift = s.chunk(2, dim=1)
        x4_cond = x4 * (1.0 + scale) + shift

        gate_logit = self.gatekeeper(x4_cond)

        x = self.up4(x4_cond, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        heatmap = self.head(x).squeeze(1)

        return heatmap, gate_logit.squeeze(-1)


# ---------------------------------------------------------------------------
# Legacy variant — pre-bilinear Up
# ---------------------------------------------------------------------------

class HeatmapUNetLegacy(nn.Module):
    """Same architecture as :class:`HeatmapUNet` but uses :class:`UpLegacy`.

    Only needed to load the pre-2026-06-16 ``unet_state_rgb/epoch5.pt``
    checkpoint, which was trained before the Up block was switched from
    ConvTranspose to bilinear. All later runs use :class:`HeatmapUNet`.
    """

    def __init__(self, state_dim: int = 156, base_ch: int = 16,
                 H: int = 240, W: int = 320):
        super().__init__()
        self.H, self.W = H, W

        self.stem = DoubleConv(3, base_ch)
        self.down1 = Down(base_ch,      base_ch * 2)
        self.down2 = Down(base_ch * 2,  base_ch * 4)
        self.down3 = Down(base_ch * 4,  base_ch * 8)
        self.down4 = Down(base_ch * 8,  base_ch * 16)

        bottleneck_ch = base_ch * 16
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_ch, bottleneck_ch),
        )

        self.up4 = UpLegacy(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = UpLegacy(base_ch * 8,  base_ch * 4, base_ch * 4)
        self.up2 = UpLegacy(base_ch * 4,  base_ch * 2, base_ch * 2)
        self.up1 = UpLegacy(base_ch * 2,  base_ch,     base_ch)

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, rgb, state):
        x = rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb.float()

        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        s = self.state_proj(state)
        x4 = x4 + s[:, :, None, None]

        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        return self.head(x).squeeze(1)


__all__ = [
    "HeatmapUNet",
    "FiLMHeatmapUNet",
    "CoordFiLMHeatmapUNet",
    "GatekeeperCoordFiLMUNet",
    "HeatmapUNetLegacy",
    "DoubleConv",
    "Down",
    "Up",
    "UpLegacy",
]
