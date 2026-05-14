"""LIBERO contact-heatmap model.

A simple U-Net-style encoder/decoder that consumes RGB + depth (and optionally
robot state) and predicts the agentview cam-projected contact heatmap and
mean-depth maps at native resolution.

Architecture (default, image-only, ~12M params):

    rgb (3, H, W)
        ⊕                                      ┌── aux head ── mass_total scalar
    depth (1, H, W) ──► ResNet-18 encoder ─────┤
                                               └──┐
                                                  ▼
                  ┌── skip s0 ──┐                 │
                  │             ▼                 │
                  └── skip s1 ──► up-block ── decoder ── 1x1 conv ─► (mass, depth) at (H, W)
                  └── skip s2 ──►              ▲
                  └── skip s3 ──►              │
                                          state MLP (optional)

Inputs
------
    rgb   (B, 3, H, W) float32, ImageNet-normalised
    depth (B, 1, H, W) float32, raw metres
    state (B, 14)      float32, pre_qpos ⊕ pre_qvel (only if use_state=True)

Outputs (dict)
--------------
    mass         (B, 1, H, W) — predicted log1p(aggregated mass)
    depth        (B, 1, H, W) — predicted aggregated mean depth (metres)
    mass_total   (B,)         — predicted scalar total log1p mass

Standard loss (see :meth:`LiberoHeatmapModel.loss`):

    L = MSE(mass) + 0.1 · masked-MSE(depth, mask=target_mass > eps)
        + 0.01 · MSE(mass_total)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class _ConvBlock(nn.Module):
    """3x3 conv → BN → ReLU, twice."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on a scalar / small vector.

    F' = γ(cond) ⊙ F + β(cond)

    Initialised so that γ ≡ 1, β ≡ 0 at construction — i.e. the layer is
    the identity function. Training is free to deviate from there.
    """

    def __init__(self, n_channels: int, cond_dim: int = 1, hidden: int = 32):
        super().__init__()
        self.scale = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_channels),
        )
        self.shift = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_channels),
        )
        # γ starts at 1: zero final weight, bias = 1.
        nn.init.zeros_(self.scale[-1].weight)
        nn.init.ones_(self.scale[-1].bias)
        # β starts at 0: zero final weight and bias.
        nn.init.zeros_(self.shift[-1].weight)
        nn.init.zeros_(self.shift[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); cond: (B, cond_dim) or (B,).
        if cond.dim() == 1:
            cond = cond.view(-1, 1)
        cond = cond.to(x.dtype)
        gamma = self.scale(cond)[..., None, None]
        beta = self.shift(cond)[..., None, None]
        return gamma * x + beta


class _UpBlock(nn.Module):
    """ConvTranspose2d up by 2× → concat with skip → ConvBlock."""

    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2)
        self.block = _ConvBlock(c_in // 2 + c_skip, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Realign if encoder/decoder shapes diverge by ±1 px (odd input).
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class LiberoHeatmapModel(nn.Module):
    """RGB+depth → (mass, depth) contact heatmap.

    Parameters
    ----------
    use_state : bool
        If True, an MLP on (pre_qpos ⊕ pre_qvel) produces a 64-D feature that
        is broadcast-tiled and concatenated to the encoder bottleneck. Adds
        ~30 k parameters. Default off — image-only baseline first.
    pretrained : bool
        Initialise the ResNet-18 encoder from ImageNet weights. The depth
        channel of the 4-channel first conv is initialised to the mean of the
        three RGB channels (a standard transfer trick).
    state_dim : int
        Dimension of the state vector (default 14: 7 qpos + 7 qvel).
    state_feat_dim : int
        Dimension of the projected state feature (default 64).
    """

    def __init__(self,
                 use_state: bool = False,
                 use_holding: bool = True,
                 holding_mode: str = "bottleneck",
                 use_failure_mode: bool = False,
                 n_failure_modes: int = 5,
                 pretrained: bool = True,
                 state_dim: int = 14,
                 state_feat_dim: int = 64,
                 holding_feat_dim: int = 64):
        super().__init__()
        if holding_mode not in ("bottleneck", "film"):
            raise ValueError(f"holding_mode must be 'bottleneck' or 'film', got {holding_mode!r}")
        self.use_state = use_state
        self.use_holding = use_holding
        self.holding_mode = holding_mode if use_holding else "bottleneck"
        self.use_failure_mode = use_failure_mode
        self.n_failure_modes = n_failure_modes

        # --------------------------- encoder ---------------------------------
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        # Replace first conv with a 4-channel version.
        new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = backbone.conv1.weight
            # Initialise depth channel to the mean of the RGB channels.
            new_conv.weight[:, 3:4] = backbone.conv1.weight.mean(dim=1, keepdim=True)
        self.conv1 = new_conv
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1   # 64 channels, stride 4
        self.layer2 = backbone.layer2   # 128 channels, stride 8
        self.layer3 = backbone.layer3   # 256 channels, stride 16
        self.layer4 = backbone.layer4   # 512 channels, stride 32

        # ---------------------- optional state head --------------------------
        bottleneck_extra = 0
        if use_state:
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(inplace=True),
                nn.Linear(128, state_feat_dim), nn.ReLU(inplace=True),
            )
            bottleneck_extra += state_feat_dim

        # ---------------------- holding-bit head -----------------------------
        # 'bottleneck' mode: feed a 64-D feature into the bottleneck-extra concat.
        # 'film' mode:       use FiLM modulation at every decoder up-block instead.
        if use_holding and self.holding_mode == "bottleneck":
            self.holding_mlp = nn.Sequential(
                nn.Linear(1, holding_feat_dim), nn.ReLU(inplace=True),
            )
            bottleneck_extra += holding_feat_dim
        self.bottleneck_extra = bottleneck_extra

        # --------------------------- decoder ---------------------------------
        # Skip-connection channels: s3=256, s2=128, s1=64, s0=64.
        self.up4 = _UpBlock(512 + bottleneck_extra, 256, 256)  # → stride 16
        self.up3 = _UpBlock(256, 128, 128)                     # → stride 8
        self.up2 = _UpBlock(128, 64, 64)                       # → stride 4
        self.up1 = _UpBlock(64, 64, 64)                        # → stride 2

        # FiLM blocks (only constructed in 'film' mode). Identity-initialised.
        # cond_dim grows when failure_mode one-hot is added.
        film_cond_dim = 0
        if use_holding and self.holding_mode == "film":
            film_cond_dim += 1
        if use_failure_mode:
            film_cond_dim += n_failure_modes
        self._film_cond_dim = film_cond_dim
        if film_cond_dim > 0:
            self.film4 = FiLM(256, cond_dim=film_cond_dim)
            self.film3 = FiLM(128, cond_dim=film_cond_dim)
            self.film2 = FiLM(64, cond_dim=film_cond_dim)
            self.film1 = FiLM(64, cond_dim=film_cond_dim)
        # Final 2× upsample to native input resolution.
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            _ConvBlock(32, 32),
        )
        self.head = nn.Conv2d(32, 2, kernel_size=1)  # (mass, depth)

        # ---------------------- auxiliary scalar head ------------------------
        self.aux_pool = nn.AdaptiveAvgPool2d(1)
        self.aux_head = nn.Linear(512 + bottleneck_extra, 1)

    # ----------------------------------------------------------------- forward
    def forward(self,
                rgb: torch.Tensor,
                depth: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                is_holding: Optional[torch.Tensor] = None,
                failure_onehot: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = torch.cat([rgb, depth], dim=1)            # (B, 4, H, W)

        # Encoder
        s0 = self.relu(self.bn1(self.conv1(x)))        # H/2, 64
        x = self.maxpool(s0)                           # H/4
        s1 = self.layer1(x)                            # H/4, 64
        s2 = self.layer2(s1)                           # H/8, 128
        s3 = self.layer3(s2)                           # H/16, 256
        s4 = self.layer4(s3)                           # H/32, 512

        # Validate inputs needed downstream.
        if self.use_holding and is_holding is None:
            raise ValueError(
                "model was built with use_holding=True but forward() got is_holding=None")
        if self.use_state and state is None:
            raise ValueError(
                "model was built with use_state=True but forward() got state=None")

        # Bottleneck state + (optional) bottleneck-mode holding fusion.
        bn = s4
        extras: list[torch.Tensor] = []
        if self.use_state:
            extras.append(self.state_mlp(state))       # (B, state_feat_dim)
        if self.use_holding and self.holding_mode == "bottleneck":
            h = is_holding.view(-1, 1).to(s4.dtype)
            extras.append(self.holding_mlp(h))         # (B, holding_feat_dim)
        if extras:
            f = torch.cat(extras, dim=1)
            f_tile = f[..., None, None].expand(-1, -1, bn.shape[2], bn.shape[3])
            bn = torch.cat([bn, f_tile], dim=1)        # (B, 512+extras, H/32, W/32)

        # Build the FiLM conditioning vector (concat of optional pieces).
        film_pieces: list[torch.Tensor] = []
        if self.use_holding and self.holding_mode == "film":
            film_pieces.append(is_holding.view(-1, 1).to(s4.dtype))
        if self.use_failure_mode:
            if failure_onehot is None:
                raise ValueError(
                    "model was built with use_failure_mode=True but forward() got failure_onehot=None")
            film_pieces.append(failure_onehot.to(s4.dtype))
        film_cond = torch.cat(film_pieces, dim=1) if film_pieces else None

        # Decoder with skip connections. FiLM-modulate at every stage when
        # any FiLM conditioning is active.
        d3 = self.up4(bn, s3)                          # stride 16
        if film_cond is not None: d3 = self.film4(d3, film_cond)
        d2 = self.up3(d3, s2)                          # stride 8
        if film_cond is not None: d2 = self.film3(d2, film_cond)
        d1 = self.up2(d2, s1)                          # stride 4
        if film_cond is not None: d1 = self.film2(d1, film_cond)
        d0 = self.up1(d1, s0)                          # stride 2
        if film_cond is not None: d0 = self.film1(d0, film_cond)
        d = self.final_up(d0)                          # stride 1
        out = self.head(d)                             # (B, 2, H, W)

        # Auxiliary scalar head from the bottleneck.
        aux = self.aux_pool(bn).flatten(1)             # (B, 512+F)
        mass_total = self.aux_head(aux).squeeze(-1)    # (B,)

        return {
            "mass": out[:, 0:1],
            "depth": out[:, 1:2],
            "mass_total": mass_total,
        }

    # --------------------------------------------------------------- training
    @staticmethod
    def loss(pred: Dict[str, torch.Tensor],
             batch: Dict[str, torch.Tensor],
             depth_weight: float = 0.1,
             mass_total_weight: float = 0.01,
             mass_mask_eps: float = 1e-3) -> tuple[torch.Tensor, Dict[str, float]]:
        """Standard composite training loss.

        L = MSE(mass)
            + ``depth_weight`` · masked-MSE(depth, mask = target_mass > eps)
            + ``mass_total_weight`` · MSE(mass_total)

        The depth mask suppresses gradient at pixels with no contact mass,
        where the target depth is undefined / zero.
        """
        mass_loss = F.mse_loss(pred["mass"], batch["target_mass"])

        mask = (batch["target_mass"] > mass_mask_eps).float()
        depth_sq = (pred["depth"] - batch["target_depth"]) ** 2 * mask
        depth_loss = depth_sq.sum() / mask.sum().clamp_min(1.0)

        total_pred = pred["mass_total"]
        total_target = batch["target_mass_total"]
        total_loss = F.mse_loss(total_pred, total_target)

        loss = mass_loss + depth_weight * depth_loss + mass_total_weight * total_loss
        components = {
            "loss": float(loss.detach()),
            "mass_loss": float(mass_loss.detach()),
            "depth_loss": float(depth_loss.detach()),
            "mass_total_loss": float(total_loss.detach()),
        }
        return loss, components


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# --------------------------------------------------------------- smoke main
if __name__ == "__main__":
    import argparse, time

    ap = argparse.ArgumentParser()
    ap.add_argument("--use_state", action="store_true")
    ap.add_argument("--holding_mode", default="bottleneck",
                    choices=["bottleneck", "film"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--h", type=int, default=240, help="image H for smoke")
    ap.add_argument("--w", type=int, default=320, help="image W for smoke")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = LiberoHeatmapModel(use_state=args.use_state,
                               holding_mode=args.holding_mode).to(device)
    trn, tot = count_parameters(model)
    print(f"params: trainable={trn/1e6:.2f}M  total={tot/1e6:.2f}M  "
          f"holding_mode={model.holding_mode}")

    rgb = torch.randn(args.batch, 3, args.h, args.w, device=device)
    depth = torch.rand(args.batch, 1, args.h, args.w, device=device) * 2.0
    state = torch.randn(args.batch, 14, device=device) if args.use_state else None
    is_holding = torch.randint(0, 2, (args.batch,), device=device).float()

    # Identity-init check for FiLM mode: flipping is_holding must yield the
    # exact same output if FiLM blocks are init'd as γ=1, β=0.
    if model.holding_mode == "film":
        with torch.no_grad():
            h0 = torch.zeros(args.batch, device=device)
            h1 = torch.ones(args.batch, device=device)
            o0 = model(rgb, depth, state=state, is_holding=h0)
            o1 = model(rgb, depth, state=state, is_holding=h1)
        diff = (o0["mass"] - o1["mass"]).abs().max().item()
        print(f"FiLM identity-init check: max|out(h=0) - out(h=1)| = {diff:.2e} "
              f"(should be 0 at init)")

    t0 = time.time()
    out = model(rgb, depth, state=state, is_holding=is_holding)
    fwd_ms = (time.time() - t0) * 1000
    print(f"\nforward pass (b={args.batch}, hw=({args.h},{args.w})) — {fwd_ms:.1f} ms")
    for k, v in out.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    # Loss smoke
    batch = {
        "target_mass": torch.rand(args.batch, 1, args.h, args.w, device=device) * 2,
        "target_depth": torch.rand(args.batch, 1, args.h, args.w, device=device) * 1.5,
        "target_mass_total": torch.rand(args.batch, device=device) * 1000,
    }
    loss, comps = model.loss(out, batch)
    print(f"\nloss = {loss.item():.4f}")
    for k, v in comps.items():
        print(f"  {k}: {v:.4f}")
    loss.backward()
    print("\nbackward OK")
