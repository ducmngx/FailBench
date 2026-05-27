"""Image-conditional UNet baseline — wraps :class:`LiberoHeatmapModel`.

Takes the v2 windowed RGB+depth input, temporally averages across T to feed
the single-frame ResNet-18 UNet, and remaps its (mass, depth) head to the
benchmark's ``pred`` contract. Aux depth and mass-total heads are exposed in
the output dict for downstream losses but the default trainer ignores them.

ImageNet normalisation is applied to RGB before the encoder (the ResNet-18
backbone is ImageNet-pretrained, depth channel is initialised to the mean of
RGB weights — see :class:`LiberoHeatmapModel`).

Modality requirements
---------------------
``modalities.rgb`` is required. ``modalities.depth`` is optional; if absent,
a zero depth channel is passed (the model still consumes 4 channels by
construction). ``modalities.state`` toggles the inner ``use_state`` path
(qpos⊕qvel only — first 14 dims of the 18-D state vector).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from planner.risk.benchmark_dataset import ModalityConfig
from planner.risk.libero_model import LiberoHeatmapModel

# ImageNet stats used by torchvision pretrained ResNets.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class BenchmarkUNet(nn.Module):
    """v2-benchmark wrapper around :class:`LiberoHeatmapModel`.

    Temporal handling: per-frame inputs are mean-pooled across the T window.
    This is the cheapest "use the window" baseline; a 3D-conv variant can be
    added later if mean-pooling leaves headroom (see the plan).
    """

    def __init__(self, *, modalities: ModalityConfig, grid_hw: tuple,
                 T: int = 8, K: int = 3, pretrained: bool = True,
                 temporal_mode: str = "mean"):
        super().__init__()
        if not modalities.rgb:
            raise ValueError("BenchmarkUNet requires modalities.rgb=True")
        if temporal_mode not in ("mean", "conv3d", "last", "late_fusion"):
            raise ValueError(f"temporal_mode must be mean|conv3d|last|late_fusion, got {temporal_mode!r}")
        self.modalities = modalities
        self.grid_hw = tuple(grid_hw)
        self.T = T
        self.temporal_mode = temporal_mode

        # FiLM conditioning: concat failure_mode (5) + failure_joints (7) into
        # a 12-D vector if both are on; whichever subset is enabled determines
        # the size. Inner model treats it as "n_failure_modes" but the meaning
        # is "failure descriptor" — FiLM is structure-agnostic.
        fm_dim = (5 if modalities.failure_mode else 0) + \
                 (7 if modalities.failure_joints else 0)
        self.inner = LiberoHeatmapModel(
            use_state=modalities.state,
            use_holding=False,           # v2 has no holding bit in the loader
            use_failure_mode=(fm_dim > 0),
            n_failure_modes=max(fm_dim, 1),
            pretrained=pretrained,
            state_dim=14,                # qpos(7) + qvel(7) only
        )

        # Optional per-channel temporal mixer. Kernel (T, 1, 1) collapses the
        # time axis into a single frame while letting the model learn which
        # frames matter. Identity-init so the first forward equals a uniform
        # temporal mean (matches the old "mean" default at step 0).
        if temporal_mode == "conv3d":
            self.rgb_tconv = nn.Conv3d(3, 3, kernel_size=(T, 1, 1), bias=True)
            self.depth_tconv = nn.Conv3d(1, 1, kernel_size=(T, 1, 1), bias=True)
            with torch.no_grad():
                # Per-channel uniform mean: weight[c, c, :, 0, 0] = 1/T.
                for tc, C in ((self.rgb_tconv, 3), (self.depth_tconv, 1)):
                    tc.weight.zero_()
                    for c in range(C):
                        tc.weight[c, c, :, 0, 0] = 1.0 / T
                    tc.bias.zero_()

        # Keep ImageNet normalization constants on the right device via buffers.
        self.register_buffer("_im_mean", _IMAGENET_MEAN, persistent=False)
        self.register_buffer("_im_std", _IMAGENET_STD, persistent=False)

    def _failure_descriptor(self, batch: dict):
        """Concat failure_mode (5) and failure_joints (7) into the FiLM cond.

        Returns None when neither is enabled. When only one is enabled, returns
        just that tensor. The inner model's FiLM blocks were built with
        ``n_failure_modes = enabled_dim`` so the cond_dim matches.
        """
        parts = []
        if self.modalities.failure_mode:
            parts.append(batch["failure_mode"])
        if self.modalities.failure_joints:
            parts.append(batch["failure_joints"])
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def _temporal_reduce_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) → (B, C, H, W)."""
        if self.temporal_mode == "mean":
            return x.mean(dim=1)
        if self.temporal_mode == "last":
            return x[:, -1]
        # conv3d: permute to (B, C, T, H, W), conv, squeeze T.
        x = x.permute(0, 2, 1, 3, 4)
        return self.rgb_tconv(x).squeeze(2)

    def _temporal_reduce_depth(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_mode == "mean":
            return x.mean(dim=1)
        if self.temporal_mode == "last":
            return x[:, -1]
        x = x.permute(0, 2, 1, 3, 4)
        return self.depth_tconv(x).squeeze(2)

    def _late_fusion_forward(self, batch: dict) -> dict:
        """Per-frame encoder → mean-pool feature maps across T → shared decoder.

        Reuses the inner model's layers directly (encoder + decoder), but the
        skip features and bottleneck are temporally pooled before decode.
        Bypasses holding/FiLM paths (we never enable them here).
        """
        m = self.inner
        rgb = batch["rgb_window"]            # (B, T, 3, H, W)
        rgb = (rgb - self._im_mean.unsqueeze(0)) / self._im_std.unsqueeze(0)
        if self.modalities.depth:
            depth = batch["depth_window"]    # (B, T, 1, H, W)
        else:
            depth = rgb.new_zeros(rgb.shape[0], rgb.shape[1], 1, rgb.shape[3], rgb.shape[4])

        B, T = rgb.shape[:2]
        x = torch.cat([rgb, depth], dim=2)                        # (B, T, 4, H, W)
        if x.shape[-2:] != self.grid_hw:
            x = nn.functional.interpolate(
                x.reshape(B * T, 4, *x.shape[-2:]),
                size=self.grid_hw, mode="bilinear", align_corners=False
            ).view(B, T, 4, *self.grid_hw)
        x = x.reshape(B * T, 4, *self.grid_hw)

        # --- Encoder on every frame, then temporally mean-pool each skip ---
        s0_bt = m.relu(m.bn1(m.conv1(x)))
        p = m.maxpool(s0_bt)
        s1_bt = m.layer1(p)
        s2_bt = m.layer2(s1_bt)
        s3_bt = m.layer3(s2_bt)
        s4_bt = m.layer4(s3_bt)

        def _pool(t: torch.Tensor) -> torch.Tensor:
            # (B*T, C, h, w) -> (B, C, h, w) via mean across T.
            return t.view(B, T, *t.shape[1:]).mean(dim=1)

        s0, s1, s2, s3, s4 = map(_pool, (s0_bt, s1_bt, s2_bt, s3_bt, s4_bt))

        # --- Decoder (mirror of LiberoHeatmapModel.forward) ---
        bn = s4
        if self.modalities.state and m.use_state:
            state = batch["state_window"][..., :14].mean(dim=1)   # (B, 14)
            sf = m.state_mlp(state)
            sf_tile = sf[..., None, None].expand(-1, -1, bn.shape[2], bn.shape[3])
            bn = torch.cat([bn, sf_tile], dim=1)

        # FiLM conditioning from failure_mode (+ failure_joints if enabled).
        film_cond = self._failure_descriptor(batch) if m._film_cond_dim > 0 else None

        d3 = m.up4(bn, s3)
        if film_cond is not None: d3 = m.film4(d3, film_cond)
        d2 = m.up3(d3, s2)
        if film_cond is not None: d2 = m.film3(d2, film_cond)
        d1 = m.up2(d2, s1)
        if film_cond is not None: d1 = m.film2(d1, film_cond)
        d0 = m.up1(d1, s0)
        if film_cond is not None: d0 = m.film1(d0, film_cond)
        d = m.final_up(d0)
        out = m.head(d)                                            # (B, 2, H, W)

        aux = m.aux_pool(bn).flatten(1)
        mass_total = m.aux_head(aux).squeeze(-1)
        return {
            "pred": out[:, 0],
            "aux_depth": out[:, 1],
            "aux_mass_total": mass_total,
        }

    def forward(self, batch: dict) -> dict:
        if self.temporal_mode == "late_fusion":
            return self._late_fusion_forward(batch)

        rgb = self._temporal_reduce_rgb(batch["rgb_window"])           # (B, 3, H, W)
        rgb = (rgb - self._im_mean) / self._im_std

        if self.modalities.depth:
            depth = self._temporal_reduce_depth(batch["depth_window"])  # (B, 1, H, W)
        else:
            B, _, H, W = rgb.shape
            depth = rgb.new_zeros((B, 1, H, W))

        # Resize images to the target grid_hw if the loader is providing a
        # different resolution. v2 stores 240×320 which matches our target.
        if rgb.shape[-2:] != self.grid_hw:
            rgb = nn.functional.interpolate(rgb, size=self.grid_hw,
                                            mode="bilinear", align_corners=False)
            depth = nn.functional.interpolate(depth, size=self.grid_hw,
                                              mode="bilinear", align_corners=False)

        state = None
        if self.modalities.state:
            # state_window: (B, T, 18) = qpos7+qvel7+ee3+grip1.
            # Inner model wants (qpos⊕qvel) = first 14 dims, averaged across T.
            state = batch["state_window"][..., :14].mean(dim=1)

        failure_onehot = self._failure_descriptor(batch)
        out = self.inner(rgb, depth, state=state, failure_onehot=failure_onehot)
        # Inner mass head is (B, 1, H, W); the trainer expects (B, H, W).
        return {
            "pred": out["mass"].squeeze(1),
            "aux_depth": out["depth"].squeeze(1),
            "aux_mass_total": out["mass_total"],
        }
