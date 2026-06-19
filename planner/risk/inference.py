"""Single-state inference adapter for the trained contact predictor.

The dataset training path (``planner.risk.benchmark_dataset.BenchmarkDataset``
+ collate) operates on batched HDF5 reads. This module exposes the same
``GatekeeperCoordFiLMUNet`` (and the simpler ablation siblings) at
**single-state** granularity — pass in one RGB window + one state window + a
failure descriptor, get back a heatmap + gatekeeper probability. Used by
the safety-rollout experiments where the action source (open-loop demo
replay) feeds the predictor a single env observation at each step.

Key entry points:

- :class:`ContactPredictor` — wraps a trained model checkpoint, exposes
  ``predict(rgb_window, state_window, failure_mode, failure_joints)``.
- :func:`marginal_heatmap` — sums per-mode predictions weighted by a prior,
  returning the "if some failure triggers, where will contacts land"
  marginal that the safety policies consume.
- :func:`risk_score` — integrates a heatmap over a dict of pixel-space
  entity masks, optionally weighted by per-entity values.
- :func:`aabb_to_image_mask` — projects a 3D axis-aligned bounding box into
  the agentview camera to produce a pixel mask suitable for integration.

Reuses the canonical failure-mode order from
:data:`planner.risk.dataset_v2._FAILURE_MODES` so this layer can't drift
from how the model was trained.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Canonical failure-mode order. Must mirror
# ``planner.risk.dataset_v2._FAILURE_MODES`` (line 21) — defined here too so
# this module doesn't drag in h5py via ``dataset_v2`` for the safety-rollout
# stack, which only needs torch + numpy. A sanity check at import time
# verifies the two stay in sync if both modules are available.
_FAILURE_MODES = ("GRIPPER_OPEN", "SLIPPERY_GRIP", "SINGLE_JOINT",
                  "MULTI_JOINT", "ALL_JOINTS")

try:
    from planner.risk.dataset_v2 import _FAILURE_MODES as _DATASET_FAILURE_MODES
    if tuple(_DATASET_FAILURE_MODES) != _FAILURE_MODES:
        raise RuntimeError(
            "_FAILURE_MODES drift: inference.py vs dataset_v2.py disagree "
            f"({_FAILURE_MODES} vs {tuple(_DATASET_FAILURE_MODES)})")
except ImportError:
    pass  # h5py not available — assume the inline tuple is canonical

from planner.risk.models.heatmapbaseline import (
    GatekeeperCoordFiLMUNet,
    HeatmapUNet,
    HeatmapUNetLegacy,
    FiLMHeatmapUNet,
    CoordFiLMHeatmapUNet,
)
from planner.risk.spatial import integrate_per_entity


# Map of arch name (as stored in checkpoints) → class. Add new ones here
# when a checkpoint comes from a freshly-named variant.
_ARCH_REGISTRY = {
    "HeatmapUNet": HeatmapUNet,
    "HeatmapUNetLegacy": HeatmapUNetLegacy,
    "FiLMHeatmapUNet": FiLMHeatmapUNet,
    "CoordFiLMHeatmapUNet": CoordFiLMHeatmapUNet,
    "GatekeeperCoordFiLMUNet": GatekeeperCoordFiLMUNet,
}

# Default joint sets per failure mode, used by marginal_heatmap when the
# caller doesn't supply explicit joints. Pitch joints (j2/j4/j6) carry the
# planner-relevant signal per dataset construction; we mirror that here.
_DEFAULT_JOINTS = {
    "GRIPPER_OPEN":  [],
    "SLIPPERY_GRIP": [],
    "SINGLE_JOINT":  [4],
    "MULTI_JOINT":   [2, 4],
    "ALL_JOINTS":    [1, 2, 3, 4, 5, 6, 7],
}


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------

@dataclass
class _CheckpointMeta:
    arch: str
    state_dim: int
    base_ch: int
    H: int
    W: int
    epoch: Optional[int]
    val_heat: Optional[float]


class ContactPredictor:
    """Trained contact-heatmap predictor at single-state granularity.

    Construct via :meth:`from_checkpoint`. Stays loaded on the configured
    device and exposes deterministic inference (``model.eval()``) until the
    object is discarded.
    """

    def __init__(self, model: torch.nn.Module, meta: _CheckpointMeta,
                 device: torch.device):
        self.model = model
        self.meta = meta
        self.device = device
        self._is_dual = isinstance(model, GatekeeperCoordFiLMUNet)
        self.model.eval()

    # ------------------------------------------------------------------ load

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path,
                        device: Optional[str | torch.device] = None,
                        arch_override: Optional[str] = None) -> "ContactPredictor":
        """Restore a predictor from a torch.save'd checkpoint.

        Checkpoint dict is expected to contain:

        - ``"model"`` — state_dict for one of the variants in
          ``_ARCH_REGISTRY``.
        - ``"arch"`` — string identifying the architecture, format
          ``"<ClassName>(state_dim=..., base_ch=...)"`` (the dual-model
          notebook saves this exact format). If absent, pass ``arch_override``.

        Optional but read if present: ``"epoch"``, ``"val_heat"``.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        arch_str = arch_override or ckpt.get("arch", "GatekeeperCoordFiLMUNet")
        meta = _parse_arch_str(arch_str, ckpt)

        cls_ = _ARCH_REGISTRY.get(meta.arch)
        if cls_ is None:
            raise ValueError(
                f"unknown arch {meta.arch!r}; known: {sorted(_ARCH_REGISTRY)}")
        model = cls_(state_dim=meta.state_dim, base_ch=meta.base_ch,
                     H=meta.H, W=meta.W).to(device)

        # Tolerate "model" or "model_state" key — match both notebook and
        # planner.risk.models train scripts.
        state_dict = ckpt.get("model") or ckpt.get("model_state")
        if state_dict is None:
            raise KeyError(f"checkpoint at {ckpt_path} lacks 'model' / "
                           "'model_state' key")
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"state_dict mismatch: missing={missing} unexpected={unexpected}")

        return cls(model=model, meta=meta, device=device)

    # --------------------------------------------------------------- predict

    @torch.no_grad()
    def predict(self, rgb_window: np.ndarray, state_window: np.ndarray,
                failure_mode: str, failure_joints: Optional[list[int]] = None,
                ) -> tuple[np.ndarray, float]:
        """Run one forward pass.

        Parameters
        ----------
        rgb_window : (T, 3, H, W) uint8 or float — the temporal RGB window the
            trainer saw. Only the last frame is used by the current
            architectures, matching ``build_inputs`` in the dual-model notebook.
        state_window : (T, 18) float — per-frame [qpos(7), qvel(7), ee(3),
            grip(1)] state, same layout the trainer uses.
        failure_mode : one of ``_FAILURE_MODES``.
        failure_joints : list of 1-based arm joint indices that failed. Empty
            list / None for GRIPPER_OPEN and SLIPPERY_GRIP.

        Returns
        -------
        heatmap : (H, W) float32 — log1p contact mass in agentview pixels.
        gate_prob : float — gatekeeper probability that this trial produces
            any contact. ``float("nan")`` for non-gated architectures.
        """
        if failure_mode not in _FAILURE_MODES:
            raise ValueError(
                f"failure_mode={failure_mode!r} not in {_FAILURE_MODES}")

        # RGB: take the last frame, batch it. uint8 stays uint8 (model
        # divides by 255 internally); float passes through.
        rgb_last = np.asarray(rgb_window)[-1]                              # (3,H,W)
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_last)).unsqueeze(0).to(
            self.device, non_blocking=True)
        if rgb_t.dtype != torch.uint8:
            rgb_t = rgb_t.float()

        # State: flatten the window + concat failure descriptor.
        state_flat = np.asarray(state_window, dtype=np.float32).flatten()  # (T*18,)
        mode_oh = np.zeros(5, dtype=np.float32)
        mode_oh[_FAILURE_MODES.index(failure_mode)] = 1.0
        joint_oh = np.zeros(7, dtype=np.float32)
        for j in (failure_joints or []):
            ji = int(j) - 1
            if 0 <= ji < 7:
                joint_oh[ji] = 1.0
        state_full = np.concatenate([state_flat, mode_oh, joint_oh])
        state_t = torch.from_numpy(state_full).unsqueeze(0).to(
            self.device, non_blocking=True)

        if self._is_dual:
            heatmap_t, gate_logit_t = self.model(rgb_t, state_t)
            gate_prob = float(torch.sigmoid(gate_logit_t).item())
        else:
            heatmap_t = self.model(rgb_t, state_t)
            gate_prob = float("nan")

        return heatmap_t.squeeze(0).cpu().numpy().astype(np.float32), gate_prob


# ---------------------------------------------------------------------------
# Marginal over failure modes
# ---------------------------------------------------------------------------

def marginal_heatmap(
    predictor: ContactPredictor,
    rgb_window: np.ndarray,
    state_window: np.ndarray,
    mode_prior: Optional[dict[str, float]] = None,
    mode_joints: Optional[dict[str, list[int]]] = None,
) -> tuple[np.ndarray, float]:
    """Predicted heatmap marginalised over failure modes.

    ``mode_prior``: dict from mode name → prior weight. Defaults to uniform
    over the 5 canonical modes. Normalised internally.

    ``mode_joints``: dict from mode name → joints to use for that mode in
    the per-mode call. Defaults to ``_DEFAULT_JOINTS``.

    Returns ``(marginal_heatmap, marginal_gate_prob)`` where the gate
    probability is the prior-weighted average of per-mode gate probabilities.
    """
    prior = dict(mode_prior or {m: 1.0 for m in _FAILURE_MODES})
    z = sum(prior.values())
    if z <= 0:
        raise ValueError("mode_prior weights sum to zero")
    prior = {k: v / z for k, v in prior.items()}

    joints = dict(_DEFAULT_JOINTS)
    if mode_joints is not None:
        joints.update(mode_joints)

    total_heat = None
    total_gate = 0.0
    for mode, w in prior.items():
        h, g = predictor.predict(rgb_window, state_window, mode,
                                 joints.get(mode, []))
        if total_heat is None:
            total_heat = np.zeros_like(h)
        total_heat += w * h
        if not np.isnan(g):
            total_gate += w * g
    return total_heat, total_gate  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Risk scoring + mask projection
# ---------------------------------------------------------------------------

def risk_score(
    heatmap: np.ndarray,
    masks: dict[str, np.ndarray],
    object_values: Optional[dict[str, float]] = None,
) -> tuple[float, dict[str, float]]:
    """Integrate ``heatmap`` over each named pixel mask, optionally weighted.

    Wraps :func:`planner.risk.spatial.integrate_per_entity` and applies a
    per-entity value multiplier. ``object_values`` defaults to uniform 1.0
    over ``masks.keys()`` (this is the explicit project-wide default for
    the initial exploration).

    Returns ``(total_risk, per_entity_risk)``.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"expected 2D heatmap, got shape {heatmap.shape}")
    per_entity_raw = integrate_per_entity(heatmap, masks)
    values = dict(object_values or {k: 1.0 for k in per_entity_raw})
    per_entity = {k: v * values.get(k, 0.0) for k, v in per_entity_raw.items()}
    total = float(sum(per_entity.values()))
    return total, per_entity


def aabb_to_image_mask(
    aabb_min: np.ndarray, aabb_max: np.ndarray,
    cam_pos: np.ndarray, cam_mat0: np.ndarray,
    fovy_deg: float, image_hw: tuple[int, int],
) -> np.ndarray:
    """Project a 3D AABB into a camera and return a binary pixel mask of
    its bounding rectangle in image coords.

    Uses the same agentview projection convention as
    :mod:`planner.risk.v2_targets` (Y-row negation so the image Y axis points
    down). Camera looks along -Z in its own frame.

    Returns ``(H, W) bool`` array, True inside the projected rectangle.
    """
    H, W = image_hw
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    mat0 = np.asarray(cam_mat0, dtype=np.float64).reshape(3, 3)

    R = mat0.T.copy()
    R[1] = -R[1]
    fy = (H / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy
    cx, cy = W / 2.0, H / 2.0

    # 8 AABB corners
    x0, y0, z0 = aabb_min
    x1, y1, z1 = aabb_max
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1],
    ], dtype=np.float64)

    cam_coords = (corners - cam_pos) @ R.T
    depth = -cam_coords[:, 2]
    in_front = depth > 1e-6
    if not in_front.any():
        return np.zeros((H, W), dtype=bool)

    u = fx * cam_coords[in_front, 0] / depth[in_front] + cx
    v = fy * cam_coords[in_front, 1] / depth[in_front] + cy

    u_min = int(np.floor(max(0, u.min())))
    u_max = int(np.ceil(min(W - 1, u.max())))
    v_min = int(np.floor(max(0, v.min())))
    v_max = int(np.ceil(min(H - 1, v.max())))
    if u_max < u_min or v_max < v_min:
        return np.zeros((H, W), dtype=bool)

    mask = np.zeros((H, W), dtype=bool)
    mask[v_min:v_max + 1, u_min:u_max + 1] = True
    return mask


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_arch_str(arch_str: str, ckpt: dict) -> _CheckpointMeta:
    """Pull (class, state_dim, base_ch) out of the saved arch string.

    Accepts e.g. ``"GatekeeperCoordFiLMUNet(state_dim=156, base_ch=16)"``.
    Falls back to defaults for missing fields.
    """
    import re
    name_match = re.match(r"\s*([A-Za-z_]\w*)", arch_str)
    if not name_match:
        raise ValueError(f"can't parse arch string {arch_str!r}")
    name = name_match.group(1)
    sd = re.search(r"state_dim\s*=\s*(\d+)", arch_str)
    bc = re.search(r"base_ch\s*=\s*(\d+)", arch_str)
    h  = re.search(r"H\s*=\s*(\d+)", arch_str)
    w  = re.search(r"W\s*=\s*(\d+)", arch_str)
    return _CheckpointMeta(
        arch=name,
        state_dim=int(sd.group(1)) if sd else 156,
        base_ch=int(bc.group(1)) if bc else 16,
        H=int(h.group(1)) if h else 240,
        W=int(w.group(1)) if w else 320,
        epoch=ckpt.get("epoch"),
        val_heat=ckpt.get("val_heat") or ckpt.get("val_loss"),
    )


__all__ = [
    "ContactPredictor",
    "marginal_heatmap",
    "risk_score",
    "aabb_to_image_mask",
]
