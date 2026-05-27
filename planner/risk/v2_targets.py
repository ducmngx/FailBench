"""Agentview heatmap targets for the v2 benchmark.

Pure-numpy projection of a trial's world-frame contacts into the static
``agentview`` camera, smoothed into a 2D mass map. No MuJoCo load needed —
the v2 schema stores ``cam_agentview_{pos,mat0,fovy,size}`` per trial.

Convention matches :class:`planner.risk.projection_labels._DynamicCameraProjector`:
camera looks along ``-Z`` in its own frame; image ``v`` axis points DOWN, so the
Y-row of the rotation is negated.

Per-trial scalar ``failure_prob`` (set by ``LiberoRunner``) is used in place of
the v10 ``failure_probs[contact_failure_id]`` lookup — every contact in a v2
trial belongs to the same sampled failure mode, so the prior is a constant
multiplier across the cloud.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

Weighting = Literal["force_prior", "force", "count"]


@dataclass(frozen=True)
class AgentviewTarget:
    heatmap: np.ndarray   # (H, W) float32, mass-preserving
    n_in_frame: int       # contacts that landed on-image and in front of cam
    weight_in_frame: float
    size: Tuple[int, int]  # (W, H) for downstream sanity


def _projector_from_trial(trial: dict):
    """Extract the static agentview pinhole projection from a trial dict.

    Returns ``(R, cam_pos, fx, fy, cx, cy, W, H)`` where ``R`` is the
    image-convention camera rotation (Y-row negated, so projection is
    ``(world - cam_pos) @ R.T`` giving camera-frame coords with image-down
    Y axis).
    """
    cam_pos = np.asarray(trial["cam_agentview_pos"], dtype=np.float64)
    mat0 = np.asarray(trial["cam_agentview_mat0"], dtype=np.float64).reshape(3, 3)
    fovy_deg = float(trial["cam_agentview_fovy"])
    W, H = (int(x) for x in trial["cam_agentview_size"])

    R = mat0.T.copy()
    R[1] = -R[1]
    fy = (H / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy  # square pixels — MuJoCo cameras only specify fovy
    cx, cy = W / 2.0, H / 2.0
    return R, cam_pos, fx, fy, cx, cy, W, H


def _contact_weights(trial: dict, mode: Weighting) -> np.ndarray:
    """Per-contact scalar weight in v2 (failure_prob is constant per trial).

    Force magnitude reads ``contact_force_world`` (linear, world frame). If
    only the older ``contact_forces`` is present we fall back to its first
    three columns (the local contact frame's linear component) — magnitude is
    invariant to frame so this is fine.
    """
    n = trial["contact_positions"].shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    if mode == "count":
        return np.ones((n,), dtype=np.float32)

    f = trial.get("contact_force_world")
    if f is None:
        f = trial["contact_forces"][:, :3]
    mag = np.linalg.norm(np.asarray(f, dtype=np.float32), axis=1)

    if mode == "force":
        return mag
    if mode == "force_prior":
        prob = float(trial.get("failure_prob", 1.0))
        return mag * np.float32(prob)
    raise ValueError(f"unknown weighting: {mode!r}")


def build_agentview_target(
    trial: dict,
    *,
    sigma_px: float = 4.0,
    weighting: Weighting = "force_prior",
) -> AgentviewTarget:
    """Build a (H, W) agentview contact-mass heatmap for one v2 trial.

    ``trial`` is the dict returned by :meth:`V2Reader.read_trial` (or one
    element of :class:`LiberoV2Dataset`). Must contain ``contact_positions``,
    a force field, ``cam_agentview_{pos,mat0,fovy,size}``, and (for
    ``weighting='force_prior'``) the ``failure_prob`` scalar attr.
    """
    R, cam_pos, fx, fy, cx, cy, W, H = _projector_from_trial(trial)
    out = np.zeros((H, W), dtype=np.float32)

    pts = np.asarray(trial["contact_positions"], dtype=np.float64)
    if pts.shape[0] == 0:
        return AgentviewTarget(out, 0, 0.0, (W, H))

    p_cam = (pts - cam_pos) @ R.T
    depth = -p_cam[:, 2]
    in_front = depth > 1e-6
    if not in_front.any():
        return AgentviewTarget(out, 0, 0.0, (W, H))

    u = fx * p_cam[in_front, 0] / depth[in_front] + cx
    v = fy * p_cam[in_front, 1] / depth[in_front] + cy
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not in_img.any():
        return AgentviewTarget(out, 0, 0.0, (W, H))

    w_all = _contact_weights(trial, weighting)
    w = w_all[in_front][in_img].astype(np.float32)

    ix = np.clip(u[in_img].astype(np.int32), 0, W - 1)
    iy = np.clip(v[in_img].astype(np.int32), 0, H - 1)
    np.add.at(out, (iy, ix), w)

    if sigma_px > 0:
        out = gaussian_filter(out, sigma=sigma_px, mode="constant", cval=0.0)

    return AgentviewTarget(out, int(in_img.sum()), float(w.sum()), (W, H))
