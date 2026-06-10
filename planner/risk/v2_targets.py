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


# --------------------------------------------------------------------------
# Projection cache — small precomputed handle that the GPU target builder
# consumes at training time.
#
# The cache stores, per trial, the (N_in_frame, 3) array of in-frame projected
# contacts as ``[u_pixel, v_pixel, force_magnitude]``. The Gaussian smoothing
# and the failure-prob multiplication are NOT baked in — those happen on the
# GPU at training time, so the user can tune ``sigma_px`` and ``weighting``
# without rebuilding the cache.
# --------------------------------------------------------------------------


def project_contacts_for_cache(trial: dict) -> tuple[np.ndarray, int, int]:
    """Project a trial's contacts to in-frame pixel coords.

    Returns ``(arr, H, W)`` where ``arr`` is shape ``(N_in_frame, 3)`` float32
    with columns ``[u_pixel, v_pixel, force_magnitude]``. Empty array if no
    contact falls in-frame.

    Matches the projection math in :func:`build_agentview_target` exactly so a
    GPU-rebuilt target reproduces the on-the-fly target up to float32 rounding.
    """
    R, cam_pos, fx, fy, cx, cy, W, H = _projector_from_trial(trial)
    pts = np.asarray(trial["contact_positions"], dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), H, W

    p_cam = (pts - cam_pos) @ R.T
    depth = -p_cam[:, 2]
    in_front = depth > 1e-6
    if not in_front.any():
        return np.zeros((0, 3), dtype=np.float32), H, W

    u = fx * p_cam[in_front, 0] / depth[in_front] + cx
    v = fy * p_cam[in_front, 1] / depth[in_front] + cy
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not in_img.any():
        return np.zeros((0, 3), dtype=np.float32), H, W

    # Force magnitude (force_prior weighting; multiply by failure_prob on GPU).
    f = trial.get("contact_force_world")
    if f is None:
        f = trial["contact_forces"][:, :3]
    mag_all = np.linalg.norm(np.asarray(f, dtype=np.float32), axis=1)
    mag = mag_all[in_front][in_img].astype(np.float32)
    u = u[in_img].astype(np.float32)
    v = v[in_img].astype(np.float32)

    return np.stack([u, v, mag], axis=1), H, W


def build_target_from_projection(
    proj_flat: "torch.Tensor",
    batch_idx: "torch.Tensor",
    failure_prob: "torch.Tensor",
    *,
    H: int,
    W: int,
    sigma_px: float = 4.0,
    log1p: bool = True,
):
    """Build a (B, H, W) contact-mass heatmap from a flat batch of projections.

    Uses a sparse-flat layout because per-trial contact counts vary by 10× or
    more (median ~2k, p99 ~15k). Padding to max wastes memory; concatenating
    + carrying a batch-index array is 3-7× more efficient.

    ``proj_flat``     : (total_N, 3) float32 — concatenated [u, v, force_mag] across the batch.
    ``batch_idx``     : (total_N,) int64    — which batch element each row belongs to.
    ``failure_prob``  : (B,) float32         — per-trial scalar prior.

    Reproduces ``build_agentview_target(weighting="force_prior")`` up to float32
    rounding. ~1-2 ms/batch on GPU.
    """
    import torch
    import torch.nn.functional as F

    B = int(failure_prob.shape[0])
    device = proj_flat.device

    # Per-contact weight = force * per-trial failure_prob[batch_idx].
    w = proj_flat[:, 2] * failure_prob[batch_idx]                   # (total_N,)
    u = proj_flat[:, 0].long().clamp_(0, W - 1)
    v = proj_flat[:, 1].long().clamp_(0, H - 1)
    flat_idx = batch_idx * (H * W) + v * W + u                      # (total_N,)

    target = torch.zeros(B * H * W, device=device, dtype=w.dtype)
    target.scatter_add_(0, flat_idx, w)
    target = target.view(B, 1, H, W)

    if sigma_px > 0:
        target = _separable_gaussian(target, sigma=sigma_px)

    target = target.squeeze(1)                                      # (B, H, W)
    if log1p:
        target = torch.log1p(target)
    return target


def _separable_gaussian(x: "torch.Tensor", sigma: float) -> "torch.Tensor":
    """Separable Gaussian via two conv2d calls. Matches scipy.ndimage with
    ``mode="constant", cval=0.0`` to about 1e-4 across the supported range.
    """
    import torch
    import torch.nn.functional as F

    radius = max(1, int(round(3 * sigma)))
    k = 2 * radius + 1
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - radius
    g = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    g = g / g.sum()
    kh = g.view(1, 1, k, 1)
    kw = g.view(1, 1, 1, k)
    x = F.conv2d(x, kh, padding=(radius, 0))
    x = F.conv2d(x, kw, padding=(0, radius))
    return x
