"""LIBERO-friendly contact-distribution labels.

Two representations, both built from a trial's raw ``contact_positions`` +
``contact_forces`` + ``failure_probs[contact_failure_id]`` weights:

* :func:`build_camera_heatmap` — project contacts into a named camera image
  plane and splat with a 2D Gaussian. Fixed (H, W); naturally aligned with the
  RGB-D inputs an image model receives.
* :func:`build_voxel_density` — rasterise contacts into a fixed 3D world-frame
  grid and smooth with a 3D Gaussian. Preserves z, so distinguishes
  table-level contacts from cabinet/shelf contacts.

Both functions take a precomputed ``contact_weights`` vector so the caller
chooses count-only / force-weighted / prior-weighted independently.

Mass-preserving: Gaussian smoothing is mass-preserving by construction, so the
heatmap/voxel sum equals the in-frame / in-bounds weight sum (within numerical
noise from boundary truncation of the kernel).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter


class _DynamicCameraProjector:
    """Pinhole projector that reads the camera's CURRENT world pose from MjData.

    Required for body-attached cameras (e.g. ``robot0_eye_in_hand``) whose
    ``model.cam_pos`` is *local to the parent body* — the static pose used by
    :class:`planner.experiments.data_capture.ContactProjector` is wrong as soon
    as the arm moves. For world-frame cameras the answer matches the static
    projector exactly.

    Caller must have called ``mujoco.mj_forward(model, data)`` so
    ``data.cam_xpos`` / ``data.cam_xmat`` reflect the current qpos.

    Y-axis is negated to point DOWN in the image (same image convention as
    ``ContactProjector``).
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 camera_name: str, width: int, height: int):
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Camera '{camera_name}' not found in model")
        self.width = width
        self.height = height
        self.cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64).copy()
        # cam_xmat is (9,) row-major; columns of (3,3) are world-frame axes.
        # Transpose so rows = camera axes, then negate Y for image convention.
        self.cam_rot = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3).T.copy()
        self.cam_rot[1] = -self.cam_rot[1]
        fovy_deg = float(model.cam_fovy[cam_id])
        fovy_rad = math.radians(fovy_deg)
        fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx, cy = width / 2.0, height / 2.0
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def project(self, world_points: np.ndarray):
        pts = np.atleast_2d(world_points).astype(np.float64)
        p_cam = (pts - self.cam_pos) @ self.cam_rot.T
        depth = -p_cam[:, 2]
        safe = np.where(depth > 1e-8, depth, 1e-8)
        u = self.K[0, 0] * p_cam[:, 0] / safe + self.K[0, 2]
        v = self.K[1, 1] * p_cam[:, 1] / safe + self.K[1, 2]
        return np.column_stack([u, v]), depth

    def in_frame(self, pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
        u, v = pixels[:, 0], pixels[:, 1]
        return (depths > 0) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)


def contact_weights_force_prior(
    contact_forces: np.ndarray,
    contact_failure_id: np.ndarray,
    failure_probs: np.ndarray,
) -> np.ndarray:
    """Per-contact weight = ‖force_xyz‖ * failure_probs[contact_failure_id].

    Severity (force magnitude) × the probability we'd actually sample this
    failure mode. Matches the paper's risk formulation.
    """
    if contact_forces.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    mag = np.linalg.norm(contact_forces[:, :3], axis=1).astype(np.float32)
    prior = failure_probs[contact_failure_id].astype(np.float32)
    return mag * prior


def build_camera_heatmap(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_positions: np.ndarray,
    contact_weights: np.ndarray,
    camera_name: str,
    width: int = 640,
    height: int = 480,
    sigma_px: float = 8.0,
    with_depth: bool = False,
) -> Tuple[np.ndarray, int, float]:
    """Project ``contact_positions`` into ``camera_name`` and splat into (H, W).

    ``data`` must have been ``mj_forward``-ed with the qpos that matches the
    contacts' world frame — for LIBERO, that's ``pre_qpos`` from the trial
    npz. The camera's world pose is read from ``data.cam_xpos`` / ``cam_xmat``
    so body-attached cameras (wrist) get the right pose.

    When ``with_depth=True`` the output is (H, W, 2) with channel 0 = smoothed
    mass and channel 1 = smoothed mean depth (metres). Mean depth is computed
    as gauss(weighted_depth_sum) / gauss(mass_sum) so that smoothing happens on
    the *sums* (averaging ratios directly produces edge artefacts at low-mass
    pixels). Cells with mass < eps get depth=0.

    Returns
    -------
    heatmap : (H, W) float32 if with_depth else (H, W, 2) float32
    n_in_frame : number of contacts that landed on-image and in front of camera
    weight_in_frame : sum of weights of the in-frame contacts (== mass-channel
                      sum within Gaussian boundary truncation)
    """
    if with_depth:
        empty = np.zeros((height, width, 2), dtype=np.float32)
    else:
        empty = np.zeros((height, width), dtype=np.float32)
    if contact_positions.shape[0] == 0:
        return empty, 0, 0.0

    proj = _DynamicCameraProjector(model, data, camera_name=camera_name,
                                   width=width, height=height)
    pixels, depths = proj.project(contact_positions)
    mask = proj.in_frame(pixels, depths)
    if not mask.any():
        return empty, 0, 0.0

    u = pixels[mask, 0]
    v = pixels[mask, 1]
    w = contact_weights[mask].astype(np.float32)
    d = depths[mask].astype(np.float32)

    mass_sum = np.zeros((height, width), dtype=np.float32)
    ix = np.clip(u.astype(np.int32), 0, width - 1)
    iy = np.clip(v.astype(np.int32), 0, height - 1)
    np.add.at(mass_sum, (iy, ix), w)

    if with_depth:
        depth_sum = np.zeros((height, width), dtype=np.float32)
        np.add.at(depth_sum, (iy, ix), w * d)

    if sigma_px > 0:
        mass_sum = gaussian_filter(mass_sum, sigma=sigma_px, mode="constant", cval=0.0)
        if with_depth:
            depth_sum = gaussian_filter(depth_sum, sigma=sigma_px, mode="constant", cval=0.0)

    if not with_depth:
        return mass_sum, int(mask.sum()), float(w.sum())

    # Mean depth = depth_sum / mass_sum, with mass=0 cells set to 0.
    eps = 1e-8
    mean_depth = np.where(mass_sum > eps, depth_sum / np.maximum(mass_sum, eps), 0.0).astype(np.float32)
    out = np.stack([mass_sum, mean_depth], axis=-1)
    return out, int(mask.sum()), float(w.sum())


# -----------------------------------------------------------------------------
# Smoothing / processing helpers
# -----------------------------------------------------------------------------


def log1p_label(label: np.ndarray) -> np.ndarray:
    """``np.log1p`` wrapper preserving dtype/shape. Reversible via ``np.expm1``.

    For multi-channel labels (depth-channel cam heatmap), apply only to the
    mass channel — depth is in physical units and should not be log-compressed.
    """
    return np.log1p(label).astype(np.float32)


def sum_normalize_label(label: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, float]:
    """Divide ``label`` by its total sum so it integrates to 1.

    Returns
    -------
    normalised : same shape as ``label``, sums to 1.0 (or 0.0 if empty)
    total : the original total sum (cast to float)
    """
    total = float(label.sum())
    if total <= eps:
        return np.zeros_like(label, dtype=np.float32), 0.0
    return (label / total).astype(np.float32), total


def aggregate_labels(labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean of ``labels`` across the leading axis.

    ``labels`` has shape (N, ...); ``weights`` has shape (N,). When all weights
    are zero, returns a zero array of the per-sample shape.
    """
    w = np.asarray(weights, dtype=np.float32)
    wsum = float(w.sum())
    if wsum <= 0:
        return np.zeros(labels.shape[1:], dtype=np.float32)
    # Reshape weights to broadcast against the per-sample shape.
    shape = (len(w),) + (1,) * (labels.ndim - 1)
    return (labels * w.reshape(shape)).sum(axis=0).astype(np.float32) / wsum


def build_voxel_density(
    contact_positions: np.ndarray,
    contact_weights: np.ndarray,
    bounds: np.ndarray,
    voxel_cm: float = 3.0,
    sigma_vox: float = 2.0,
) -> Tuple[np.ndarray, int, float]:
    """Rasterise contacts onto a 3D world-frame grid, smoothed by σ voxels.

    Parameters
    ----------
    bounds : (3, 2) float
        [[x_min, x_max], [y_min, y_max], [z_min, z_max]] in metres.
    voxel_cm : float
        Voxel edge length in cm. 3.0 → about 27×27×20 voxels for a typical
        LIBERO workspace.
    sigma_vox : float
        Gaussian smoothing σ in voxel units.

    Returns
    -------
    density : (nz, ny, nx) float32
    n_in_bounds : number of contacts that fell inside ``bounds``
    weight_in_bounds : sum of weights for in-bounds contacts
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    voxel_m = voxel_cm / 100.0

    nx = max(1, int(np.ceil((bounds[0, 1] - bounds[0, 0]) / voxel_m)))
    ny = max(1, int(np.ceil((bounds[1, 1] - bounds[1, 0]) / voxel_m)))
    nz = max(1, int(np.ceil((bounds[2, 1] - bounds[2, 0]) / voxel_m)))

    density = np.zeros((nz, ny, nx), dtype=np.float32)
    if contact_positions.shape[0] == 0:
        return density, 0, 0.0

    p = contact_positions.astype(np.float64)
    in_bounds = (
        (p[:, 0] >= bounds[0, 0]) & (p[:, 0] < bounds[0, 1]) &
        (p[:, 1] >= bounds[1, 0]) & (p[:, 1] < bounds[1, 1]) &
        (p[:, 2] >= bounds[2, 0]) & (p[:, 2] < bounds[2, 1])
    )
    if not in_bounds.any():
        return density, 0, 0.0

    pi = p[in_bounds]
    w = contact_weights[in_bounds].astype(np.float32)

    ix = np.clip(((pi[:, 0] - bounds[0, 0]) / voxel_m).astype(np.int32), 0, nx - 1)
    iy = np.clip(((pi[:, 1] - bounds[1, 0]) / voxel_m).astype(np.int32), 0, ny - 1)
    iz = np.clip(((pi[:, 2] - bounds[2, 0]) / voxel_m).astype(np.int32), 0, nz - 1)

    np.add.at(density, (iz, iy, ix), w)

    if sigma_vox > 0:
        density = gaussian_filter(density, sigma=sigma_vox, mode="constant", cval=0.0)

    return density, int(in_bounds.sum()), float(w.sum())


def derive_bounds_from_contacts(
    contact_positions_list,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    pad_cm: float = 18.0,
    round_to_cm: float = 5.0,
) -> np.ndarray:
    """Heuristic world-frame bounds from a collection of trials' contact clouds.

    Concatenates all positions, takes the [low_pct, high_pct] percentile per
    axis, pads outward by ``pad_cm`` (default 18 cm ≈ 3σ at σ_vox=2, voxel=3cm
    — keeps Gaussian-smoothed mass inside the grid), then rounds outward to
    the next ``round_to_cm`` grid line. Returns (3, 2) float64.
    """
    pts = [p for p in contact_positions_list if p.shape[0] > 0]
    if not pts:
        # Fallback: a 1 m cube centred at the origin.
        return np.array([[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]], dtype=np.float64)
    p = np.concatenate(pts, axis=0)
    lo = np.percentile(p, low_pct, axis=0) - pad_cm / 100.0
    hi = np.percentile(p, high_pct, axis=0) + pad_cm / 100.0
    step = round_to_cm / 100.0
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
    return np.stack([lo, hi], axis=1).astype(np.float64)
