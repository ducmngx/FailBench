"""Spatial contact-density targets for FailBench risk modeling.

Builds a per-config 2D contact heatmap over the table plane, weighted by the
failure-mode prior `failure_probs` and smoothed with a Gaussian kernel. The
heatmap is the supervised target; per-entity interaction scores are derived
post-hoc by integrating the heatmap over each entity's projected footprint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter

import mujoco

from planner.experiments.data_capture import _ROBOT_BODY_NAMES

_GEOM_BOX = 6


SCENE_TABLE_Z: dict[str, float] = {
    "scene_level2":    0.40,
    "scene_kitchen":   0.760,
    "scene_workshop":  0.824,
    "scene_grocery":   0.760,
    "scene_cluttered": 0.760,
}


def above_table_mask(positions: np.ndarray, scene: str, margin: float = 0.0) -> np.ndarray:
    """Boolean mask: True where contact world-z >= table_z + margin."""
    return positions[:, 2] >= (SCENE_TABLE_Z[scene] + margin)


@dataclass(frozen=True)
class SceneGrid:
    """Fixed 2D X-Y binning for one scene, in world coordinates (meters)."""
    scene: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int
    bin_cm: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        # matplotlib imshow extent: (x_min, x_max, y_min, y_max)
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    @property
    def xrange(self) -> tuple[float, float]:
        return (self.x_min, self.x_max)

    @property
    def yrange(self) -> tuple[float, float]:
        return (self.y_min, self.y_max)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SceneGrid":
        return SceneGrid(**d)


def derive_scene_grid(model: mujoco.MjModel, scene: str,
                      bin_cm: float = 1.0, pad_cm: float = 5.0) -> SceneGrid:
    """Build the X-Y grid for a scene from the largest horizontal table box.

    Uses the same table-detection rule as `_derive_scene_heights` in
    `scripts/generate_task_trajs.py`: largest-area horizontal box geom in any
    body whose name contains "table". Pads by `pad_cm` in xy so any contact
    just past the table edge still bins inside the grid.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    best_area = 0.0
    best = None  # (cx, cy, sx, sy)
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not bname or "table" not in bname.lower():
            continue
        if int(model.geom_type[gid]) != _GEOM_BOX:
            continue
        size = model.geom_size[gid]
        if size[0] <= 0.1 or size[1] <= 0.1:
            continue
        area = size[0] * size[1]
        if area > best_area:
            best_area = area
            cx, cy = float(data.geom_xpos[gid][0]), float(data.geom_xpos[gid][1])
            sx, sy = float(size[0]), float(size[1])
            best = (cx, cy, sx, sy)

    if best is None:
        raise RuntimeError(f"derive_scene_grid: no table box found for {scene}")

    cx, cy, sx, sy = best
    pad = pad_cm / 100.0
    bin_m = bin_cm / 100.0
    x_min, x_max = cx - sx - pad, cx + sx + pad
    y_min, y_max = cy - sy - pad, cy + sy + pad
    nx = int(np.ceil((x_max - x_min) / bin_m))
    ny = int(np.ceil((y_max - y_min) / bin_m))
    # Re-snap maxes to bin edges so width / nx == bin_m exactly.
    x_max = x_min + nx * bin_m
    y_max = y_min + ny * bin_m
    return SceneGrid(scene=scene, x_min=x_min, x_max=x_max,
                     y_min=y_min, y_max=y_max, nx=nx, ny=ny, bin_cm=bin_cm)


def compute_target(npz: dict | np.lib.npyio.NpzFile, scene: str, grid: SceneGrid,
                   sigma_cm: float = 2.0,
                   force_weighted: bool = False,
                   margin: float = -0.01) -> tuple[np.ndarray, int, float]:
    """Default margin -0.01 m: MuJoCo contact frames for resting-object
    contacts sit ~5 mm below the nominal table top z; keep them."""
    """Compute the smoothed 2D contact-density target for one trial.

    Returns (heatmap (ny, nx) float32, n_contacts_above_table, total_weight).
    """
    pos = np.asarray(npz["contact_positions"])
    if len(pos) == 0:
        return np.zeros(grid.shape, dtype=np.float32), 0, 0.0
    fids = np.asarray(npz["contact_failure_id"])
    fprobs = np.asarray(npz["failure_probs"], dtype=np.float32)

    above = above_table_mask(pos, scene, margin)
    if not above.any():
        return np.zeros(grid.shape, dtype=np.float32), 0, 0.0
    p = pos[above]
    w = fprobs[fids[above]].astype(np.float32)
    if force_weighted:
        forces = np.asarray(npz["contact_forces"])[above, :3]
        w = w * np.linalg.norm(forces, axis=1).astype(np.float32)

    # histogram2d returns shape (nx, ny) — transpose to (ny, nx) for image-like axes.
    H, _, _ = np.histogram2d(p[:, 0], p[:, 1],
                             bins=(grid.nx, grid.ny),
                             range=[grid.xrange, grid.yrange],
                             weights=w)
    H = H.T.astype(np.float32)
    sigma_bins = sigma_cm / grid.bin_cm
    if sigma_bins > 0:
        H = gaussian_filter(H, sigma=sigma_bins, mode="constant", cval=0.0)
    return H, int(above.sum()), float(w.sum())


# ---------------------------------------------------------------------------
# Entity footprints + integrator
# ---------------------------------------------------------------------------


def _is_robot_body(name: str | None) -> bool:
    return name in _ROBOT_BODY_NAMES if name else False


def scene_entities(model: mujoco.MjModel) -> list[str]:
    """Ordered list of non-robot named bodies that have at least one geom.

    Skips the world body, robot links, and any unnamed body. Order follows
    body id for stability across runs.
    """
    out = []
    seen = set()
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name or name in seen:
            continue
        if _is_robot_body(name):
            continue
        # require at least one geom on this body
        has_geom = any(model.geom_bodyid[gid] == bid for gid in range(model.ngeom))
        if not has_geom:
            continue
        out.append(name)
        seen.add(name)
    return out


def entity_footprints(model: mujoco.MjModel, grid: SceneGrid) -> dict[str, np.ndarray]:
    """Binary mask per entity over the scene grid covering its X-Y AABB at rest.

    AABB is derived from each body's geoms at the model's reference state
    (mj_forward on a fresh MjData). For each entity body we union the AABBs
    of all its geoms and rasterise the resulting rectangle onto the grid.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bin_m = grid.bin_cm / 100.0
    masks: dict[str, np.ndarray] = {}
    for name in scene_entities(model):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        x_lo, x_hi = np.inf, -np.inf
        y_lo, y_hi = np.inf, -np.inf
        any_geom = False
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != bid:
                continue
            any_geom = True
            cx, cy = float(data.geom_xpos[gid][0]), float(data.geom_xpos[gid][1])
            # geom_size semantics differ per type; use the largest two extents
            # as a conservative AABB in xy. Robust enough for footprint masks.
            sx, sy = float(model.geom_size[gid][0]), float(model.geom_size[gid][1])
            x_lo = min(x_lo, cx - sx); x_hi = max(x_hi, cx + sx)
            y_lo = min(y_lo, cy - sy); y_hi = max(y_hi, cy + sy)
        if not any_geom:
            continue
        # Convert world-frame AABB to grid index ranges, clip to grid.
        ix_lo = max(0, int(np.floor((x_lo - grid.x_min) / bin_m)))
        ix_hi = min(grid.nx, int(np.ceil((x_hi - grid.x_min) / bin_m)))
        iy_lo = max(0, int(np.floor((y_lo - grid.y_min) / bin_m)))
        iy_hi = min(grid.ny, int(np.ceil((y_hi - grid.y_min) / bin_m)))
        if ix_hi <= ix_lo or iy_hi <= iy_lo:
            continue
        m = np.zeros(grid.shape, dtype=bool)
        m[iy_lo:iy_hi, ix_lo:ix_hi] = True
        masks[name] = m
    return masks


def integrate_per_entity(heatmap: np.ndarray,
                         footprints: dict[str, np.ndarray]) -> dict[str, float]:
    """Sum heatmap inside each entity's footprint mask. Not normalised."""
    return {name: float(heatmap[mask].sum()) for name, mask in footprints.items()}


# ---------------------------------------------------------------------------
# Grid metadata (sidecar json per scene dir)
# ---------------------------------------------------------------------------


def save_grid(grid: SceneGrid, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(grid.to_dict(), indent=2) + "\n")


def load_grid(path: Path) -> SceneGrid:
    return SceneGrid.from_dict(json.loads(path.read_text()))
