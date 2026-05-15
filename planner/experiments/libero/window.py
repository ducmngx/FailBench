"""Pre-failure window + goal + calibration helpers for the v2 LIBERO dataset.

A v2 trial augments a v1-style single-frame trial with:

* a T-frame **window** of pre-failure observations ending at ``fail_idx``
  (RGB+depth per camera + per-frame state with real, finite-diff'd qvel),
* a K-step **goal** feature taken from the un-failed demo at future offsets
  after ``fail_idx`` (full state: qpos, qvel, ee_pos, gripper_ctrl),
* per-trial **camera calibration** and **scene geometry** so any downstream
  label form (image-plane / world top-down / voxel / per-entity) is
  constructible without re-simulating.

This module is pure-Python utilities; it does not touch HDF5 output or
multiprocess plumbing.
"""
from __future__ import annotations

import json
from typing import Iterable, Sequence

import numpy as np
import mujoco

from planner.experiments.libero.adapter import LiberoDemo
from planner.risk.spatial import (
    SCENE_TABLE_Z,
    derive_scene_grid,
    scene_entities,
    entity_footprints,
)


LIBERO_CONTROL_HZ = 20.0  # LIBERO/robosuite default control rate; obs sampled at this freq.


# --------------------------------------------------------------------------
# Window index sampling
# --------------------------------------------------------------------------


def sample_window_indices(fail_idx: int, demo_len: int,
                          T: int = 8, stride: int = 5) -> np.ndarray:
    """T evenly-spaced demo timesteps ending at ``fail_idx``.

    Result is monotonic-nondecreasing, in [0, demo_len). When ``fail_idx`` is
    too small to fit the requested span, earlier frames are clipped to 0 (the
    demo's first frame) — the prefix becomes a "still" hold at the start pose
    rather than an out-of-range index.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if demo_len <= 0:
        raise ValueError(f"demo_len must be positive, got {demo_len}")
    fail_idx = int(np.clip(fail_idx, 0, demo_len - 1))
    raw = fail_idx - stride * np.arange(T - 1, -1, -1)
    return np.clip(raw, 0, demo_len - 1).astype(np.int32)


# --------------------------------------------------------------------------
# Per-frame state extraction
# --------------------------------------------------------------------------


def _finite_diff(qpos_track: np.ndarray, idx: int, demo_len: int,
                 dt: float) -> np.ndarray:
    """Central finite difference of qpos around demo index ``idx``.

    Edge handling: forward/backward difference at the demo boundaries.
    Returns a (7,) joint-velocity vector in rad/s.
    """
    if idx <= 0:
        return (qpos_track[1] - qpos_track[0]) / dt
    if idx >= demo_len - 1:
        return (qpos_track[-1] - qpos_track[-2]) / dt
    return (qpos_track[idx + 1] - qpos_track[idx - 1]) / (2.0 * dt)


def compute_window_state(demo: LiberoDemo, frame_idx: np.ndarray,
                         ee_states: np.ndarray) -> dict:
    """Per-window-frame state arrays.

    ``ee_states`` is the demo's ``obs/ee_states`` array (T, 6) read once at
    HDF5-load time and threaded in to avoid re-opening the file here.

    Returns a dict matching the v2 schema:

        window_qpos          (T, 7)  f32
        window_qvel          (T, 7)  f32   finite-diff
        window_ee_pos        (T, 3)  f32
        window_gripper_ctrl  (T, 1)  f32   mean finger qpos as a scalar
    """
    T = int(frame_idx.shape[0])
    demo_len = demo.arm_qpos.shape[0]
    dt = 1.0 / LIBERO_CONTROL_HZ
    qpos = np.empty((T, 7), dtype=np.float32)
    qvel = np.empty((T, 7), dtype=np.float32)
    ee_pos = np.empty((T, 3), dtype=np.float32)
    grip = np.empty((T, 1), dtype=np.float32)
    for k, idx in enumerate(frame_idx.tolist()):
        qpos[k] = demo.arm_qpos[idx].astype(np.float32)
        qvel[k] = _finite_diff(demo.arm_qpos, idx, demo_len, dt).astype(np.float32)
        ee_pos[k] = ee_states[idx, :3].astype(np.float32)
        grip[k, 0] = float(np.mean(demo.finger_qpos[idx]))
    return {
        "window_qpos": qpos,
        "window_qvel": qvel,
        "window_ee_pos": ee_pos,
        "window_gripper_ctrl": grip,
    }


# --------------------------------------------------------------------------
# Goal / intent feature
# --------------------------------------------------------------------------


def compute_goal(demo: LiberoDemo, fail_idx: int,
                 ee_states: np.ndarray,
                 offsets: Sequence[int] = (5, 15, 30)) -> dict:
    """Full state at ``fail_idx + offset`` for each ``offset`` in ``offsets``.

    Offsets beyond ``demo_len - 1`` are clipped to the final demo step. The
    returned arrays are oriented ``(K, ...)`` matching the schema.
    """
    demo_len = demo.arm_qpos.shape[0]
    dt = 1.0 / LIBERO_CONTROL_HZ
    offsets = list(offsets)
    K = len(offsets)
    qpos = np.empty((K, 7), dtype=np.float32)
    qvel = np.empty((K, 7), dtype=np.float32)
    ee_pos = np.empty((K, 3), dtype=np.float32)
    grip = np.empty((K, 1), dtype=np.float32)
    goal_offsets = np.empty((K,), dtype=np.int32)
    for k, off in enumerate(offsets):
        idx = int(np.clip(fail_idx + off, 0, demo_len - 1))
        qpos[k] = demo.arm_qpos[idx].astype(np.float32)
        qvel[k] = _finite_diff(demo.arm_qpos, idx, demo_len, dt).astype(np.float32)
        ee_pos[k] = ee_states[idx, :3].astype(np.float32)
        grip[k, 0] = float(np.mean(demo.finger_qpos[idx]))
        goal_offsets[k] = int(off)
    return {
        "goal_qpos": qpos,
        "goal_qvel": qvel,
        "goal_ee_pos": ee_pos,
        "goal_gripper_ctrl": grip,
        "goal_offsets": goal_offsets,
    }


# --------------------------------------------------------------------------
# Camera calibration
# --------------------------------------------------------------------------


def extract_cam_calibration(model: mujoco.MjModel, data: mujoco.MjData,
                            cam_name: str, image_size: tuple) -> dict:
    """Snapshot a camera's pose + intrinsics at the current ``data`` state.

    ``data.cam_xpos`` / ``data.cam_xmat`` reflect the camera's world pose
    *after* any kinematic chain the camera is attached to (so wrist cam moves
    with the arm). Caller should ``mj_forward`` before calling for the current
    sim state to be reflected.

    Returns a dict with the keys expected by the v2 schema for that camera:
    pos (3,), mat0 (3, 3), fovy scalar, size (W, H).
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera {cam_name!r} not found in model")
    pos = data.cam_xpos[cam_id].astype(np.float64).copy()
    mat = data.cam_xmat[cam_id].reshape(3, 3).astype(np.float64).copy()
    fovy = float(model.cam_fovy[cam_id])
    W, H = int(image_size[0]), int(image_size[1])
    return {
        "pos": pos,
        "mat0": mat,
        "fovy": fovy,
        "size": np.array([W, H], dtype=np.int32),
    }


def sample_window_cam_trajectory(model: mujoco.MjModel,
                                 data: mujoco.MjData,
                                 cam_name: str,
                                 set_state_fn,
                                 frame_states: np.ndarray,
                                 image_size: tuple) -> dict:
    """Walk through ``frame_states`` and record cam pose at each frame.

    Used for wrist cam which moves with the arm. ``set_state_fn(s)`` should
    restore the sim to the demo's state at one window frame (typically
    ``LiberoRunner._set_full_state``). After each restore we ``mj_forward``
    to propagate the camera transform.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera {cam_name!r} not found in model")
    T = int(frame_states.shape[0])
    pos = np.empty((T, 3), dtype=np.float64)
    mat = np.empty((T, 3, 3), dtype=np.float64)
    for k in range(T):
        set_state_fn(frame_states[k])
        mujoco.mj_forward(model, data)
        pos[k] = data.cam_xpos[cam_id]
        mat[k] = data.cam_xmat[cam_id].reshape(3, 3)
    fovy = float(model.cam_fovy[cam_id])
    W, H = int(image_size[0]), int(image_size[1])
    return {
        "pos_window": pos,
        "mat0_window": mat,
        "fovy": fovy,
        "size": np.array([W, H], dtype=np.int32),
    }


# --------------------------------------------------------------------------
# Scene metadata
# --------------------------------------------------------------------------


_BODY_TABLE_TOKENS = ("table", "main_table", "wood_table")


def _scene_aabb(model: mujoco.MjModel, data: mujoco.MjData) -> tuple:
    """Loose world AABB of the workspace.

    Spans the union of all non-world, non-robot body AABBs. Used as the
    bounds for future voxel-grid label construction.
    """
    from planner.experiments.data_capture import _ROBOT_BODY_NAMES

    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not name or name == "world":
            continue
        # Skip robot bodies — they're not workspace.
        if name in _ROBOT_BODY_NAMES:
            continue
        if name.startswith(("robot0_", "gripper0_", "mount0_")):
            continue
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != bid:
                continue
            cx, cy, cz = data.geom_xpos[gid]
            sx, sy, sz = model.geom_size[gid]
            lo = np.minimum(lo, [cx - sx, cy - sy, cz - sz])
            hi = np.maximum(hi, [cx + sx, cy + sy, cz + sz])
    if not np.isfinite(lo).all():
        lo = np.array([-1.0, -1.0, 0.0])
        hi = np.array([1.0, 1.0, 1.5])
    return lo, hi


def _detect_table_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Largest horizontal box geom in any body whose name mentions a table.

    Mirrors ``derive_scene_grid``'s rule from ``planner/risk/spatial.py``,
    but works directly on a robosuite/LIBERO MJCF (different body names).
    Falls back to the most common 0.91 m LIBERO table height if no match.
    """
    best_area = 0.0
    best_z = None
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not any(tok in bname.lower() for tok in _BODY_TABLE_TOKENS):
            continue
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        sx, sy = float(model.geom_size[gid][0]), float(model.geom_size[gid][1])
        if sx <= 0.05 or sy <= 0.05:
            continue
        area = sx * sy
        if area > best_area:
            best_area = area
            cz = float(data.geom_xpos[gid][2])
            sz = float(model.geom_size[gid][2])
            best_z = cz + sz  # top surface
    return float(best_z) if best_z is not None else 0.91


def _scene_entity_list(model: mujoco.MjModel, data: mujoco.MjData) -> list:
    """Per-entity name + world AABB at the current sim state.

    Returns a list of dicts; each entity is a non-robot, non-table named body
    that owns at least one geom. AABB is in world coordinates and is the union
    over the body's geom AABBs at this moment.
    """
    from planner.experiments.data_capture import _ROBOT_BODY_NAMES

    entities = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not name or name == "world":
            continue
        if name in _ROBOT_BODY_NAMES:
            continue
        if name.startswith(("robot0_", "gripper0_", "mount0_")):
            continue
        if any(tok in name.lower() for tok in _BODY_TABLE_TOKENS):
            continue
        lo = np.full(3, np.inf, dtype=np.float64)
        hi = np.full(3, -np.inf, dtype=np.float64)
        any_geom = False
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] != bid:
                continue
            any_geom = True
            cx, cy, cz = data.geom_xpos[gid]
            sx, sy, sz = model.geom_size[gid]
            lo = np.minimum(lo, [cx - sx, cy - sy, cz - sz])
            hi = np.maximum(hi, [cx + sx, cy + sy, cz + sz])
        if not any_geom:
            continue
        entities.append({
            "name": name,
            "aabb_min": lo.tolist(),
            "aabb_max": hi.tolist(),
        })
    return entities


def extract_scene_metadata(model: mujoco.MjModel,
                           data: mujoco.MjData) -> dict:
    """Per-trial scene metadata for label-form construction.

    Returns table z, workspace AABB, and a list of entity AABBs as a
    JSON-serialisable dict. Caller writes these to v2 trial attrs.
    """
    aabb_min, aabb_max = _scene_aabb(model, data)
    return {
        "scene_table_z": _detect_table_z(model, data),
        "scene_aabb_min": aabb_min.astype(np.float64),
        "scene_aabb_max": aabb_max.astype(np.float64),
        "scene_entities_json": json.dumps(_scene_entity_list(model, data)),
    }


# --------------------------------------------------------------------------
# Object-pose snapshots
# --------------------------------------------------------------------------


def _object_body_ids(model: mujoco.MjModel) -> list:
    """Body IDs for scene objects: every non-world, non-robot, non-table body
    that owns at least one geom. Stable across calls (id-ordered)."""
    from planner.experiments.data_capture import _ROBOT_BODY_NAMES

    ids = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not name or name == "world":
            continue
        if name in _ROBOT_BODY_NAMES:
            continue
        if name.startswith(("robot0_", "gripper0_", "mount0_")):
            continue
        if any(tok in name.lower() for tok in _BODY_TABLE_TOKENS):
            continue
        if not any(model.geom_bodyid[gid] == bid for gid in range(model.ngeom)):
            continue
        ids.append(bid)
    return ids


def snapshot_object_poses(model: mujoco.MjModel, data: mujoco.MjData,
                          body_ids: list) -> tuple:
    """Body world position + quaternion arrays for the given body ids.

    Returns ``(pos (n_obj, 3) f32, quat (n_obj, 4) f32)`` where quat is in
    MuJoCo ``(w, x, y, z)`` order.
    """
    n = len(body_ids)
    pos = np.empty((n, 3), dtype=np.float32)
    quat = np.empty((n, 4), dtype=np.float32)
    for k, bid in enumerate(body_ids):
        pos[k] = data.xpos[bid]
        quat[k] = data.xquat[bid]
    return pos, quat


def object_names(model: mujoco.MjModel, body_ids: list) -> list:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            for bid in body_ids]
