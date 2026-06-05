"""RoboCasa-specific scene-helpers that produce a ``SceneOverrides`` for the runner.

RoboCasa kitchens have hundreds of named bodies (walls, floor, every cabinet
panel, every fixture) — LIBERO's "non-robot, non-table" auto-detection picks
them all up. We need the small, semantic allowlist of *manipulated* objects.
That allowlist lives in ``ep_meta["object_cfgs"]`` (per-demo) and the matching
MuJoCo body names follow RoboCasa's ``"<cfg_name>_main"`` naming convention.

This module also derives:

- a counter-aware ``scene_table_z`` by reading the manipulated object's
  z-coordinate at the seed state and walking down to the supporting surface;
- a tight workspace ``scene_aabb`` around the allowed objects plus a margin;
- a per-entity AABB list for downstream per-entity risk labels.
"""

from __future__ import annotations

import json
from typing import Iterable, List, Optional

import mujoco
import numpy as np

from planner.experiments.libero.runner import SceneOverrides


def _resolve_body_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _resolve_object_bodies(model: mujoco.MjModel,
                           allowlist_names: Iterable[str]) -> tuple[List[int], List[str]]:
    """Map ep_meta object names to MuJoCo body IDs.

    RoboCasa convention: an ``object_cfgs[i] = {"name": "cookware", ...}`` entry
    materialises into a body named ``"cookware_main"``. We try both the bare
    name and ``"<name>_main"`` (favouring the suffixed form because that's the
    actual parent body).
    """
    ids: List[int] = []
    resolved: List[str] = []
    for raw in allowlist_names:
        for candidate in (f"{raw}_main", raw):
            bid = _resolve_body_id(model, candidate)
            if bid >= 0:
                ids.append(bid)
                resolved.append(candidate)
                break
    return ids, resolved


def _descendant_bids(model: mujoco.MjModel, root_bid: int) -> set[int]:
    """All body IDs in the subtree rooted at ``root_bid`` (inclusive).

    RoboCasa composes manipulated objects as body hierarchies — the named
    parent (``<name>_main``) is just a transform node with no direct geoms;
    visual / collision geoms live on grandchild bodies (``*_main_group/*_g{N}``).
    The AABB walker must follow the full subtree, not just direct children.
    """
    children: dict[int, list[int]] = {}
    for b in range(model.nbody):
        children.setdefault(int(model.body_parentid[b]), []).append(b)
    out: set[int] = set()
    stack = [root_bid]
    while stack:
        bid = stack.pop()
        if bid in out:
            continue
        out.add(bid)
        stack.extend(children.get(bid, []))
    return out


def _body_aabb(model: mujoco.MjModel, data: mujoco.MjData, bid: int) -> tuple[np.ndarray, np.ndarray]:
    """World-frame AABB of all geoms anywhere in the subtree rooted at ``bid``.

    Returns ``(inf, -inf)`` when the subtree owns no geoms at all.
    """
    bids = _descendant_bids(model, bid)
    lo = np.full(3, np.inf, dtype=np.float64)
    hi = np.full(3, -np.inf, dtype=np.float64)
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) not in bids:
            continue
        c = data.geom_xpos[gid]
        s = model.geom_size[gid]
        lo = np.minimum(lo, c - s)
        hi = np.maximum(hi, c + s)
    return lo, hi


def _support_table_z(model: mujoco.MjModel,
                     data: mujoco.MjData,
                     obj_bids: List[int]) -> float:
    """Approximate the surface the manipulated object rests on.

    Walks down from each allowed-object's z-min and takes the median of the
    candidate surface heights — works for cookware on a stovetop, mugs on a
    counter, etc. Falls back to the LIBERO default if no objects are present.
    """
    if not obj_bids:
        return 0.91
    zs: List[float] = []
    for bid in obj_bids:
        lo, _ = _body_aabb(model, data, bid)
        if np.isfinite(lo[2]):
            zs.append(float(lo[2]))
    if not zs:
        return 0.91
    return float(np.median(zs))


def _build_scene_metadata(model: mujoco.MjModel,
                          data: mujoco.MjData,
                          obj_bids: List[int],
                          obj_names: List[str]) -> dict:
    """Tight RoboCasa scene metadata: table_z from support, AABB around allowed objects."""
    if obj_bids:
        los = []
        his = []
        entities = []
        for bid, name in zip(obj_bids, obj_names):
            lo, hi = _body_aabb(model, data, bid)
            if not np.isfinite(lo).all():
                continue
            los.append(lo)
            his.append(hi)
            entities.append({
                "name": name,
                "aabb_min": lo.tolist(),
                "aabb_max": hi.tolist(),
            })
        if los:
            aabb_min = np.minimum.reduce(los) - 0.20  # 20 cm margin for failure-induced motion
            aabb_max = np.maximum.reduce(his) + 0.20
        else:
            aabb_min = np.array([-1.0, -1.0, 0.0])
            aabb_max = np.array([1.0, 1.0, 1.5])
    else:
        aabb_min = np.array([-1.0, -1.0, 0.0])
        aabb_max = np.array([1.0, 1.0, 1.5])
        entities = []

    return {
        "scene_table_z": _support_table_z(model, data, obj_bids),
        "scene_aabb_min": aabb_min.astype(np.float64),
        "scene_aabb_max": aabb_max.astype(np.float64),
        "scene_entities_json": json.dumps(entities),
    }


def build_scene_overrides(model: mujoco.MjModel,
                          data: mujoco.MjData,
                          ep_meta: dict) -> SceneOverrides:
    """Top-level: produce a SceneOverrides from a loaded RoboCasa MJCF + ep_meta."""
    from planner.experiments.robocasa.adapter import object_allowlist_from_ep_meta

    names = object_allowlist_from_ep_meta(ep_meta)
    obj_bids, resolved_names = _resolve_object_bodies(model, names)
    scene_meta = _build_scene_metadata(model, data, obj_bids, resolved_names)
    return SceneOverrides(
        object_body_ids=obj_bids,
        object_names=resolved_names,
        scene_metadata=scene_meta,
    )
