"""HDF5 → in-memory RoboCasa demo, with cached self-contained MJCF.

RoboCasa robomimic-format HDF5 layout (per ``mimicdroid/playback_dataset.py``):

    data.attrs["env_args"], data.attrs["total"]
    data/demo_N/
        attrs["model_file"]   ← MuJoCo XML string (robosuite-rendered)
        attrs["ep_meta"]      ← JSON: layout_id, style_id, object_cfgs, ...
        attrs["num_samples"]  ← T
        states                (T, 1 + nq + nv) flattened [time | qpos | qvel]
        actions               (T, 12)  OSC_POSE + mobile-base + gripper
        actions_abs           (T, 12)  absolute-frame variant
        obs/robot0_joint_pos          (T, 7)
        obs/robot0_gripper_qpos       (T, 2)
        obs/robot0_eef_pos            (T, 3)
        obs/robot0_eef_quat           (T, 4)
        obs/ee_states                 ← NOT present in RoboCasa; we synthesize
        obs/robot0_agentview_{left,right,center}_image  (T, 128, 128, 3)
        obs/robot0_eye_in_hand_image                    (T, 128, 128, 3)
        rewards, dones                ← scalars per step

This adapter produces a ``LiberoDemo`` from the LIBERO module, so
``LiberoRunner`` consumes RoboCasa demos with no per-source dispatch in the
runner itself.

Two RoboCasa-specific concerns this adapter handles:

1. **Per-author absolute asset paths.** MJCFs from the binhng/* HF mirror were
   generated on three different workstations (soroush / aaronl / abhim), each
   baking in absolute paths to ``robosuite/`` and ``robocasa/``. A regex
   substitution remaps them all.

2. **Manipulated-object allowlist.** RoboCasa kitchens have hundreds of bodies
   (cabinet panels, fixtures, doors). LIBERO's ``object_body_ids`` heuristic
   would surface all of them. ``ep_meta["object_cfgs"]`` lists only the
   movable / manipulated objects per demo — use that.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from typing import List, Optional

import h5py
import numpy as np

from planner.experiments.libero.adapter import LiberoDemo


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_MJCF_CACHE = os.path.join(_REPO_ROOT, "datasets", "robocasa", "mjcf_cache")
_ROBOSUITE_ROOT = os.path.join(_REPO_ROOT, "external", "robosuite_for_robocasa", "robosuite")
_ROBOCASA_ROOT = os.path.join(_REPO_ROOT, "external", "robocasa", "robocasa")


# --------------------------------------------------------------------------
# MJCF path remap
# --------------------------------------------------------------------------

# binhng/robocasa MJCFs bake in absolute author-machine paths. Observed so far:
#   /home/soroush/code/robosuite-dev/robosuite/...
#   /home/soroush/code/robocasa-dev/robocasa/...
#   /data1/aaronl/rpl-robocasa/robosuite-dev/...
#   /data1/aaronl/rpl-robocasa/robocasa-dev/...
#   /home/abhim/robocasa/robosuite/robosuite/...
#   /home/abhim/robocasa/robocasa/...
# Regex matches any leading absolute path ending in /<pkg>/models/ — minimal
# greedy form so the trailing /models/ disambiguates which package it is.
_ROBOSUITE_PAT = re.compile(r"/[^\"<>\s]+?/robosuite/models/")
_ROBOCASA_PAT = re.compile(r"/[^\"<>\s]+?/robocasa/models/")


def _rewrite_xml(xml_str: str,
                 robosuite_root: str = _ROBOSUITE_ROOT,
                 robocasa_root: str = _ROBOCASA_ROOT) -> str:
    """Map per-author absolute paths to our local install + neutralise meshdir.

    Also injects ``inertia="shell"`` on every ``<mesh>`` that doesn't already
    specify one. RoboCasa ships some visual meshes with sub-mm volume (e.g.
    ``utensil_holder_main_group_model_1_vis``) that fail MuJoCo's strict
    volume-positivity check during compilation. Shell-inertia computes mass
    from the surface area instead of volume, so it works on degenerate meshes
    without skipping the demo. Geometry of failure-injection simulations is
    dominated by collision meshes (which we don't touch), so the inertia
    change is harmless for our use case.
    """
    xml = _ROBOSUITE_PAT.sub(f"{robosuite_root}/models/", xml_str)
    xml = _ROBOCASA_PAT.sub(f"{robocasa_root}/models/", xml)
    xml = xml.replace('meshdir="meshes/"',
                      f'meshdir="{robocasa_root}/models/assets/"')
    # Two-part inertia fix for mujoco>=3.3 strictness:
    #   (a) Strip shellinertia="true" attributes from <geom> elements.
    #       Without (b), 3.3 complains that the mesh asset has no inertia
    #       specification when a geom requests shell inertia.
    #   (b) Inject inertia="shell" on <mesh> assets that don't already carry it.
    #       Without (a), the geom-level shellinertia is interpreted strictly
    #       and visual-only mesh geoms (contype=0) get rejected.
    # Both are needed; tested against TurnOffStove demo_231 (microwave + utensil
    # holder fixtures with tiny visual meshes).
    xml = re.sub(r'\s*shellinertia="(?:true|True)"', "", xml)

    def _inject_inertia(match: "re.Match[str]") -> str:
        tag = match.group(0)
        if 'inertia="' in tag:
            return tag
        return tag.replace("<mesh ", '<mesh inertia="shell" ', 1)
    xml = re.sub(r"<mesh\s[^>]*?>", _inject_inertia, xml)
    return xml


def _cache_path(model_xml: str, cache_dir: str) -> str:
    sha = hashlib.sha1(model_xml.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{sha}.xml")


def materialise_mjcf(model_xml: str, cache_dir: str = _DEFAULT_MJCF_CACHE) -> str:
    """Write the per-demo MJCF to disk (one-time per unique XML) and return path.

    Atomic write — multiple workers materialising the same demo race safely.
    """
    os.makedirs(cache_dir, exist_ok=True)
    out = _cache_path(model_xml, cache_dir)
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        rewritten = _rewrite_xml(model_xml)
        tmp = f"{out}.tmp.{os.getpid()}.{id(model_xml)}"
        with open(tmp, "w") as f:
            f.write(rewritten)
        os.replace(tmp, out)
    return out


# --------------------------------------------------------------------------
# Object allowlist from ep_meta
# --------------------------------------------------------------------------


def object_allowlist_from_ep_meta(ep_meta: dict) -> List[str]:
    """Extract movable-object names from RoboCasa's ep_meta JSON.

    Returns the list of ``object_cfgs[i]["name"]`` entries. Each name matches
    the body name in the MJCF prefixed by ``obj_`` per RoboCasa convention,
    but body resolution is deferred to the scene-helpers layer (which has the
    model in hand) — this function just returns the raw names.
    """
    cfgs = ep_meta.get("object_cfgs", [])
    return [c["name"] for c in cfgs if "name" in c]


def _parse_ep_meta(raw) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return dict(raw)


# --------------------------------------------------------------------------
# HDF5 loader
# --------------------------------------------------------------------------


def list_demos(hdf5_path: str) -> List[str]:
    """Return demo keys sorted by their integer suffix (handles non-sequential)."""
    with h5py.File(hdf5_path, "r") as f:
        keys = list(f["data"].keys())
    return sorted(keys, key=lambda k: int(k.split("_")[-1]))


def load_demo(hdf5_path: str, demo_key: str = "demo_0") -> LiberoDemo:
    """Read one RoboCasa demo into a :class:`LiberoDemo`.

    The dataclass is shared with LIBERO so :class:`LiberoRunner` consumes both
    sources unchanged. ``init_state`` is derived from ``states[0]`` since
    RoboCasa demos store the start state inside the per-step array rather than
    in a separate attr.
    """
    with h5py.File(hdf5_path, "r") as f:
        if demo_key not in f["data"]:
            raise KeyError(f"demo {demo_key!r} not in {hdf5_path}; "
                           f"available: {list(f['data'].keys())[:5]}...")
        grp = f[f"data/{demo_key}"]

        model_xml = grp.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")

        states = np.asarray(grp["states"]).astype(np.float64)
        init_state = states[0].copy() if len(states) > 0 else None

        # RoboCasa stores the 7-DoF arm qpos directly under obs/robot0_joint_pos.
        arm_qpos = np.asarray(grp["obs/robot0_joint_pos"]).astype(np.float64)
        finger_qpos = (
            np.asarray(grp["obs/robot0_gripper_qpos"]).astype(np.float64)
            if "obs/robot0_gripper_qpos" in grp
            else np.zeros((arm_qpos.shape[0], 2), dtype=np.float64)
        )

    task_stem = os.path.splitext(os.path.basename(hdf5_path))[0]
    task_id = f"robocasa/{task_stem}"
    traj_id = int(demo_key.split("_")[-1])

    return LiberoDemo(
        hdf5_path=hdf5_path,
        demo_key=demo_key,
        task_id=task_id,
        traj_id=traj_id,
        model_xml=model_xml,
        init_state=init_state,
        arm_qpos=arm_qpos,
        finger_qpos=finger_qpos,
        full_states=states,
        bddl_file_name=None,
    )


# --------------------------------------------------------------------------
# Convenience accessors for run_v2
# --------------------------------------------------------------------------


def read_ep_meta(hdf5_path: str, demo_key: str) -> dict:
    """Convenience: read the JSON-encoded ep_meta dict for one demo."""
    with h5py.File(hdf5_path, "r") as f:
        return _parse_ep_meta(f[f"data/{demo_key}"].attrs.get("ep_meta", None))


def read_ee_states_synthetic(hdf5_path: str, demo_key: str) -> np.ndarray:
    """LiberoRunner.run_v2 reads obs/ee_states (T, 6). RoboCasa stores eef_pos
    (T, 3) and eef_quat (T, 4) separately; synthesize a 6-D (pos + quat[:3])
    surrogate so the runner doesn't have to branch.

    NOTE: this preserves shape only — semantics differ slightly. If the goal
    extractor needs the quaternion's w component it should read robot0_eef_quat
    directly. Used here purely to satisfy run_v2's existing shape contract.
    """
    with h5py.File(hdf5_path, "r") as f:
        grp = f[f"data/{demo_key}"]
        pos = np.asarray(grp["obs/robot0_eef_pos"]).astype(np.float64)  # (T,3)
        quat = np.asarray(grp["obs/robot0_eef_quat"]).astype(np.float64)  # (T,4)
    # Use (x, y, z, qx, qy, qz) — drop qw. Matches the 6-D shape LIBERO ee_states uses.
    return np.concatenate([pos, quat[:, 1:4]], axis=1)  # (T, 6)
