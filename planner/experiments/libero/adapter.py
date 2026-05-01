"""HDF5 → in-memory LIBERO demo, with cached self-contained MJCF.

LIBERO HDF5 layout (from external/LIBERO/scripts/create_dataset.py):

    data.attrs["bddl_file_name"], data.attrs["bddl_file_content"]
    data/demo_N/
        attrs["model_file"]   ← MuJoCo XML string (robosuite-rendered)
        attrs["init_state"]   ← flattened MuJoCo state for t=0
        states                (T, nq+nv) flattened sim states
        actions               (T, 7)     OSC actions (unused for replay)
        obs/joint_states      (T, 7)     arm qpos
        obs/gripper_states    (T, 2)     finger qpos
        obs/ee_states         (T, 6)
        obs/agentview_rgb, obs/eye_in_hand_rgb, ...

We extract a pure-MuJoCo replay specification (model_xml + qpos timeline) so the
runner can step the sim with our own physics and inject failures. Robosuite is
*not* needed at trial time; it is only needed (via the sidecar venv) for the
one-time download of the dataset.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np


@dataclass
class LiberoDemo:
    """One LIBERO demo materialised for FailBench replay."""
    hdf5_path: str
    demo_key: str                  # e.g. "demo_0"
    task_id: str                   # derived from hdf5 filename + demo_key
    traj_id: int                   # demo index
    model_xml: str                 # self-contained MJCF (asset paths resolved)
    init_state: Optional[np.ndarray]   # flattened (qpos+qvel) for t=0; may be None
    arm_qpos: np.ndarray           # (T, 7)
    finger_qpos: np.ndarray        # (T, 2)
    full_states: Optional[np.ndarray]  # (T, nq+nv) — optional, for sanity replay
    bddl_file_name: Optional[str]


# --------------------------------------------------------------------------
# MJCF cache
# --------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_MJCF_CACHE = os.path.join(_REPO_ROOT, "datasets", "libero", "mjcf_cache")
_LIBERO_ASSETS_ROOT = os.path.join(_REPO_ROOT, "external", "LIBERO", "libero", "libero")


def _find_robosuite_root() -> Optional[str]:
    """Locate the robosuite package inside the LIBERO sidecar venv.

    Returns ``None`` if the venv hasn't been populated yet — callers can fall
    back to leaving robosuite-prefixed paths untouched and let MuJoCo fail
    loudly with a missing-file error.
    """
    cand = glob.glob(os.path.join(
        _REPO_ROOT, "external", "LIBERO", ".venv",
        "lib", "python*", "site-packages", "robosuite"))
    return cand[0] if cand else None


# Path prefixes the LIBERO recorder bakes into MJCF strings (originating on
# the dataset author's machine). Each maps the *first occurrence* of the key
# token to a local install directory; everything before the token is dropped.
def _build_path_remap() -> dict:
    remap = {"libero": _LIBERO_ASSETS_ROOT,
             "chiliocosm": _LIBERO_ASSETS_ROOT}
    rs = _find_robosuite_root()
    if rs is not None:
        remap["robosuite-master/robosuite"] = rs
        remap["robosuite"] = rs
    return remap


def _rewrite_asset_path(old: str, remap: dict) -> Optional[str]:
    """Map an absolute asset path to a local one. Returns None if no rule fits."""
    parts = old.split("/")
    # Multi-segment keys (e.g. "robosuite-master/robosuite") need a sliding match.
    for key, new_root in remap.items():
        key_parts = key.split("/")
        n = len(key_parts)
        for i in range(len(parts) - n + 1):
            if parts[i:i + n] == key_parts:
                return os.path.normpath(os.path.join(new_root, *parts[i + n:]))
    return None


def _rewrite_xml(xml_str: str) -> str:
    """Rewrite mesh/texture file= attributes to point at local installs."""
    remap = _build_path_remap()
    root = ET.fromstring(xml_str)
    asset = root.find("asset")
    if asset is None:
        return xml_str
    for elem in list(asset.findall("mesh")) + list(asset.findall("texture")):
        old = elem.get("file")
        if not old:
            continue
        new = _rewrite_asset_path(old, remap)
        if new is not None:
            elem.set("file", new)
    return ET.tostring(root, encoding="utf-8").decode("utf-8")


def _cache_path(model_xml: str, cache_dir: str) -> str:
    sha = hashlib.sha1(model_xml.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{sha}.xml")


def materialise_mjcf(model_xml: str, cache_dir: str = _DEFAULT_MJCF_CACHE) -> str:
    """Write the per-demo MJCF to disk (one-time per unique XML) and return path.

    Robosuite emits absolute mesh paths into the XML string, so as long as the
    referenced asset files still exist on this machine the cached XML is fully
    self-contained for ``mujoco.MjModel.from_xml_path``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    out = _cache_path(model_xml, cache_dir)
    # Atomic write: a concurrent reader must never see a partial file. Write
    # to a unique temp path, then os.replace into place. If another worker
    # wins the race, both copies have identical content (sha-keyed), so
    # whichever lands "last" is fine.
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        rewritten = _rewrite_xml(model_xml)
        tmp = f"{out}.tmp.{os.getpid()}.{id(model_xml)}"
        with open(tmp, "w") as f:
            f.write(rewritten)
        os.replace(tmp, out)
    return out


# --------------------------------------------------------------------------
# HDF5 loader
# --------------------------------------------------------------------------


def list_demos(hdf5_path: str):
    """Return the list of demo keys in an HDF5 file (e.g. ['demo_0', ...])."""
    with h5py.File(hdf5_path, "r") as f:
        return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))


def load_demo(hdf5_path: str, demo_key: str = "demo_0") -> LiberoDemo:
    """Read one demo from an HDF5 file into a :class:`LiberoDemo`."""
    with h5py.File(hdf5_path, "r") as f:
        grp = f[f"data/{demo_key}"]
        model_xml = grp.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")

        init_state = None
        if "init_state" in grp.attrs:
            init_state = np.asarray(grp.attrs["init_state"]).astype(np.float64)

        arm_qpos = np.asarray(grp["obs/joint_states"]).astype(np.float64)
        finger_qpos = (
            np.asarray(grp["obs/gripper_states"]).astype(np.float64)
            if "obs/gripper_states" in grp else
            np.zeros((arm_qpos.shape[0], 2), dtype=np.float64)
        )

        full_states = None
        if "states" in grp:
            full_states = np.asarray(grp["states"]).astype(np.float64)

        bddl_name = None
        if "bddl_file_name" in f["data"].attrs:
            bn = f["data"].attrs["bddl_file_name"]
            bddl_name = bn.decode("utf-8") if isinstance(bn, bytes) else str(bn)

    task_stem = os.path.splitext(os.path.basename(hdf5_path))[0]
    task_id = f"libero/{task_stem}"
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
        full_states=full_states,
        bddl_file_name=bddl_name,
    )
