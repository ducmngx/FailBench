"""Sample 6-DoF grasps from precomputed GraspGen YAMLs.

Loads SHA-keyed grasp YAMLs produced by ``scripts/precompute_grasps.py`` and
transforms them into world-frame TCP poses ready to feed to ``plan_to_ee_pose``.

Frame convention (verified, see memory/feedback_graspgen_frame.md):
    YAML ``position`` is the gripper BASE, not the fingertip TCP.
    TCP = position + GRASPGEN_PANDA_DEPTH * approach_axis
    where approach_axis = Z column of the grasp rotation.
"""

import hashlib
import json
import os
from typing import Optional, Tuple

import numpy as np
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRASPGEN_PANDA_DEPTH = 0.105  # from external/GraspGen/config/grippers/franka_panda.yaml


def _quat_wxyz_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ])


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class GraspSampler:
    """Loads cached GraspGen YAMLs and samples world-frame TCP poses."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(_REPO_ROOT, "cache", "graspgen")
        index_path = os.path.join(self.cache_dir, "index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No GraspGen cache index at {index_path}. "
                f"Run  python scripts/precompute_grasps.py  first."
            )
        with open(index_path) as f:
            self.index: dict = json.load(f)
        self._yaml_cache: dict[str, list[dict]] = {}

    def _load_grasps(self, sha: str) -> list[tuple[str, dict]]:
        """Return grasps as (grasp_id, grasp_data) tuples sorted by -confidence."""
        if sha in self._yaml_cache:
            return self._yaml_cache[sha]
        yml_path = os.path.join(self.cache_dir, f"{sha}.yml")
        with open(yml_path) as f:
            data = yaml.safe_load(f)
        grasps = sorted(data["grasps"].items(),
                        key=lambda kv: -kv[1]["confidence"])
        self._yaml_cache[sha] = grasps
        return grasps

    def _resolve_sha(self, mesh_path: str) -> str:
        """Find the cached SHA for a mesh, falling back to hashing if unseen."""
        try:
            rel = os.path.relpath(os.path.abspath(mesh_path), _REPO_ROOT)
        except ValueError:
            rel = mesh_path
        if rel in self.index:
            return self.index[rel]
        # Fall back to hashing (e.g., mesh moved since precompute)
        sha = _sha256_file(mesh_path)
        if not os.path.exists(os.path.join(self.cache_dir, f"{sha}.yml")):
            raise KeyError(
                f"No cached grasps for {mesh_path} (sha {sha[:10]}). "
                f"Run  python scripts/precompute_grasps.py --mesh {mesh_path}"
            )
        return sha

    def sample_ranked(
        self,
        mesh_path: str,
        T_obj_world: np.ndarray,
        seed: int = 0,
        n: int = 8,
        top_k: int = 40,
        max_approach_z: float = -0.85,
    ) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]]:
        """Return up to ``n`` world-frame grasp candidates in shuffled order.

        Each candidate is ``(tcp_pos_world, quat_wxyz_world, approach_axis_world,
        grasp_id, confidence)``. The shuffle is deterministic in ``seed`` so
        retries can be reproduced. Callers typically iterate this list and
        pick the first one that passes an IK-feasibility check.
        """
        sha = self._resolve_sha(mesh_path)
        grasps = self._load_grasps(sha)[:top_k]
        if not grasps:
            raise RuntimeError(f"Empty grasp set for {mesh_path}")

        R_obj_world = T_obj_world[:3, :3]

        eligible: list[tuple[str, dict]] = []
        for gid, g in grasps:
            q = g["orientation"]
            R_obj = _quat_wxyz_to_matrix(q["w"], *q["xyz"])
            approach_world = R_obj_world @ R_obj[:, 2]
            if approach_world[2] <= max_approach_z:
                eligible.append((gid, g))

        if not eligible:
            # Fallback: most-downward top-5 regardless of threshold
            scored = sorted(
                grasps,
                key=lambda kv: (R_obj_world @ _quat_wxyz_to_matrix(
                    kv[1]["orientation"]["w"], *kv[1]["orientation"]["xyz"])[:, 2])[2],
            )
            eligible = scored[:5]

        rng = np.random.RandomState(seed)
        order = rng.permutation(len(eligible))
        chosen = [eligible[i] for i in order[:n]]

        out: list[Tuple[np.ndarray, np.ndarray, np.ndarray, str, float]] = []
        for gid, g in chosen:
            q = g["orientation"]
            R_obj = _quat_wxyz_to_matrix(q["w"], *q["xyz"])
            t_obj = np.asarray(g["position"], dtype=float)

            T_grasp_obj = np.eye(4)
            T_grasp_obj[:3, :3] = R_obj
            T_grasp_obj[:3, 3] = t_obj

            T_world = T_obj_world @ T_grasp_obj
            approach_axis = T_world[:3, 2].copy()
            approach_axis /= max(np.linalg.norm(approach_axis), 1e-9)

            tcp_pos = T_world[:3, 3] + GRASPGEN_PANDA_DEPTH * approach_axis
            quat_wxyz = _matrix_to_quat_wxyz(T_world[:3, :3])
            out.append((tcp_pos, quat_wxyz, approach_axis, gid, float(g["confidence"])))
        return out

    def sample(
        self,
        mesh_path: str,
        T_obj_world: np.ndarray,
        seed: int = 0,
        top_k: int = 40,
        max_approach_z: float = -0.85,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Sample one grasp; return (tcp_pos, quat_wxyz, approach_axis, grasp_id)."""
        cands = self.sample_ranked(mesh_path, T_obj_world, seed=seed, n=1,
                                    top_k=top_k, max_approach_z=max_approach_z)
        tcp, quat, axis, gid, _conf = cands[0]
        return tcp, quat, axis, gid
