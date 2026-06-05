"""Smoke-test: replay one binhng/robocasa demo step-by-step in MuJoCo, render
frames, and report contact stats.

Validates Phase 0 → Phase 1 transition:
    - Per-demo MJCF + states-based replay works end-to-end.
    - Renderer produces sensible kitchen frames at multiple progress points.
    - ContactExtractor-style filtering finds non-robot contacts mid-trajectory.

Run under the robocasa conda env (mimicdroid-robocasa + ShahRutav/robosuite):
    /home/aaron/miniconda3/envs/robocasa/bin/python scripts/robocasa/smoke_replay_demo.py \
        --hdf5 datasets/robocasa/raw/TurnOffStove.hdf5

The script does NOT inject failures — it's a pure replay smoke. Failure
injection lands in Phase 1 inside the adapter/runner extension.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOSUITE_ROOT = REPO_ROOT / "external" / "robosuite_for_robocasa" / "robosuite"
ROBOCASA_ROOT = REPO_ROOT / "external" / "robocasa" / "robocasa"

# Absolute paths the binhng/robocasa MJCFs bake in (Soroush Nasiriany's
# workstation). The adapter's path remap will need to handle these.
EMBEDDED_ROBOSUITE = "/home/soroush/code/robosuite-dev/robosuite"
EMBEDDED_ROBOCASA = "/home/soroush/code/robocasa-dev/robocasa"


def remap_mjcf(xml: str) -> str:
    xml = xml.replace(EMBEDDED_ROBOSUITE, str(ROBOSUITE_ROOT))
    xml = xml.replace(EMBEDDED_ROBOCASA, str(ROBOCASA_ROOT))
    # Replace relative meshdir with absolute robocasa assets root so any other
    # relative file= attributes resolve.
    xml = xml.replace(
        'meshdir="meshes/"', f'meshdir="{ROBOCASA_ROOT}/models/assets/"'
    )
    return xml


def seed_state(model: mujoco.MjModel, data: mujoco.MjData, flat: np.ndarray) -> None:
    expected = 1 + model.nq + model.nv
    if flat.shape[0] == expected:
        offset = 1
    elif flat.shape[0] == model.nq + model.nv:
        offset = 0
    else:
        raise ValueError(
            f"state length {flat.shape[0]} != nq+nv ({model.nq + model.nv}) "
            f"or 1+nq+nv ({expected})"
        )
    data.qpos[:] = flat[offset : offset + model.nq]
    data.qvel[:] = flat[offset + model.nq : offset + model.nq + model.nv]
    mujoco.mj_forward(model, data)


def robot_geom_ids(model: mujoco.MjModel) -> set[int]:
    """All geom ids belonging to robot/base/gripper bodies."""
    prefixes = ("robot0_", "gripper0_", "base0_", "mount0_")
    robot_bodies: set[int] = set()
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.startswith(prefixes):
            robot_bodies.add(bid)
    return {
        gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) in robot_bodies
    }


def extract_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, robot_geoms: set[int], min_force: float = 1.0
) -> list[tuple[int, int, np.ndarray, float]]:
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in robot_geoms and g2 in robot_geoms:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        if np.linalg.norm(force[:3]) < min_force:
            continue
        out.append((g1, g2, c.pos.copy(), float(np.linalg.norm(force[:3]))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=REPO_ROOT / "datasets" / "robocasa" / "raw" / "TurnOffStove.hdf5",
    )
    parser.add_argument("--demo", type=str, default=None,
                        help="Demo key (default: first demo in file)")
    parser.add_argument("--camera", type=str, default="robot0_agentview_center")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "out" / "robocasa_smoke_replay.png",
    )
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        demos = list(f["data"].keys())
        demo_key = args.demo or demos[0]
        g = f[f"data/{demo_key}"]
        xml = g.attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode("utf-8")
        states = g["states"][...]
        actions = g["actions"][...]
        print(f"loaded {args.hdf5.name}/{demo_key}: T={states.shape[0]} "
              f"state_dim={states.shape[1]} action_dim={actions.shape[1]}")

    xml = remap_mjcf(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    print(f"model: nq={model.nq} nv={model.nv} ngeom={model.ngeom} ncam={model.ncam}")

    robot_geoms = robot_geom_ids(model)
    print(f"robot/base/gripper geoms: {len(robot_geoms)} of {model.ngeom}")

    renderer = mujoco.Renderer(model, args.height, args.width)
    # Suppress collision geomgroups: robosuite collision geoms (group 0) overlay
    # textured visual geoms (group 1). Hide group 0 if both present.
    scene_option = mujoco.MjvOption()
    groups = set(int(g) for g in model.geom_group)
    if 0 in groups and 1 in groups:
        scene_option.geomgroup[0] = 0
        scene_option.geomgroup[1] = 1

    T = states.shape[0]
    progress = [0.0, 0.25, 0.5, 0.75, 1.0]
    frames: list[tuple[float, np.ndarray, int, float]] = []  # (p, rgb, ncon, max_force)
    for p in progress:
        t = min(int(round(p * (T - 1))), T - 1)
        seed_state(model, data, states[t])
        renderer.update_scene(data, args.camera, scene_option)
        rgb = renderer.render()
        cs = extract_contacts(model, data, robot_geoms)
        ncon = len(cs)
        maxf = max((c[3] for c in cs), default=0.0)
        frames.append((p, rgb, ncon, maxf))
        print(f"  t={t:>3}/{T - 1:<3} (p={p:.2f}): "
              f"non-robot contacts (>1N)={ncon:<3} max_force={maxf:7.2f} N")

    renderer.close()

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, (p, rgb, ncon, maxf) in zip(axes.flat, frames):
        ax.imshow(rgb)
        ax.set_title(f"p={p:.2f}  contacts={ncon}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{args.hdf5.name}/{demo_key}  camera={args.camera}  "
        f"nq={model.nq} nv={model.nv}",
        fontsize=11,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main() or 0)
