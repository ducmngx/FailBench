"""Render front_cam vs topdown_cam for each scene at the robot home pose.

Writes a 5×2 mosaic to runs/preview_topdown.png so we can eyeball the new
topdown camera positions before committing to data re-collection.

Usage:
    python scripts/preview_topdown.py
    python scripts/preview_topdown.py --output runs/preview_topdown.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mujoco


SCENES = ["scene_level2", "scene_kitchen", "scene_workshop",
          "scene_grocery", "scene_cluttered"]

# Franka home pose — same 7 joints used elsewhere; gripper open.
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04],
                     dtype=np.float64)


def render(model: mujoco.MjModel, data: mujoco.MjData,
           camera_name: str, height: int = 480, width: int = 640) -> np.ndarray:
    r = mujoco.Renderer(model, height, width)
    try:
        r.update_scene(data, camera_name)
        return r.render()
    finally:
        r.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_dir", type=Path, default=Path("scenes"))
    ap.add_argument("--output", type=Path, default=Path("runs/preview_topdown.png"))
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    return ap.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in SCENES:
        xml = args.scenes_dir / scene / "scene.xml"
        print(f"loading {xml}")
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        # Set the robot's first 9 qpos to the home pose (works on both elevated
        # and non-elevated panda includes).
        n = min(len(HOME_QPOS), model.nq)
        data.qpos[:n] = HOME_QPOS[:n]
        mujoco.mj_forward(model, data)

        # Sanity: confirm both cameras exist
        cam_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                     for i in range(model.ncam)]
        if "front_cam" not in cam_names or "topdown_cam" not in cam_names:
            print(f"  !! missing cameras in {scene}: {cam_names}")
            continue

        front = render(model, data, "front_cam", args.height, args.width)
        top = render(model, data, "topdown_cam", args.height, args.width)
        rows.append((scene, front, top))

    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 3.4 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, -1)
    for r, (scene, front, top) in enumerate(rows):
        axes[r, 0].imshow(front); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"{scene}  —  front_cam", fontsize=10)
        axes[r, 1].imshow(top);   axes[r, 1].axis("off")
        axes[r, 1].set_title(f"{scene}  —  topdown_cam", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.output, dpi=110)
    plt.close()
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
