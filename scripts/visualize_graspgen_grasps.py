"""Viser visualizer for GraspGen YAML outputs.

Lays out one or more meshes along X and renders their top-K grasps as Franka
gripper wireframes, color-coded by confidence (red=low, green=high).

Must run with the GraspGen venv (its viser + grasp_gen install), not
failbench_env:

    external/GraspGen/.venv/bin/python scripts/visualize_graspgen_grasps.py \
        --pair scenes/scene_kitchen/assets/cubesmall.stl /tmp/graspgen_smoke/cubesmall_grasps.yml \
        --pair scenes/scene_kitchen/assets/apple.stl     /tmp/graspgen_smoke/apple_grasps.yml

Then open  http://localhost:8080  in a browser. Ctrl-C to stop.
"""
import argparse
import os
import time

import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml

from grasp_gen.utils.viser_utils import (
    create_visualizer,
    visualize_grasp,
    visualize_mesh,
)


def color_from_score(s: float):
    s = max(0.0, min(1.0, float(s)))
    return [int(255 * (1 - s)), int(255 * s), 0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pair", nargs=2, action="append", metavar=("MESH", "YAML"), required=True,
        help="Mesh file + GraspGen YAML. Repeat for multiple objects.",
    )
    p.add_argument("--top_k", type=int, default=10, help="Grasps per mesh (default 10).")
    p.add_argument("--spacing", type=float, default=0.35, help="X spacing between meshes (m).")
    p.add_argument("--port", type=int, default=8080, help="Viser server port.")
    p.add_argument("--gripper", default="franka_panda", help="Gripper wireframe name.")
    return p.parse_args()


def main():
    args = parse_args()
    vis = create_visualizer(clear=True, port=args.port)
    print(f"Viser server at  http://localhost:{args.port}")

    for i, (mesh_path, yml_path) in enumerate(args.pair):
        name = os.path.splitext(os.path.basename(mesh_path))[0]
        if not (os.path.exists(mesh_path) and os.path.exists(yml_path)):
            print(f"skip {name}: missing {mesh_path!r} or {yml_path!r}")
            continue

        mesh = trimesh.load(mesh_path, force="mesh")
        offset = np.eye(4)
        offset[0, 3] = i * args.spacing
        visualize_mesh(vis, f"/{name}/mesh", mesh, color=[180, 180, 200], transform=offset)

        d = yaml.safe_load(open(yml_path))
        grasps = sorted(d["grasps"].values(), key=lambda g: -g["confidence"])[: args.top_k]

        for k, g in enumerate(grasps):
            T = tra.quaternion_matrix([g["orientation"]["w"], *g["orientation"]["xyz"]])
            T[:3, 3] = g["position"]
            visualize_grasp(
                vis,
                f"/{name}/grasp_{k:02d}",
                offset @ T,
                color=color_from_score(g["confidence"]),
                gripper_name=args.gripper,
                linewidth=2.0,
            )

        print(f"  {name}: mesh + top {len(grasps)} grasps at x={i * args.spacing:.2f}")

    print("\nLeave this process running to keep the server alive. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
