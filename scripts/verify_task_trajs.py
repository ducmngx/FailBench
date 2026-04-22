#!/usr/bin/env python
"""Batch-verify generated trajectory pkl files via physics replay.

Replays each trajectory headlessly through MuJoCo and checks:
  - No unwanted arm-environment collisions
  - Object successfully grasped and lifted
  - Object placed near goal position

Usage:
    # Verify all trajectories for a task
    python scripts/verify_task_trajs.py --scene scene_level2 --task stack_nominal

    # Verify all tasks for a scene
    python scripts/verify_task_trajs.py --scene scene_level2

    # Verify and delete failures
    python scripts/verify_task_trajs.py --scene scene_level2 --delete_failures

    # Verify a single file
    python scripts/verify_task_trajs.py --traj_file scenes/scene_level2/trajs/scene_level2_stack_nominal_00.pkl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.trajectory_verifier import verify_trajectory


def _collect_pkls(trajs_dir, task=None):
    files = []
    for fname in sorted(os.listdir(trajs_dir)):
        if not fname.endswith(".pkl"):
            continue
        if task and f"_{task}_" not in fname:
            continue
        files.append(os.path.join(trajs_dir, fname))
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--traj_file", help="Verify a single pkl file")
    group.add_argument("--scene", help="Verify all trajectories for this scene")

    parser.add_argument("--task", default=None, help="Filter to a specific task")
    parser.add_argument("--scene_xml", default=None, help="Override scene XML path")
    parser.add_argument("--delete_failures", action="store_true",
                        help="Delete pkl files that fail verification")
    parser.add_argument("--collision_threshold", type=float, default=5.0,
                        help="Arm-env collision force threshold in N (default 5.0)")
    parser.add_argument("--place_tolerance", type=float, default=0.05,
                        help="Max xy distance from goal in m (default 0.05)")

    args = parser.parse_args()

    if args.traj_file:
        pkl_paths = [args.traj_file]
        parts = os.path.basename(args.traj_file).split("_")
        scene = parts[0] + "_" + parts[1]
        scene_xml = args.scene_xml or f"scenes/{scene}/scene.xml"
    elif args.scene:
        scene = args.scene
        trajs_dir = f"scenes/{scene}/trajs"
        if not os.path.isdir(trajs_dir):
            print(f"No trajs directory at {trajs_dir}")
            sys.exit(1)
        pkl_paths = _collect_pkls(trajs_dir, args.task)
        if not pkl_paths:
            print(f"No trajectories found for scene={scene} task={args.task}")
            sys.exit(1)
        scene_xml = args.scene_xml or f"scenes/{scene}/scene.xml"
    else:
        parser.print_help()
        sys.exit(1)

    n_pass = 0
    n_fail = 0

    for pkl in pkl_paths:
        result = verify_trajectory(
            scene_xml, pkl,
            collision_force_threshold=args.collision_threshold,
            place_xy_tolerance=args.place_tolerance,
        )

        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {os.path.basename(pkl)}"
              f"  collision={result.max_collision_force:.1f}N"
              f"  place_err={result.place_error:.3f}m"
              f"  grasp={'ok' if result.grasp_ok else 'FAIL'}"
              f"  | {result.details}")

        if result.passed:
            n_pass += 1
        else:
            n_fail += 1
            if args.delete_failures:
                os.unlink(pkl)
                print(f"       deleted {pkl}")

    print(f"\n{n_pass} passed, {n_fail} failed out of {len(pkl_paths)} total")
