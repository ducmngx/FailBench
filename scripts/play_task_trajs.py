#!/usr/bin/env python
"""Visualise generated task trajectories in the MuJoCo viewer.

Loads each trajectory pkl, teleports the grasped object to the EE,
replays the interpolated trajectory, then holds the final pose for a
moment before moving to the next trajectory.

Usage:
    # Play all tasks for scene_level2
    python scripts/play_task_trajs.py --scene scene_level2

    # Play one specific task
    python scripts/play_task_trajs.py --scene scene_level2 --task pick_place_nominal

    # Play one specific trajectory file
    python scripts/play_task_trajs.py --traj_file scenes/scene_level2/trajs/scene_level2_pick_place_nominal_00.pkl

Controls:
    The viewer runs automatically. Each trajectory plays in full, then
    pauses for --hold seconds before the next one starts.
    Close the viewer window to stop early.
"""

import argparse
import os
import pickle
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.utils.trajectory_interpolation import interpolate_trajectory

TASK_ORDER = [
    "pick_place_nominal",
    "pick_place_far",
    "pick_place_cluttered",
    "pick_alt_object",
    "pick_and_stack",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_traj(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    scene_name = next(iter(data))
    entry = data[scene_name]
    return entry


def _collect_trajs(scene, task, trajs_dir):
    """Return sorted list of pkl paths for a given scene + optional task filter."""
    files = []
    for fname in sorted(os.listdir(trajs_dir)):
        if not fname.endswith(".pkl"):
            continue
        if task and f"_{task}_" not in fname:
            continue
        files.append(os.path.join(trajs_dir, fname))
    return files


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def play(pkl_paths, scene_xml, hold_secs=2.0, speed=1.0,
         interp_points=100, steps_per_point=8):
    """Play all trajectories in pkl_paths using a shared MuJoCo viewer."""
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    # Save initial qpos for full env reset between trajectories
    mujoco.mj_forward(model, data)
    init_qpos = data.qpos.copy()
    init_qvel = data.qvel.copy()

    joint_limits = np.column_stack([
        model.jnt_range[:7, 0],
        model.jnt_range[:7, 1],
    ])

    n_total = len(pkl_paths)
    print(f"\nPlaying {n_total} trajectories. Close viewer to stop.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i, pkl_path in enumerate(pkl_paths):
            if not viewer.is_running():
                break

            entry = _load_traj(pkl_path)
            task_id = entry.get("task_id", "?")
            traj_id = entry.get("traj_id", i)
            grasped = entry.get("grasped_object", "object3")
            goal = entry.get("goal_pos")

            print(f"[{i+1}/{n_total}] {os.path.basename(pkl_path)}")
            print(f"         task={task_id}  traj={traj_id}  "
                  f"object={grasped}  goal={np.round(goal, 3) if goal is not None else 'N/A'}")

            # Reset full environment to initial state
            data.qpos[:] = init_qpos
            data.qvel[:] = init_qvel
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()

            # Segmented format
            segments = entry.get("segments")
            if segments:
                # Open gripper for approach
                data.ctrl[7] = 255.0
                for _ in range(200):
                    mujoco.mj_step(model, data)
                grip_ctrl = data.ctrl[7]

                for seg in segments:
                    seg_name = seg["name"]
                    seg_traj = seg["trajectory"]
                    action = seg.get("action_after")
                    print(f"           {seg_name} ({len(seg_traj)} wp)", end="")

                    dense = interpolate_trajectory(
                        seg_traj,
                        num_points_per_segment=interp_points,
                        method="cubic",
                        joint_limits=joint_limits,
                    )

                    for pt in dense:
                        if not viewer.is_running():
                            break
                        data.ctrl[:7] = pt
                        data.ctrl[7] = grip_ctrl
                        for _ in range(steps_per_point):
                            mujoco.mj_step(model, data)
                        viewer.sync()
                        time.sleep(0.002 / max(speed, 0.1))

                    if action == "grasp":
                        print(" → GRASP", end="")
                        # Gradual close
                        for step in range(30):
                            data.ctrl[7] = 255.0 * (1 - step / 30 * 0.95)
                            for _ in range(20):
                                mujoco.mj_step(model, data)
                            viewer.sync()
                            time.sleep(0.002)
                        grip_ctrl = data.ctrl[7]
                        # Settle
                        for _ in range(300):
                            mujoco.mj_step(model, data)
                            viewer.sync()
                            time.sleep(0.001)
                    elif action == "release":
                        print(" → RELEASE", end="")
                        data.ctrl[7] = 255.0
                        for _ in range(300):
                            mujoco.mj_step(model, data)
                            viewer.sync()
                            time.sleep(0.001)
                        grip_ctrl = data.ctrl[7]
                    print()
            else:
                # Legacy flat format
                traj = entry["trajectory"]
                dense = interpolate_trajectory(
                    traj, num_points_per_segment=interp_points,
                    method="cubic", joint_limits=joint_limits)
                grip_ctrl = data.ctrl[7]
                for pt in dense:
                    if not viewer.is_running():
                        break
                    data.ctrl[:7] = pt
                    data.ctrl[7] = grip_ctrl
                    for _ in range(steps_per_point):
                        mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.002 / max(speed, 0.1))

            # Hold final pose
            t0 = time.time()
            while viewer.is_running() and time.time() - t0 < hold_secs:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.002)

        print("\nPlayback complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--traj_file", help="Play a single specific pkl file")
    group.add_argument("--scene", help="Play all trajectories for this scene")

    parser.add_argument("--task", default=None,
                        choices=TASK_ORDER + [None],
                        help="Filter to a specific task (use with --scene)")
    parser.add_argument("--scene_xml", default=None,
                        help="Override scene XML path")
    parser.add_argument("--hold",  type=float, default=2.0,
                        help="Seconds to hold final pose before next trajectory (default 2)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0)")
    parser.add_argument("--interp_points", type=int, default=100,
                        help="Interpolation points per segment (default 100)")

    args = parser.parse_args()

    if args.traj_file:
        # Single file — infer scene from filename
        pkl_paths = [args.traj_file]
        scene = os.path.basename(args.traj_file).split("_")[0] + "_" + \
                os.path.basename(args.traj_file).split("_")[1]
        scene_xml = args.scene_xml or f"scenes/{scene}/scene.xml"
    elif args.scene:
        scene = args.scene
        trajs_dir = f"scenes/{scene}/trajs"
        if not os.path.isdir(trajs_dir):
            print(f"No trajs directory found at {trajs_dir}")
            sys.exit(1)
        pkl_paths = _collect_trajs(scene, args.task, trajs_dir)
        if not pkl_paths:
            print(f"No trajectories found for scene={scene} task={args.task}")
            sys.exit(1)
        scene_xml = args.scene_xml or f"scenes/{scene}/scene.xml"
    else:
        parser.print_help()
        sys.exit(1)

    play(
        pkl_paths=pkl_paths,
        scene_xml=scene_xml,
        hold_secs=args.hold,
        speed=args.speed,
        interp_points=args.interp_points,
    )
