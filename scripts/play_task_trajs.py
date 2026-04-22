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

from planner.grasp_lock import GraspLock
from planner.utils.trajectory_interpolation import interpolate_trajectory

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


def _filter_meshes_only(pkl_paths):
    """Keep only pkls whose grasp_meta.source == 'graspgen'.

    Legacy pkls without grasp_meta are dropped — the flag is for focused
    inspection of GraspGen-produced trajectories, not legacy content.
    """
    kept = []
    for p in pkl_paths:
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            entry = data[next(iter(data))]
            meta = entry.get("grasp_meta")
            if meta is not None and meta.get("source") == "graspgen":
                kept.append(p)
        except Exception:
            continue
    return kept


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def play(pkl_paths, scene_xml, hold_secs=2.0, speed=1.0,
         interp_points=100, steps_per_point=8,
         show_meta=False, pause_at_grasp=False, flag_bad=False,
         flagged_log=None, strict_attach=False, show_depth=False,
         depth_cams=("ee_cam",)):
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

    depth_renderers = []
    if show_depth:
        import cv2
        from planner.experiments.data_capture import OffscreenRenderer
        for cam in depth_cams:
            r = OffscreenRenderer(model, height=240, width=320, camera_name=cam)
            depth_renderers.append((cam, r))
            cv2.namedWindow(f"depth: {cam}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"depth: {cam}", 480, 360)

    def _pump_depth():
        if not depth_renderers:
            return
        for cam, r in depth_renderers:
            d = r.render_depth(data)
            near, far = 0.05, 2.0
            d = np.clip(d, near, far)
            norm = ((d - near) / (far - near) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)
            cv2.imshow(f"depth: {cam}", colored)
        cv2.waitKey(1)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i, pkl_path in enumerate(pkl_paths):
            if not viewer.is_running():
                break

            entry = _load_traj(pkl_path)
            task_id = entry.get("task_id", "?")
            traj_id = entry.get("traj_id", i)
            grasped = entry.get("grasped_object", "object3")
            goal = entry.get("goal_pos")
            meta = entry.get("grasp_meta")

            print(f"[{i+1}/{n_total}] {os.path.basename(pkl_path)}")
            print(f"         task={task_id}  traj={traj_id}  "
                  f"object={grasped}  goal={np.round(goal, 3) if goal is not None else 'N/A'}")
            if show_meta and meta is not None:
                src = meta.get("source", "?")
                gid = meta.get("grasp_id") or "-"
                conf = meta.get("confidence")
                conf_s = f"{conf:.3f}" if isinstance(conf, (int, float)) else "-"
                ik = meta.get("ik_attempts", "-")
                rc = meta.get("retry_count", "-")
                axis = meta.get("approach_axis")
                axis_s = (f"[{axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f}]"
                          if axis is not None else "-")
                print(f"         grasp: src={src}  id={gid}  conf={conf_s}  "
                      f"ik_tries={ik}  retry={rc}  axis={axis_s}")

            # Reset full environment to initial state
            data.qpos[:] = init_qpos
            data.qvel[:] = init_qvel
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()

            # Segmented format
            segments = entry.get("segments")
            if segments:
                # Command the arm to the trajectory's starting config BEFORE
                # settling so the arm doesn't swing toward zero-config (which
                # can knock scene objects during the 200-step settle).
                first_pt = np.asarray(segments[0]["trajectory"][0])
                data.ctrl[:7] = first_pt[:7]
                data.qpos[:7] = first_pt[:7]
                data.qvel[:7] = 0.0
                data.ctrl[7] = 255.0
                mujoco.mj_forward(model, data)
                for _ in range(200):
                    mujoco.mj_step(model, data)
                grip_ctrl = data.ctrl[7]
                lock = GraspLock(model)

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
                            lock.update(data)
                        viewer.sync()
                        _pump_depth()
                        time.sleep(0.002 / max(speed, 0.1))

                    if action == "grasp":
                        print(" → GRASP", end="")
                        if pause_at_grasp:
                            print()
                            try:
                                input("         [ENTER to execute grasp, Ctrl-C to abort]: ")
                            except (EOFError, KeyboardInterrupt):
                                print()
                                return
                        for step in range(30):
                            data.ctrl[7] = 255.0 * (1 - step / 30 * 0.95)
                            for _ in range(20):
                                mujoco.mj_step(model, data)
                            viewer.sync()
                            time.sleep(0.002)
                        grip_ctrl = data.ctrl[7]
                        for _ in range(300):
                            mujoco.mj_step(model, data)
                            viewer.sync()
                            time.sleep(0.001)
                        if strict_attach:
                            if not lock.attach_strict(model, data, grasped):
                                print(" → GRASP-FAILED (no contact)", end="")
                        else:
                            lock.attach(model, data, grasped)
                    elif action == "release":
                        print(" → RELEASE", end="")
                        lock.release(data)
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

            if flag_bad and flagged_log is not None and viewer.is_running():
                try:
                    ans = input(f"         flag this traj as bad? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if ans == "y":
                    with open(flagged_log, "a") as fl:
                        fl.write(f"{pkl_path}\n")
                    print(f"         → logged to {flagged_log}")

        print("\nPlayback complete.")
        if flag_bad and flagged_log is not None and os.path.exists(flagged_log):
            with open(flagged_log) as fl:
                n_flagged = sum(1 for _ in fl)
            print(f"Flagged {n_flagged} trajectory/ies in {flagged_log}")

    if depth_renderers:
        import cv2
        for _, r in depth_renderers:
            r.close()
        cv2.destroyAllWindows()


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
                        help="Filter to a specific task (use with --scene)")
    parser.add_argument("--scene_xml", default=None,
                        help="Override scene XML path")
    parser.add_argument("--hold",  type=float, default=2.0,
                        help="Seconds to hold final pose before next trajectory (default 2)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0)")
    parser.add_argument("--interp_points", type=int, default=100,
                        help="Interpolation points per segment (default 100)")
    parser.add_argument("--meshes_only", action="store_true",
                        help="Skip trajs whose grasp_meta.source != 'graspgen'.")
    parser.add_argument("--show_meta", action="store_true",
                        help="Print grasp_meta (grasp_id, confidence, ik_attempts) per traj.")
    parser.add_argument("--pause_at_grasp", action="store_true",
                        help="Wait for ENTER at each grasp action to freeze-frame for inspection.")
    parser.add_argument("--flag_bad", action="store_true",
                        help="After each traj, prompt y/N; flagged pkl paths are appended to .flagged.txt.")
    parser.add_argument("--strict-attach", action="store_true", dest="strict_attach",
                        help="Require finger-object contact before engaging GraspLock; "
                             "print GRASP-FAILED when fingers didn't actually close on the object.")
    parser.add_argument("--show_depth", action="store_true",
                        help="Pop up a depth-camera window (OpenCV) alongside the 3D viewer.")
    parser.add_argument("--depth_cam", default="ee_cam",
                        help="Comma-separated camera names for depth windows "
                             "(default ee_cam; e.g. 'ee_cam,front_cam').")

    args = parser.parse_args()

    if args.traj_file:
        # Single file — infer scene from filename
        pkl_paths = [args.traj_file]
        scene = os.path.basename(args.traj_file).split("_")[0] + "_" + \
                os.path.basename(args.traj_file).split("_")[1]
        scene_xml = args.scene_xml or f"scenes/{scene}/scene.xml"
        flagged_log = None
    elif args.scene:
        scene = args.scene
        trajs_dir = f"scenes/{scene}/trajs"
        if not os.path.isdir(trajs_dir):
            print(f"No trajs directory found at {trajs_dir}")
            sys.exit(1)
        pkl_paths = _collect_trajs(scene, args.task, trajs_dir)
        if args.meshes_only:
            pkl_paths = _filter_meshes_only(pkl_paths)
        if not pkl_paths:
            print(f"No trajectories found for scene={scene} task={args.task}")
            sys.exit(1)
        scene_xml = args.scene_xml or f"scenes/{scene}/scene.xml"
        flagged_log = os.path.join(trajs_dir, ".flagged.txt") if args.flag_bad else None
    else:
        parser.print_help()
        sys.exit(1)

    play(
        pkl_paths=pkl_paths,
        scene_xml=scene_xml,
        hold_secs=args.hold,
        speed=args.speed,
        interp_points=args.interp_points,
        show_meta=args.show_meta,
        pause_at_grasp=args.pause_at_grasp,
        flag_bad=args.flag_bad,
        flagged_log=flagged_log,
        strict_attach=args.strict_attach,
        show_depth=args.show_depth,
        depth_cams=tuple(c.strip() for c in args.depth_cam.split(",") if c.strip()),
    )
