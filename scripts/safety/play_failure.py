#!/usr/bin/env python3
"""Play a single failure on a trajectory and render object health bars.

Replays a LIBERO demo up to a chosen failure point, injects a hardware failure,
settles the physics, and writes an mp4 of the agentview camera with a live
health bar per object that depletes as the OopsieVerse damage model
(:class:`planner.risk.damage.DamageAccumulator`) accrues d_mech. Objects below
half health are flagged BREAKING; below 10, DESTROYED.

Usage::

    conda run -n failbench_env python -m scripts.safety.play_failure \\
        --task pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate \\
        --split libero_spatial --demo demo_0 --mode ALL_JOINTS --fail_frac 0.5 \\
        --out figures/failure_play.mp4
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import sys
from pathlib import Path
import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

H, W = 240, 320
BREAK_HEALTH, DESTROY_HEALTH = 50.0, 10.0


def short(name: str) -> str:
    return (name.replace("_main", "").replace("akita_", "")
            .replace("glazed_rim_porcelain_", "").replace("_1", "")[:16])


def make_failure(mode_name, joints_arg):
    from planner.experiments.config import FailureMode, FailureConfig
    from planner.risk.inference import _DEFAULT_JOINTS
    kw = dict(mode=FailureMode[mode_name])
    js = joints_arg or [f"joint{j}" for j in _DEFAULT_JOINTS[mode_name]]
    if js and mode_name in ("SINGLE_JOINT", "MULTI_JOINT"):
        kw["joint_names"] = js
    if mode_name == "SLIPPERY_GRIP":
        kw["grip_value"] = 180.0
    return FailureConfig(**kw)


def compose(rgb, health, baseline, banner, banner_col):
    """RGB (left) + health-bar panel (right). ``health``/``baseline`` are dicts
    {obj: health}; baseline is full health for the % computation."""
    import cv2
    PW, PH = 560, 420
    PANEL = 360
    img = cv2.cvtColor(cv2.resize(rgb, (PW, PH)), cv2.COLOR_RGB2BGR)
    canvas = np.full((PH, PW + PANEL, 3), 28, np.uint8)
    canvas[:, :PW] = img

    # failure banner over the camera
    cv2.rectangle(canvas, (0, 0), (PW, 34), (20, 20, 20), -1)
    cv2.putText(canvas, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                banner_col, 2, cv2.LINE_AA)

    x0, bw, bh, row = PW + 22, PANEL - 130, 26, 64
    cv2.putText(canvas, "OBJECT HEALTH", (x0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (230, 230, 230), 2, cv2.LINE_AA)
    for i, (obj, hp) in enumerate(health.items()):
        y = 58 + i * row
        frac = max(0.0, min(1.0, hp / 100.0))
        # green -> yellow -> red
        if frac > 0.5:
            col = (60, 200, 60)
        elif frac > 0.25:
            col = (40, 200, 230)
        else:
            col = (50, 50, 230)
        cv2.putText(canvas, short(obj), (x0, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y), (x0 + bw, y + bh), (70, 70, 70), -1)
        if frac > 0:
            cv2.rectangle(canvas, (x0, y), (x0 + int(frac * bw), y + bh), col, -1)
        cv2.rectangle(canvas, (x0, y), (x0 + bw, y + bh), (200, 200, 200), 1)
        cv2.putText(canvas, f"{hp:5.1f}", (x0 + bw + 8, y + bh - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        if hp < DESTROY_HEALTH:
            cv2.putText(canvas, "DESTROYED", (x0 + 6, y + bh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        elif hp < BREAK_HEALTH:
            cv2.putText(canvas, "BREAKING", (x0 + 6, y + bh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="libero_spatial")
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--traj", type=Path, default=None,
                    help="play a trajectory file (.npz arm_qpos) instead of the "
                         "demo — demos are usually too safe to show damage")
    ap.add_argument("--object", default="bowl")
    ap.add_argument("--mode", default="ALL_JOINTS",
                    choices=("GRIPPER_OPEN", "SLIPPERY_GRIP", "SINGLE_JOINT",
                             "MULTI_JOINT", "ALL_JOINTS"))
    ap.add_argument("--joints", default=None,
                    help="comma joints for SINGLE/MULTI (e.g. joint2,joint4)")
    ap.add_argument("--fail_frac", type=float, default=0.5)
    ap.add_argument("--gallery", action="store_true",
                    help="play several failures (one per mode) injected at "
                         "DIFFERENT points spread along the trajectory")
    ap.add_argument("--modes", default=None,
                    help="comma list of modes for --gallery (default: all 5)")
    ap.add_argument("--settle_steps", type=int, default=400)
    ap.add_argument("--render_every", type=int, default=4)
    ap.add_argument("--lead_frames", type=int, default=12)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("figures/failure_play.mp4"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import cv2
    from scripts.libero.gen_diverse_trajs import setup_scene
    from planner.experiments.libero.runner import LiberoRunner, LiberoTrialConfig
    from planner.experiments.data_capture import OffscreenRenderer
    from planner.risk.damage import DamageAccumulator
    from planner.risk.severity import entity_severity

    ctx = setup_scene(args.task, args.split, args.demo, args.object,
                      with_predictor=False)
    runner = LiberoRunner(ctx["demo"], LiberoTrialConfig(
        resistance_mode="gravcomp_pd", post_failure_settle_steps=args.settle_steps))
    model, data, h = runner.model, runner.data, runner.handles
    rend = OffscreenRenderer(model, height=H, width=W, camera_name=h.agentview_cam)

    carried = ctx["carried"]["name"]
    track = [carried] + [e["name"] for e in ctx["ents"]
                         if e["name"] != carried
                         and entity_severity(e["name"], scale="paper") >= 2.0]
    track = track[:5]
    print(f"tracking health for: {track}")

    # trajectory source: a saved arm trajectory (--traj) or the demo's states
    if args.traj:
        from scripts.safety.eval_trajectories import traj_from_npz, seed_step
        traj = traj_from_npz(args.traj, model)
        fq = np.asarray(ctx["demo"].finger_qpos, float)
        Td = len(fq)
        closed_fingers = fq[Td // 4:3 * Td // 4].mean(0) if Td >= 4 else fq.mean(0)
        T = len(traj["frames"])
        seed = lambda t: seed_step(runner, ctx, traj, t, closed_fingers, "friction")
        src_label = f"{args.traj.stem}"
    else:
        fs = np.asarray(ctx["demo"].full_states, float)
        T = len(fs)

        def seed(t):
            runner._set_full_state(fs[t])
            return np.array([data.qpos[adr] for adr in h.arm_qpos_adrs])
        src_label = f"demo {args.demo}"

    def play_one(ev_frac, mode_name, joints):
        """Play one failure (inject at ev_frac, settle) → list of frames."""
        runner.injector.restore_all()                 # clean any prior failure
        fail_step = int(np.clip(ev_frac, 0, 1) * (T - 1))
        fc = make_failure(mode_name, joints)
        jl = (" " + ",".join(fc.joint_names)) if getattr(fc, "joint_names", None) else ""
        clip = []
        arm_target = None
        lead0 = max(0, fail_step - args.lead_frames)
        for t in range(lead0, fail_step + 1):
            arm_target = seed(t)
            clip.append(compose(np.asarray(rend.render(data)),
                                {o: 100.0 for o in track}, None,
                                f"executing...  (fail @ {ev_frac:.0%})", (180, 220, 180)))
        runner._inject_failure(fc)
        # synthetic grips wedge the object; force a clean release so a gripper
        # failure actually drops it (demo grasps release on their own)
        if args.traj and mode_name in ("GRIPPER_OPEN", "SLIPPERY_GRIP"):
            from scripts.safety.eval_trajectories import release_carried
            release_carried(runner, ctx)
        acc = DamageAccumulator(model, data, h.robot_geom_ids,
                                held_body_ids={ctx["carried"]["body_id"]})
        for step in range(args.settle_steps):
            runner._apply_resistance(arm_target)
            mujoco.mj_step(model, data)
            acc.step()
            if step % args.render_every == 0 or step == args.settle_steps - 1:
                health = {o: float(acc.per_body_health.get(o, 100.0)) for o in track}
                clip.append(compose(np.asarray(rend.render(data)), health, None,
                                    f"FAILURE: {mode_name}{jl}  @ {ev_frac:.0%}",
                                    (60, 60, 255)))
        for _ in range(max(1, args.fps // 2)):
            clip.append(clip[-1])
        final = {short(o): round(float(acc.per_body_health.get(o, 100.0)), 1)
                 for o in track}
        print(f"  {mode_name:13s} @frac {ev_frac:.2f} (step "
              f"{int(ev_frac*(T-1)):3d}/{T-1}): final health {final}")
        runner.injector.restore_all()
        return clip

    # build the event list: one failure, or a gallery spread across the traj
    if args.gallery:
        ms = (args.modes.split(",") if args.modes else
              ["GRIPPER_OPEN", "SLIPPERY_GRIP", "SINGLE_JOINT",
               "MULTI_JOINT", "ALL_JOINTS"])
        # spread the injection points across the carry/transport phase so no two
        # failures fire at the same spot
        spots = np.linspace(0.30, 0.85, len(ms))
        events = [(float(spots[i]), m, None) for i, m in enumerate(ms)]
    else:
        events = [(args.fail_frac, args.mode,
                   args.joints.split(",") if args.joints else None)]

    print(f"{src_label}: {T} states; {len(events)} failure(s)")
    frames = []
    for ev_frac, mode_name, joints in events:
        frames += play_one(ev_frac, mode_name, joints)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Hh, Ww = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (Ww, Hh))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"wrote {args.out}  ({len(frames)} frames @ {args.fps}fps)")
    try:
        rend.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
