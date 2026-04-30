#!/usr/bin/env python3
"""Record an mp4 of a LIBERO demo + failure injection.

Replays the demo kinematically up to ``fail_progress``, records pre-failure
frames every ``stride`` steps, fires the failure, then steps physics and
records post-failure frames every ``stride`` steps. The pre-failure and
post-failure clips are written to ``<output>``.

Usage::

    conda run -n failbench_env python -m scripts.libero.record_failure_video \\
        --hdf5 datasets/libero/raw/libero_spatial/<task>.hdf5 \\
        --demo demo_0 --mode SINGLE_JOINT --joints joint4 \\
        --fail_progress 0.6 --output out/single_joint.mp4
"""

from __future__ import annotations

import argparse
import os

import cv2
import mujoco
import numpy as np

from planner.experiments.config import FailureConfig, FailureMode
from planner.experiments.data_capture import OffscreenRenderer
from planner.experiments.libero.adapter import load_demo, materialise_mjcf
from planner.experiments.libero.failure import (
    LiberoFailureInjector, parse_joint_spec)
from planner.experiments.libero.naming import resolve_model_handles


def _kinematic_step(model, data, h, arm_q, finger_q):
    for adr, q in zip(h.arm_qpos_adrs, arm_q):
        data.qpos[adr] = q
    if len(h.finger_qpos_adrs) == 2:
        for adr, q in zip(h.finger_qpos_adrs, finger_q):
            data.qpos[adr] = q
    for adr in h.arm_dof_adrs:
        data.qvel[adr] = 0.0
    for adr in h.finger_dof_adrs:
        data.qvel[adr] = 0.0
    mujoco.mj_forward(model, data)


def _set_gripper_normalised(model, data, h, t):
    if h.gripper_actuator_id >= 0:
        lo, hi = model.actuator_ctrlrange[h.gripper_actuator_id]
        data.ctrl[h.gripper_actuator_id] = float(lo + t * (hi - lo))
        return
    for aid in h.finger_actuator_ids:
        lo, hi = model.actuator_ctrlrange[aid]
        open_end, close_end = (lo, hi) if abs(lo) < abs(hi) else (hi, lo)
        data.ctrl[aid] = float(close_end + t * (open_end - close_end))


_DEFAULT_KP = np.array([600.0, 600.0, 600.0, 600.0, 300.0, 120.0, 120.0])


def _apply_resistance(model, data, h, injector, target_qpos, kp=_DEFAULT_KP):
    """Mirror of LiberoRunner._apply_resistance — gravity-comp + PD on healthy joints."""
    kd = 2.0 * np.sqrt(kp)
    mujoco.mj_forward(model, data)
    for i, jid in enumerate(h.arm_joint_ids):
        if jid in injector.failed_joint_ids:
            continue
        aid = h.arm_actuator_ids[i]
        if aid < 0:
            continue
        q = float(data.qpos[h.arm_qpos_adrs[i]])
        qd = float(data.qvel[h.arm_dof_adrs[i]])
        grav = float(data.qfrc_bias[h.arm_dof_adrs[i]])
        tau = grav + kp[i] * (target_qpos[i] - q) - kd[i] * qd
        lo, hi = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(tau, lo, hi))


def _inject(injector, model, data, h, fc):
    mode = fc.mode
    if mode == FailureMode.GRIPPER_OPEN:
        _set_gripper_normalised(model, data, h, 1.0)
    elif mode == FailureMode.SLIPPERY_GRIP:
        _set_gripper_normalised(model, data, h, np.clip(fc.grip_value / 255.0, 0, 1))
    elif mode == FailureMode.SINGLE_JOINT:
        injector.fail_single(parse_joint_spec(fc.joint_names[0]))
    elif mode == FailureMode.MULTI_JOINT:
        injector.fail_multi([parse_joint_spec(n) for n in fc.joint_names])
    elif mode == FailureMode.ALL_JOINTS:
        injector.fail_all()


def _annotate(frame, label, color=(255, 255, 255)):
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)
    return frame


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--demo", default="demo_0")
    p.add_argument("--fail_progress", type=float, default=0.6)
    p.add_argument("--mode", default="SINGLE_JOINT",
                   choices=[m.name for m in FailureMode])
    p.add_argument("--joints", default="joint4")
    p.add_argument("--grip_value", type=float, default=180.0)
    p.add_argument("--post_steps", type=int, default=600,
                   help="Sim steps after failure injection")
    p.add_argument("--stride", type=int, default=4,
                   help="Render every Nth sim/replay step")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--camera", default=None,
                   help="Camera name (default: agentview)")
    p.add_argument("--resistance", choices=["none", "gravcomp_pd"], default="none")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    demo = load_demo(args.hdf5, args.demo)
    model = mujoco.MjModel.from_xml_path(materialise_mjcf(demo.model_xml))
    data = mujoco.MjData(model)
    h = resolve_model_handles(model)

    if demo.init_state is not None:
        nq, nv = model.nq, model.nv
        data.qpos[:nq] = demo.init_state[:nq]
        data.qvel[:nv] = demo.init_state[nq:nq + nv]
        mujoco.mj_forward(model, data)

    cam = args.camera or h.agentview_cam
    renderer = OffscreenRenderer(model, height=args.height, width=args.width,
                                 camera_name=cam)

    fc = FailureConfig(
        mode=FailureMode[args.mode],
        joint_names=[s.strip() for s in args.joints.split(",") if s.strip()],
        grip_value=args.grip_value,
    )
    injector = LiberoFailureInjector(model, data, h)

    T = demo.arm_qpos.shape[0]
    fail_idx = max(1, min(int(args.fail_progress * (T - 1)), T - 1))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps,
                             (args.width, args.height))

    # Pre-failure: kinematic replay
    for t in range(fail_idx + 1):
        _kinematic_step(model, data, h, demo.arm_qpos[t], demo.finger_qpos[t])
        if t % args.stride == 0:
            rgb = renderer.render(data)
            label = f"t={t}/{T}  PRE  progress={t/(T-1):.2f}"
            writer.write(_annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), label))

    # Hold a few frames at the failure boundary
    for _ in range(args.fps // 2):
        rgb = renderer.render(data)
        writer.write(_annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               f"FAILURE: {args.mode}", (0, 0, 255)))

    # Capture the hold target before failure injection (the demo's pose at the
    # failure boundary). Healthy joints will be PD-controlled toward this if
    # active resistance is enabled.
    last_qpos_cmd = demo.arm_qpos[fail_idx].astype(np.float64).copy()

    # Inject failure
    _inject(injector, model, data, h, fc)

    # Post-failure: free physics (optionally with active resistance on healthy joints)
    for s in range(args.post_steps):
        if args.resistance == "gravcomp_pd":
            _apply_resistance(model, data, h, injector, last_qpos_cmd)
        mujoco.mj_step(model, data)
        if s % args.stride == 0:
            rgb = renderer.render(data)
            label = f"s={s}/{args.post_steps}  POST  {args.mode}  [{args.resistance}]"
            writer.write(_annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   label, (0, 255, 255)))

    writer.release()
    renderer.close()
    print(f"Wrote {args.output}  ({fail_idx} pre frames + {args.post_steps} post steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
