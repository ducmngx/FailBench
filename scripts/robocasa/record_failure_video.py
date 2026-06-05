"""Record an mp4 of a binhng/robocasa demo with a failure injected mid-trajectory.

Replays the demo kinematically up to ``fail_progress``, fires the chosen failure,
then steps physics with gravity-comp + PD active resistance on healthy joints
(arm + base + torso) so the mobile base doesn't drift. The arm joints listed in
``--joints`` go limp (zero torque); everything else holds its pre-failure pose.

Self-contained — does not depend on the (not-yet-written) RoboCasa adapter.

Usage::

    /home/aaron/miniconda3/envs/robocasa/bin/python scripts/robocasa/record_failure_video.py \
        --hdf5 datasets/robocasa/raw/TurnOffStove.hdf5 \
        --demo demo_1037 --mode SINGLE_JOINT --joints joint2 \
        --fail_progress 0.6 --output out/robocasa_failures/turnoff_stove_j2.mp4
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import re
import sys
from pathlib import Path
from typing import List

import cv2
import h5py
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOSUITE_ROOT = REPO_ROOT / "external" / "robosuite_for_robocasa" / "robosuite"
ROBOCASA_ROOT = REPO_ROOT / "external" / "robocasa" / "robocasa"

# binhng/robocasa MJCFs were generated on multiple machines and bake in absolute
# paths from each. Map any "<author-path>/robosuite/models/..." or
# "<author-path>/robocasa/models/..." to our local copies via regex.
#   Author paths seen so far:
#     /home/soroush/code/{robosuite-dev,robocasa-dev}/...
#     /data1/aaronl/rpl-robocasa/{robosuite-dev,robocasa-dev}/...
#     /home/abhim/robocasa/{robosuite,robocasa}/...

_ROBOSUITE_PAT = re.compile(r"/[^\"<>\s]+?/robosuite/models/")
_ROBOCASA_PAT = re.compile(r"/[^\"<>\s]+?/robocasa/models/")


def remap_mjcf(xml: str) -> str:
    xml = _ROBOSUITE_PAT.sub(f"{ROBOSUITE_ROOT}/models/", xml)
    xml = _ROBOCASA_PAT.sub(f"{ROBOCASA_ROOT}/models/", xml)
    xml = xml.replace(
        'meshdir="meshes/"', f'meshdir="{ROBOCASA_ROOT}/models/assets/"'
    )
    return xml


def _id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


class RobotHandles:
    """Resolve joint/actuator/qpos/qvel indices for PandaMobile + Omron base."""

    ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]
    ARM_ACTS = [f"robot0_torq_j{i}" for i in range(1, 8)]
    BASE_ACTS = [
        "base0_actuator_mobile_forward",
        "base0_actuator_mobile_side",
        "base0_actuator_mobile_yaw",
        "base0_actuator_torso_height",
    ]
    BASE_JOINTS = [
        "base0_joint_mobile_forward",
        "base0_joint_mobile_side",
        "base0_joint_mobile_yaw",
        "base0_joint_torso_height",
    ]
    FINGER_JOINTS = [
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
    ]
    FINGER_ACTS = [
        "gripper0_right_gripper_finger_joint1",
        "gripper0_right_gripper_finger_joint2",
    ]

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.arm_jids = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.ARM_JOINTS]
        self.arm_qadrs = [int(model.jnt_qposadr[j]) for j in self.arm_jids]
        self.arm_dadrs = [int(model.jnt_dofadr[j]) for j in self.arm_jids]
        self.arm_aids = [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.ARM_ACTS]

        self.base_jids = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.BASE_JOINTS]
        self.base_qadrs = [int(model.jnt_qposadr[j]) for j in self.base_jids]
        self.base_dadrs = [int(model.jnt_dofadr[j]) for j in self.base_jids]
        self.base_aids = [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.BASE_ACTS]

        self.finger_jids = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.FINGER_JOINTS]
        self.finger_qadrs = [int(model.jnt_qposadr[j]) for j in self.finger_jids]
        self.finger_aids = [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.FINGER_ACTS]


# Arm PD gains — tuned for PandaMobile torque actuators (ctrlrange ±80 for j1-5, ±12 for j6-7)
ARM_KP = np.array([60.0, 60.0, 60.0, 40.0, 20.0, 10.0, 10.0])
ARM_KD = 2.0 * np.sqrt(ARM_KP)
BASE_KP = np.array([300.0, 300.0, 100.0, 5000.0])  # mobile-fwd, mobile-side, yaw, torso
BASE_KD = 2.0 * np.sqrt(BASE_KP)


def apply_resistance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    h: RobotHandles,
    failed_arm_idx: set[int],
    arm_target: np.ndarray,
    base_target: np.ndarray,
    finger_target: np.ndarray,
):
    """Gravity-comp + PD on every healthy joint. Failed arm joints get zero torque."""
    mujoco.mj_forward(model, data)
    # Arm
    for i in range(7):
        aid = h.arm_aids[i]
        if i in failed_arm_idx:
            data.ctrl[aid] = 0.0
            continue
        q = float(data.qpos[h.arm_qadrs[i]])
        qd = float(data.qvel[h.arm_dadrs[i]])
        grav = float(data.qfrc_bias[h.arm_dadrs[i]])
        tau = grav + ARM_KP[i] * (arm_target[i] - q) - ARM_KD[i] * qd
        lo, hi = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(tau, lo, hi))
    # Base (always healthy)
    for i in range(4):
        aid = h.base_aids[i]
        q = float(data.qpos[h.base_qadrs[i]])
        qd = float(data.qvel[h.base_dadrs[i]])
        grav = float(data.qfrc_bias[h.base_dadrs[i]])
        tau = grav + BASE_KP[i] * (base_target[i] - q) - BASE_KD[i] * qd
        lo, hi = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(tau, lo, hi))
    # Fingers: write ctrls directly toward their target qpos (these actuators are position-controlled)
    for i in range(2):
        aid = h.finger_aids[i]
        lo, hi = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(finger_target[i], lo, hi))


def inject_failure(
    h: RobotHandles, mode: str, joints: List[str], grip_value: float
) -> tuple[set[int], np.ndarray | None]:
    """Returns (failed_arm_idx set, override_finger_target or None)."""
    failed_arm_idx: set[int] = set()
    override_fingers: np.ndarray | None = None
    if mode == "SINGLE_JOINT":
        j = int(joints[0].replace("joint", "")) - 1
        failed_arm_idx.add(j)
    elif mode == "MULTI_JOINT":
        for jn in joints:
            failed_arm_idx.add(int(jn.replace("joint", "")) - 1)
    elif mode == "ALL_JOINTS":
        failed_arm_idx.update(range(7))
    elif mode == "GRIPPER_OPEN":
        # Override finger target to fully-open
        # ctrlrange: j1 (0, 0.04), j2 (-0.04, 0)  → open = (+0.04, -0.04)
        override_fingers = np.array([0.04, -0.04])
    elif mode == "SLIPPERY_GRIP":
        t = float(np.clip(grip_value / 255.0, 0.0, 1.0))
        # Partial close: lerp between open and closed
        override_fingers = np.array([(1 - t) * 0.04, -(1 - t) * 0.04])
    return failed_arm_idx, override_fingers


def annotate(frame, label, color=(255, 255, 255), y=28):
    cv2.putText(frame, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)
    return frame


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True, type=Path)
    p.add_argument("--demo", default=None,
                   help="Demo key (default: first demo in file)")
    p.add_argument("--mode", default="SINGLE_JOINT",
                   choices=["SINGLE_JOINT", "MULTI_JOINT", "ALL_JOINTS",
                            "GRIPPER_OPEN", "SLIPPERY_GRIP"])
    p.add_argument("--joints", default="joint4",
                   help="Comma-separated joint names: joint1..joint7")
    p.add_argument("--fail_progress", type=float, default=0.6)
    p.add_argument("--grip_value", type=float, default=180.0)
    p.add_argument("--post_steps", type=int, default=600)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--camera", default="robot0_agentview_center")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        demo_key = args.demo or demos[0]
        g = f[f"data/{demo_key}"]
        xml = g.attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode("utf-8")
        states = g["states"][...]
        print(f"[{args.hdf5.name}/{demo_key}] T={states.shape[0]}  state_dim={states.shape[1]}")

    xml = remap_mjcf(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    h = RobotHandles(model)
    nq, nv = model.nq, model.nv

    T = states.shape[0]
    fail_idx = max(1, min(int(args.fail_progress * (T - 1)), T - 1))

    # Seed pre-failure state from states[0] so the world starts at the right config
    data.qpos[:] = states[0, 1 : 1 + nq]
    data.qvel[:] = states[0, 1 + nq :]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, args.height, args.width)
    scene_option = mujoco.MjvOption()
    groups = set(int(gp) for gp in model.geom_group)
    if 0 in groups and 1 in groups:
        scene_option.geomgroup[0] = 0
        scene_option.geomgroup[1] = 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(args.output), fourcc, args.fps, (args.width, args.height)
    )

    # Pre-failure: kinematic replay from states
    for t in range(fail_idx + 1):
        data.qpos[:] = states[t, 1 : 1 + nq]
        data.qvel[:] = states[t, 1 + nq :]
        mujoco.mj_forward(model, data)
        if t % args.stride == 0:
            renderer.update_scene(data, args.camera, scene_option)
            rgb = renderer.render()
            label = f"t={t}/{T - 1}  PRE  p={t / max(T - 1, 1):.2f}"
            writer.write(annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), label))

    # Snapshot pre-failure targets
    arm_target = np.array([float(data.qpos[h.arm_qadrs[i]]) for i in range(7)])
    base_target = np.array([float(data.qpos[h.base_qadrs[i]]) for i in range(4)])
    finger_target = np.array([float(data.qpos[h.finger_qadrs[i]]) for i in range(2)])

    # Inject
    failed_arm_idx, override_fingers = inject_failure(
        h, args.mode, [s.strip() for s in args.joints.split(",") if s.strip()],
        args.grip_value
    )
    if override_fingers is not None:
        finger_target = override_fingers

    # Failure boundary banner
    fail_label = f"FAILURE: {args.mode}"
    if args.mode in ("SINGLE_JOINT", "MULTI_JOINT"):
        fail_label += f"  {args.joints}"
    for _ in range(args.fps // 2):
        renderer.update_scene(data, args.camera, scene_option)
        rgb = renderer.render()
        writer.write(annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                              fail_label, (0, 0, 255)))

    # Post-failure: physics with active resistance on healthy joints
    for s in range(args.post_steps):
        apply_resistance(model, data, h, failed_arm_idx,
                         arm_target, base_target, finger_target)
        mujoco.mj_step(model, data)
        if s % args.stride == 0:
            renderer.update_scene(data, args.camera, scene_option)
            rgb = renderer.render()
            label = f"s={s}/{args.post_steps}  POST  {args.mode}"
            writer.write(annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                  label, (0, 255, 255)))

    writer.release()
    renderer.close()
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
