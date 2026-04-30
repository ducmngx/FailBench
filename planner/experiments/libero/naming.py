"""Robosuite ↔ FailBench name resolution.

LIBERO/robosuite uses ``robot0_joint{1..7}`` for the Panda arm joints,
``gripper0_finger_joint1/2`` for fingers, and ``robot0_eef_*`` / ``agentview`` /
``robot0_eye_in_hand`` for sites and cameras. FailBench expects ``joint{1..7}``,
``end_effector``, ``front_cam``, ``ee_cam``. This module resolves names by trying
both conventions through ``mj_name2id`` so the rest of the LIBERO runner can
treat them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import mujoco


# Candidate prefixes to try when looking up Panda arm joints in a robosuite model.
# Order matters: robosuite's actual names come first, FailBench's fallback last.
_ARM_JOINT_CANDIDATES = [
    "robot0_joint{i}",
    "joint{i}",
]

_GRIPPER_FINGER_JOINT_CANDIDATES = [
    ["gripper0_finger_joint1", "gripper0_finger_joint2"],
    ["finger_joint1", "finger_joint2"],
]

# Robosuite Panda gripper exposes two per-finger actuators with opposite ranges
# rather than a single scalar; we drive both. Fall back to FailBench's single
# ``actuator8`` when running on our own panda.xml.
_GRIPPER_FINGER_ACTUATOR_CANDIDATES = [
    ["gripper0_gripper_finger_joint1", "gripper0_gripper_finger_joint2"],
    ["gripper0_finger_joint1", "gripper0_finger_joint2"],
]
_GRIPPER_SCALAR_ACTUATOR_CANDIDATES = [
    "gripper0_gripper",
    "actuator8",
]

_EE_SITE_CANDIDATES = [
    "gripper0_grip_site",
    "robot0_grip_site",
    "robot0_ee",
    "robot0_eef",
    "end_effector",
]

_AGENTVIEW_CAM_CANDIDATES = ["agentview", "frontview", "front_cam"]
_EE_CAM_CANDIDATES = ["robot0_eye_in_hand", "eye_in_hand", "ee_cam"]

# Robot body-name prefixes used to build the "is-robot" mask in ContactExtractor.
ROBOSUITE_ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mount0_")


@dataclass
class ModelHandles:
    """Resolved IDs / addresses for the runtime model."""
    # Arm
    arm_joint_ids: List[int]          # length 7
    arm_qpos_adrs: List[int]          # length 7
    arm_dof_adrs: List[int]           # length 7
    arm_actuator_ids: List[int]       # length 7 (-1 if no per-joint actuator)
    arm_joint_names: List[str]        # the names that resolved (for failure injector)
    # Gripper
    finger_qpos_adrs: List[int]       # length 2
    finger_dof_adrs: List[int]        # length 2
    gripper_actuator_id: int          # scalar gripper controller; -1 if absent
    finger_actuator_ids: List[int]    # length 2 (per-finger, robosuite); empty if scalar exists
    # EE
    ee_site_id: int                   # -1 if absent
    # Cameras
    agentview_cam: Optional[str]
    ee_cam: Optional[str]
    # Robot geom set (for ContactExtractor)
    robot_geom_ids: set


def _first_match(model: mujoco.MjModel, obj_type, candidates) -> int:
    for name in candidates:
        jid = mujoco.mj_name2id(model, obj_type, name)
        if jid >= 0:
            return jid
    return -1


def resolve_model_handles(model: mujoco.MjModel) -> ModelHandles:
    """Resolve all the names the LIBERO runner needs.

    Raises ValueError when an essential element (the 7 arm joints) cannot be
    located — every other field degrades to a sentinel (-1 / None).
    """
    arm_joint_ids: List[int] = []
    arm_joint_names: List[str] = []
    for i in range(1, 8):
        jid = -1
        resolved_name = None
        for tmpl in _ARM_JOINT_CANDIDATES:
            name = tmpl.format(i=i)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                resolved_name = name
                break
        if jid < 0:
            raise ValueError(
                f"Could not resolve arm joint {i} from candidates "
                f"{[t.format(i=i) for t in _ARM_JOINT_CANDIDATES]}"
            )
        arm_joint_ids.append(jid)
        arm_joint_names.append(resolved_name)

    arm_qpos_adrs = [int(model.jnt_qposadr[j]) for j in arm_joint_ids]
    arm_dof_adrs = [int(model.jnt_dofadr[j]) for j in arm_joint_ids]

    # Map joint -> actuator (transmission target).
    arm_actuator_ids: List[int] = []
    for jid in arm_joint_ids:
        aid = -1
        for a in range(model.nu):
            if model.actuator_trnid[a, 0] == jid:
                aid = a
                break
        arm_actuator_ids.append(aid)

    # Fingers: pick the first candidate set whose names both resolve.
    finger_qpos_adrs: List[int] = []
    finger_dof_adrs: List[int] = []
    for cand_pair in _GRIPPER_FINGER_JOINT_CANDIDATES:
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in cand_pair]
        if all(i >= 0 for i in ids):
            finger_qpos_adrs = [int(model.jnt_qposadr[i]) for i in ids]
            finger_dof_adrs = [int(model.jnt_dofadr[i]) for i in ids]
            break

    gripper_actuator_id = _first_match(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, _GRIPPER_SCALAR_ACTUATOR_CANDIDATES)

    finger_actuator_ids: List[int] = []
    if gripper_actuator_id < 0:
        for cand_pair in _GRIPPER_FINGER_ACTUATOR_CANDIDATES:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                   for n in cand_pair]
            if all(i >= 0 for i in ids):
                finger_actuator_ids = ids
                break

    ee_site_id = _first_match(model, mujoco.mjtObj.mjOBJ_SITE, _EE_SITE_CANDIDATES)

    agentview_cam = None
    for n in _AGENTVIEW_CAM_CANDIDATES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) >= 0:
            agentview_cam = n
            break
    ee_cam = None
    for n in _EE_CAM_CANDIDATES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) >= 0:
            ee_cam = n
            break

    # Robot geom set: bodies whose name starts with any robosuite robot prefix,
    # plus FailBench-style names if present.
    robot_body_ids = set()
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.startswith(ROBOSUITE_ROBOT_BODY_PREFIXES):
            robot_body_ids.add(bid)
        elif bname in {"link0", "link1", "link2", "link3", "link4", "link5",
                       "link6", "link7", "hand", "left_finger", "right_finger"}:
            robot_body_ids.add(bid)
    robot_geom_ids = {gid for gid in range(model.ngeom)
                      if int(model.geom_bodyid[gid]) in robot_body_ids}

    return ModelHandles(
        arm_joint_ids=arm_joint_ids,
        arm_qpos_adrs=arm_qpos_adrs,
        arm_dof_adrs=arm_dof_adrs,
        arm_actuator_ids=arm_actuator_ids,
        arm_joint_names=arm_joint_names,
        finger_qpos_adrs=finger_qpos_adrs,
        finger_dof_adrs=finger_dof_adrs,
        gripper_actuator_id=gripper_actuator_id,
        finger_actuator_ids=finger_actuator_ids,
        ee_site_id=ee_site_id,
        agentview_cam=agentview_cam,
        ee_cam=ee_cam,
        robot_geom_ids=robot_geom_ids,
    )
