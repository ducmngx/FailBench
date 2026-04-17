"""Physics-based trajectory verifier.

Replays a trajectory pkl through MuJoCo physics (headless) and checks:
1. No unwanted arm-environment collisions (robot links vs table/obstacles)
2. Grasp success (object lifted above table after grasp + lift)
3. Place success (object lands near goal after release)

Used by generate_task_trajs.py to reject bad trajectories before saving,
and by verify_task_trajs.py for batch validation of existing files.
"""

import os
import pickle
from dataclasses import dataclass

import mujoco
import numpy as np

from planner.grasp_lock import GraspLock
from planner.utils.trajectory_interpolation import interpolate_trajectory

# Robot arm bodies (exclude fingers — they're supposed to touch the object)
_ARM_BODIES = {"link0", "link1", "link2", "link3", "link4",
               "link5", "link6", "link7", "hand"}


@dataclass
class VerificationResult:
    passed: bool
    grasp_ok: bool
    place_ok: bool
    collision_free: bool
    details: str
    max_collision_force: float
    place_error: float
    object_final_z: float


def _build_arm_geom_set(model):
    """Return set of geom IDs belonging to robot arm bodies (not fingers)."""
    arm_body_ids = set()
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in _ARM_BODIES:
            arm_body_ids.add(bid)
    return {gid for gid in range(model.ngeom)
            if model.geom_bodyid[gid] in arm_body_ids}


def _get_table_z(model, data):
    """Find table top z from the scene model."""
    table_z = 0.0
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and "table" in bname.lower() and model.geom_type[gid] == 6:
            size = model.geom_size[gid]
            if size[0] > 0.1:  # tabletop, not a leg
                table_z = max(table_z, data.geom_xpos[gid][2] + size[2])
    return table_z


def _check_collisions(model, data, arm_geoms, grasped_geom_id):
    """Check for unwanted arm-environment contacts at current sim state.

    Returns the maximum contact force magnitude from unwanted contacts,
    or 0.0 if clean.
    """
    max_force = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2

        # We only care about contacts where one geom is an arm body
        g1_arm = g1 in arm_geoms
        g2_arm = g2 in arm_geoms
        if not (g1_arm or g2_arm):
            continue

        # Skip arm-arm self-contacts
        if g1_arm and g2_arm:
            continue

        # Skip expected contacts: arm/finger touching grasped object
        other = g2 if g1_arm else g1
        if other == grasped_geom_id:
            continue

        # This is an unwanted arm-environment contact — measure force
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        mag = np.linalg.norm(force[:3])
        max_force = max(max_force, mag)

    return max_force


def verify_trajectory(
    scene_xml: str,
    pkl_path: str,
    interp_points: int = 100,
    steps_per_point: int = 8,
    collision_force_threshold: float = 5.0,
    grasp_z_margin: float = 0.10,
    place_xy_tolerance: float = 0.05,
    settle_steps: int = 300,
) -> VerificationResult:
    """Replay a trajectory pkl through physics and verify quality.

    Parameters
    ----------
    scene_xml : path to MuJoCo scene XML
    pkl_path : path to trajectory pkl file
    interp_points : interpolation points per segment (match generation settings)
    steps_per_point : sim steps per interpolated point
    collision_force_threshold : Newtons; arm-env contact above this → collision
    grasp_z_margin : meters above table_z to confirm successful pick
    place_xy_tolerance : meters; max xy distance from goal after release
    settle_steps : sim steps to settle after grasp/release actions
    """
    # Load trajectory
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
    scene_name = next(iter(pkl_data))
    entry = pkl_data[scene_name]
    segments = entry["segments"]
    grasped_object = entry.get("grasped_object", "object3")
    goal_pos = entry.get("goal_pos")

    # Load model
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Identify geoms
    arm_geoms = _build_arm_geom_set(model)
    obj_geom_name = f"{grasped_object}_geom"
    grasped_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obj_geom_name)
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, grasped_object)

    table_z = _get_table_z(model, data)
    joint_limits = np.column_stack([model.jnt_range[:7, 0], model.jnt_range[:7, 1]])

    # Tracking
    max_collision_force = 0.0
    grasp_ok = False
    place_ok = False
    issues = []

    # Open gripper and settle the object to its natural resting position.
    # Command the arm to the trajectory's starting config so it doesn't swing
    # toward zero-config during settle and knock scene objects around.
    first_pt = np.asarray(segments[0]["trajectory"][0])
    data.ctrl[:7] = first_pt[:7]
    data.qpos[:7] = first_pt[:7]
    data.qvel[:7] = 0.0
    data.ctrl[7] = 255.0
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    obj_settled_z = data.xpos[obj_body_id][2]
    # Reset to t=0 so trajectory replay starts from a clean state
    data2 = mujoco.MjData(model)
    data2.ctrl[:7] = first_pt[:7]
    data2.qpos[:7] = first_pt[:7]
    data2.qvel[:7] = 0.0
    data2.ctrl[7] = 255.0
    mujoco.mj_forward(model, data2)
    for _ in range(500):
        mujoco.mj_step(model, data2)
    data = data2
    grip_ctrl = data.ctrl[7]
    lock = GraspLock(model)

    # Replay each segment
    for seg in segments:
        seg_name = seg["name"]
        seg_traj = seg["trajectory"]
        action = seg.get("action_after")

        dense = interpolate_trajectory(
            seg_traj,
            num_points_per_segment=interp_points,
            method="cubic",
            joint_limits=joint_limits,
        )

        for pt in dense:
            data.ctrl[:7] = pt
            data.ctrl[7] = grip_ctrl
            for _ in range(steps_per_point):
                mujoco.mj_step(model, data)
                lock.update(data)

            # Check collisions at each interpolated point
            force = _check_collisions(model, data, arm_geoms, grasped_geom_id)
            max_collision_force = max(max_collision_force, force)

        # Execute gripper actions
        if action == "grasp":
            for step in range(30):
                data.ctrl[7] = 255.0 * (1 - step / 30 * 0.95)
                for _ in range(20):
                    mujoco.mj_step(model, data)
            grip_ctrl = data.ctrl[7]
            for _ in range(settle_steps):
                mujoco.mj_step(model, data)
            lock.attach(model, data, grasped_object)

        elif action == "release":
            lock.release(data)
            data.ctrl[7] = 255.0
            grip_ctrl = data.ctrl[7]
            for _ in range(settle_steps):
                mujoco.mj_step(model, data)

        # Check grasp success after lift segment.
        # Use settled_z as baseline so mesh objects (with actual resting height
        # different from XML-declared center) are handled correctly.
        if seg_name == "lift":
            obj_z = data.xpos[obj_body_id][2]
            grasp_threshold = obj_settled_z + grasp_z_margin
            grasp_ok = obj_z > grasp_threshold
            if not grasp_ok:
                issues.append(f"grasp failed: obj_z={obj_z:.3f} vs settled_z+margin={grasp_threshold:.3f}")

    # Check place success after final settle
    obj_final_pos = data.xpos[obj_body_id].copy()
    obj_final_z = obj_final_pos[2]

    if goal_pos is not None:
        place_error = np.linalg.norm(obj_final_pos[:2] - goal_pos[:2])
        place_ok = place_error < place_xy_tolerance
        if not place_ok:
            issues.append(f"place error: {place_error:.3f}m (tol={place_xy_tolerance}m)")
    else:
        place_error = 0.0
        place_ok = True

    collision_free = max_collision_force < collision_force_threshold
    if not collision_free:
        issues.append(f"arm collision: {max_collision_force:.1f}N (tol={collision_force_threshold}N)")

    passed = grasp_ok and place_ok and collision_free
    details = "; ".join(issues) if issues else "all checks passed"

    return VerificationResult(
        passed=passed,
        grasp_ok=grasp_ok,
        place_ok=place_ok,
        collision_free=collision_free,
        details=details,
        max_collision_force=max_collision_force,
        place_error=place_error,
        object_final_z=obj_final_z,
    )
