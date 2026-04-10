#!/usr/bin/env python
"""Generate full-mission pick-and-place trajectories using the IK+RRT planner.

Each trajectory has 5 segments: approach → descend(grasp) → lift → transport → place(release).
Segments are planned sequentially via RRT, each starting from the end config of the previous.

Heights (approach, lift, carry, place) are derived at runtime from the MuJoCo scene model,
so no manual per-scene tuning is needed. The approach direction is sampled on a hemisphere
around the object to maximise pre-failure joint configuration diversity.

Output format (one .pkl per trajectory):
    {scene_name: {"task_id", "traj_id", "grasped_object", "goal_pos", "segments": [...]}}

Usage:
    python scripts/generate_task_trajs.py \\
        --scene scene_level2 \\
        --task pick_place_nominal \\
        --n_trajs 3 \\
        --seed 0
"""

import argparse
import os
import pickle
import re
import sys
from typing import Optional

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.examples.pick_and_place_safety_L2 import PandaPickAndPlace_L2

# ---------------------------------------------------------------------------
# Scene registry
# ---------------------------------------------------------------------------

_ALT_OBJECT = {
    "scene_level2":    "object1",
    "scene_kitchen":   "mug",
    "scene_workshop":  "bolt",
    "scene_grocery":   "can_master_chef",
    "scene_cluttered": "tape",
}


def _detect_robot_xml(scene_xml: str) -> str:
    """Parse the scene XML to find which robot MJCF is included."""
    scene_dir = os.path.dirname(os.path.abspath(scene_xml))
    with open(scene_xml) as f:
        text = f.read()
    m = re.search(r'<include\s+file="([^"]*panda[^"]*)"', text)
    if m is None:
        raise RuntimeError(f"No panda include found in {scene_xml}")
    return os.path.normpath(os.path.join(scene_dir, m.group(1)))


# ---------------------------------------------------------------------------
# Scene geometry queries — derive heights from MuJoCo model at runtime
# ---------------------------------------------------------------------------

# Robot body names — used to exclude robot geoms from obstacle search
_ROBOT_BODIES = {"world", "link0", "link1", "link2", "link3", "link4",
                 "link5", "link6", "link7", "hand", "left_finger", "right_finger"}


def _derive_scene_heights(model, data, obj_name, search_radius=0.30):
    """Derive approach/lift/carry/place heights from the scene model.

    Queries the MuJoCo model for:
      - Object position and half-height (from its geom)
      - Table top z (largest horizontal box geom in a body with 'table' in its name)
      - Tallest obstacle within `search_radius` of the object (excluding robot geoms)

    Returns a dict of heights used by _compute_waypoints().
    """
    mujoco.mj_forward(model, data)

    # Object position and half-height
    obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    obj_pos = data.xpos[obj_bid].copy()
    obj_geom_name = f"{obj_name}_geom"
    obj_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obj_geom_name)
    if obj_gid >= 0:
        obj_half_h = float(model.geom_size[obj_gid][2])  # z half-extent for box
    else:
        obj_half_h = 0.03  # fallback

    # Table top z — find the largest horizontal box in any body named *table*
    table_z = 0.0
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and "table" in bname.lower() and model.geom_type[gid] == 6:  # box
            size = model.geom_size[gid]
            top = data.geom_xpos[gid][2] + size[2]
            if size[0] > 0.1:  # must be a tabletop, not a leg
                table_z = max(table_z, top)

    if table_z == 0.0:
        table_z = obj_pos[2] - obj_half_h  # fallback: bottom of object

    # Tallest obstacle within search_radius of the object (excluding robot and floor)
    max_obstacle_top = table_z
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname in _ROBOT_BODIES:
            continue
        if model.geom_type[gid] == 0:  # plane (floor)
            continue
        if bname and "table" in bname.lower():
            continue

        gpos = data.geom_xpos[gid]
        dist_xy = np.linalg.norm(gpos[:2] - obj_pos[:2])
        if dist_xy > search_radius:
            continue

        # Top of this geom (works for box; conservative for other types)
        if model.geom_type[gid] == 6:  # box
            top = gpos[2] + model.geom_size[gid][2]
        else:
            top = gpos[2] + model.geom_size[gid][0]  # radius as conservative height
        max_obstacle_top = max(max_obstacle_top, top)

    # Compute derived heights
    clearance = 0.18  # generous margin — keeps lower arm links above table during transport
    grasp_offset = 0.0  # EE at object center for firm finger contact
    lift_z = max_obstacle_top + clearance  # safely above obstacles
    carry_z = max(lift_z, max_obstacle_top + clearance + 0.05)
    place_offset = obj_half_h + 0.04  # EE above surface with margin for controller overshoot

    heights = {
        "table_z": table_z,
        "obj_half_h": obj_half_h,
        "grasp_h": grasp_offset,            # above object CENTER for grasp
        "approach_radius": 0.08,             # base radius for hemisphere sampling
        "approach_radius_var": 0.04,         # variation in approach radius
        "approach_min_elev": 0.0,            # 0° from vertical (can be directly above)
        "approach_max_elev": np.pi / 6,      # 30° max from vertical (mostly upright)
        "approach_min_z": max_obstacle_top + clearance,  # well above obstacles for easy RRT
        "lift_z": lift_z,                    # absolute z for lift target
        "carry_z": carry_z,                  # absolute z for transport
        "place_offset": place_offset,        # above placement surface
        "max_obstacle_top": max_obstacle_top,
    }

    print(f"  Derived heights: table_z={table_z:.3f} max_obstacle_top={max_obstacle_top:.3f} "
          f"carry_z={carry_z:.3f} lift_z={lift_z:.3f}")
    return heights


# ---------------------------------------------------------------------------
# Hemisphere approach sampling
# ---------------------------------------------------------------------------

def _sample_approach_pos(obj_pos, heights, rng):
    """Sample an approach position on the upper hemisphere around the object.

    Instead of always approaching from directly above, samples a direction
    on a hemisphere (azimuth 0–2π, elevation 30°–60° from vertical) at a
    randomised distance. This produces fundamentally different arm configurations
    for each trajectory.

    The approach z is clamped to stay above the tallest nearby obstacle.
    """
    theta = rng.uniform(0, 2 * np.pi)                # azimuth: full circle
    phi = rng.uniform(heights["approach_min_elev"],
                      heights["approach_max_elev"])    # elevation from vertical
    r = heights["approach_radius"] + rng.uniform(0, heights["approach_radius_var"])

    offset = r * np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])
    approach = obj_pos + offset

    # Clamp z to stay above obstacles
    approach[2] = max(approach[2], heights["approach_min_z"])

    return approach


# ---------------------------------------------------------------------------
# Goal samplers — return (place_x, place_y) placement position on table
# ---------------------------------------------------------------------------

def _goal_sampler(scene: str, task: str):
    """Return callable rng → (x, y) placement position.

    Goal samplers are scene-specific because the table geometry and obstacle
    layout differ. Only the placement XY is sampled; z is derived from heights.
    """
    if scene == "scene_level2":
        # Obstacle grid occupies x∈[-0.2, 0.2], y∈[-0.25, -0.45].
        # Table extends x∈[-0.30, 0.30], y∈[-0.19, -0.51].
        # Goal regions must be CLEAR of obstacles (at least 0.06m from any obstacle center).
        # Clear zones: near-edge (y > -0.22), far-edge (y < -0.48),
        #              left-edge (x < -0.23), right-edge (x > 0.23).

        # All obstacle centers for collision avoidance
        _obstacle_centers = [
            (-0.2,-0.25),(-0.1,-0.25),(0.0,-0.25),(0.1,-0.25),(0.2,-0.25),
            (-0.2,-0.35),(-0.1,-0.35),(0.0,-0.35),(0.1,-0.35),(0.2,-0.35),
            (-0.2,-0.45),(-0.1,-0.45),(0.0,-0.45),(0.1,-0.45),(0.2,-0.45),
        ]

        def _is_clear(x, y, min_dist=0.06):
            """Check goal is at least min_dist from every obstacle center."""
            for ox, oy in _obstacle_centers:
                if abs(x - ox) < min_dist and abs(y - oy) < min_dist:
                    return False
            return True

        if task == "pick_place_nominal":
            # Clear areas on table edges, outside obstacle grid.
            # Avoid extreme diagonal reaches (e.g., x=-0.30, y=-0.48)
            # which cause lower arm links to sweep through the table.
            def sampler(rng):
                for _ in range(50):
                    zone = rng.choice(["near", "far", "left", "right"])
                    if zone == "near":
                        x, y = rng.uniform(-0.20, 0.25), rng.uniform(-0.19, -0.22)
                    elif zone == "far":
                        x, y = rng.uniform(-0.15, 0.25), rng.uniform(-0.48, -0.51)
                    elif zone == "left":
                        x, y = rng.uniform(-0.27, -0.23), rng.uniform(-0.22, -0.42)
                    else:
                        x, y = rng.uniform(0.23, 0.30), rng.uniform(-0.22, -0.48)
                    if _is_clear(x, y):
                        return x, y
                return x, y  # fallback

        elif task == "pick_place_far":
            # Farthest reachable edges
            def sampler(rng):
                for _ in range(50):
                    if rng.choice([True, False]):
                        x, y = rng.uniform(0.25, 0.35), rng.uniform(-0.22, -0.48)
                    else:
                        x, y = rng.uniform(-0.35, -0.25), rng.uniform(-0.22, -0.48)
                    if _is_clear(x, y):
                        return x, y
                return x, y

        elif task == "pick_place_cluttered":
            # Intentionally through the grid — but place in a clear gap
            def sampler(rng):
                for _ in range(50):
                    x = rng.uniform(-0.20, 0.20)
                    y = rng.uniform(-0.48, -0.52)
                    if _is_clear(x, y):
                        return x, y
                return x, y

        elif task == "pick_alt_object":
            # Same clear zones as nominal
            def sampler(rng):
                for _ in range(50):
                    zone = rng.choice(["near", "far", "left", "right"])
                    if zone == "near":
                        x, y = rng.uniform(-0.20, 0.25), rng.uniform(-0.19, -0.22)
                    elif zone == "far":
                        x, y = rng.uniform(-0.15, 0.25), rng.uniform(-0.48, -0.51)
                    elif zone == "left":
                        x, y = rng.uniform(-0.27, -0.23), rng.uniform(-0.22, -0.42)
                    else:
                        x, y = rng.uniform(0.23, 0.30), rng.uniform(-0.22, -0.48)
                    if _is_clear(x, y):
                        return x, y
                return x, y

        elif task == "pick_and_stack":
            obstacle_xy = [
                (-0.2, -0.35), (-0.1, -0.35), (0.0, -0.35), (0.1, -0.35),
                (-0.1, -0.45), (0.0, -0.45), (0.1, -0.45), (0.2, -0.45),
            ]
            def sampler(rng):
                ox, oy = obstacle_xy[rng.randint(len(obstacle_xy))]
                return ox + rng.uniform(-0.02, 0.02), oy + rng.uniform(-0.02, 0.02)
        else:
            raise ValueError(f"Unknown task '{task}' for scene '{scene}'")
    else:
        raise NotImplementedError(f"Goal sampler for '{scene}' not defined yet.")
    return sampler


# ---------------------------------------------------------------------------
# Segment planning
# ---------------------------------------------------------------------------

_SEGMENT_DEFS = [
    # (name, action_after, task_type, downward_constraint)
    # Downward constraint on all segments keeps the gripper pointing down,
    # preventing the arm from sweeping sideways into the table.
    ("approach",  None,      "transit", True),
    ("descend",   "grasp",   "pick",    True),
    ("lift",      None,      "pick",    True),
    ("transport", None,      "transit", True),
    ("place",     "release", "place",   True),
]


def _compute_waypoints(obj_pos, goal_xy, heights, task, approach_pos):
    """Compute the 5 EE target positions for a full mission.

    Args:
        obj_pos: Object world position (3,)
        goal_xy: Placement (x, y) tuple
        heights: Dict from _derive_scene_heights()
        task: Task name string
        approach_pos: Pre-sampled approach position from hemisphere
    """
    gx, gy = goal_xy
    grasp = obj_pos.copy();  grasp[2] += heights["grasp_h"]
    lift = obj_pos.copy();   lift[2] = heights["lift_z"]
    transport = np.array([gx, gy, heights["carry_z"]])

    if task == "pick_and_stack":
        place_z = heights["max_obstacle_top"] + heights["place_offset"]
    else:
        place_z = heights["table_z"] + heights["place_offset"]
    place = np.array([gx, gy, place_z])

    return [approach_pos, grasp, lift, transport, place]


def _plan_full_mission(planner, obj_pos, goal_xy, heights, task, seed_offset,
                       approach_pos):
    """Plan all 5 segments of a full pick-and-place mission.

    Returns list of segment dicts or None on planning failure.
    """
    targets = _compute_waypoints(obj_pos, goal_xy, heights, task, approach_pos)
    segments = []

    planner._set_home_position()

    for i, (seg_name, action_after, task_type, downward) in enumerate(_SEGMENT_DEFS):
        target_pos = targets[i]
        planner.seed = seed_offset + i * 10

        ok = planner.plan_to_ee_pose(target_pos, task_type=task_type,
                                     use_downward_constraint=downward)
        if not ok or not planner.current_path:
            print(f"      Planning failed at segment '{seg_name}' → target={target_pos.round(3)}")
            return None

        traj = [np.array(wp) for wp in planner.current_path]
        segments.append({
            "name": seg_name,
            "trajectory": traj,
            "action_after": action_after,
        })

        # Advance planner state to end of this segment
        last_wp = traj[-1]
        planner.scene_data.qpos[:planner.robot_dof] = last_wp[:planner.robot_dof]
        mujoco.mj_forward(planner.scene_model, planner.scene_data)

    return segments


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

def generate(scene, task, n_trajs, seed, scene_xml=None, robot_xml=None,
             out_dir=None, max_retries=5):
    scene_xml = scene_xml or f"scenes/{scene}/scene.xml"
    robot_xml = robot_xml or _detect_robot_xml(scene_xml)
    out_dir = out_dir or f"scenes/{scene}/trajs"
    os.makedirs(out_dir, exist_ok=True)

    grasped_object = _ALT_OBJECT[scene] if task == "pick_alt_object" else "object3"
    sampler = _goal_sampler(scene, task)

    print(f"Scene: {scene}  Task: {task}  Trajs: {n_trajs}  Seed: {seed}")
    print(f"Robot XML: {robot_xml}")
    print(f"Grasped object: {grasped_object}\n")

    print("Initialising planner ...")
    planner = PandaPickAndPlace_L2(scene_xml, robot_xml, seed=seed)
    # Speed up RRT for batch generation while keeping collision safety.
    # step_size=0.02 balances speed with fine enough collision detection
    # around narrow obstacles like the table edge.
    planner.rrt_planner.step_size = 0.02
    planner.rrt_planner.max_iterations = 3000

    obj_pos = planner.get_object_position(grasped_object)
    if obj_pos is None:
        print(f"ERROR: object '{grasped_object}' not found in scene")
        return

    # Derive heights from the scene model (no hardcoded values)
    heights = _derive_scene_heights(planner.scene_model, planner.scene_data,
                                     grasped_object)

    rng = np.random.RandomState(seed)
    saved = 0

    for traj_idx in range(n_trajs):
        out_path = os.path.join(out_dir, f"{scene}_{task}_{traj_idx:02d}.pkl")
        if os.path.exists(out_path):
            print(f"[{traj_idx:02d}] Already exists, skipping.")
            saved += 1
            continue

        print(f"[{traj_idx:02d}] Planning full mission ...")
        success = False

        for attempt in range(max_retries):
            goal_xy = sampler(rng)
            seed_offset = seed + traj_idx * 1000 + attempt * 100

            approach_pos = _sample_approach_pos(obj_pos.copy(), heights, rng)
            print(f"      attempt {attempt+1}/{max_retries}  "
                  f"goal=({goal_xy[0]:.3f}, {goal_xy[1]:.3f})  "
                  f"approach={approach_pos.round(3)}")

            segments = _plan_full_mission(planner, obj_pos.copy(), goal_xy, heights,
                                          task, seed_offset, approach_pos)
            if segments is None:
                continue

            total_wps = sum(len(s["trajectory"]) for s in segments)
            goal_pos = _compute_waypoints(obj_pos.copy(), goal_xy, heights, task,
                                           approach_pos)[-1]

            pkl_data = {
                scene: {
                    "task_id": task,
                    "traj_id": traj_idx,
                    "grasped_object": grasped_object,
                    "goal_pos": goal_pos,
                    "segments": segments,
                }
            }
            with open(out_path, "wb") as f:
                pickle.dump(pkl_data, f)

            seg_summary = " → ".join(f"{s['name']}({len(s['trajectory'])}wp)" for s in segments)
            print(f"      Saved {total_wps} total waypoints: {seg_summary}")
            saved += 1
            success = True
            break

        if not success:
            print(f"      FAILED after {max_retries} attempts")

    print(f"\nDone. {saved}/{n_trajs} trajectories saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, choices=list(_ALT_OBJECT))
    parser.add_argument("--task", required=True,
                        choices=["pick_place_nominal", "pick_place_far",
                                 "pick_place_cluttered", "pick_alt_object",
                                 "pick_and_stack"])
    parser.add_argument("--n_trajs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene_xml", default=None)
    parser.add_argument("--robot_xml", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_retries", type=int, default=5)
    args = parser.parse_args()

    generate(scene=args.scene, task=args.task, n_trajs=args.n_trajs, seed=args.seed,
             scene_xml=args.scene_xml, robot_xml=args.robot_xml,
             out_dir=args.out_dir, max_retries=args.max_retries)
