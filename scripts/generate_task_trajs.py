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
        --task clean_nominal \\
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
from planner.grasp_sampler import GraspSampler
from planner.tasks import get_grasped_object, load_tasks, make_goal_sampler
from planner.trajectory_verifier import verify_trajectory


def _detect_robot_xml(scene_xml: str) -> str:
    """Parse the scene XML to find which robot MJCF is included."""
    scene_dir = os.path.dirname(os.path.abspath(scene_xml))
    with open(scene_xml) as f:
        text = f.read()
    m = re.search(r'<include\s+file="([^"]*panda[^"]*)"', text)
    if m is None:
        raise RuntimeError(f"No panda include found in {scene_xml}")
    return os.path.normpath(os.path.join(scene_dir, m.group(1)))


def _resolve_mesh_path(scene_xml: str, body_name: str) -> Optional[str]:
    """Resolve the visual mesh file for a body, or None if it's a primitive."""
    import xml.etree.ElementTree as ET
    scene_dir = os.path.dirname(os.path.abspath(scene_xml))
    root = ET.parse(scene_xml).getroot()
    candidates = [os.path.join(scene_dir, "assets"), scene_dir]
    comp = root.find("compiler")
    if comp is not None and comp.get("meshdir"):
        candidates.insert(0, os.path.normpath(os.path.join(scene_dir, comp.get("meshdir"))))
    body = next((b for b in root.iter("body") if b.get("name") == body_name), None)
    if body is None:
        return None
    mesh_name = None
    for geom in body.findall("geom"):
        m = geom.get("mesh")
        if m and not m.endswith(("_c0", "_c1", "_c2", "_c3", "_c4")):
            mesh_name = m
            break
    if mesh_name is None:
        return None
    m_el = next((m for m in root.iter("mesh") if m.get("name") == mesh_name), None)
    if m_el is None:
        return None
    f = m_el.get("file")
    for cand in candidates:
        p = os.path.normpath(os.path.join(cand, f))
        if os.path.exists(p):
            return p
    return None


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
    _GEOM_CYLINDER = 5
    _GEOM_BOX = 6
    if obj_gid >= 0:
        gtype = int(model.geom_type[obj_gid])
        if gtype == _GEOM_CYLINDER:
            obj_half_h = float(model.geom_size[obj_gid][1])  # half-height for cylinder
        elif gtype == _GEOM_BOX:
            obj_half_h = float(model.geom_size[obj_gid][2])  # z half-extent for box
        else:
            obj_half_h = 0.03  # mesh: use a sensible fallback
    else:
        obj_half_h = 0.03  # fallback
        gtype = -1

    # Table top z — find the LARGEST-AREA horizontal box in any body named *table*.
    # Picking the largest (not just any big-x one) avoids treating small named
    # shelves / platforms attached to the table body as the tabletop.
    table_z = 0.0
    table_area = 0.0
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and "table" in bname.lower() and model.geom_type[gid] == 6:  # box
            size = model.geom_size[gid]
            if size[0] > 0.1 and size[1] > 0.1:  # must be a tabletop, not a leg or shelf strip
                area = size[0] * size[1]
                if area > table_area:
                    table_area = area
                    table_z = data.geom_xpos[gid][2] + size[2]

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

    # Get robot base z from link0 position (needed for workspace-limited lift_z)
    link0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    robot_base_z = float(data.xpos[link0_id][2]) if link0_id >= 0 else 0.0

    # Optional top shelf: scenes that include a geom named "top_shelf_geom"
    # expose its top z for place_height: on_shelf tasks.
    shelf_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "top_shelf_geom")
    if shelf_gid >= 0:
        shelf_top_z = float(data.geom_xpos[shelf_gid][2] + model.geom_size[shelf_gid][2])
    else:
        shelf_top_z = None

    # Compute derived heights
    # Desired clearance above obstacles for lift/approach
    desired_clearance = 0.18
    is_primitive = obj_gid >= 0 and gtype in (_GEOM_CYLINDER, _GEOM_BOX)
    if is_primitive:
        # EE at cylinder/box top face so finger pads can close around the sides.
        # Exception: for tall upright cylinders (half_length > 2 × radius, e.g. a
        # screwdriver-handle rod), grasping at the top puts the fingers past the
        # object. Target the mid-shaft by setting grasp_offset=0.
        if gtype == _GEOM_CYLINDER and obj_half_h > 2.0 * float(model.geom_size[obj_gid][0]):
            grasp_offset = 0.0
        else:
            grasp_offset = obj_half_h
    else:
        grasp_offset = 0.0       # EE at object center for mesh objects
    approach_max_elev = np.pi / 6  # 30° max elevation from vertical (all objects)

    # Workspace-limited lift_z: cap at a height reachable from the object's horizontal
    # position with downward EE constraint. Objects far from the base center (e.g.
    # coffeemug at x=-0.35) have less headroom before the Panda arm fully extends.
    panda_max_reach = 0.855  # approximate max reach from link0 center (m)
    obj_horiz_dist_sq = float(obj_pos[0] ** 2 + obj_pos[1] ** 2)
    max_reachable_lift_z = robot_base_z + np.sqrt(
        max(panda_max_reach ** 2 - obj_horiz_dist_sq, 0.04)
    )
    desired_lift_z = max_obstacle_top + desired_clearance
    lift_z = min(desired_lift_z, max_reachable_lift_z - 0.06)  # 6 cm workspace safety margin
    lift_z = max(lift_z, max_obstacle_top + 0.08)              # at least 8 cm above obstacles

    carry_z = max_obstacle_top + 0.10     # 10 cm above tallest obstacle; transport uses
    # downward=False so far-reach goals remain reachable via tilted arm configurations
    place_offset = obj_half_h + 0.04  # EE above surface with margin for controller overshoot

    # Tight-clutter detection: tall nearby obstacles force approach_min_z high
    # above the object. The primitive hemisphere (30–60° lateral) fails in that
    # regime — the arm can't come in from the side without clipping obstacles,
    # and the descend wrist ends up misaligned with the object axes. Treat such
    # primitives like meshes: near-vertical approach + downward descend.
    tight_primitive_clutter = (
        is_primitive and (max_obstacle_top + 0.02 - obj_pos[2]) > 0.15
    )
    use_lateral_approach = is_primitive and not tight_primitive_clutter

    heights = {
        "table_z": table_z,
        "obj_half_h": obj_half_h,
        "grasp_h": grasp_offset,            # above object CENTER for grasp
        "approach_radius": 0.08,             # base radius for hemisphere sampling
        "approach_radius_var": 0.04,         # variation in approach radius
        "approach_min_elev": np.pi / 6 if use_lateral_approach else 0.0,
        "approach_max_elev": np.pi / 3 if use_lateral_approach else approach_max_elev,
        "approach_min_z": (max_obstacle_top + 0.02) if use_lateral_approach else lift_z,
        "lift_z": lift_z,                    # absolute z for lift target
        "carry_z": carry_z,                  # absolute z for transport
        "place_offset": place_offset,        # above placement surface
        "max_obstacle_top": max_obstacle_top,
        "robot_base_z": robot_base_z,
        "is_primitive": is_primitive,
        "use_lateral_approach": use_lateral_approach,
        "shelf_top_z": shelf_top_z,
    }

    approach_min_z_print = (max_obstacle_top + 0.02) if use_lateral_approach else lift_z
    approach_max_elev_print = np.pi / 3 if use_lateral_approach else approach_max_elev
    print(f"  Derived heights: table_z={table_z:.3f} max_obstacle_top={max_obstacle_top:.3f} "
          f"lift_z={lift_z:.3f} carry_z={carry_z:.3f} approach_min_z={approach_min_z_print:.3f} "
          f"approach_max_elev={np.degrees(approach_max_elev_print):.1f}°"
          + (" [primitive: lateral approach 30-60°, descend unconstrained]" if is_primitive else ""))
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
# Segment planning
# ---------------------------------------------------------------------------

_SEGMENT_DEFS = [
    # (name, action_after, task_type, downward_constraint)
    # Downward constraint on approach/descend/lift/place keeps the gripper pointing
    # down during pick and place, preventing the arm from sweeping sideways into the
    # table. Transport uses no orientation constraint so far-reach goals on large
    # tables (e.g. scene_kitchen y ≈ -0.85) remain reachable.
    ("approach",  None,      "transit", True),
    ("descend",   "grasp",   "pick",    True),
    ("lift",      None,      "pick",    True),
    ("transport", None,      "transit", False),
    ("place",     "release", "place",   True),
]


def _compute_waypoints(obj_pos, goal_xy, heights, place_height, approach_pos,
                       grasp_pos=None):
    """Compute the 5 EE target positions for a full mission.

    Args:
        obj_pos: Object world position (3,)
        goal_xy: Placement (x, y) tuple
        heights: Dict from _derive_scene_heights()
        place_height: "table_surface" or "on_target" (from task YAML)
        approach_pos: Pre-sampled approach position
        grasp_pos: Optional explicit grasp TCP position (from GraspGen).
            If None, falls back to obj_pos + grasp_h offset.
    """
    gx, gy = goal_xy
    if grasp_pos is not None:
        grasp = np.asarray(grasp_pos, dtype=float).copy()
    else:
        grasp = obj_pos.copy();  grasp[2] += heights["grasp_h"]
    lift = obj_pos.copy();   lift[2] = heights["lift_z"]
    transport = np.array([gx, gy, heights["carry_z"]])

    if place_height == "on_target":
        place_z = heights["max_obstacle_top"] + heights["place_offset"]
    elif place_height == "on_shelf":
        if heights.get("shelf_top_z") is None:
            raise ValueError("place_height: on_shelf requires a geom named 'top_shelf_geom' in the scene XML")
        place_z = heights["shelf_top_z"] + heights["place_offset"]
    else:
        place_z = heights["table_z"] + heights["place_offset"]
    place = np.array([gx, gy, place_z])

    return [approach_pos, grasp, lift, transport, place]


def _plan_full_mission(planner, obj_pos, goal_xy, heights, place_height, seed_offset,
                       approach_pos, segment_defs=None, grasp_pos=None, grasp_quat=None):
    """Plan all 5 segments of a full pick-and-place mission.

    When ``grasp_quat`` is provided, the approach/descend/lift segments use
    the sampled 6-DoF orientation instead of the downward-pointing default.

    Returns list of segment dicts or None on planning failure.
    """
    if segment_defs is None:
        segment_defs = _SEGMENT_DEFS
    targets = _compute_waypoints(obj_pos, goal_xy, heights, place_height, approach_pos,
                                 grasp_pos=grasp_pos)
    segments = []

    # Segments that inherit the sampled grasp orientation. The gripper must
    # hold its orientation through the entire carry — rotating mid-transport
    # would slip the object out of the fingers. Only ``place`` reverts to the
    # downward constraint so the object is set down flat.
    _GRASP_QUAT_SEGMENTS = {"approach", "descend", "lift", "transport"}

    planner._set_home_position()

    for i, (seg_name, action_after, task_type, downward) in enumerate(segment_defs):
        target_pos = targets[i]
        planner.seed = seed_offset + i * 10

        quat = grasp_quat if (grasp_quat is not None and seg_name in _GRASP_QUAT_SEGMENTS) else None
        ok = planner.plan_to_ee_pose(target_pos, task_type=task_type,
                                     use_downward_constraint=downward,
                                     target_quat_wxyz=quat)
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
             out_dir=None, max_retries=5, strict_attach=False):
    scene_xml = scene_xml or f"scenes/{scene}/scene.xml"
    robot_xml = robot_xml or _detect_robot_xml(scene_xml)
    out_dir = out_dir or f"scenes/{scene}/trajs"
    os.makedirs(out_dir, exist_ok=True)

    tasks_yaml = load_tasks(scene)
    if task not in tasks_yaml["tasks"]:
        raise ValueError(
            f"Unknown task '{task}' for scene '{scene}'. "
            f"Available: {list(tasks_yaml['tasks'])}"
        )
    task_def = tasks_yaml["tasks"][task]
    grasped_object = get_grasped_object(task_def, tasks_yaml.get("grasped_object", "object3"))
    obstacle_centers = [tuple(c) for c in tasks_yaml["obstacles"]["centers"]]
    min_clearance = tasks_yaml["obstacles"].get("min_clearance", 0.06)
    sampler = make_goal_sampler(task_def, obstacle_centers, min_clearance)
    place_height = task_def.get("place_height", "table_surface")

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

    # Load GraspGen-based 6-DoF sampler for mesh pick targets. Primitive picks
    # keep the analytic top/side grasp — they don't need mesh inference.
    grasp_sampler = None
    mesh_path = None
    if not heights.get("is_primitive"):
        mesh_path = _resolve_mesh_path(scene_xml, grasped_object)
        if mesh_path is not None:
            try:
                grasp_sampler = GraspSampler()
                print(f"Using GraspGen 6-DoF sampler for mesh: {mesh_path}")
            except (FileNotFoundError, KeyError) as e:
                print(f"GraspGen cache unavailable ({e}); falling back to analytic grasp.")
                grasp_sampler = None

    # For primitive objects (box/cylinder) in open scenes, use unconstrained
    # descend so the IK can find lateral side-grasp configurations. Top-down
    # descent to coffeemug height (~0.860) consistently fails because lower
    # arm links hit the table. BUT in tight-clutter scenes (e.g. scene_grocery
    # with tall YCB neighbors), lateral approach can't thread past obstacles
    # — fall back to downward descend, same as mesh-style picks.
    if heights.get("use_lateral_approach"):
        seg_defs = [
            ("approach",  None,      "transit", True),
            ("descend",   "grasp",   "pick",    False),   # unconstrained: lateral side grasp
            ("lift",      None,      "pick",    True),    # downward: reorient to secure grip during lift
            ("transport", None,      "transit", False),
            ("place",     "release", "place",   True),
        ]
    else:
        seg_defs = _SEGMENT_DEFS
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

            grasp_pos = None
            grasp_quat = None
            grasp_meta = {
                "source": "analytic",
                "grasp_id": None,
                "confidence": None,
                "approach_axis": None,
                "ik_attempts": 0,
                "retry_count": attempt,
            }
            if grasp_sampler is not None:
                # Sample a ranked list of 6-DoF grasp candidates in world frame,
                # then IK-screen each until one is reachable. Avoids burning the
                # outer retry budget on grasps that can't be reached.
                T_obj_world = np.eye(4)
                T_obj_world[:3, 3] = obj_pos
                grasp_seed = seed + traj_idx * 1000 + attempt
                candidates = grasp_sampler.sample_ranked(
                    mesh_path, T_obj_world, seed=grasp_seed, n=8
                )
                chosen = None
                ik_attempts = 0
                for cand in candidates:
                    cand_tcp, cand_quat, cand_axis, cand_gid, cand_conf = cand
                    ik_attempts += 1
                    if planner.check_ik_feasibility(cand_tcp, cand_quat, max_attempts=8):
                        chosen = cand
                        break
                if chosen is None:
                    print(f"      no IK-feasible grasp in top {len(candidates)} "
                          f"candidates; retrying with new seed")
                    continue
                grasp_pos, grasp_quat, approach_axis, grasp_id, confidence = chosen
                # GraspGen's +Z is the gripper's advance direction, so
                # approach_axis.z is negative for top-down grasps. Subtracting
                # along approach_axis places the waypoint "behind" the grasp,
                # i.e. above the object for top-down approaches.
                approach_pos = grasp_pos - 0.12 * approach_axis
                approach_pos[2] = max(approach_pos[2], heights["table_z"] + 0.02)
                grasp_meta = {
                    "source": "graspgen",
                    "grasp_id": grasp_id,
                    "confidence": float(confidence),
                    "approach_axis": approach_axis.tolist(),
                    "ik_attempts": ik_attempts,
                    "retry_count": attempt,
                }
            else:
                approach_pos = _sample_approach_pos(obj_pos.copy(), heights, rng)
            print(f"      attempt {attempt+1}/{max_retries}  "
                  f"goal=({goal_xy[0]:.3f}, {goal_xy[1]:.3f})  "
                  f"approach={approach_pos.round(3)}"
                  + (f"  grasp={grasp_pos.round(3)}"
                     f" [{grasp_meta['grasp_id']} conf={grasp_meta['confidence']:.3f}"
                     f" ik_tries={grasp_meta['ik_attempts']}]"
                     if grasp_pos is not None else ""))

            segments = _plan_full_mission(planner, obj_pos.copy(), goal_xy, heights,
                                          place_height, seed_offset, approach_pos,
                                          segment_defs=seg_defs,
                                          grasp_pos=grasp_pos, grasp_quat=grasp_quat)
            if segments is None:
                continue

            total_wps = sum(len(s["trajectory"]) for s in segments)
            goal_pos = _compute_waypoints(obj_pos.copy(), goal_xy, heights, place_height,
                                          approach_pos, grasp_pos=grasp_pos)[-1]

            # Save to temp file for physics verification
            tmp_path = out_path + ".tmp"
            pkl_data = {
                scene: {
                    "task_id": task,
                    "traj_id": traj_idx,
                    "grasped_object": grasped_object,
                    "goal_pos": goal_pos,
                    "segments": segments,
                    "grasp_meta": grasp_meta,
                }
            }
            with open(tmp_path, "wb") as f:
                pickle.dump(pkl_data, f)

            # Physics verification — replay and check collisions, grasp, place
            print(f"      Verifying via physics replay ...")
            verify_kwargs = {"strict_attach": strict_attach}
            # Tall objects (half-height > 5 cm) can't complete a full 10 cm lift
            # at the workspace edge — accept a 6 cm clearance. They also tip on
            # release, so the post-settle XY can drift by ~their half-length from
            # the release point; widen place tolerance to match.
            if heights.get("obj_half_h", 0.0) > 0.05:
                verify_kwargs["grasp_z_margin"] = 0.06
                verify_kwargs["place_xy_tolerance"] = 0.10
            vr = verify_trajectory(scene_xml, tmp_path, **verify_kwargs)
            if not vr.passed:
                print(f"      REJECTED: {vr.details}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                continue

            os.rename(tmp_path, out_path)
            seg_summary = " → ".join(f"{s['name']}({len(s['trajectory'])}wp)" for s in segments)
            print(f"      Verified OK. Saved {total_wps} waypoints: {seg_summary}")
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
    parser.add_argument("--scene", required=True,
                        help="Scene name (must have scenes/<scene>/tasks.yaml)")
    parser.add_argument("--task", required=True,
                        help="Task name from tasks.yaml, or 'all' to run every task")
    parser.add_argument("--n_trajs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene_xml", default=None)
    parser.add_argument("--robot_xml", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--strict-attach", action="store_true", dest="strict_attach",
                        help="Require finger-object contact before engaging GraspLock; "
                             "trajectories without real contact are rejected as grasp failures.")
    args = parser.parse_args()

    if args.task == "all":
        from planner.tasks import load_tasks as _load_tasks
        _yaml = _load_tasks(args.scene)
        all_tasks = _yaml["tasks"]
        tasks_to_run = [n for n, tdef in all_tasks.items() if tdef.get("enabled", True)]
        skipped = [n for n in all_tasks if n not in tasks_to_run]
        if skipped:
            print(f"Skipping disabled tasks: {skipped}")
        print(f"Running {len(tasks_to_run)} tasks: {tasks_to_run}")
        for t in tasks_to_run:
            generate(scene=args.scene, task=t, n_trajs=args.n_trajs, seed=args.seed,
                     scene_xml=args.scene_xml, robot_xml=args.robot_xml,
                     out_dir=args.out_dir, max_retries=args.max_retries,
                     strict_attach=args.strict_attach)
    else:
        generate(scene=args.scene, task=args.task, n_trajs=args.n_trajs, seed=args.seed,
                 scene_xml=args.scene_xml, robot_xml=args.robot_xml,
                 out_dir=args.out_dir, max_retries=args.max_retries,
                 strict_attach=args.strict_attach)
