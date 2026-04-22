# FailBench Pipeline Architecture

This document is the reference for how trajectory generation, failure injection, and data serialization actually work in the current codebase. Use it when you need to touch the code — not just run it. For user-facing command recipes, see [data_generation.md](data_generation.md) and the [README](../README.md).

---

## Module map

```
scripts/
  generate_task_trajs.py        Stage 1 orchestrator (IK + RRT)
  verify_task_trajs.py          Stage 2 physics-audit of pkls
  play_task_trajs.py            Stage 3 MuJoCo viewer + depth windows
  test_pipeline.py              Stage 4 end-to-end smoke test
  precompute_grasps.py          GraspGen cache builder

planner/
  tasks.py                      tasks.yaml loader + goal samplers
  grasp_sampler.py              GraspGen candidate loader + IK pre-filter helper
  grasp_lock.py                 Kinematic sticky-grip attach/release
  kinematics/inverse_kinematics.py   Stateless Mink-based IKSolver (used by RRT)
  algorithms/RRTplanner.py      JointSpaceRRTConnect (bidirectional joint-space RRT)
  algorithms/abstract_planner.py     AbstractRRTPlanner base, collision checking
  examples/pick_and_place_safety_L2.py   PandaPickAndPlace_L2 — main planner entry
  utils/trajectory_interpolation.py      Cubic/linear densification
  experiments/
    config.py                   ExperimentConfig, FailureConfig, FailureMode
    runner.py                   ExperimentRunner — segmented replay + failure injection
    manager.py                  BatchExperimentManager, save_sample_npz, manifest I/O
    data_capture.py             OffscreenRenderer, ContactExtractor, RobotStateCollector

failure_injection/
  agressive_injector.py         AggressiveFailureInjector (5 failure modes)
  collision_estimation.py       Body-vs-body contact force estimation
  explain.md                    Physics notes for each failure mode
```

---

## Stage 1: Trajectory generation

Entry point: `scripts/generate_task_trajs.py`.

```
load_tasks(scene)                     # planner/tasks.py
   │
   ▼
PandaPickAndPlace_L2(scene_xml, robot_xml)   # planner/examples/pick_and_place_safety_L2.py
   │
   ├─ _derive_scene_heights(model, data, object_name)
   │     lift_z, carry_z, place_z, table_z ← model geometry
   │
   ├─ for traj_id in range(n_trajs):
   │     goal_xy = sampler(rng)                       # from planner/tasks.make_goal_sampler
   │     approach_pos = _sample_approach_pos(...)      # hemisphere around object
   │     grasp_cands = GraspSampler.sample_ranked()    # GraspGen or analytic top-down
   │     for cand in grasp_cands:
   │         if planner.check_ik_feasibility(cand):
   │             break                                 # pick first IK-feasible candidate
   │
   │     for segment in [approach, descend, lift, transport, place]:
   │         planner.plan_to_ee_pose(tcp, quat, task_type=...)
   │             │
   │             ├─ IKSolver.solve(target, seed=current_config)   # Mink
   │             └─ JointSpaceRRTConnect.plan(current_cfg, goal_cfg)
   │                   ├─ sample_random_config()
   │                   ├─ CollisionEstimator.check()   # arm vs env geoms
   │                   └─ connect trees → shortcut → waypoints
   │
   └─ verify_trajectory(...) → pass/fail; retry up to --max_retries
         │
         ▼
      pkl: {task_id, traj_id, grasped_object, goal_pos, segments[5], grasp_meta}
```

Key implementation details:

**Adaptive heights.** `_derive_scene_heights()` in `scripts/generate_task_trajs.py` walks the MuJoCo model at runtime:
- `table_z`: largest horizontal box geom in any body matching `*table*`.
- Obstacle clearance: tallest non-robot geom within 30 cm of the pick object.
- `carry_z = obstacle_clearance + 0.18 m`, `lift_z = obstacle_clearance + margin`, `place_z = table_z + object_half_h + 0.04`.

Nothing is hardcoded per-scene. Changing scene geometry automatically reshapes the plans.

**Hemisphere approach sampling.** `_sample_approach_pos()` samples around the object:
- Azimuth uniform 0–2π, elevation 0–30° from vertical, radius 0.08 ± 0.04 m.
- `z` clamped to ≥ tallest obstacle + clearance.
- Combined with a **downward EE orientation constraint** on every segment (enforced inside `plan_to_ee_pose`), this keeps the gripper pointing down throughout the mission and drives joint-configuration diversity for the failure planner's training signal.

**Segment schedule.** `_SEGMENT_DEFS` lists the five segments and their `task_type` / `use_downward_constraint` flags. Segments are planned sequentially, each seeded from the previous segment's end configuration; the gripper is actuated between segments via `action_after ∈ {"grasp", "release"}`.

**Grasp selection.**
- For primitive pick targets (e.g., `scene_level2` green block), `grasp_sampler.py` falls back to an analytic top-down grasp.
- For mesh pick targets, it loads precomputed 6-DoF poses from `cache/graspgen/<sha>.yml` (see [graspgen_setup.md](graspgen_setup.md)), filters by confidence and Z, and returns a shuffled `n`-candidate list.
- The generator calls `PandaPickAndPlace.check_ik_feasibility()` on each candidate and picks the first IK-feasible one. This avoids wasting an RRT plan on a grasp the arm can't reach.

**Strict attach.** Invoked by `--strict-attach`. `GraspLock.attach_strict` (in `planner/grasp_lock.py`) refuses to kinematically bind the object to the hand unless a finger body is in genuine contact. This surfaces fake grasps that blind attach would have silently masked.

**pkl format.**
```python
{
  "<scene_name>": {
    "task_id": str,
    "traj_id": int,
    "grasped_object": str,
    "goal_pos": np.ndarray(3),
    "segments": [
      {"name": "approach",  "trajectory": List[np.ndarray(9,)], "action_after": None},
      {"name": "descend",   "trajectory": [...], "action_after": "grasp"},
      {"name": "lift",      "trajectory": [...], "action_after": None},
      {"name": "transport", "trajectory": [...], "action_after": None},
      {"name": "place",     "trajectory": [...], "action_after": "release"},
    ],
    "grasp_meta": {
      "source": "graspgen" | "analytic",
      "grasp_id": str | None,
      "confidence": float | None,
      "approach_axis": [x, y, z] | None,
      "ik_attempts": int,
      "retry_count": int,
    },
  }
}
```

Each waypoint is a 9-vector `[q0..q6, gripper_left, gripper_right]`.

---

## Motion planner internals

**IK.** `planner/kinematics/inverse_kinematics.py` wraps [Mink](https://github.com/kevinzakka/mink) in a stateless `IKSolver` (OSQP backend, configurable position/orientation tolerance, posture regularization, damping). It is used two ways:

1. **From the RRT planner** for single-query goal-config resolution and for `get_random_valid_config()` during tree expansion.
2. **Indirectly through `PandaPickAndPlace.plan_to_ee_pose()`**, which runs a loop: sample an IK solution → check joint limits and workspace constraints → feed into RRT. `plan_to_ee_pose` advances an RNG across attempts (does *not* re-seed each attempt — earlier versions had this bug).

The API-level reference for `IKSolver` / `EndEffectorTarget` / `IKConfig` lives in [ik_docs.md](ik_docs.md). That document is accurate for the class itself; note however that the current data-generation call graph goes through `plan_to_ee_pose`, which wraps `IKSolver.solve` with workspace and orientation constraints specific to pick-and-place.

**RRT.** `planner/algorithms/RRTplanner.py` provides:
- `JointSpaceRRT` — single-tree RRT with a 30%-biased random sampler that sprinkles Gaussian noise around existing tree nodes.
- `JointSpaceRRTConnect` — bidirectional, used by default in the generator. Deterministic under `--seed`; `rng` is initialized inside `plan()`.

Generator parameters (see `PandaPickAndPlace.__init__`): `step_size=0.02`, `max_iterations=3000`, max 3 IK goals tried per segment. Collision checks use `CollisionEstimator` / `OptimizedCollisionEstimator` (in `failure_injection/`) over all robot-vs-env geom pairs, including unnamed table geoms.

**Downward EE constraint.** Every `plan_to_ee_pose` call in the generator passes `use_downward_constraint=True` for approach, descend, transport, and place (not lift). This is the single most important collision-avoidance lever — without it, the arm sweeps sideways through the table during long transports.

**Dense interpolation.** At replay time (both `play_task_trajs.py` and `ExperimentRunner`), `trajectory_interpolation.py` densifies each segment to 100 points (cubic spline by default, linear fallback) and `ExperimentRunner` advances 8 sim steps per dense point. Tight tracking keeps the executed path close to the RRT plan.

---

## Stage 2: Verification

`scripts/verify_task_trajs.py` → `planner.experiments.verify_trajectory` (same code path used inline during generation). Each pkl is replayed headlessly through MuJoCo with:
- Arm-environment collision force threshold **5 N** (default; `--collision_threshold` to override).
- Grasp success: object `z` after the lift segment must exceed `table_z + object_half_h + 2 cm`.
- Place accuracy: object xy within **5 cm** of `goal_pos` after release (`--place_tolerance` to override).

`--strict-attach` enforces real finger-object contact during the grasp event. `--delete_failures` removes pkls that fail any check. Tall-primitive picks (e.g., `scene_workshop` cylinder) use relaxed lift/place tolerances — see CLAUDE.md notes.

---

## Stage 3: Playback

`scripts/play_task_trajs.py` reuses the same interpolation + gripper-action logic as the runner, but renders into the MuJoCo `passive_viewer` instead of offscreen buffers. `--show_depth --depth_cam <cam1,cam2>` spawns one OpenCV window per camera using `OffscreenRenderer.render_depth()` (TURBO colormap, 0.05–2.0 m range).

The arm is pre-positioned at the trajectory's first config before each settle so the grasped object isn't knocked by a zero-config arm swing.

---

## Stage 4: Failure injection and dataset output

Entry point today: `scripts/test_pipeline.py` (single-trial smoke test). Bulk generator `scripts/generate_dataset_tasks.py` is TODO — see CLAUDE.md Next Steps.

```
ExperimentConfig(scene_xml, robot_xml, trajectory_file,
                 task_id, traj_id, fail_fraction, failure_configs=[...],
                 extra_cameras=["ee_cam"], ...)
        │
        ▼
ExperimentRunner.run()                        # planner/experiments/runner.py
  1. Load scene model + pkl trajectory
  2. Reset scene; pre-position arm at segment[0].trajectory[0]
  3. Attach grasped object via GraspLock (strict or blind)
  4. Densify each segment (cubic spline, 100 pts × 8 sim steps)
  5. Replay segments sequentially:
       - At action_after == "grasp"/"release": actuate gripper, settle
       - Stop replay at `traj_progress == fail_fraction`
  6. Capture pre-failure RGB+depth from front_cam and each extra camera,
     plus robot state (qpos/qvel/ee_pos/gripper_ctrl)
  7. For each FailureConfig:
       - Fork MjData
       - AggressiveFailureInjector.inject(mode, joint_names, grip_value)
       - Step for `post_failure_settle_steps` (default 500)
       - ContactExtractor harvests contacts (threshold 1 N, robot-self filtered)
  8. Return DataSample
        │
        ▼
save_sample_npz(sample, path)                 # planner/experiments/manager.py
append_manifest([row], path)
```

### Failure modes (`planner/experiments/config.py`)

| `FailureMode` | Effect |
|---|---|
| `GRIPPER_OPEN` | Gripper control forced to 0 — object falls |
| `SLIPPERY_GRIP` | Gripper control set to `grip_value` (default 180/255) — partial grip |
| `SINGLE_JOINT` | Named joint's actuator gain, stiffness, damping, friction zeroed |
| `MULTI_JOINT` | Same, applied to multiple named joints |
| `ALL_JOINTS` | Applied to all seven arm joints — total collapse |

Physics details per mode: [`failure_injection/explain.md`](../failure_injection/explain.md).

Default probability mix (from `config._default_failures`): `gripper_open 0.35`, `slippery_grip 0.25`, `single_joint(j4) 0.15`, `single_joint(j6) 0.10`, `multi_joint(j4,j6) 0.10`, `all_joints 0.05`.

### Canonical fail fractions

`[0.1, 0.25, 0.4, 0.55, 0.7, 0.85]` over the full mission timeline. `fail_fraction=None` picks one at random per trial.

### npz schema (current)

| Key | Shape | dtype | Description |
|---|---|---|---|
| `pre_rgb` | (H, W, 3) | uint8 | Pre-failure front_cam RGB |
| `pre_depth` | (H, W) | float32 | Pre-failure front_cam depth (if captured) |
| `<cam>_rgb` | (H, W, 3) | uint8 | Per extra camera in `extra_cameras` |
| `<cam>_depth` | (H, W) | float32 | Per extra camera |
| `post_rgb` | (H, W, 3) | uint8 | Optional — final post-failure frame |
| `pre_qpos` | (7,) | float64 | Arm joint positions at failure |
| `pre_qvel` | (7,) | float64 | Arm joint velocities |
| `pre_ee_pos` | (3,) | float64 | End-effector position |
| `pre_gripper_ctrl` | (1,) | float64 | Gripper control value |
| `pre_qvel_norm` | (1,) | float64 | \|\|qvel\|\| |
| `contact_positions` | (N, 3) | float32 | 3D contact positions across all failure modes |
| `contact_forces` | (N, 6) | float32 | Force (3) + torque (3) per contact |
| `contact_geom_pairs` | (N, 2) | int32 | Geom IDs |
| `contact_failure_id` | (N,) | int32 | Index into `failure_modes` |
| `failure_modes` | (M,) | str | Per-run failure mode names |
| `failure_probs` | (M,) | float32 | Sampling probabilities |
| `impacted_geom_ids` | (K,) | int32 | Unique geoms contacted across all modes |
| `task_id` | (1,) | str | |
| `traj_id` | (1,) | int32 | |
| `traj_progress` | (1,) | float32 | Fraction ∈ [0, 1] where failure fired |
| `seed` | (1,) | int32 | |

### Manifest CSV columns

```
experiment_id, task_id, traj_id, trajectory_file, seed,
traj_progress, num_contacts, num_failure_modes, had_any_collision,
impacted_geom_ids, pre_qvel_norm, npz_file
```

Output directory is chosen by the caller (no fixed v9-style path). `scenes/*/datasets/` is gitignored; datasets are reproducible from the committed trajectory pkls.

---

## Camera conventions

Only `front_cam` (scene-fixed front-right view, fovy=45°) and `ee_cam` (wrist-mounted on link7, fovy=90°) are used. This is deliberate: both have lab-reproducible physical counterparts (tripod + RealSense on flange). An overhead camera existed in earlier revisions but has been removed.

`ContactProjector` in `data_capture.py` projects 3D contact positions into 2D pixel coords for any named camera. **Gotcha:** for named cameras, the Y-row of `cam_rot` must be negated before projection — `validate()` catches this.

---

## Concurrency model

`BatchExperimentManager` in `manager.py` supports multiprocess batches (`num_workers`). Each worker spawns its own `ExperimentRunner` with a separate MuJoCo model. No shared state; deterministic per-`seed` outputs.

Trajectory *generation* is currently single-process — RRT is cheap enough that wall-clock is dominated by verification replay, not planning.

---

## Deterministic reproducibility

- `generate_task_trajs.py --seed N`: same N → same trajectories (goal sampling, approach sampling, IK tie-breaks, RRT).
- `ExperimentConfig.seed`: controls failure sampling (when `failure_sample_mode="sample"`) and randomized fail-fraction selection (when `fail_fraction=None`).
- Task ordering under `--task all` is `tasks.yaml` insertion order.

Changing scene geometry invalidates prior trajectory pkls, since heights are model-derived. Always regenerate after scene XML edits.

---

## Related docs

- [README](../README.md) — install + runnable recipes
- [data_generation.md](data_generation.md) — tasks.yaml deep reference and stage-by-stage CLI
- [graspgen_setup.md](graspgen_setup.md) — GraspGen install + cache details
- [ik_docs.md](ik_docs.md) — `IKSolver` API reference (Mink wrapper)
- [`failure_injection/explain.md`](../failure_injection/explain.md) — per-mode failure physics
