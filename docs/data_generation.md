# Data Generation Guide

This document describes how to generate trajectory datasets for FailBench, from trajectory planning through failure injection and npz output.

---

## Overview

Data generation has three stages:

```
1. generate_task_trajs.py      →  .pkl trajectory files per task
2. verify_task_trajs.py        →  physics-validate and prune bad trajectories
3. generate_dataset_tasks.py   →  failure injection → .npz + manifest.csv
```

Stages 1 and 2 are integrated: `generate_task_trajs.py` automatically runs physics verification before saving each trajectory. `verify_task_trajs.py` is available separately to audit existing files. Stage 3 consumes the verified pkls and produces the failure-injection dataset used by the downstream contact-prediction model.

---

## Stage 1: Generate Trajectories

### Basic Usage

```bash
# Generate 10 trajectories for one task
python scripts/generate_task_trajs.py \
    --scene scene_level2 \
    --task clean_nominal \
    --n_trajs 10 \
    --seed 0

# Generate all tasks for a scene at once
python scripts/generate_task_trajs.py \
    --scene scene_level2 \
    --task all \
    --n_trajs 10 \
    --seed 0
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--scene` | required | Scene name (must have `scenes/<scene>/tasks.yaml`) |
| `--task` | required | Task name from tasks.yaml, or `all` |
| `--n_trajs` | 10 | Number of trajectories to generate |
| `--seed` | 0 | RNG seed for reproducibility |
| `--max_retries` | 5 | Max planning+verification attempts per trajectory |
| `--scene_xml` | auto | Override scene XML path |
| `--robot_xml` | auto | Override robot XML path (normally auto-detected) |
| `--out_dir` | auto | Output directory (default: `scenes/<scene>/trajs/`) |

### Output

One `.pkl` file per trajectory: `scenes/<scene>/trajs/<scene>_<task>_NN.pkl`

Each pkl contains:
```python
{
  "scene_level2": {
    "task_id":        "clean_nominal",
    "traj_id":        0,
    "grasped_object": "object3",
    "goal_pos":       array([x, y, z]),
    "segments": [
      {"name": "approach",  "trajectory": [...], "action_after": None},
      {"name": "descend",   "trajectory": [...], "action_after": "grasp"},
      {"name": "lift",      "trajectory": [...], "action_after": None},
      {"name": "transport", "trajectory": [...], "action_after": None},
      {"name": "place",     "trajectory": [...], "action_after": "release"},
    ]
  }
}
```

### Physics Verification

Every trajectory is automatically physics-verified before saving. If a trajectory fails any check, it is discarded and the planner retries:

- **Arm-environment collision**: robot links vs table/obstacles, threshold 5N
- **Grasp success**: object lifted above table + 2cm margin after the lift segment
- **Place accuracy**: object lands within 5cm xy of goal after release

Use `--max_retries 10` for harder tasks (e.g., long diagonal carries).

---

## Stage 2: Verify Existing Trajectories

Use `verify_task_trajs.py` to audit or clean up existing pkl files.

```bash
# Verify all trajectories for a task (report only)
python scripts/verify_task_trajs.py --scene scene_level2 --task sort_nominal

# Verify all tasks for a scene
python scripts/verify_task_trajs.py --scene scene_level2

# Verify and delete failures
python scripts/verify_task_trajs.py --scene scene_level2 --delete_failures

# Verify a single file
python scripts/verify_task_trajs.py \
    --traj_file scenes/scene_level2/trajs/scene_level2_stack_nominal_00.pkl
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--scene` | — | Verify all trajectories for this scene |
| `--task` | None | Filter to a specific task (use with `--scene`) |
| `--traj_file` | — | Verify a single pkl file |
| `--delete_failures` | False | Delete pkl files that fail verification |
| `--collision_threshold` | 5.0 | Arm-env collision force in N |
| `--place_tolerance` | 0.05 | Max xy distance from goal in m |

### Sample Output

```
[PASS] scene_level2_clean_nominal_00.pkl  collision=0.0N  place_err=0.009m  grasp=ok  | all checks passed
[FAIL] scene_level2_clean_nominal_01.pkl  collision=1157.9N  place_err=0.110m  grasp=ok  | place error: 0.110m; arm collision: 1157.9N

8 passed, 1 failed out of 9 total
```

---

## Visual Playback (optional QA)

Play trajectories in the MuJoCo viewer to visually inspect quality.

```bash
# Play all trajectories for a task
python scripts/play_task_trajs.py --scene scene_level2 --task clean_nominal

# Play all tasks for a scene (120 trajectories)
python scripts/play_task_trajs.py --scene scene_level2

# Play a single file
python scripts/play_task_trajs.py \
    --traj_file scenes/scene_level2/trajs/scene_level2_stack_nominal_00.pkl
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--scene` | — | Play all trajectories for this scene |
| `--task` | None | Filter to a specific task |
| `--traj_file` | — | Play a single pkl file |
| `--hold` | 2.0 | Seconds to hold final pose between trajectories |
| `--speed` | 1.0 | Playback speed multiplier |
| `--interp_points` | 100 | Interpolation density per segment |

Close the viewer window to stop playback early.

---

## Stage 3: Failure-Injection Dataset

The output of this stage is the labeled dataset used to train the contact/failure model. For each verified trajectory, failures are injected at K stratified points along the full pick-and-place mission (approach / descend / lift / transport / place), and all 6 default failure modes are forked from each pre-failure state.

### Quick start

```bash
# Preflight — 1 scene, 2 tasks, 100 trials, ~4 min on 4 workers
python scripts/generate_dataset_tasks.py \
    --scenes scene_level2 \
    --tasks clean_nominal stack_nominal \
    --k_per_segment 1 \
    --output_version v10_preflight \
    --workers 4

# Full PoC run — 5 scenes, 54 enabled tasks, ~16,775 trials, ~3 h on 8 workers, ~41 GB
python scripts/generate_dataset_tasks.py \
    --k_per_segment 5 \
    --output_version v10 \
    --workers 8
```

### Output layout

```
datasets/<output_version>/<scene>/<task>/
├── manifest.csv          — one row per trial: experiment_id, task_id, traj_id, traj_progress,
│                           num_contacts, num_failure_modes, had_any_collision, impacted_geom_ids,
│                           pre_qvel_norm, npz_file
└── exp_*.npz             — one file per trial
```

### npz schema (per trial)

| Key | Shape | Description |
|---|---|---|
| `pre_rgb`, `pre_depth` | (480, 640, 3) / (480, 640) | Front-cam at the failure instant |
| `ee_cam_rgb`, `ee_cam_depth` | (480, 640, 3) / (480, 640) | Wrist-cam at the failure instant |
| `post_rgb` | (480, 640, 3) | Front-cam after last-fork settle (qualitative only) |
| `pre_qpos`, `pre_qvel` | (7,) | Arm joint positions / velocities at failure |
| `pre_ee_pos`, `pre_gripper_ctrl` | (3,) / (1,) | End-effector position / gripper ctrl signal |
| `traj_progress` | (1,) | Where in [0, 1] the failure was injected |
| `task_id`, `traj_id`, `seed` | scalars | Trial identity |
| `failure_modes`, `failure_probs` | (6,) | Modes forked from this pre-failure state |
| `contact_positions`, `contact_forces`, `contact_geom_pairs` | (N, 3/6/2) | Flat contact rows across all 6 forks |
| `contact_failure_id` | (N,) | Which fork (0..5) each contact came from |
| `impacted_geom_ids` | variable | Unique geoms touched across all forks |

One npz carries six failure outcomes from one pre-failure state, tagged by `contact_failure_id`. Expand to 6 training examples by fanning out on the mode index.

### CLI flags (`generate_dataset_tasks.py`)

| Flag | Default | Description |
|---|---|---|
| `--scenes` | all 5 | Subset of scenes to generate (e.g. `--scenes scene_kitchen scene_grocery`) |
| `--tasks` | all enabled | Optional filter: only these task ids across each scene |
| `--k_per_segment` | 5 | Continuous fail_fraction draws per mission segment. K_total_per_traj = 5 × this (5 segments) |
| `--output_version` | `v10` | Output subdir under `datasets/` |
| `--workers` | 8 | Parallel worker processes (each runs one trial at a time; recommended ≤ CPU cores) |
| `--base_seed` | 20260424 | RNG seed — every trial's seed = base_seed + global_idx |

### Stratification

Each mission segment (approach / descend / lift / transport / place) gets `K_per_segment` independent continuous draws of `fail_fraction ∈ [lo, hi]`, where `(lo, hi)` are the segment's fractional bounds in the concatenated dense timeline (computed per-pkl from sparse-waypoint counts — segments are not equal length in general). At `K_per_segment=5` that's **25 samples per trajectory**, guaranteeing per-phase coverage.

### Throughput and scaling

Measured on a local 4-worker preflight (100 trials, 4.2 min): **~0.40 trials/s on 4 workers**, roughly linear to worker count. Expected full-PoC run (16,775 trials, 8 workers) ≈ **3 h wall time, ~41 GB** at ~2.45 MB/npz compressed.

### Cluster dispatch (single scene per node)

The script works well as an array job by splitting scenes across nodes:

```bash
# Node 1
python scripts/generate_dataset_tasks.py --scenes scene_level2 --output_version v10 --workers 16
# Node 2
python scripts/generate_dataset_tasks.py --scenes scene_kitchen --output_version v10 --workers 16
# ...etc
```

Each invocation writes to its own per-scene subdirectory, so concurrent runs on shared storage don't collide. Use the `enabled` flag in each scene's `tasks.yaml` to toggle which tasks participate under `--tasks all` semantics (default: all enabled).

### Visualization

```bash
jupyter lab notebooks/visualize_dataset.ipynb
```

The notebook inspects any `datasets/<version>/<scene>/<task>/` slice: schema summary, traj_progress / qvel / contact-count distributions, impacted-geom frequency, per-sample pre/ee/post RGB + depth, contact 3D scatter colored by failure mode, front_cam 2D overlay via `planner.experiments.data_capture.ContactProjector`, and a same-pre-state → 6-failure-outcomes grid. Edit the `DATASET_DIR` / `SCENE_XML` variables at the top and re-run all cells to point at a different slice.

### Known runtime behavior

- **`Insufficient arena memory` warnings** in `MUJOCO_LOG.TXT`: occasional, benign. MuJoCo's default constraint arena occasionally falls short during heavy-contact failure cascades (e.g. ALL_JOINTS dropping the arm into a stack); the simulation continues with incomplete constraints.
- **Rare `mj_stackAlloc: out of memory` fatal errors**: small fraction of trials (≈1–2%) crash in-worker under extreme contact loads. The crashed worker's trial is skipped (returns None) and the batch continues — no manifest row is written for it. Expect a 98–99% yield versus the planned trial count.
- **Workers are short-lived** (one trial each via `maxtasksperchild=1`): MuJoCo's GL/compiler global state doesn't accumulate across trials, at the cost of a ~0.5 s process-spawn penalty per trial.

---

## Task Definitions (tasks.yaml)

Tasks are defined in per-scene YAML files at `scenes/<scene>/tasks.yaml`. No Python changes are needed to add new tasks.

### Schema

```yaml
scene: scene_level2
grasped_object: object3          # default pick target

obstacles:
  min_clearance: 0.06            # min distance from goal to any obstacle center
  centers:                       # obstacle XY positions (for clearance checks)
    - [-0.08, -0.27]
    - [ 0.06, -0.27]
    - ...

tasks:
  clean_nominal:
    category: clean
    description: "Pick object3, drop in near-edge clear zone"
    place_height: table_surface  # or "on_target" (elevated onto obstacle top)
    goal:
      type: zones                # or "choice"
      zones:
        - {x: [-0.20, 0.15], y: [-0.19, -0.22]}
      check_clearance: true      # retry if goal too close to obstacle

  stack_nominal:
    category: stack
    place_height: on_target
    goal:
      type: choice
      targets: [[-0.12, -0.36], [0.12, -0.36]]
      jitter: 0.02

  sort_nominal:
    category: sort
    grasped_object: object1      # override default pick target
    place_height: table_surface
    goal:
      type: zones
      zones:
        - {x: [0.23, 0.30], y: [-0.22, -0.35]}
      check_clearance: true
```

### Goal Types

| Type | Description |
|---|---|
| `zones` | Sample uniform (x, y) from a rectangle. If `check_clearance: true`, retries up to 50× to stay ≥ `min_clearance` from obstacles. |
| `choice` | Pick randomly from `targets` list and add uniform jitter in `[-jitter, jitter]`. |

### Place Heights

| Value | Effect |
|---|---|
| `table_surface` | EE placed at `table_z + place_offset` |
| `on_target` | EE placed at `max_obstacle_top + place_offset` (for stacking) |

### Adding a New Task

1. Open `scenes/<scene>/tasks.yaml`
2. Add a new entry under `tasks:` following the schema above
3. Run `generate_task_trajs.py --task <new_task>` — no Python changes needed

### Adding a New Scene

1. Create `scenes/<scene>/scene.xml` (MuJoCo model)
2. Create `scenes/<scene>/tasks.yaml` following the schema above
3. Run `generate_task_trajs.py --scene <scene> --task all`

The planner auto-detects the robot XML from the scene's `<include>` tag and derives all heights (carry height, lift height, place height) at runtime from the model — no hardcoded values needed.

---

## Task Taxonomy

### Semantic Categories

| Category | Description |
|---|---|
| `stack` | Pick object, place on top of an existing obstacle |
| `clean` | Pick object, carry to a designated drop zone |
| `sort` | Pick alternate object, deliver to a specific zone |
| `handover` | Pick object, carry to table edge |

### Geometric Subtasks

| Suffix | What it tests |
|---|---|
| `_nominal` | Short carry, open space |
| `_far` | Long carry, arm extended at delivery |
| `_over` | Goal on far side of obstacle field, cross-table traverse |

---

## Smoke Test

Runs one full cycle: generate → ExperimentRunner → verify npz fields.

```bash
python scripts/test_pipeline.py
```

---

## Current Dataset Status

### Trajectory library (Stage 1+2, complete)

| Scene | Enabled tasks | Verified pkls | Notes |
|---|---|---|---|
| scene_level2 | 12 | 120 | |
| scene_kitchen | 12 | 113 | 4 mesh-pick tasks disabled (low grasp yield) |
| scene_cluttered | 9 | 137 | `sort_*` disabled (workspace-edge alt-pick limits) |
| scene_workshop | 9 | 149 | `sort_*` disabled |
| scene_grocery | 12 | 152 | Unique right-wall + top-shelf geometry; candidate OOD scene |
| **Total** | **54** | **671** | |

### Failure-injection dataset (Stage 3)

| Run | Config | Status |
|---|---|---|
| `v10_preflight` (scene_level2 × 2 tasks) | K_per_segment=1, 4 workers | Verified — 98/100 npz, 4.2 min |
| `v10` (full PoC) | K_per_segment=5, all 54 tasks, 8 workers | Pending cluster run (~16,775 trials, ~3 h, ~41 GB) |
