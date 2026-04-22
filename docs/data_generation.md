# Data Generation Guide

This document describes how to generate trajectory datasets for FailBench, from trajectory planning through failure injection and npz output.

---

## Overview

Data generation has three stages:

```
1. generate_task_trajs.py   →  .pkl trajectory files per task
2. verify_task_trajs.py     →  physics-validate and prune bad trajectories
3. generate_dataset_tasks.py (TODO)  →  failure injection → .npz + manifest.csv
```

Stages 1 and 2 are integrated: `generate_task_trajs.py` automatically runs physics verification before saving each trajectory. `verify_task_trajs.py` is available separately to audit existing files.

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

## Stage 3: Visual Playback

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

| Scene | Tasks | Trajs per task | Total | Status |
|---|---|---|---|---|
| scene_level2 | 12 | 10 | 120 | Complete, verified |
| scene_kitchen | 12 | — | — | In progress |
| scene_workshop | 12 | — | — | TODO |
| scene_grocery | 12 | — | — | TODO |
| scene_cluttered | 12 | — | — | TODO |
