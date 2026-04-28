# FailBench: Simulating Robot Failures in MuJoCo

**FailBench** is a MuJoCo-based simulation framework for studying Franka Panda robot behavior during sudden hardware failures. It generates labeled datasets of contact patterns, RGB/depth images, and robot state across multiple failure modes and tasks.

**Research goal**: Train a planner that chooses safer robot configurations by learning which pre-failure joint configs lead to worse outcomes. The dataset covers a wide range of pre-failure configurations across different arm poses, carry directions, and mission phases.

![Demo](docs/media/mujoco_arm_planner_demo.gif)

---

## Quick links

- [Installation](#installation) — clone → conda → external assets → GraspGen
- [Quickstart](#quickstart) — 3 commands to a verified trajectory
- [Pipeline](#pipeline) — generate → verify → play → inject failures → npz
- [Risk modeling](docs/risk_modeling.md) — learn per-config contact density, integrate per entity → planner safety cost
- [Cluster / headless deployment](#cluster--headless-deployment)
- [Task schema](#task-schema) and [adding new scenes/tasks](docs/data_generation.md)
- [Troubleshooting](#troubleshooting)

---

## Installation

Supported on Linux (tested on Ubuntu 22.04). Requires a CUDA 12.x toolkit if you plan to use GraspGen (mesh-pick scenes).

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/<your-org>/FailBench.git
cd FailBench
# Or, if you already cloned without --recurse-submodules:
git submodule update --init external/GraspGen
```

### 2. Conda environment

```bash
conda env create -f environment.yml
conda activate failbench_env
```

Key dependencies: MuJoCo 3.3.4, Python 3.10, Mink 0.0.11 (IK), PyTorch, OpenCV.

### 3. External asset libraries

YCB / vikashplus / kevinzakka mesh libraries are git-ignored and fetched via a setup script:

```bash
cd external_assets
bash setup.sh
cd ..
```

This clones ~1 GB of third-party meshes into `external_assets/vikashplus_*/` and `external_assets/kevinzakka_*/`. It is idempotent — re-running skips already-cloned repos.

### 4. GraspGen (required for mesh-pick scenes)

`scene_kitchen`, `scene_workshop`, `scene_grocery`, and `scene_cluttered` use precomputed 6-DoF grasps from [GraspGen](https://github.com/nvlabs/GraspGen). `scene_level2` uses a primitive analytic grasp and does **not** need GraspGen.

GraspGen lives in an isolated UV venv at `external/GraspGen/.venv` so it never interferes with `failbench_env`.

```bash
# uv must be on PATH:
#   curl -LsSf https://astral.sh/uv/install.sh | sh

conda deactivate                          # install_graspgen.sh refuses to run inside a conda env
bash scripts/install_graspgen.sh          # creates external/GraspGen/.venv, builds pointnet2_ops
bash scripts/download_graspgen_models.sh  # ~few hundred MB into external/GraspGen/checkpoints/

conda activate failbench_env
python scripts/precompute_grasps.py       # caches per-mesh grasps under cache/graspgen/
```

Details and troubleshooting: [docs/graspgen_setup.md](docs/graspgen_setup.md).

---

## Quickstart

With `failbench_env` active and (optionally) GraspGen installed:

```bash
# 1. Generate 5 trajectories for one task (no GraspGen needed)
python scripts/generate_task_trajs.py \
    --scene scene_level2 --task clean_nominal --n_trajs 5 --seed 0

# 2. Play them back in the MuJoCo viewer
python scripts/play_task_trajs.py --scene scene_level2 --task clean_nominal

# 3. End-to-end smoke test (generate → inject failure → npz)
python scripts/test_pipeline.py
```

---

## Pipeline

```
scenes/<scene>/tasks.yaml
        │
        ▼
 generate_task_trajs.py ──► scenes/<scene>/trajs/*.pkl   (IK + RRT, 5 segments)
        │                         │
        │                         ▼
        │                 verify_task_trajs.py           (optional audit)
        │                         │
        │                         ▼
        │                 play_task_trajs.py             (viewer + depth)
        ▼
ExperimentRunner (planner.experiments.runner)
        │
        ▼
 .npz + manifest.csv                                     (dataset samples)
```

Trajectories are **full missions** with 5 segments: `approach → descend (grasp) → lift → transport → place (release)`. Heights and obstacle clearances are derived at runtime from the MuJoCo model — no hardcoded per-scene values. Approach direction is sampled on a hemisphere around the object to maximise pre-failure joint configuration diversity.

**Touching the code?** Read [docs/pipeline.md](docs/pipeline.md) first — it maps each script to the module it drives (IK via Mink in `planner/kinematics/`, RRT-Connect in `planner/algorithms/`, the `PandaPickAndPlace_L2` planner in `planner/examples/`, GraspGen integration in `planner/grasp_sampler.py` + `planner/grasp_lock.py`) and documents the exact npz/manifest schema.

---

## Stage 1 — Generate trajectories

```bash
# One task
python scripts/generate_task_trajs.py \
    --scene scene_level2 --task clean_nominal --n_trajs 10 --seed 0

# All enabled tasks for a scene
python scripts/generate_task_trajs.py \
    --scene scene_kitchen --task all --n_trajs 10 --seed 0 --strict-attach
```

| Flag | Default | Meaning |
|---|---|---|
| `--scene` | required | Scene name (must have `scenes/<scene>/tasks.yaml`) |
| `--task` | required | Task name from `tasks.yaml`, or `all` |
| `--n_trajs` | 10 | Number of trajectories to generate |
| `--seed` | 0 | RNG seed for reproducibility |
| `--max_retries` | 5 | Max plan+verify attempts per trajectory |
| `--strict-attach` | off | Reject trajectories whose fingers never contact the object (highly recommended for mesh picks) |
| `--scene_xml` / `--robot_xml` / `--out_dir` | auto | Overrides; defaults are auto-detected |

Output: one `.pkl` per trajectory at `scenes/<scene>/trajs/<scene>_<task>_NN.pkl`. Each pkl contains ordered segments plus a `grasp_meta` block with grasp source, confidence, IK attempts, and approach axis.

Full reference: **[docs/data_generation.md](docs/data_generation.md)**.

---

## Stage 2 — Verify / audit existing trajectories

```bash
# Audit all tasks for a scene
python scripts/verify_task_trajs.py --scene scene_level2

# Verify one task, delete any that fail
python scripts/verify_task_trajs.py \
    --scene scene_kitchen --task stack_nominal --delete_failures

# Single-file check
python scripts/verify_task_trajs.py \
    --traj_file scenes/scene_level2/trajs/scene_level2_clean_nominal_00.pkl
```

Checks arm-environment collisions, grasp success (object lifted above table), and place accuracy. See [docs/data_generation.md](docs/data_generation.md) for thresholds and sample output.

---

## Stage 3 — Visualise

```bash
# Play everything for a task
python scripts/play_task_trajs.py --scene scene_level2 --task clean_nominal

# Pop up depth views alongside the 3D viewer (OpenCV windows, TURBO colormap)
python scripts/play_task_trajs.py \
    --scene scene_grocery --task shelf_over \
    --show_depth --depth_cam ee_cam,front_cam

# GraspGen triage sweep — pause at each grasp, print provenance, flag bad ones
python scripts/play_task_trajs.py \
    --scene scene_kitchen --meshes_only --show_meta \
    --pause_at_grasp --flag_bad --strict-attach
```

| Flag | Effect |
|---|---|
| `--show_depth` | Pop up one OpenCV depth window per camera in `--depth_cam` |
| `--depth_cam ee_cam,front_cam` | Comma-separated list of cameras to render depth for |
| `--meshes_only` | Keep only trajs whose `grasp_meta.source == "graspgen"` |
| `--show_meta` | Print grasp id, confidence, IK attempts, approach axis per traj |
| `--pause_at_grasp` | Wait for ENTER at each grasp action (freeze-frame inspection) |
| `--flag_bad` | Prompt y/N after each traj; flagged paths appended to `.flagged.txt` |
| `--strict-attach` | Print `GRASP-FAILED` for trajs whose fingers never contacted the object |
| `--hold` / `--speed` / `--interp_points` | Playback tuning (defaults: 2 s, 1.0×, 100) |

---

## Stage 4 — Inject failures and emit npz

Bulk dataset generation (`scripts/generate_dataset_tasks.py`) is **not yet implemented**. Today the failure-injection path is exercised via the end-to-end smoke test:

```bash
python scripts/test_pipeline.py
```

This generates one trajectory, runs `planner.experiments.runner.ExperimentRunner` at `fail_fraction ∈ {0.1, 0.6}`, and verifies the resulting `DataSample` serialises to an `.npz` with the expected schema. To build your own single-trial runs programmatically:

```python
from planner.experiments.config import ExperimentConfig, FailureMode
from planner.experiments.runner import ExperimentRunner
from planner.experiments.manager import save_sample_npz

config = ExperimentConfig(
    scene_xml_path="scenes/scene_level2/scene.xml",
    robot_xml_path="scenes/scene_level2/panda.xml",
    trajectory_file="scenes/scene_level2/trajs/scene_level2_clean_nominal_00.pkl",
    task_id="clean_nominal", traj_id=0, seed=42,
    fail_fraction=0.55,
    failure_config=...,                    # see planner/experiments/config.py
    experiment_id="demo_0",
    extra_cameras=["ee_cam"],
)
sample = ExperimentRunner(config).run()
save_sample_npz(sample, "demo_0.npz")
```

### Failure modes

| Mode | Description |
|---|---|
| `GRIPPER_OPEN` | Gripper fully opens, drops object |
| `SLIPPERY_GRIP` | Partial gripper closure, reduced grip |
| `SINGLE_JOINT` | One arm joint frozen |
| `MULTI_JOINT` | Multiple joints frozen |
| `ALL_JOINTS` | All arm joints frozen |

Canonical failure fractions: `[0.1, 0.25, 0.4, 0.55, 0.7, 0.85]` along the full mission.

### npz schema (summary)

`pre_rgb`, `pre_qpos`, `pre_qvel`, `pre_ee_pos`, `task_id`, `traj_id`, `traj_progress`, `contact_positions`, `contact_forces`, `contact_geom_pairs`, `contact_failure_id`, `failure_modes`, `failure_probs`, plus `<cam>_rgb` / `<cam>_depth` per extra camera. Full schema + manifest columns + internals: **[docs/pipeline.md](docs/pipeline.md)**.

Bulk generator roadmap: iterate enabled tasks × trajs × canonical failure fractions × failure modes → `.npz` files + `manifest.csv`. Tracked as the next milestone.

---

## Task schema

Tasks are defined per-scene in `scenes/<scene>/tasks.yaml`. No Python edits are needed to add tasks — only YAML.

```yaml
scene: scene_level2
grasped_object: object3

obstacles:
  min_clearance: 0.06
  centers:
    - [-0.08, -0.27]
    - [ 0.06, -0.27]

tasks:
  clean_nominal:
    category: clean
    description: "Pick object3, drop in near-edge clear zone"
    place_height: table_surface          # table_surface | on_target | on_shelf
    goal:
      type: zones                        # zones | choice
      zones:
        - {x: [-0.20, 0.15], y: [-0.19, -0.22]}
      check_clearance: true

  stack_nominal:
    category: stack
    place_height: on_target
    goal:
      type: choice
      targets: [[-0.12, -0.36], [0.12, -0.36]]
      jitter: 0.02

  pick_banana:
    enabled: false                       # skipped by --task all, still reachable via --task pick_banana
    category: sort
    grasped_object: banana
    ...
```

**Semantic categories**: `stack`, `clean`, `sort`, `handover`, `shelf`. **Geometric suffixes**: `_nominal` (short), `_far` (long carry), `_over` (cross-obstacle-field). Deep reference and "adding a new scene" walkthrough: [docs/data_generation.md](docs/data_generation.md).

---

## Cluster / headless deployment

Reaching first-trajectory on a fresh cluster node in order:

```bash
# 1. Code + submodules
git clone --recurse-submodules <repo-url> FailBench && cd FailBench

# 2. Conda env
conda env create -f environment.yml && conda activate failbench_env

# 3. Third-party meshes (~1 GB)
( cd external_assets && bash setup.sh )

# 4. GraspGen venv + checkpoints (skip if only using scene_level2)
conda deactivate
bash scripts/install_graspgen.sh
bash scripts/download_graspgen_models.sh
conda activate failbench_env
python scripts/precompute_grasps.py

# 5. Headless rendering — set BEFORE running anything that opens a MuJoCo context
export MUJOCO_GL=egl                     # preferred (GPU-accelerated on NVIDIA nodes)
# export MUJOCO_GL=osmesa                # fallback if EGL is unavailable

# 6. First trajectory
python scripts/generate_task_trajs.py \
    --scene scene_level2 --task clean_nominal --n_trajs 5 --seed 0 --strict-attach
```

### Environment variables

| Variable | When to set | Why |
|---|---|---|
| `MUJOCO_GL=egl` | Any headless node with an NVIDIA GPU | Enables offscreen rendering without a display. Required for `play_task_trajs.py`'s `--show_depth`, `ExperimentRunner` RGB/depth capture, and `verify_task_trajs.py`. |
| `MUJOCO_GL=osmesa` | Nodes without GPU EGL support | Software rendering fallback. Slower but portable. |
| `CUDA_HOME=/usr/local/cuda-12.x` | GraspGen install only, when system `nvcc` is older than 12.0 | `pointnet2_ops` requires `nvcc` whose major version matches the torch CUDA build (12.x). `scripts/install_graspgen.sh` autodetects `/usr/local/cuda-12*` but export manually if it fails. |

### Hardware expectations

| Stage | CPU | GPU |
|---|---|---|
| Trajectory generation (MuJoCo + IK + RRT) | Yes | Not used |
| GraspGen precompute (`precompute_grasps.py`) | — | Yes (CUDA 12.x, one-time per mesh) |
| `ExperimentRunner` / test_pipeline | Yes | GPU only if `MUJOCO_GL=egl` |
| Future `generate_dataset_tasks.py` | Yes, parallelisable across tasks/trajs | GPU optional (rendering speed) |

### Disk budget

| Path | Size |
|---|---|
| `external_assets/` | ~1 GB (third-party meshes) |
| `external/GraspGen/.venv` + `checkpoints/` | ~several GB |
| `cache/graspgen/` | ~few MB per mesh (gitignored, regeneratable) |
| `scenes/*/trajs/` | ~4 KB per pkl × 10s–100s per scene |

### Reproducibility

- `--seed` fixes the RNG for goal sampling, approach sampling, and IK tie-breaks. Two identical commands produce identical pkls.
- `--task all` iterates tasks in `tasks.yaml` insertion order — stable across runs.
- All per-scene heights are model-derived at runtime, so a scene geometry change will shift the trajs it produces; regenerate after scene XML edits.

### SLURM / Docker hints

For a batch node without X:

```bash
#SBATCH ...
export MUJOCO_GL=egl
source ~/miniconda3/etc/profile.d/conda.sh && conda activate failbench_env
cd /path/to/FailBench
python scripts/generate_task_trajs.py --scene $SCENE --task all --n_trajs 10 --seed $SLURM_ARRAY_TASK_ID --strict-attach
```

GraspGen's Docker images live in `external/GraspGen/docker/` if you prefer a containerised install over the UV venv.

---

## Repository layout

```
franka_emika_panda/      Robot MJCF + meshes (base and elevated panda.xml)
scenes/<name>/           One directory per scene
  scene.xml              MuJoCo scene (includes a panda XML by relative path)
  tasks.yaml             Task definitions for this scene
  trajs/                 Pre-computed .pkl trajectories
  datasets/v<N>/         Output: manifest.csv + exp_*.npz
external_assets/         Third-party mesh libraries (gitignored; setup via setup.sh)
external/GraspGen/       Submodule — 6-DoF grasp generator (isolated UV venv)
cache/graspgen/          Precomputed grasp YAMLs keyed by mesh SHA (gitignored)
planner/
  experiments/           ExperimentConfig, ExperimentRunner, data capture, npz I/O
  risk/                  Spatial contact-density target + entity integration + heatmap regressor
  examples/              Reference pick-and-place planner (IK+RRT)
  tasks.py               tasks.yaml loader + goal samplers
  grasp_lock.py          Sticky gripper (kinematic attach)
  grasp_sampler.py       GraspGen candidate sampler
scripts/
  generate_task_trajs.py    Stage 1: IK+RRT trajectory generation
  verify_task_trajs.py      Stage 2: physics audit of existing pkls
  play_task_trajs.py        Stage 3: MuJoCo viewer + optional depth windows
  test_pipeline.py          Stage 4: end-to-end smoke test (failure injection)
  build_density_targets.py  Stage 5: per-config 2D contact-density targets
  train_demo.py             Stage 6: heatmap regressor demo (one scene)
  precompute_grasps.py      GraspGen cache builder
  install_graspgen.sh       GraspGen UV venv installer
notebooks/
  inspect_dataset.ipynb     Whole-dataset inspection + per-scene density heatmaps
  eval_model.ipynb          Heatmap regressor evaluation + camera-overlay projection
docs/
  data_generation.md     Deep reference for tasks.yaml + stages 1-3
  graspgen_setup.md      GraspGen install / caveats
  pipeline.md            Architecture / npz schema reference
  risk_modeling.md       Stages 5-6 (target dataset, model, eval)
```

---

## Current dataset status

| Scene | Enabled tasks | Trajectories | Grasp source |
|---|---:|---:|---|
| scene_level2 | 12 | 120 | analytic (primitive pick) |
| scene_kitchen | 12 | 113 | GraspGen + analytic |
| scene_workshop | 9 | 149 | GraspGen (cylinder) |
| scene_cluttered | 9 | 137 | GraspGen |
| scene_grocery | 12 | 152 | GraspGen |
| **Total** | **54** | **671** | — |

All verified under `--strict-attach` with ≥97% real finger-object contact during playback.

---

## Troubleshooting

**`mujoco.FatalError: gladLoadGL error` / `GLFW` errors on a headless node**
Set `MUJOCO_GL=egl` (or `osmesa`) before running any script that opens a MuJoCo context. Applies to all rendering paths — viewer, offscreen RGB/depth, and verifier.

**`pointnet2_ops` build fails during `install_graspgen.sh`**
System `nvcc` is older than CUDA 12.0. Install `cuda-toolkit-12-1` (or similar) and export `CUDA_HOME=/usr/local/cuda-12.x`, then re-run the installer.

**Mesh-pick scenes produce 0 trajectories**
Check in order: (1) `cache/graspgen/index.json` exists and references the mesh SHA, (2) `external/GraspGen/checkpoints/` is populated, (3) the scene's `grasped_object` is a mesh (not a primitive), (4) try without `--strict-attach` to see whether the blind-attach path succeeds — if it does, the fingers aren't making real contact and you need to investigate the grasp candidates.

**`scene_grocery` handover tasks all reject with "arm collision"**
Ensure `scenes/scene_grocery/tasks.yaml` has the front-edge handover zones (x∈[-0.10, 0.20] for `handover_nominal`, x∈[0.10, 0.22] for `handover_far`). Older zones at x>0.28 collide with the right-side wall at x=0.58.

**`cv2` / `ModuleNotFoundError: opencv` when using `--show_depth`**
Run from the `failbench_env` conda env (ships OpenCV), not the base env or the GraspGen venv.

---

## Acknowledgments

This project is inspired by and builds upon the excellent work of existing simulation platforms such as:

- [RoboCasa](https://robocasa.ai/)
- [Robosuite](https://robosuite.ai/)

We extend their design philosophies and modular stacks to focus specifically on simulating and understanding robotic failures.

---

## Citation

```bibtex
@inproceedings{[FAILBENCH2025],
  title={TBA},
  author={Duc M. Nguyen, Saad Ghani, Andrew Marshall, Allison Andreyev, Gregory J. Stein and Xuesu Xiao},
  booktitle={TBA},
  year={2025}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
