# GraspGen setup

GraspGen (NVLabs) is installed as a git submodule at `external/GraspGen` with its own isolated UV venv. It does **not** share packages with the `failbench_env` conda environment — the two coexist safely.

## Requirements

- NVIDIA GPU with driver ≥ 525 (CUDA 12.1 compatible). Tested on RTX 3070 (8 GB), driver 535.
- Python 3.10 (installed automatically by `uv`).
- `uv` on PATH: install with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Internet access to fetch PyTorch wheels and HuggingFace checkpoints.

## Install

```bash
# 1. Fetch the submodule (first time only)
git submodule update --init external/GraspGen

# 2. Deactivate any active conda env, then install
conda deactivate          # repeat until $CONDA_DEFAULT_ENV is empty
bash scripts/install_graspgen.sh

# 3. Download model checkpoints (~hundreds of MB, gitignored)
bash scripts/download_graspgen_models.sh
```

The installer refuses to run while a conda env is active — this is the primary guard against polluting `failbench_env`.

## Verify isolation

```bash
# GraspGen venv — torch 2.1.0, CUDA available
external/GraspGen/.venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"

# failbench_env — whatever torch version it had before, unchanged
conda activate failbench_env
python -c "import torch; print(torch.__version__)"
conda deactivate
```

## Run inference (sanity check)

```bash
external/GraspGen/.venv/bin/python \
  external/GraspGen/scripts/demo_object_mesh.py \
  --mesh_file scenes/scene_kitchen/assets/cubesmall.stl \
  --mesh_scale 1.0 \
  --gripper_config external/GraspGen/checkpoints/checkpoints/graspgen_franka_panda.yml \
  --num_grasps 10 \
  --output_file /tmp/cubesmall_grasps.yml \
  --no-visualization
```

Expected: `/tmp/cubesmall_grasps.yml` with 10 ranked 6-DoF grasp poses.

## Precompute grasps for the trajectory generator

The trajectory generator reads precomputed grasps from `cache/graspgen/<sha>.yml`.
Run this once (and again whenever a pick-target mesh changes):

```bash
# from failbench_env — the script shells out to the GraspGen venv itself
conda activate failbench_env
python scripts/precompute_grasps.py              # scan all scenes/*/tasks.yaml
python scripts/precompute_grasps.py --force      # regenerate
python scripts/precompute_grasps.py --mesh scenes/scene_kitchen/assets/cubesmall.stl
```

Output: `cache/graspgen/<sha>.yml` per unique mesh and `cache/graspgen/index.json`
mapping repo-relative mesh paths to their SHA. Primitive pick targets (box /
cylinder geoms) are skipped — the analytic top/side grasp is used there.

## Architecture

### Pipeline overview

```
precompute_grasps.py          (runs once, uses GraspGen venv)
        │
        ▼
cache/graspgen/<sha>.yml      (SHA-keyed per unique mesh, gitignored)
        │
        ▼
grasp_sampler.py              (loads YAML, samples 6-DoF TCP poses)
        │
        ▼
generate_task_trajs.py        (IK+RRT with sampled orientation)
        │
        ▼
trajectory_verifier.py        (physics replay with grasp lock)
        │
        ▼
scenes/<scene>/trajs/*.pkl    (verified trajectories)
```

### When GraspGen is used vs analytic

| Object type | Grasp method | Orientation |
|---|---|---|
| **Mesh-backed** (cubesmall, apple, banana, etc.) | GraspGen 6-DoF from cache | Sampled per trajectory |
| **Primitive** (box, cylinder geoms) | Analytic top-down / side | Fixed downward or unconstrained |

The sampled orientation carries through the approach, descend, lift, and
transport segments. Only the place segment reverts to downward constraint
so the object is set down flat.

If the GraspGen cache is unavailable (not precomputed), mesh picks fall
back to the analytic path automatically with a warning.

### Grasp sampler details (`planner/grasp_sampler.py`)

- Loads cached YAML by SHA lookup from `cache/graspgen/index.json`
- Filters grasps by approach direction: keeps only mostly-downward
  approaches (`approach_axis.z ≤ -0.85`) to avoid table collisions
- Falls back to the 5 most-downward grasps if filter eliminates all
- Converts GraspGen's gripper-base position to fingertip TCP:
  `tcp = base + 0.105m * approach_axis` (Panda finger depth)
- Transforms from object-local frame to world frame
- Returns `(tcp_pos, quat_wxyz, approach_axis)` per call

### Frame convention

GraspGen's YAML `position` field is the **gripper base** (flange), not the
fingertip TCP. The gripper's +Z axis is the advance/approach direction.
For a top-down grasp, approach_axis points downward (Z ≈ -1 in world).

See `memory/feedback_graspgen_frame.md` for the original verification.

## How grasping works (sticky gripper)

Physics-based finger closure is unreliable on mesh objects with complex
collision hulls — MuJoCo's contact solver can't sustain a grip when the
fingertip TCP sits inside the object's volume.

Instead, FailBench uses a **kinematic grasp lock** (`planner/grasp_lock.py`):
when the `"grasp"` action fires, the object's free joint is locked to the
hand body by recording their relative transform and enforcing it every sim
step. On `"release"` (or `GRIPPER_OPEN` / `SLIPPERY_GRIP` failure injection),
the lock is released and the object falls under gravity.

The finger close/open animation still runs for visual fidelity. The lock is
integrated into all three replay sites:

| File | Role |
|---|---|
| `planner/trajectory_verifier.py` | Physics verification during traj generation |
| `scripts/play_task_trajs.py` | MuJoCo viewer playback |
| `planner/experiments/runner.py` | Dataset generation with failure injection |

For failure injection: `GRIPPER_OPEN` and `SLIPPERY_GRIP` release the lock
(object drops). Joint-freeze failures (`SINGLE_JOINT`, `MULTI_JOINT`,
`ALL_JOINTS`) keep the lock active — the object stays in the gripper while
the arm freezes, matching real-world behavior.

## Tested objects

Verified end-to-end (precompute → generate → verify → play) on scene_kitchen:

| Object | Shape | Pass rate (3 trajs) | Notes |
|---|---|---|---|
| rubberduck | irregular, medium | 3/3 | |
| banana | elongated | 3/3 | |
| apple | small sphere | 2/3 (20 retries) | Near clutter, some IK failures |
| alarmclock | large box | 2/3 (20 retries) | Near workspace edge |

Failures are arm collision / IK reachability issues (object position in
scene), not grasp mechanics. Increasing `--max_retries` resolves them.

## Troubleshooting

- **`install_uv_pointnet.sh` fails with nvcc errors**: the installer auto-detects `/usr/local/cuda-12.*`. If no CUDA 12.x toolkit is found, install one: `sudo apt install cuda-toolkit-12-1`.
- **OOM during inference on 8 GB GPU**: drop `--num_grasps` from 200 to 50.
- **`uv` not on PATH**: add `source $HOME/.local/bin/env` to `~/.bashrc`.
- **HuggingFace download is slow / hangs**: `huggingface-cli login` if your network requires auth, or set `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads.
- **0/N pass rate on a new mesh pick task**: check object position — if it's near the workspace edge (|x| > 0.25 or y < -0.55), IK struggles with 6-DoF orientations. Increase `--max_retries` to 20+, or move the object closer to the robot base.
- **"No GraspGen cache index" error**: run `python scripts/precompute_grasps.py` first. The cache is gitignored — each collaborator regenerates locally.

## Rollback

```bash
git submodule deinit -f external/GraspGen
git rm -f external/GraspGen
rm -rf .git/modules/external/GraspGen external/GraspGen
rm -rf cache/graspgen
```

`failbench_env` is untouched throughout.

## License

GraspGen is distributed under the NVIDIA Research License. The submodule contains a pointer only (not a redistribution). Confirm acceptable use with your institution before shipping publicly.
