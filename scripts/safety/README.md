# Safety-rollout experiments

End-to-end pipeline for showing the contact predictor reduces realized
risk on LIBERO tasks.

## What's in this directory

| File | Purpose |
|---|---|
| `smoke_env_predictor.py` | Stage-0 verification: env loads, predictor queries, failure injection visibly affects qvel. Run this first on a new machine. |
| `eval_predictor.py` | Quantitative checkpoint eval — produces Table 1 (LIBERO ablation) and Table 2 (cross-corpus generalization). Runs without LIBERO env, only needs the v2 HDF5s. |
| `safety_rollout.py` | The actual experiment runner. Iterates `task × init_state × failure_mode × fail_progress × policy`, writes per-task `results.{parquet,csv}`. |
| `sweep_safety_rollouts.sbatch` | SLURM batch script for the full 3-task cluster sweep. |

Output goes to `out/safety_rollouts/<task>/results.csv` (locally), or
`$SCRATCH/failbench/safety_rollouts/<task>/results.csv` (cluster).
Analyse via `notebooks/safety_rollouts/analysis.ipynb`.

## Local install (LIBERO sidecar venv)

The trainer venv (`failbench_env`) doesn't have robosuite/mujoco; LIBERO
lives in its own sidecar venv at `external/LIBERO/.venv`. The full dep
set we settled on locally:

```bash
external/LIBERO/.venv/bin/pip install \
    bddl==1.0.1 future==0.18.2 hydra-core==1.2.0 easydict==1.9 \
    einops==0.4.1 opencv-python==4.6.0.66 cloudpickle==2.1.0 \
    gym==0.25.2 matplotlib==3.5.3 h5py hdf5plugin
external/LIBERO/.venv/bin/pip install torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

(robosuite==1.4.0 and libero itself were already installed by the
earlier dataset-generation setup.)

## Activation pattern (every shell)

```bash
source external/LIBERO/.venv/bin/activate
export PYTHONPATH="$PWD/external/LIBERO:$PWD:${PYTHONPATH:-}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HDF5_USE_FILE_LOCKING=FALSE
```

## Stage 0 — smoke

```bash
python -u -m scripts.safety.smoke_env_predictor \
    --ckpt notebooks/model_playground/cluster_download/06172026/dualgated_state_rgb/best_ep08_val0.0648.pt \
    --bddl external/LIBERO/libero/libero/bddl_files/libero_90/KITCHEN_SCENE5_put_the_black_bowl_on_the_plate.bddl \
    --mode SINGLE_JOINT --joints 4 --inject_at 30 --n_steps 80
```

Expected (5/5 checks passing):

```
[1/5] env loaded + reset in ~3s
[2/5] arch=GatekeeperCoordFiLMUNet ep=8 val_heat=0.06479...
[3/5] window ready
[4/5] heat max~0.4  gate_prob~0.4
[5/5] ✓ failed joint's velocity is bounded
```

## Stage 5 — single-task smoke (~6 min)

```bash
python -u -m scripts.safety.safety_rollout \
    --ckpt <ckpt.pt> \
    --libero_root external/LIBERO \
    --demo_root datasets/libero/raw \
    --tasks pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate \
    --n_inits 3 --progresses 0.4,0.7 --modes SINGLE_JOINT,GRIPPER_OPEN \
    --policies baseline,scaling,search \
    --out_root out/safety_rollouts_smoke
```

Output: 36 rollouts. Per-policy aggregate in `results.csv`.

## Stage 5 — single-task full (~55 min)

```bash
python -u -m scripts.safety.safety_rollout \
    --ckpt <ckpt.pt> \
    --libero_root external/LIBERO \
    --demo_root datasets/libero/raw \
    --tasks pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate \
    --n_inits 10 \
    --modes GRIPPER_OPEN,SLIPPERY_GRIP,SINGLE_JOINT,MULTI_JOINT,ALL_JOINTS \
    --progresses 0.1,0.25,0.4,0.55,0.7,0.85 \
    --policies baseline,scaling \
    --out_root out/safety_rollouts_opt1
```

600 rollouts. Skips `search` for time (degenerate at default σ — see Caveats).

## Cluster sweep — 3 tasks, all modes, full progresses

```bash
sbatch scripts/safety/sweep_safety_rollouts.sbatch
```

5,400 rollouts with all three policies; ~19 h. ~3.6 h if `--policies baseline,scaling`.

## Caveats from the smoke run (worth knowing)

1. **`ActionScalingPolicy` at default `--scaling_tau 0.0` fires every step.**
   Mean realized risk drops 833× but it's not really *adaptive* — the
   predictor's verdict doesn't gate anything. For a real "selective
   slowdown" claim, set `--scaling_tau` to ~p70 of the baseline's
   `pred_risk_t0` distribution (look at `out/safety_rollouts_opt1/<task>/results.csv`,
   take `pred_risk_t0.quantile(0.70)` over baseline rows).

2. **`CandidateSearchPolicy` is currently degenerate.** σ=0.02 produces
   action perturbations whose 1-step predicted risks are indistinguishable
   from the demo action, so the search picks an arbitrary noise candidate.
   The 105,000× risk reduction we see is from *random* small action noise
   breaking the catastrophic demo trajectory, not from predictor-guided
   selection. To make search informative: increase `--search_sigma` (e.g.
   0.05 or 0.1) so candidates produce meaningfully different next states.

3. **`n_safety_triggers` is the number of steps the scaling policy
   modulated the action.** With `tau=0.0` it equals episode length;
   with a proper τ it should be roughly the post-failure steps only.

## What the result tells you (per-rollout CSV columns)

| Column | Meaning |
|---|---|
| `success_once` | LIBERO env's binary success flag at episode end |
| `n_steps` | Episode length |
| `contact_mass_total` | Σ over non-robot bodies of `‖F_world‖ · dt` |
| `contact_mass_per_body` | JSON dict, same broken out by body name |
| `realized_risk` | Σᵢ value(eᵢ)·contact_mass(eᵢ) with uniform 1.0 values |
| `pred_risk_pre_failure` | Predictor's marginal risk score at step `fail_step-1` |
| `pred_risk_t0` | Same, at the first env step (before failure) |
| `n_safety_triggers` | How many steps the policy modulated the action |
| `rollout_seconds` | Wall-clock per rollout |
