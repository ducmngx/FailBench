# RL Training Handoff — Moving Safety-Aware Training to Another Machine

Companion to `docs/safety_rl_training_plan.md`. This doc covers the *operational* side: what to copy to the training machine, what to install, how to verify the env works, what to ship back.

For background and the algorithm decision (Option A: PPO from BC), see the plan doc.

---

## What you're moving

### Code

```
planner/policy/safe_rl_env.py                 Gym env wrapper
planner/policy/SAFE_RL_ENV.md                 env reference doc
planner/policy/safe_action.py                 ObsWindow + query_risk helpers
planner/policy/libero_env_failure.py          failure injection
planner/risk/damage.py                        DamageAccumulator (d_mech)
planner/risk/inference.py                     ContactPredictor + marginal_heatmap
planner/risk/models/heatmapbaseline.py        model architectures
scripts/safety/safety_rollout.py              build_entity_masks (reused at env init)
docs/safety_rl_training_plan.md               algorithm + hyperparameter reference
docs/safety_rl_training_handoff.md            this doc
```

A `git pull origin refactor/cleanup` on the training machine gives all of it.

### Data — about ~5 GB total

| Item | Size | Where |
|---|---|---|
| Predictor checkpoint (Gatekeeper, val_heat=0.0648) | ~50 MB | `notebooks/model_playground/cluster_download/06172026/dualgated_state_rgb/best_ep08_val0.0648.pt` |
| LIBERO demos for first 4 tasks | ~4 GB | `datasets/libero/raw/libero_object/<task>_demo.hdf5` and `libero_spatial/...` |
| LIBERO repo + BDDL files | included in `external/LIBERO/` submodule | submodule init |

The 4 demo files for the first-pass tasks:

```
datasets/libero/raw/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5
datasets/libero/raw/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5
datasets/libero/raw/libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5
datasets/libero/raw/libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5
```

Each is ~1 GB. Transfer via:

```bash
rsync -avh --progress \
  datasets/libero/raw/libero_object/pick_up_the_{tomato_sauce,milk}_and_place_it_in_the_basket_demo.hdf5 \
  <user>@<machine>:<repo>/datasets/libero/raw/libero_object/

rsync -avh --progress \
  datasets/libero/raw/libero_spatial/pick_up_the_black_bowl_{on_the_cookie_box,in_the_top_drawer_of_the_wooden_cabinet}_and_place_it_on_the_plate_demo.hdf5 \
  <user>@<machine>:<repo>/datasets/libero/raw/libero_spatial/
```

---

## Setup on the training machine

### 1. Clone + checkout the branch

```bash
git clone <repo-url> FailBench
cd FailBench
git checkout refactor/cleanup
git submodule update --init --recursive external/LIBERO
```

### 2. Two virtual environments

The repo already uses a split-env layout. Mirror it on the new machine:

| env | Purpose | How to create |
|---|---|---|
| `failbench_env` (conda) | predictor inference, training scripts (sb3, torch) | `conda env create -f environment.yml` |
| `external/LIBERO/.venv` (sidecar pip venv) | robosuite 1.4.0 + LIBERO env at training time | `cd external/LIBERO && python -m venv .venv && .venv/bin/pip install -e . robosuite==1.4.0 termcolor` |

For RL training you also need `stable-baselines3 >= 2.3` inside the LIBERO venv (because the env is what the trainer drives):

```bash
external/LIBERO/.venv/bin/pip install \
    "stable-baselines3[extra]>=2.3" \
    "gymnasium" \
    "tensorboard" \
    "wandb"  # optional
```

Also bring torch with CUDA:

```bash
external/LIBERO/.venv/bin/pip install torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 3. Headless rendering

Mandatory for `SubprocVecEnv` on any GPU machine that doesn't have a display server:

```bash
# In your training launcher script, BEFORE forking workers
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Or set them inside Python at module top before any LIBERO import:

```python
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
```

### 4. GPU visibility quirk (only if using GMU Hopper)

Hopper's `/etc/profile.d/hide_cuda_visible_devices.sh` sets `CUDA_VISIBLE_DEVICES=-1` readonly. Override in Python BEFORE `import torch`:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-<paste from nvidia-smi -L>"
import torch
```

UUIDs change per node — re-run `nvidia-smi -L` after each `salloc`. Skip this on non-Hopper machines.

### 5. Smoke test before any training

```bash
PYTHONPATH=external/LIBERO:. external/LIBERO/.venv/bin/python -m planner.policy.safe_rl_env \
  --bddl external/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl \
  --demo datasets/libero/raw/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5 \
  --ckpt notebooks/model_playground/cluster_download/06172026/dualgated_state_rgb/best_ep08_val0.0648.pt \
  --failure_prob 1.0 --steps 30 \
  --render mp4 --render_path out/smoke.mp4
```

Expected: prints obs space + action space, runs 30 random-action steps, writes `out/smoke.mp4` with health bars + heatmap overlay. If this works, the trainer's foundation is solid.

---

## Training entry points to write

The plan doc has the full task list (TaskIDs #42–#46). The two short scripts you need first:

### `scripts/safety/augment_demos.py` (Phase 1)

Run the predictor on every demo timestep, save `pred_per_body` + `gate_prob` per timestep + per-demo aggregate stats. ~80 lines.

### `scripts/safety/train_bc.py` (Phase 2)

Load augmented demos (curated by `max_pred_risk < threshold`), train a small MLP for 20 epochs with MSE on demo actions. Save `bc_policy.pt`. ~120 lines.

### `scripts/safety/train_ppo.py` (Phase 3 + 4)

SB3 PPO + `MultiInputPolicy`. Load BC weights into actor. `SubprocVecEnv(n=16)`. `model.learn(total_timesteps=2_000_000)`. Skeleton sketched in the plan doc. ~150 lines.

I haven't pre-written these because they want to live alongside your trainer-specific logging and run-management code on the training machine.

---

## What to ship back to the analysis machine

When training finishes (or pauses), bring back:

```
runs/<run_id>/
  tensorboard/                  TB event files
  bc_policy.pt                  BC checkpoint
  ppo_<step>.zip                SB3 PPO checkpoint(s)
  eval_metrics.csv              one row per eval step (task success%, safe success%, mean damage, mean pred_cost)
  rollout_videos/               handful of mp4s for paper figures
  config.json                   the exact CLI args + hyperparameters
```

`config.json` makes runs reproducible. Don't lose it — for the paper figure you'll want to retrace which (λ_pred, λ_dmg, failure_prob, seed) produced each curve.

---

## Memory-side things to bring with you

The training machine doesn't need them, but if you go to debug edge cases later, these memory notes from `~/.claude/.../memory/` are the most relevant:

- `feedback_libero_env_flip.md` — Y-flip on every env image before predictor input (already wired into `SafeLiberoEnv._build_obs`, but if you write a debugger that reads RGB outside the env, you need to know)
- `feedback_gatekeeper_corpus.md` — predictor trained LIBERO-only (so cross-corpus results need a caveat)
- `feedback_gripper_failure_semantics.md` — current `GRIPPER_OPEN` injection is the legacy "zero actuator gain" semantics that matches v2 training distribution. Do not change it without retraining.
- `feedback_hopper_cuda_devices.md` — Hopper override only

---

## Quick reference — what the training command will look like

After the three scripts above exist:

```bash
PYTHONPATH=external/LIBERO:. external/LIBERO/.venv/bin/python -m scripts.safety.augment_demos \
  --task pick_up_the_tomato_sauce_and_place_it_in_the_basket \
  --ckpt notebooks/.../best_ep08_val0.0648.pt

PYTHONPATH=external/LIBERO:. external/LIBERO/.venv/bin/python -m scripts.safety.train_bc \
  --task pick_up_the_tomato_sauce_and_place_it_in_the_basket \
  --curation_threshold 1500 \
  --out runs/01-bc-tomato-sauce

PYTHONPATH=external/LIBERO:. external/LIBERO/.venv/bin/python -m scripts.safety.train_ppo \
  --task pick_up_the_tomato_sauce_and_place_it_in_the_basket \
  --bc_ckpt runs/01-bc-tomato-sauce/bc_policy.pt \
  --lambda_pred 1e-3 --lambda_dmg 1.0 --failure_prob 0.1 \
  --total_timesteps 2_000_000 \
  --n_envs 16 \
  --out runs/01-ppo-tomato-sauce-predictor-plus-damage
```

The three reward variants come from different `--lambda_pred` / `--lambda_dmg` combinations on the same Phase-4 script.

---

## If something breaks on the new machine

Order of diagnosis:

1. Does the smoke command from §5 produce a non-empty mp4? If not, the env wiring is broken (most likely `MUJOCO_GL` not set or LIBERO venv missing).
2. Does `python -c 'from planner.risk.inference import ContactPredictor; p = ContactPredictor.from_checkpoint("...")'` succeed? If not, the predictor checkpoint isn't reaching the right path or torch versions disagree.
3. Does `safe_rl_env.SafeLiberoEnv(...)` print `failure injected: {...}` on `reset()`? If not, the failure injector isn't seeing the demos.
4. Does `info["pred_risk_total"]` change across steps? If not, the predictor inference is no-op.

These four checks cover ~95% of cross-machine setup failures.
