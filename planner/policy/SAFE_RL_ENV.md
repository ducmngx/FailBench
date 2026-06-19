# SafeLiberoEnv

Gym-compatible LIBERO environment with FailBench's contact predictor and OopsieVerse-style mechanical damage tracking. Drop-in for stable-baselines3, cleanrl, or any Gym/Gymnasium-style trainer.

Module: `planner.policy.safe_rl_env`

## What it gives you

- **Standard Gym/Gymnasium API** — 5-tuple `step()` returning `(obs, reward, terminated, truncated, info)`.
- **Shaped reward** combining LIBERO's task success signal, the predictor's anticipated-contact cost, and the OopsieVerse `d_mech` damage delta.
- **Per-episode failure injection** — sample one of 5 hardware-failure modes at a random progress step, with configurable rate.
- **Init-state replay** from the task's recorded demos so each episode starts from the demo's exact scene configuration (in-distribution for the predictor).
- **Rich `info` dict** — every reward component, predictor diagnostics, per-body damage, and the injected failure event are exposed every step for logging.

## Reward formulation

```
r_t = r_task(s_t)
    − λ_pred · pred_cost(s_t)
    − λ_dmg  · Δd_mech(s_t)
```

| Term | Meaning | Default weight |
|---|---|---|
| `r_task` | LIBERO's binary success flag (1 on success step, 0 otherwise) | — |
| `pred_cost` | Marginal heatmap mass integrated over per-object masks, summed over failure modes weighted by `mode_prior` (uniform default) | `λ_pred = 1e-3` |
| `Δd_mech` | Per-step accumulated mechanical-damage change, OopsieVerse formula with our LIBERO fragility table | `λ_dmg = 1.0` |

The components are returned separately in `info` (`r_task`, `r_pred`, `r_damage`) so a trainer can route each to its own logger curve.

### Why a Δ on damage instead of total

We reward the policy for **not increasing** damage at each step, not for the cumulative amount. This avoids a phantom "punish the last steps of an episode that already had damage in earlier steps" signal — if no new damage happens this step, `r_damage = 0` regardless of accumulated damage so far.

## Observation space

`gym.spaces.Dict` with:

| Key | Shape | dtype | Description |
|---|---|---|---|
| `proprio` | `(18,)` | `float32` | `concat([qpos(7), qvel(7), ee_pos(3), gripper_qpos(1)])` |
| `agentview_rgb` | `(240, 320, 3)` | `uint8` | Y-down (v2-compatible) camera view |

`agentview_rgb` is included only when `rgb_in_obs=True` (default). Set to `False` to train a state-only policy while the predictor internally still gets RGB.

## Action space

`gym.spaces.Box(low=-1, high=+1, shape=(action_dim,))` — robosuite OSC pose-delta + gripper. `action_dim = 7` for the LIBERO Panda.

## info dict (every step)

| Key | Value |
|---|---|
| `r_task`, `r_pred`, `r_damage` | Reward decomposition |
| `pred_risk_total` | `sum(pred_per_body)` for this step's predictor query |
| `pred_per_body` | `{body_name: float}` mass attribution from heatmap·mask integration |
| `gate_prob` | Gatekeeper classifier probability (NaN if predictor disabled or non-gated) |
| `damage_total` | Cumulative `d_mech` damage this episode |
| `damage_step` | Damage added this step |
| `per_body_damage` | `{body_name: float}` cumulative per-body damage |
| `failure_injected` | `{mode, joints, fail_step}` or `None` |
| `step_idx` | 1-based step count since `reset()` |
| `success` | LIBERO's `info["success"]` (also baked into `r_task`) |

## Constructor reference

```python
SafeLiberoEnv(
    *,
    bddl_file: str,                      # required
    demo_hdf5: str | None = None,        # init-state seeding
    ckpt: str | None = None,             # predictor disabled if None
    lambda_pred: float = 1e-3,
    lambda_dmg: float = 1.0,
    failure_prob: float = 0.0,           # 0 = no failures (Stage A baseline)
    failure_modes: tuple = (             # which modes to sample
        "GRIPPER_OPEN", "SLIPPERY_GRIP",
        "SINGLE_JOINT", "MULTI_JOINT", "ALL_JOINTS"),
    progress_range: tuple = (0.2, 0.85), # sample fail_step in this range
    max_episode_steps: int = 250,
    image_h: int = 240, image_w: int = 320,
    predictor_every_k: int = 1,          # reuse last pred for K-1 steps
    mode_prior: dict | None = None,      # weights for marginal_heatmap
    severity: dict | None = None,        # per-body weights for risk_score
    rgb_in_obs: bool = True,
    seed: int | None = None,
    predictor_device: str | None = None, # cuda / cpu / MIG-...
)
```

## Failure injection mechanics

- On each `reset()`, sample a Bernoulli with rate `failure_prob`.
- If it fires, pick a mode uniformly from `failure_modes`, look up canonical joints (or use `mode_joints` override), and pick a progress `p ~ Uniform(progress_range)`. The failure fires at `step ≈ p × n_actions` where `n_actions` comes from the demo (or `max_episode_steps` fallback).
- Failure is realized through `EnvFailureScheduler` — same code path as the dataset generation. **Uses legacy `actuator_gainprm` zeroing semantics** for gripper-class failures to match the v2 training distribution. See `feedback_gripper_failure_semantics.md` in `~/.claude/.../memory/`.

## Important gotchas

| Gotcha | Detail |
|---|---|
| **Y-axis flip** | LIBERO returns OpenGL framebuffer (Y-up); the v2 corpus the predictor was trained on stored Y-down. We flip on the way in. Don't double-flip in your trainer. |
| **Predictor reward scale** | `pred_risk` is unnormalized and lives in `[0, ~5000]` for our checkpoints. Default `λ_pred=1e-3` brings it into the same magnitude as the sparse task reward (0 or 1). If you log un-weighted, the cost will look enormous — read `info["r_pred"]` (post-scaling) for the actual contribution. |
| **Sparse task reward** | LIBERO success is binary, fired only when the predicate is satisfied. PPO from scratch with a sparse reward is hard. Consider adding a potential function based on bowl-to-plate distance for early shaping, or warm-starting from BC. |
| **Per-env memory** | Each env loads its own predictor checkpoint (~50 MB) plus LIBERO env state (~1.5 GB headless). Plan ~2 GB per `SubprocVecEnv` worker. |
| **Predictor inference cost** | ~50 ms per forward pass on a 3070. With `predictor_every_k=1` you spend roughly as much time on predictor as on the env step. Use `predictor_every_k=4` for a 4× sim throughput boost at minor cost-signal latency. |

## Vectorization

```python
from stable_baselines3.common.vec_env import SubprocVecEnv
from planner.policy.safe_rl_env import SafeLiberoEnv

def make():
    return SafeLiberoEnv(
        bddl_file="...", demo_hdf5="...", ckpt="...",
        lambda_pred=1e-3, lambda_dmg=1.0, failure_prob=0.1,
    )

vec = SubprocVecEnv([make for _ in range(16)])
```

Use `predictor_device="cpu"` if you'd rather keep all GPUs for the policy network. Predictor is a small UNet and runs fine on CPU at ~150ms/step (slower than GPU but doesn't contend).

## Three suggested training configs

Match the three reward-shape regimes from the OopsieVerse follow-up plan:

```python
# A. OopsieVerse reproduction (damage-only, no predictor, no failures)
SafeLiberoEnv(..., ckpt=None, lambda_pred=0.0, lambda_dmg=1.0,
              failure_prob=0.0)

# B. Predictor-aware (no damage cost, modest failure rate)
SafeLiberoEnv(..., ckpt=CKPT, lambda_pred=1e-3, lambda_dmg=0.0,
              failure_prob=0.1)

# C. Combined
SafeLiberoEnv(..., ckpt=CKPT, lambda_pred=5e-4, lambda_dmg=0.5,
              failure_prob=0.1)
```

## Smoke test (CLI)

```bash
PYTHONPATH=external/LIBERO:. external/LIBERO/.venv/bin/python -m planner.policy.safe_rl_env \
  --bddl external/LIBERO/libero/libero/bddl_files/libero_spatial/<task>.bddl \
  --demo datasets/libero/raw/libero_spatial/<task>_demo.hdf5 \
  --ckpt notebooks/model_playground/cluster_download/06172026/dualgated_state_rgb/best_ep08_val0.0648.pt \
  --failure_prob 1.0 --steps 25 \
  --lambda_pred 1e-3 --lambda_dmg 1.0
```

Prints first/last 3 steps of a random rollout plus any step where a failure injects.

## File layout

```
planner/policy/
  safe_rl_env.py             # this env
  safe_action.py             # ObsWindow, query_risk, policy wrappers
  libero_env_failure.py      # EnvFailureScheduler
planner/risk/
  damage.py                  # DamageAccumulator + LIBERO_DAMAGE_PARAMS
  inference.py               # ContactPredictor, marginal_heatmap
scripts/safety/
  safety_rollout.py          # build_entity_masks (reused at env init)
external/LIBERO/.venv/       # sidecar venv with robosuite==1.4.0 + libero
```

## Where the reward weights come from (read this before tuning)

- `lambda_pred=1e-3` was chosen so that the typical per-step `pred_risk ≈ 1000` becomes a `-1.0` cost per step, comparable to the `+1.0` sparse task reward over an episode.
- `lambda_dmg=1.0` is comparable to OopsieVerse's reported PPO config (Place Plate, paper §V.B). Δd_mech values are `O(0.001 – 0.5)` per step; this gives a meaningful but not dominating penalty.
- Neither is "right" — they're starting points. **Always tune** on your task. Reasonable sweep: `λ_pred ∈ {1e-4, 1e-3, 1e-2}`, `λ_dmg ∈ {0.1, 1.0, 10.0}`, full factorial × 3 tasks.
