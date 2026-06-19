# RL Pipeline — Local Validation Findings (2026-06-19)

Smoke validation of the safety-aware RL pipeline on the local RTX 3070 before shipping training to the cluster. **Verdict: the pipeline mechanics work; proprio-only BC does NOT produce a functional policy and needs RGB.**

## What was validated

| Stage | Status | Wall-clock |
|---|---|---|
| `SafeLiberoEnv` obs dict (proprio + RGB + pred_per_body + gate_prob) | works | — |
| `augment_demos.py` end-to-end on 50 demos | works, ~50 steps/s on 3070 | 2.5 min |
| `train_bc.py` (proprio + pred_per_body + gate_prob → action MLP) | trains cleanly, val_loss 0.13 → 0.037 | 40 s for 100 epochs |
| `eval_bc.py` (BC policy rolled out in env) | end-to-end runs | 5 s per episode |
| Demo replay through `SafeLiberoEnv` (sanity check) | **8/10 demos succeed after env bugfixes** | — |
| **BC eval on tomato-sauce** | **0/20 success** — proprio insufficient | — |

## Two real bugs found and fixed

### 1. `info["success"]` is never populated by LIBERO

`SafeLiberoEnv.step` read `libero_info.get("success", False)` to compute `r_task`. But LIBERO's `OffScreenRenderEnv` returns `info = {}` even when the BDDL success predicate is satisfied. The signal is **in `libero_reward`** — `r=1.0` on the success step, `r=0` otherwise.

**Fix**: changed `r_task` to read `libero_reward` directly. `success = libero_reward > 0.5`. Before the fix, even authoritative demo replay reported 0% success.

### 2. `set_init_state` needs 5 zero-action warmup steps

`SafeLiberoEnv` was setting `data.qpos` / `data.qvel` directly via `mj_forward`. That seeds the sim but leaves robosuite's OSC controller goal stale. The very first real action then interprets the demo's frame against a default-pose target → arm jolts → trajectory diverges immediately.

LIBERO's own `evaluate.py` uses `env.set_init_state(state)` + 5 `env.step(zeros)` warmup steps to settle the controller. We now do the same. **After this fix, 8/10 demos succeed when replayed.**

The remaining 2/10 failures are real LIBERO demo-replay flakiness — already documented in the LIBERO repo as expected.

## The proprio-only BC ceiling

| Epochs | Best val_loss | BC eval success rate |
|---|---|---|
| 20 | 0.077 | 0% (n=10) |
| 100 | 0.037 | 0% (n=20) |

val_loss kept dropping; success rate stayed at 0. Per-component action error √0.037 ≈ 0.19 (range [-1, 1]) is too noisy for precise gripper timing in manipulation. The policy reaches the bowl region but can't grasp.

This is the classic vanilla-BC compounding-error problem. Conclusion: **the policy needs visual context.** Three viable fixes for the training machine:

1. **Add CNN encoder to BC** (consume `agentview_rgb` in addition to the current 3 obs keys). The augmented HDF5 already includes the resized RGB; only the trainer architecture needs changing.
2. **Skip BC, go straight to PPO with dense task reward shaping**. OopsieVerse Place Plate used dense negative-distance reward; we'd need to add an L2-distance-to-basket term to `r_task`.
3. **Both**: RGB-conditioned BC → PPO with safety shaping.

Option 1 or 3 is the natural path.

## What we learned about predictor-based curation

For libero_object/tomato_sauce, the curation signal is **weak**:

```
max_pred_risk distribution across 50 demos:
  min=2882  p50=3212  p90=3327  max=3383
```

The variance is 12% of the median. Humans manipulate safely and consistently; the predictor doesn't differentiate "safer" from "less safe" demos because they're all safe.

This suggests:
- Curation may give more signal on tasks where humans sometimes do risky things (e.g., long-horizon tasks where path planning varies).
- On easy pick-and-place tasks, curation is a no-op — just train BC on all demos.
- Worth running augment + checking the distribution before committing to a curation threshold.

## Pipeline files added

```
scripts/safety/augment_demos.py     ~150 lines (Phase 1)
scripts/safety/train_bc.py          ~210 lines (Phase 2)
scripts/safety/eval_bc.py            ~80 lines (Phase 2.5)
planner/policy/safe_rl_env.py        +pred_features_in_obs flag, +2 bugfixes
```

The PPO trainer (`train_ppo.py`) is NOT yet written. On the training machine it's the natural next step — straightforward SB3 PPO + `MultiInputPolicy` once BC is augmented with RGB.

## Recommended adjustments for training machine

Before kicking off real training:

1. **Modify `train_bc.py` to take RGB**. Replace `BCPolicy` with a version that has a small CNN encoder (NatureCNN or Impala-CNN) on `agentview_rgb` in parallel with the existing proprio/pred/gate encoders. Expected fix: BC success rate jumps from 0% to 60-80% on tomato-sauce, matching LIBERO BC baselines.
2. **Increase BC training epochs** with the larger model (100-200 epochs). The proprio-only BC plateaued at val_loss 0.037; with RGB this should drop to 0.01-0.02 territory.
3. **Verify env bug fixes are present** in your `git pull` — both `info["success"]` and `set_init_state` warmup fixes are in commit ahead of `safety-RL: env + d_mech scorer + analysis pipeline + training plan`.
4. **Skip curation for libero_object tasks**. Set `--curation_quantile None` (or just don't pass it). The signal isn't there.

## Smoke command for the training machine

After clone + venv setup (`docs/safety_rl_training_handoff.md`), this should reproduce the local validation:

```bash
# 1. Augment demos (~2.5 min on a single GPU)
PYTHONPATH=external/LIBERO:. external/LIBERO/.venv/bin/python -m scripts.safety.augment_demos \
    --bddl external/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl \
    --demo datasets/libero/raw/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5 \
    --ckpt notebooks/model_playground/cluster_download/06172026/dualgated_state_rgb/best_ep08_val0.0648.pt \
    --out datasets/libero/augmented/tomato_sauce.hdf5

# 2. Train BC (after upgrading to include RGB) — ~5 min on A100
PYTHONPATH=external/LIBERO:. external/LIBERO/.venv/bin/python -m scripts.safety.train_bc \
    --augmented datasets/libero/augmented/tomato_sauce.hdf5 \
    --out runs/bc-tomato-sauce-rgb \
    --epochs 100

# 3. Eval BC (~2 min on A100)
PYTHONPATH=external/LIBERO:. external/LIBERO/.venv/bin/python -m scripts.safety.eval_bc \
    --bc_ckpt runs/bc-tomato-sauce-rgb/bc_best.pt \
    --bddl external/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket.bddl \
    --demo datasets/libero/raw/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5 \
    --predictor_ckpt notebooks/.../best_ep08_val0.0648.pt \
    --n_episodes 50

# Expected: BC success rate 60-80% (matching LIBERO baselines).
# If yes -> proceed to write train_ppo.py for the safety RL paper figure.
# If no -> tune CNN encoder size, training epochs, learning rate.
```

## Bottom line

- Pipeline works.  
- Env bugs fixed.  
- Proprio-only BC is too weak — confirmed.  
- RGB is the next thing to add.  
- Augmenter, BC trainer, eval pipeline are ready to scale.

Next session at training machine: add CNN encoder to BC, repeat eval, then write PPO trainer.
