# Safety-Aware RL Training Plan — PPO vs DSRL Comparison

**Goal**: train a policy that consumes our `GatekeeperCoordFiLMUNet` contact predictor as a safety advisor and produces measurably safer behavior on LIBERO manipulation tasks. Build on OopsieVerse's results without replicating their compute scale.

**Status**: training environment + predictor + augmentation pipeline ready. Need to commit to one of two RL training directions.

This document compares the two viable paths, lays out compute requirements per direction, and ends with a concrete recommendation.

---

## Starting state — what we already have

| Component | File | Status |
|---|---|---|
| Gym-compatible LIBERO env with predictor + d_mech reward | `planner/policy/safe_rl_env.py` | done |
| Env documentation | `planner/policy/SAFE_RL_ENV.md` | done |
| Failure injection (5 modes) | `planner/policy/libero_env_failure.py` | done, legacy gripper semantics |
| OopsieVerse-style mechanical damage scorer | `planner/risk/damage.py` | done, validated |
| Pretrained contact predictor (val_heat=0.0648) | `notebooks/.../best_ep08_val0.0648.pt` | done |
| Predictor inference helpers | `planner/risk/inference.py` | done |
| 600-rollout LIBERO sweep with new metric | `out/safety_rollouts_damage/.../results.csv` | done, ρ=0.641 |
| ~50 demos per task | `datasets/libero/raw/libero_*/`...`_demo.hdf5` | LIBERO standard |
| OopsieVerse RL hyperparameters (PPO + DSRL) | researched, in this doc | done |

What's missing: a pretrained flow-matching policy on LIBERO (needed for DSRL only). PPO-from-BC has all ingredients in hand.

---

## Two paths at a glance

|  | **Option A: PPO from BC warmstart** | **Option B: DSRL on flow-matching policy** |
|---|---|---|
| **What gets trained** | full visuomotor policy (CNN + MLP) | small "noise policy" MLP only |
| **What's frozen** | nothing (full backbone updates) | IL encoder + flow backbone |
| **Reference task** | OopsieVerse PPO `Place Plate` (0%→80%) | OopsieVerse DSRL `Shelve Cereal Box` (13%→33%) |
| **Total env steps** | 1.5M (BC) + 2M (PPO) per task | 5.2 × 10⁵ per task |
| **Compute** | high (full CNN backbone updates) | low (frozen backbone) |
| **Prep work** | augment demos, write BC trainer | train a flow-matching IL policy first |
| **Architecture transfer pain** | yes (BC weights → SB3 PPO actor) | none (sidecar network) |
| **Hyperparameters** | well-documented (SB3 defaults + RLinf) | well-documented (OopsieVerse Appendix D.4) |
| **Paper claim** | "predictor-aware PPO beats damage-only PPO" | "predictor steers a frozen IL policy toward safer behavior" |
| **Most similar to OopsieVerse** | PPO Place Plate, but on LIBERO | DSRL Shelve Item, but on LIBERO |
| **Risk of failure** | medium — full backbone training under sparse reward is hard | low-medium — depends on flow backbone quality |
| **Wall-clock per task** | 3–5 days on A100 MIG | 1–2 days on A100 MIG |

---

## Option A — PPO from BC warmstart

### The pipeline

1. **Augment demos** with predictor features (1×). Run the predictor on every demo timestep, save `pred_per_body`, `gate_prob` alongside the original (obs, action) pairs.
2. **Curate demos** by the predictor's alarm level. Drop top 25-33% most-alarming demos. Match OopsieVerse §V.A "Health-Filtered Episodes" curation. Saves training time and biases BC toward safer behavior.
3. **BC pretrain** on curated demos. Same architecture as the PPO actor. 20 epochs, Adam 1e-3, MSE on demo action.
4. **PPO fine-tune** with shaped reward `r = r_task − λ_pred · pred_cost − λ_dmg · Δd_mech`. Load BC weights into SB3 PPO actor.
5. **Ablate**: vanilla / damage-only / predictor+damage / combined.

### Architecture (Option B from earlier discussion — proprio + per-body + gate + RGB)

```
agentview_rgb (240, 320, 3) ──→ CNN encoder (NatureCNN or Impala-CNN)
proprio (18,)               ──→ MLP encoder (128)
pred_per_body (K=13,)       ──→ MLP encoder (64)
gate_prob (1,)              ──→ MLP encoder (16)
                                          │
                                          ▼
                                       concat ──→ Trunk MLP (512, 512) ──→ actor / critic heads
```

Three predictor information channels into the policy:
- **Reward gradient**: `−λ_pred · sum(heatmap)` shapes the value function.
- **Per-body attention hint**: policy reads which objects are at risk.
- **Gate prob**: policy reads how confident the predictor is.

The policy retains its own visual processing through the CNN encoder so it can sanity-check the predictor's reading against the raw scene.

### Reward weights — starting point

| Term | Weight | Magnitude per step | Rationale |
|---|---|---|---|
| `r_task` | 1.0 (LIBERO sparse) | 0 or +1 | LIBERO success predicate |
| `r_pred` | λ_pred = 1e-3 | typically −0.5 to −2.5 | brings pred_risk [0, 2000] into same magnitude as task reward |
| `r_damage` | λ_dmg = 1.0 | typically −0.001 to −0.5 | OopsieVerse PPO Place Plate w=2; ours is comparable |

Sweep at second-pass: `λ_pred ∈ {1e-4, 1e-3, 1e-2}`, `λ_dmg ∈ {0.1, 1.0, 10.0}` after first task ablation converges.

### PPO hyperparameters

| Param | Value | Source |
|---|---|---|
| algorithm | PPO with clipping | sb3 PPO |
| learning rate | 3e-4 | SB3 default, CleanRL continuous |
| clip ε | 0.2 | universal |
| n_steps (per env) | 128 | CleanRL continuous |
| batch_size | 256 | scaled with n_envs |
| n_epochs | 10 | RLinf for VLA |
| GAE λ | 0.95 | universal |
| γ | 0.99 | universal |
| ent_coef | 0.0 | SB3 default |
| vf_coef | 0.5 | SB3 default |
| max_grad_norm | 0.5 | SB3 default |
| n_envs (SubprocVecEnv) | 4 (local 3070) / 16 (A100 MIG) | memory bound |
| total_timesteps per task | 2M | analogy to robosuite Lift |

### Expected wall-clock per task

| Phase | RTX 3070 | A100 MIG 3g.40gb | Full A100 |
|---|---|---|---|
| Demo augmentation (one-time) | 5 min | 3 min | 2 min |
| BC pretrain (20 epochs, 50 demos) | 30 min | 15 min | 10 min |
| PPO 2M steps, 4 envs | ~90 hours | — | — |
| PPO 2M steps, 16 envs | OOM | 40-60 hours | 25-35 hours |
| **Total per task (single seed)** | impractical | 2–3 days | 1–1.5 days |

For 3 reward variants × 3 tasks × 3 seeds (paper figure): 27 PPO runs ≈ 7 cluster-weeks on a single A100 MIG.

### Risk factors

- **Sparse task reward** + complex obs space. PPO from scratch with sparse +1 on LIBERO success is known-hard. Mitigation: BC warmstart gives the policy a reasonable starting distribution; OopsieVerse PPO Place Plate succeeded because they added dense distance shaping, which we lack.
- **BC→PPO weight transfer**. SB3's `MultiInputPolicy` parameter names are version-specific. The key mapping needs to be verified once and pinned.
- **Memory ceiling on RTX 3070**. CNN-in-policy + predictor-in-worker + LIBERO env = ~2 GB per worker; 4 workers max. Local debugging only.

---

## Option B — DSRL on a pretrained flow-matching policy

### The pipeline

1. **Train (or borrow) a flow-matching IL policy** on the 50 demos per task. Action chunk length T=8. Standard π₀-style transformer with a vision encoder. **This is the prerequisite** — we don't have one yet.
2. **Freeze the IL policy** (vision encoder + flow backbone).
3. **Train a small "noise policy"** `π_θ(μ, σ | obs)` that outputs a Gaussian over the initial flow noise `a⁰ ∈ ℝ^{8×7}`. Update via PPO with safety-shaped reward.
4. **Reward**: `r_task + clipped/normalized predictor cost` so the magnitudes match the sparse +1.
5. **Rollout**: sample noise from π_θ, run frozen IL on (obs, noise), execute the 8-step chunk, collect reward.

### Architecture

```
obs (RGB + proprio) ──┬──→ frozen IL encoder ──→ frozen flow backbone (NEVER UPDATES)
                       │                                ▲
                       │                                │ noise a⁰ ∈ ℝ^{8×7}
                       │                                │
                       └──→ trainable Noise Policy MLP ──┘
                            (5-layer, hidden 1024, SiLU)
                            outputs (μ, σ) of Gaussian over a⁰
```

The trainable surface is tiny — just the noise policy. The flow backbone never sees a gradient. This is what makes DSRL ~5–10× cheaper than full PPO in wall-clock.

### Hyperparameters from OopsieVerse Appendix D.4

| Field | Value | Note |
|---|---|---|
| Noise policy arch | 5-layer MLP, hidden 1024, SiLU | small but expressive |
| Mean clamp | [−1, 1] | keeps noise in flow backbone's training distribution |
| Std clamp | [0.1, 1.0] | prevents collapse and excess exploration |
| Sample clamp | [−2, 2] | hard tail truncation |
| Initial std | 0.5 | wider than fine-tuned VLA, narrower than uniform |
| Optimizer | Adam, lr=**1e-4** | lower than from-scratch PPO; delicate noise tuning |
| Clip ε | 0.2 | standard PPO |
| PPO epochs per iter | 5 | half of standard PPO |
| Minibatch size | 512 | larger because noise space is 56-D |
| Entropy coef | 0.0 | std clamp handles exploration |
| Grad clip | 1.0 | tight |
| Replay buffer | 2048 | small, recent-data-only |
| γ | 0.95 | shorter effective horizon (chunked execution) |
| GAE λ | 0.95 | standard |
| Total env steps | **5.2 × 10⁵** | dramatic compute savings vs full PPO |

### Reward — adapted for FailBench

OopsieVerse:
```
r_t = +1 (on completion)
    − normalized(d_mech + d_therm + d_fluid)    ∈ [-1, 0]
```

Our predictor-flavored:
```
r_t = +1 (on completion)
    − α · normalized(pred_risk)                   ∈ [-1, 0]
    − β · normalized(Δd_mech)                     ∈ [-1, 0]   (optional)
```

Normalization is critical here — without it the dense predictor cost (range [0, 2000+] per step over 8 chunked steps) would dwarf the sparse +1 success bonus and the policy would refuse to move. Three variants worth ablating:

- **Predictor-only**: α > 0, β = 0
- **Damage-only**: α = 0, β > 0  (closer reproduction of OopsieVerse)
- **Combined**: α > 0, β > 0

### Expected wall-clock per task

| Phase | RTX 3070 | A100 MIG 3g.40gb | Full A100 |
|---|---|---|---|
| Flow-matching IL pretrain (~2 days, **one-time per task**) | impractical | 2 days | 1 day |
| DSRL 520K env steps | impractical | 20–30 hours | 10–15 hours |
| **Total per task (single seed)** | not viable locally | 3 days | 1.5–2 days |

For 3 reward variants × 3 tasks × 3 seeds: 27 DSRL runs ≈ 4 cluster-weeks **plus** 3 × 2-day IL pretrains for the new tasks.

### Risk factors

- **Need to pretrain (or find) a flow-matching IL policy first.** Adds ~2 days/task to the schedule. Could use OpenVLA-OFT checkpoints if they fit LIBERO's action space — needs investigation.
- **Action-chunk length is fixed by the IL backbone.** Less flexible than per-step PPO. T=8 is OopsieVerse's choice — π₀ style.
- **Frozen backbone may have learned unsafe behaviors.** DSRL can only steer; it can't fix fundamental visuomotor bugs in the IL policy. The IL policy needs to be at least "successful most of the time" for DSRL to add safety margin on top.

---

## Compute environment options

We have access to three:

### 1. Local — RTX 3070 (8 GB)

- Good for: smoke tests, env wiring, single-rollout demos.
- Not viable for: any meaningful PPO/DSRL training. Memory is the bottleneck (4 envs max).
- **Use only for development and unit testing.**

### 2. GMU Hopper cluster — MIG slices, A100 3g.40gb

- Per-job MIG slice has ~40 GB VRAM. Fits 16-worker SubprocVecEnv comfortably.
- Job queue, MUSST set `CUDA_VISIBLE_DEVICES=MIG-...` per node (see `feedback_hopper_cuda_devices.md`).
- Setup overhead per job: ~5 min to set up the env + checkpoint paths.
- Best for: paper-quality experiments at scale (the 3×3×3 ablation matrix).
- **Use for the production training runs.**

### 3. Possible A100 access (full card)

- Roughly 2–3× faster than a 3g.40gb MIG slice for these workloads.
- If we can sustain access for a week, full Option A or B paper figure becomes practical.
- **Use if available for the heaviest sweeps.**

### Quick compute math (3-task × 3-variant × 3-seed paper figure)

| Compute | Option A (PPO from BC) | Option B (DSRL) |
|---|---|---|
| A100 MIG only | 5–7 weeks | 3–4 weeks + 1 week IL pretrains |
| Full A100 only | 2.5–3.5 weeks | 1.5–2 weeks + 0.5 week IL pretrains |
| MIG + occasional full A100 | 4 weeks | 2.5 weeks |

Option B is ~1.5× faster in total wall-clock, even after counting the IL pretraining overhead.

---

## Recommendation

**Start with Option A (PPO from BC).** It uses what we already have, doesn't depend on training a flow-matching policy first, and produces the most directly-comparable-to-OopsieVerse result (PPO with shaped reward, our predictor in place of their ground-truth damage scorer).

Reserve Option B (DSRL) as a follow-up experiment if Option A produces a strong baseline and you want a second paper-grade demonstration with a much smaller training surface — particularly useful if the reviewers ask "does this work for IL policies, not just from-scratch RL?"

### Decision rule

| If you want to … | Pick |
|---|---|
| Reproduce OopsieVerse's strongest baseline (Place Plate 0%→80%) and show our predictor improves it | **A** |
| Test whether the predictor can refine an existing policy without retraining | **B** |
| Match a small-compute deadline | **B** (after IL pretrain is amortized) |
| Maximize confidence in the result (the trainer is standard) | **A** |
| Train under a known-good ML recipe (SB3 PPO is bullet-proof) | **A** |

### Suggested order of execution

1. **Week 1**: Option A — Phase 0 (env tweaks for RGB + pred features in obs), Phase 1 (augment demos), Phase 2 (BC alone on tomato sauce → basket task). Verify BC reaches ~80% success.
2. **Week 2**: Option A — Phase 3 (BC→PPO transfer) + Phase 4 (PPO on tomato sauce with all three reward variants). Two seeds each.
3. **Week 3**: Option A — extend to milk → basket (harder, tall object). Two seeds each.
4. **Week 4**: Option A — extend to cookie box (libero_spatial, over-obstacle). Generate first Pareto plot.
5. **Week 5–6**: Option A — add cabinet drawer task; run 3 seeds across all 4 tasks for final paper figure.
6. **Week 7+**: Option B as supplementary experiment if results are strong.

Total wall-clock for Option A paper figure: ~6 weeks on a single A100 MIG (with the cluster queue absorbing parallelism). Or ~3 weeks on a dedicated full A100.

### Concrete first step (this week)

Add the `pred_features_in_obs` flag to `SafeLiberoEnv` and confirm the augmented demo file format matches what BC training will read. This unblocks both options simultaneously — Option B also wants predictor features in obs (passed to the noise policy).

---

## References

- OopsieVerse paper PDF: `/home/aaron/Downloads/oopsieverse.pdf` (Appendix D.4 has DSRL details, D.6 has PPO Place Plate)
- OopsieVerse repo: https://github.com/UT-Austin-RobIn/oopsieverse (env + scorer only, no trainer)
- DSRL paper: Wagenmaker et al. 2025, "Steering your diffusion policy with latent space reinforcement learning", arXiv:2506.15799
- PPO paper: Schulman et al. 2017, arXiv:1707.06347
- RLinf-VLA: arXiv:2510.06710 — pure-RL on LIBERO is essentially absent; consensus path is BC/VLA + PPO/GRPO
- SB3 PPO docs: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- robosuite controllers: https://robosuite.ai/docs/modules/controllers.html

## Memory references

- `feedback_libero_env_flip.md` — Y-flip on every env image before predictor input
- `feedback_gatekeeper_corpus.md` — predictor trained LIBERO-only
- `feedback_hopper_cuda_devices.md` — MIG override in Python before `import torch`
- `feedback_gripper_failure_semantics.md` — legacy actuator-gain semantics in env failure injection
- `project_safety_rollout.md` — predictor ρ = 0.641 vs realized damage (post-bugfix)
