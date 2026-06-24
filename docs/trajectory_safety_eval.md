# Trajectory safety evaluation suite

Physics-grounded evaluation of robot trajectories under hardware failure. You give
it a trajectory (from any source); it samples many random failures along the
trajectory, settles MuJoCo physics, measures realized object damage with the
OopsieVerse health model, and reports expected/total/worst-case damage. No contact
predictor is involved at eval time — this is pure ground truth, so trajectories can
be compared and a predictor's ranking can be validated against it.

## Components

| File | Role |
|---|---|
| `scripts/safety/eval_trajectories.py` | Monte-Carlo failure-cost evaluator (the suite). |
| `scripts/safety/play_failure.py` | Renders a failure as mp4 with live per-object **health bars** (`--gallery` = one failure per mode at spread injection points). |
| `scripts/safety/viz_traj_damage.py` | 3-panel figure: safety comparison, damage-vs-failure-time, env/carried split. |
| `scripts/safety/diagnose_damage.py` | Instruments one failure — per-pair contact forces and per-body damage (held filter on/off) — for "why (no) damage?" debugging. |
| `planner/risk/damage.py` | `DamageAccumulator` — the health/damage model. |
| `planner/risk/severity.py` | Object value/severity, incl. `load_severity_config` / `resolve_severity`. |
| `scripts/libero/gen_diverse_trajs.py` | Trajectory generator + `setup_scene()` / `build_candidates()` reused by the eval. |
| `configs/severity.yaml` | Example manual per-object severity config. |

## How the eval works

For one trajectory it draws `--n_samples` failures, each a **random time** along the
trajectory × a **random mode** (uniform, or `--mode_dist v2` for the realistic
likelihood). For each: seed the sim at that step, inject the failure, settle
`--settle_steps`, accumulate per-object `d_mech`. Trajectories share one
common-random-numbers sample plan (`--seed`) so comparisons are paired.

The 5 failure modes (`aligned_failure_modes`): `GRIPPER_OPEN`, `SLIPPERY_GRIP`,
`SINGLE_JOINT` (joint4), `MULTI_JOINT` (joint2,4), `ALL_JOINTS`.

### Metrics (per trajectory)
- **E[cost]** ± SE — expected realized damage if a failure strikes at a random
  moment (the headline; lower = safer), with Monte-Carlo standard error.
- **total** — sum over samples; **std / median / p90 / max** — the cost distribution.
- **catastrophe (carried / bystander)** — fraction of samples that drop the carried
  object / a bystander below half health (split so the route-invariant "you dropped
  it" doesn't mask the route-dependent collateral damage).

Cost channel `--cost {all,env,carried}` selects which objects count (all / bystanders
only / carried only).

## Severity config (object value ↔ HP loss)

`--severity_config configs/severity.yaml` is a manual `{object_substring: value}`
map. It weights each object's HP loss in the cost — `cost = Σₑ severity(e)·damage(e)`
— so damage to a high-value object counts proportionally more, and the **same
severity feeds the contact predictor**, so both sides value objects identically.
Without a config the cost is raw (unweighted) damage.

```yaml
akita_black_bowl: 10
ramekin: 10
plate: 3
table: 0      # structural, don't care
```

## Predictor validation

With `--source gen` the suite generates routing-style candidates (each carrying the
predictor's risk score) and prints a **predictor-vs-physics** block:
`Spearman(pred_risk, E[cost])`, the predictor's pick vs. the physics-safest
trajectory, and whether they agree (`rho_pred_vs_physics` in the JSON). The goal is
that the trajectory the predictor prefers scores lowest in E[cost].

## Damage model (`DamageAccumulator`)

Two channels, summed per body each step:
1. **Sustained force** (crushing): `rate · max(α·F∥ + β·F⊥ − threshold, 0)` —
   force integrated over time; dominates when the arm presses an object.
2. **Impact energy** (drops/hits): `IMPACT_DAMAGE_COEFF · α · ΔKE` — damage from the
   kinetic energy a body loses in a collision; captures brief impacts the
   time-integral misses.

Conventions:
- Object-vs-object contacts damage **both** bodies (a bowl hitting a plate hurts the
  plate, not just the bowl).
- A **held** object (`held_body_ids`) ignores the robot's grip force (it only takes
  damage from environment impacts) — the grip is not damage.
- Fragile-object `rate` is bumped ~10× vs. the original (which was tuned for
  sustained loads and crushed brief impacts to ~0).
- These are global changes: every realized-damage number (this suite,
  `safety_rollout`) reflects them.

`release_carried()` (in `eval_trajectories.py`) is used on gripper-failure modes so
synthetic IK grips — which pin the object's centre in a gripper narrower than the
object — actually drop it (demo grasps release on their own).

## Usage

```bash
# Evaluate given trajectories (any source) — 100 random failures each, report E[cost]
conda run -n failbench_env python -m scripts.safety.eval_trajectories \
    --traj a.npz b.npz --task <task> --split libero_spatial --n_samples 100 \
    --severity_config configs/severity.yaml --json out/eval/mine.json

# Validate the predictor: generate routes, compare predicted risk vs realized damage
conda run -n failbench_env python -m scripts.safety.eval_trajectories \
    --source gen --task <task> --split libero_spatial --object bowl \
    --n_samples 100 --settle_steps 300 --severity_config configs/severity.yaml

# Evaluate a LIBERO demo
conda run -n failbench_env python -m scripts.safety.eval_trajectories \
    --source demo --demos demo_0,demo_1 --task <task> --split libero_spatial

# Play a failure with health bars / a per-mode gallery
conda run -n failbench_env python -m scripts.safety.play_failure \
    --task <task> --split libero_spatial --demo demo_0 --gallery \
    --out figures/failure_gallery.mp4
```

`.npz`/`.npy` trajectories hold `arm_qpos` (T,7) or `qpos` (T,nq), optionally
`carry_flags` (T,). `--task/--split/--demo` only supply the scene (MJCF + which body
is carried).

## Status / caveats

- Drops in LIBERO scenes are low-energy (carries are ~10 cm; objects are light), so
  realized damage is modest; significant damage comes from joint-collapse crushing.
- Predictor-vs-physics across tasks is mixed: strong on some (bowl_between ρ≈+0.66 at
  N=100) but weak/negative where carry height drives predicted risk and real drop
  damage in opposite directions (bowl_stove). N≥100 is needed for a stable ρ (N=30 is
  noise-dominated).
