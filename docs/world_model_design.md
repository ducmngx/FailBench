# Action + failure-conditioned world model — design doc

**Status:** draft, not implemented.
**Goal:** given a pre-failure state, a planned action sequence, and a failure descriptor (which mode fires, when, on which joints), predict the future robot trajectory.

This is a counterfactual rollout model: *"what happens to the arm if I send these joint targets, and at fraction f of the sequence joint j freezes?"*

The contact-prediction benchmark (`docs/contact_prediction_libero_spatial.md`)
is a special case of this — its target is the final contact pattern. Here we
predict the whole intermediate trajectory.

---

## 1. Data requirements

v2 as-is is insufficient. The runner currently records:

- pre-failure 8-frame window (state, RGB, depth)
- failure descriptor
- single final contact snapshot

For sequential supervision we need to add to the trial NPZ:

| field | shape | dtype | meaning |
|---|---|---|---|
| `post_qpos` | (T_post, 9) | float32 | qpos every sim step after failure injection |
| `post_qvel` | (T_post, 9) | float32 | qvel every sim step after failure injection |
| `post_action` | (T_post, A) | float32 | the action commanded each step (whatever the demo would have sent — already known) |
| `post_failure_mask` | (T_post,) | bool | whether each step is post-failure (always True for these, included for clarity when we eventually pad / interleave) |

`T_post = 30` (1.5 s at 20 Hz) is a reasonable starting point — long enough for
the arm to settle/hit something, short enough not to blow up storage. The
runner already simulates the post-failure settle internally; we just need to
log it.

Storage budget per trial: 30 × 9 × 4 × 3 ≈ 3 kB extra. On 45 k trials that's
~130 MB total — trivial vs. the current 93 GB.

Call this dataset **v3** to stay versioned and not break v2 consumers.

---

## 2. Inputs to the model (per trial)

| name | shape | encoder |
|---|---|---|
| `pre_state` | (T_pre=8, 18) | linear → d_model |
| `goal` | (K=3, 11) | linear → d_model (optional context) |
| `action_seq` | (T_post=30, A) | linear → d_model (the "what we plan to do" stream) |
| `failure_mode` | (5,) one-hot | linear → d_model |
| `failure_joints` | (7,) binary | linear → d_model |
| `fail_progress` | scalar | sinusoidal → d_model (when the failure fires inside the action seq) |

Scene tokens are **optional** for v1 of the world model. Keep them off
initially — kinematic-only is the cheap pilot. If trajectories diverge in
ways that depend on scene contact (table, obstacle), add per-frame DINOv2 of
the pre-failure RGB as a single context token.

---

## 3. Architecture: causal Transformer with a failure event token

```
                  prefix tokens
       ┌───────────────────────────────────┐
       │ pre_state_1 ... pre_state_8        │  (history)
       │ failure_event                      │  (mode+joints+progress, single token)
       │ goal_1 ... goal_K                  │  (optional)
       └───────────────────────────────────┘
                       │
                       │   followed by interleaved per-step
                       ▼
       ┌───────────────────────────────────┐
       │ a_1  s_1  a_2  s_2  ...  a_T  s_T │  (rollout stream)
       └───────────────────────────────────┘
```

- **Prefix tokens** are conditioning context; they have full mutual attention
  among themselves and the rollout stream attends to all of them.
- **Rollout stream**: actions and states are interleaved per step. Mask is
  causal — `s_t` attends to `{prefix, a_1..a_t, s_1..s_{t-1}}`. This is the
  standard Decision-Transformer / Trajectory-Transformer pattern.
- **Failure event** is one token. Its `fail_progress` field encodes *when*
  the failure fires; the model has to learn to apply the failure effect to
  the right rollout step. (Simpler than masking per-step.)

### Concrete shape

| | value |
|---|---|
| d_model | 256 |
| n_heads | 4 |
| n_layers | 6 |
| T_pre | 8 |
| T_post | 30 |
| K (goal) | 3 |
| token count | 8 + 1 + 3 + 30·2 = 72 |
| params | ~5 M |

Sits comfortably on the 3070 at batch 64.

### Output head

For each rollout state position `t`, predict `(Δqpos_t, Δqvel_t)` as 18
floats. Predicting deltas (not absolutes) is critical — it makes the model's
job stationary and stops it from having to learn the joint-limit boundaries
from scratch.

Reconstruct absolute trajectory by cumulative sum at eval time:
`qpos_t = qpos_{t-1} + Δqpos_t`.

---

## 4. Loss

```
L = MSE(Δqpos_pred, Δqpos_true) + λ_v · MSE(Δqvel_pred, Δqvel_true)
```

with `λ_v = 0.1` (qvel is noisier; downweight).

**Two extra terms to consider after a baseline lands:**

- *Free-joint loss mask*: failed joints have `Δqpos ≈ 0` because their
  actuators are dead. The model will trivially fit that — fine, but exclude
  failed joints from the MSE so the loss isn't dominated by predicting zero.
- *Long-horizon scheduled sampling*: at later epochs, replace some
  teacher-forced `s_t` with the model's own `ŝ_t` to harden against
  autoregressive drift.

---

## 5. Training protocol

| | value |
|---|---|
| dataset | v3 libero_spatial first (1/3 cost; same as benchmark) |
| split | demo-stratified 90/10 to start (we know in-dist is the easier setting) |
| batch | 64 |
| optimizer | AdamW, lr 3e-4, wd 1e-4 |
| schedule | LinearLR warmup (2 ep) → CosineAnnealingLR (28 ep) |
| epochs | 30 |
| seed | 0 (then 1, 2 for variance bars once we have one working run) |
| AMP | bf16 (3070 supports it; speeds up the rollout attention pattern) |

Expect ~3 min/epoch — smaller than the contact Transformer because no per-frame CNN.

---

## 6. Evaluation

Three layered metrics, all on the val split:

1. **Per-step Δqpos MSE.** The training loss; report on the val set.
2. **k-step rollout MSE** at k = 1, 5, 15, 30. Autoregressive — model feeds
   its own predictions back. This is what matters for planning.
3. **Terminal-contact agreement.** Render the model's final-step qpos through
   MuJoCo for a few hundred val trials and compute weighted MSE vs the v2
   contact heatmap. *Sanity check that the world model agrees with the contact
   benchmark on the terminal state.*

Plot trajectories: 9-panel grid (one per joint) showing predicted vs
ground-truth qpos over the 30 post-failure steps for a few sampled val
trials. Most informative single visual.

---

## 7. Counterfactual coverage caveat (Option B from the conversation)

This v1 design trains *one* action-trajectory per (state, failure) pair —
whatever the demo happened to do. For pure trajectory prediction that's
fine; for *planning over counterfactual actions* the model has to extrapolate
across actions it never saw.

Two ways to address later, without changing the architecture:

- **Action perturbation regeneration.** For each pre-failure state, run K
  perturbed action variants and 1 failure mode each. K=5 multiplies data 5×.
- **Action augmentation at train time.** Add Gaussian noise to `action_seq`
  and require the model to predict the matched perturbed trajectory (cheap
  but only valid if the simulator was queried for those actions).

The first is more honest; the second is a free pretraining trick.

---

## 8. Risks / things to test early

1. **Failure-event token under-attention.** A single token among 72 might
   get drowned. If it does, switch to repeating the failure descriptor as
   a per-rollout-step input concatenated to `a_t`. Test by ablating the
   failure token entirely — if val loss barely changes, that's the bug.
2. **Autoregressive drift.** k=1 may look great and k=30 may diverge.
   Standard fix is scheduled sampling. Report k=1/5/15/30 from the start to
   surface drift early.
3. **Per-joint balance.** The 7 arm joints have very different ranges and
   noise levels. Either standardise per-joint or use a smooth-L1 loss to
   stop the large joints from dominating gradient.

---

## 9. Files to create

| path | purpose |
|---|---|
| `planner/risk/world_model/__init__.py` | exports |
| `planner/risk/world_model/dataset.py` | v3 loader; assumes `post_qpos`/`post_qvel`/`post_action` arrays present |
| `planner/risk/world_model/model.py` | the causal Transformer above |
| `planner/risk/world_model/rollout.py` | autoregressive eval; k-step MSE; trajectory plotting helper |
| `scripts/world_model/regenerate_v3.py` | extends `LiberoRunner` to log post-trajectories, writes v3 NPZs |
| `scripts/world_model/train.py` | trainer mirroring `scripts/benchmark/train_one.py` |
| `scripts/world_model/eval.py` | k-step rollout metrics + plots |

---

## 10. What the post-failure state actually contains

A subtle but important property of the LIBERO runner: failed joints are
*passive*, not *frozen*. `LiberoRunner._apply_resistance` does two things
each step after the failure injection:

- **Healthy joints**: get `τ = qfrc_bias + Kp·(q* − q) − Kd·qd` (gravity
  compensation + per-joint PD toward the demo's last-commanded pose). They
  actively resist.
- **Failed joints**: had their actuator gains zeroed in
  `LiberoFailureInjector._kill_joint`. No torque is applied, but the joint
  itself is still free. Gravity pulls it, neighboring links drag it,
  contacts push it back. `qpos`/`qvel` evolve through MuJoCo's integrator.

So `post_qpos[t]` for a failed joint at frame t is *real free-fall
dynamics* — not frozen at the failure moment. This is exactly the supervision
signal we want: "given the failure set, this is how the arm actually drops."
No special handling needed.

**One regime change inside the post-failure window:**

| phase | what determines qpos_{t+1} | image matters? |
|---|---|---|
| Pre-contact free-fall | gravity + multibody coupling on current qpos/qvel | no — state is sufficient |
| Post-contact dynamics | scene geometry (table, obstacle locations) | yes — state alone can't disambiguate which obstacle was hit |

The cutoff is whatever frame the arm first contacts something. For typical
LIBERO arm poses, that's the first 0.3–0.8 s after failure (6–16 frames
at 20 Hz). The model has a much easier time on pre-contact frames than
post-contact ones — expect the k-step rollout error to spike sharply at
the moment of first contact.

## 11. When to add image conditioning

Following directly from §10, there are three natural thresholds:

1. **Pre-contact free-fall — image is wasted.** Next-state depends only on
   current qpos/qvel + gravity vector + multibody coupling. The state
   vector already encodes everything the model needs. Adding image just
   forces the model to learn to ignore it.

2. **Post-contact dynamics — image starts to matter.** Two failures that
   look identical in state can diverge after contact because the obstacles
   are in different places. State doesn't encode obstacle pose; image
   does. **This is the first natural threshold to enable vision.**

3. **Cross-scene generalisation — image is required.** As soon as
   libero_object / libero_goal enter training, the scene varies trial-to-
   trial and the model can no longer memorise "where the table is" from
   state. Same lesson as §17 of the contact-prediction work: vision wins
   OOD.

### Recommended sequence

a. **Pilot: state-only on libero_spatial.** Cheapest baseline, isolates
   the pre-contact regime where state is sufficient. Establishes the floor.
b. **Add vision specifically for the post-contact tail.** Concretely, gate
   image attention by `frames_since_contact`: tokens before the first
   predicted contact attend to state only; tokens after also attend to a
   small set of image tokens. Ablation gives a publishable single number
   ("image reduces post-contact MSE by X % with no change to pre-contact").
c. **Scale up to libero_object/goal.** Image goes on from the start; OOD
   benefit should widen.

### Lazy alternative if (b) feels too clever

Turn vision on from the start of the pilot but log per-frame loss broken
down by `frames_since_contact` (0 = pre-contact, ≥ 1 = post-contact). The
breakdown plot will show whether image is doing work, where, and by how
much — without needing a gated architecture. Use this if the gated design
costs more than a day to build; the diagnostic curve is what we care
about.

## 12. Concrete order of operations

1. Extend `LiberoRunner` to capture and return `post_qpos/qvel/action` arrays
   (verify on a single demo, write to NPZ via a new schema key).
2. Add v3 regen script using existing 8-EGL-worker setup — pilot on
   libero_spatial only (~3 h).
3. Write `dataset.py` against v3, verify with a notebook (`world_model_v3_inspect.ipynb`).
4. Write `model.py` for the causal Transformer; smoke-test on 100 trials.
5. Write `train.py`; run a quick 5-epoch pilot to confirm the loss goes down.
6. Full 30-epoch run, report k-step MSE.
7. Terminal-contact agreement check vs v2 benchmark — sanity-checks both
   the world model and the contact benchmark labels.

Steps 1-2 are the long pole (regen). Steps 3-5 in parallel can land in a
day. The whole pilot is a 2-3 day investment before the first real result.
