# FailBench Heatmap Regressor — Experiment Report

Goal: predict the per-config 2D contact-density heatmap (the spatial term
`P_interaction` in our risk model) from pre-failure robot state and scene
context, then integrate per-entity scores for the planner's `Σ score(eᵢ)·S(eᵢ)`
cost.

We add complexity in **one axis at a time** so that any change in val MSE /
Spearman ρ is attributable to that single change. All stages share the same:

- Dataset: `datasets/v10/<scene>/<task>/exp_*.npz` + `targets.npz`
- Loss: pixel-wise MSE on standardised heatmaps
- Optimizer: AdamW (lr 3e-4, wd 1e-4), cosine schedule
- Split: traj-key-level 90/10 (no leakage of (task, traj_id) across train/val)
- Hardware: RTX 3070

Metrics:
- **val MSE (destd)**: pixel MSE in original heatmap units (smaller is better)
- **baseline MSE**: error if we predicted the train per-pixel mean (fixed reference)
- **% over baseline**: `1 - val_mse / baseline_mse`
- **Spearman ρ**: rank correlation of per-obstacle integrated scores
  (predicted vs target) on val set

## Dataset size — what each stage actually trained on

The full v10 dataset has **16,532 configs across 5 scenes**:

| scene | configs | grid_shape |
|---|---|---|
| scene_level2     | 2,834 | (43, 70)   |
| scene_kitchen    | 2,750 | (86, 130)  |
| scene_workshop   | 3,725 | (93, 133)  |
| scene_grocery    | 3,800 | (66, 130)  |
| scene_cluttered  | 3,423 | (75, 110)  |
| **Total**        | **16,532** | — |

Per-stage coverage (90/10 traj-key split):

| Stage | scenes used | train configs | val configs | v10 covered |
|---|---|---|---|---|
| 0 — state-only MLP | scene_level2 | 2,561 | 273 | 2,834 / 16,532 (17%) |
| 1 — conv decoder | scene_level2 | 2,561 | 273 | 2,834 / 16,532 (17%) |
| 2 — task / goal ablations | scene_level2 | 2,561 | 273 | 2,834 / 16,532 (17%) |
| 3 — vision (+ ablation) | scene_level2 | 2,561 | 273 | 2,834 / 16,532 (17%) |
| 4 — multi-scene + depth | all 5 | 14,884 | 1,648 | **16,532 / 16,532 (100%)** |
| 5 — multi-scene baseline | all 5 | 14,884 | 1,648 | **16,532 / 16,532 (100%)** |

**Stages 0–3 used scene_level2 only — 17% of v10**, by design (within-scene
attribution before introducing scene-level variance). Only Stages 4 and 5
trained on the full dataset.

## Framing note (2026-04-29) — focusing on reconstruction

After Stage 5, the per-scene Spearman ρ across stages was tracked alongside
MSE because it maps directly onto the planner's `Σ score(eᵢ)·S(eᵢ)` cost.
Most of what's noisy in the report is in that ρ axis — the MSE story is a
clean monotone progression with no real regressions.

**Decision**: from this point on, training scripts optimise and
checkpoint **only on reconstruction MSE**. Per-obstacle Spearman becomes
an *offline evaluation* metric, computed on a saved checkpoint via
`notebooks/eval_model.ipynb` whenever needed.

Concretely:
- `train_demo.py` and `train_multiscene.py` no longer compute ρ during
  training (the per-validation MuJoCo loads + `integrate_per_entity` were
  the slowest validation step).
- Only `best_mse.pt` is saved; `best_rho.pt` is gone. `best.pt` is kept
  as a legacy alias of `best_mse.pt`.
- The "Reading the Stage 5 results" subsection still applies — we keep
  the analysis there as recorded interpretation. New runs just don't
  produce ρ in the training history.

The earlier-staged narrative (the MSE/ρ decoupling, the ρ-peaks-early
pattern) stays valuable as documented context. We reintroduce ρ as a
training objective only when reconstruction quality plateaus and we
need to start trading pixel error for planner-relevant ranking — see
the **Stage 9 (proposed) — Loss redesign** section near the end of the
report.

---

## Stage 0 — State-only MLP baseline (current)

### Pre-stage thinking

Smallest-thing-that-could-work: take what's already in `pre_qpos`, `pre_qvel`,
`pre_ee_pos` and learn a map to a flat heatmap. No vision, no goal info, no
scene awareness. We expect this to be a weak floor — it's a config→outcome
predictor that doesn't know where in the world the arm is going or what's near
it.

### Setup

- **Input** (17): `concat(pre_qpos[7], pre_ee_pos[3], pre_qvel[7])`
- **Model**: `HeatmapMLP` — Linear(17→256→512→1024→ny·nx), SiLU + Dropout 0.1
- **Output**: dense `ny·nx` linear head reshaped to `(ny, nx)`; for
  scene_level2 the grid is `(43, 70)` → 3,010 cells
- **Scene**: scene_level2 only

### Results — scene_level2 (200 epochs)

| metric | value |
|---|---|
| baseline MSE (predict mean) | 12.66 |
| best val MSE (destd) | 3.66 |
| final val MSE (destd) | 3.71 |
| % reduction over baseline | **71%** |
| best Spearman ρ (per-obstacle) | ≈ 0.42 |
| training time | ~2m 46s |

### Post-stage thinking

The model clears the trivial floor by a wide margin (71% over predict-mean),
which says state alone *does* carry signal — different qpos / qvel really do
imply different post-failure contact distributions. But Spearman ρ ≈ 0.42 is
weak: the per-obstacle ranking induced by the predicted heatmap only loosely
matches the truth. For a downstream planner that picks "go left, not right,"
0.42 is not enough.

#### Training dynamics

- **Convergence (val MSE within 5% of best): epoch 61 / 200.** Two thirds of
  the schedule is just the model fine-tuning the cosine LR tail. Roughly
  60-80 epochs would have given the same quality.
- **Best Spearman ρ = 0.461 at epoch 57**, but the last-20-epoch mean is
  **0.421**. The model's best *ranking* fit happens *before* the best
  *MSE* fit (epoch 123) and then degrades. This is the first hint of a
  pattern that recurs across every stage: pixel-MSE optimisation moves
  mass *within* obstacle footprints in ways that don't help (and sometimes
  hurt) per-obstacle ranks. Fitting the heatmap and fitting the obstacle
  ranking are not the same problem.
- **Train/val final gap = 1.73× (val_loss 0.700 vs train_loss 0.405).**
  That's a healthy gap — present but not catastrophic. The dense Linear
  head has 3M params for 3010 cells; it has plenty of capacity to overfit
  but a Stage-0 model on 2.5k configs hits a soft ceiling around the
  training-mass distribution.

#### What the predictions look like

`preds.png` shows the qualitative pattern at the best-MSE checkpoint:
predictions are visibly *smoother* than targets (the dense head is bad at
producing sharp blobs) and are biased toward *the average heatmap* — the
model puts mass in the workspace centre because that's where most contacts
land across all configs. Configs whose actual contacts cluster off-centre
(far edge picks, alt-object picks) get under-predicted on those off-centre
cells; their per-obstacle ranks suffer.

#### Hypotheses for what's missing (in order of suspected impact)

1. The Linear head has no spatial prior — neighbouring heatmap cells are
   independent params; this likely explains the salt-and-pepper noise and
   makes per-obstacle integrals jittery. → Stage 1.
2. The model has no idea where the arm is *heading* (goal_pos, task). The
   "predict the average" failure mode above is exactly what task context
   should fix. → Stage 2.
3. The model can't see clutter / free space — limits the kind of
   *config-specific* prediction that would beat the average. → Stage 3-4.
4. One scene only ⇒ no transfer claim. → Stage 5.

#### What this tells us about the metric, before any later stage

The **MSE/ρ decoupling pattern** (best ρ at ep 57, best MSE at ep 123,
ρ degrading after its peak) is already visible here. It will only get
sharper in later stages, and ultimately drives the dual-checkpoint
(`best_mse.pt` + `best_rho.pt`) tracking we add in Stage 2.

---

## Stage 1 — ConvTranspose decoder (structural prior, same input)

### Pre-stage thinking

The Stage 0 head is `Linear(1024 → 3010)` — it predicts each of the 3,010
heatmap cells as an independent function of the bottleneck. There is no inductive
bias that says "cell (i, j) is next to cell (i, j+1)." Since target heatmaps
are smooth (Gaussian-blurred σ=2 cm), a conv decoder should fit them with
fewer parameters and generalise better.

We expect:
- Smoother predictions (less salt-and-pepper noise)
- Lower val MSE; rough budget 10–20% reduction vs Stage 0
- Same or better Spearman ρ (smoother heatmaps → more stable per-entity integrals)

If this *doesn't* help, the bottleneck is the encoder, not the head — useful
diagnostic that points us at Stages 2-4 sooner.

### Setup

- **Input**: same 17-dim state vector
- **Model**: `HeatmapConvDecoder`
  - Encoder: Linear(17→256→512), SiLU + Dropout 0.1
  - Project: Linear(512 → 64·6·9) → reshape `(64, 6, 9)`
  - Decoder: 3× UpBlock (Upsample×2 + Conv3×3 + SiLU + Conv3×3 + SiLU)
    channels 64→32→16→8, spatial 6×9 → 12×18 → 24×36 → 48×72
  - Head: Conv3×3 → 1 channel, bilinear-resize to (43, 70)
- Same loss, optimizer, schedule, split, epochs as Stage 0

### Results — scene_level2 (200 epochs, seed 0)

| metric | Stage 0 (MLP) | Stage 1 (conv) | Δ |
|---|---|---|---|
| params | ~3.5M | 1.95M | −44% |
| baseline MSE (predict mean) | 12.66 | 12.66 | — |
| best val MSE (destd) | 3.66 | **2.83** | **−23%** |
| final val MSE (destd) | 3.71 | 2.91 | −22% |
| % reduction over baseline | 71% | **78%** | +7 pts |
| best Spearman ρ | 0.42 | 0.41 | ≈ flat |
| Spearman ρ at best-MSE epoch | 0.42 | 0.37 | −0.05 |
| training time | ~2m 46s | ~2m 30s | similar |

Run dir: `runs/heatmap_scene_level2_conv_20260428-152353/`

### Post-stage thinking

The conv decoder delivers a clean MSE win (−23%) with **fewer parameters**
(1.95M vs 3.5M), which is the expected payoff of giving the head a spatial
inductive bias.

#### Training dynamics

- **Best val MSE 2.83 at epoch 159, ~converged at epoch 91.** Stage 0 hit
  its (worse) optimum at epoch 123 / converged at 61. So Stage 1 is
  *slower to converge* in epoch count — the conv decoder takes longer to
  shape a smooth-but-correct heatmap than the Linear head takes to memorise
  3010 independent cell values. Worth knowing for budget: don't expect a
  conv decoder to hit its peak in 50 epochs.
- **Train/val gap final 1.85× (val 0.588 vs train 0.318).** Slightly
  larger gap than Stage 0's 1.73×, despite the smaller param count. The
  conv head is more *capable per parameter* — it actually uses its capacity
  on patterns that don't all transfer.

#### The Spearman story is more dramatic than the table shows

The "best ρ = 0.41" entry hides a striking detail: **ρ peaked at epoch 4
with 0.409, then *monotonically degraded* to 0.374 by epoch 200.** The
model's best per-obstacle ranking happens essentially at random
initialisation — when its predictions are still close to the mean
heatmap. Every step of pixel-MSE training after that reduces MSE *and*
hurts ρ.

This is the same MSE/ρ decoupling we saw in Stage 0, but **sharper**
because the conv head can reduce MSE more aggressively. With more
expressive pixel modelling, the model has more opportunity to shift mass
*within* an obstacle's footprint to fit per-cell averages — which doesn't
change the obstacle's integrated score, and can even hurt rank order if
mass leaks across footprint boundaries.

This is the moment we should have introduced dual-checkpoint
(`best_mse` + `best_rho`) tracking. We didn't, and as a result the
Stage 1 `best.pt` is the MSE-best checkpoint at epoch 159, where ρ is
0.37 — *worse* than the same model at epoch 4. We fix this in Stage 2.

#### Two readings of "ρ stayed flat"

1. **Per-entity score is encoder-bound, not decoder-bound.** Smoother
   heatmaps fit the supervised target better in pixel-MSE terms, but the
   rank order of per-obstacle integrated scores barely moves. The
   bottleneck for the planner-relevant metric is upstream — the model
   doesn't know enough about where the arm is going. That's Stage 2.

2. **The decoder may have made ranking *harder*.** A smoother heatmap
   blurs across nearby obstacle footprints (level2 obstacles are 5-10 cm
   apart), making per-obstacle integrals less discriminative. The
   inductive-bias-for-smoothness assumption assumes targets are smooth
   *but spatially localised*; level2 targets are smooth on top of dense
   obstacle clusters where the smoothing wins on MSE but loses on
   per-entity signal.

#### Decision

**Keep the conv decoder** as the new Stage 0+ default for Stages 2+. It's
a free MSE win and a cleaner baseline to layer task / vision on top of.
But the flat-then-degrading Spearman is a clear "bring more signal" signal,
not a "fix the head" signal — exactly what the staged plan calls for.

The conv head's win on MSE *does not* prove the head was the bottleneck
for the downstream planner cost. We confirmed it was the bottleneck for
the supervised loss; the planner cost will need separate evidence as we
keep adding signal.

---

## Stage 2 — Add goal_pos + task one-hot

### Pre-stage thinking

State alone can't tell "carrying object3 to far-left" from "carrying it back to
home" if the arm happens to pass through similar configs. Add explicit task
context:
- `goal_pos (3,)` — already in the trajectory pkl, joined at dataset-load time
- `task_id` — one-hot over the 12 scene_level2 tasks (kept simple — embedding
  optimisation deferred)

Cheapest thing to try before the vision pipeline. Concat new features into the
encoder input.

### Setup

- Model: same `HeatmapConvDecoder` as Stage 1
- `HeatmapDataset` extended with `include_goal` / `include_task` flags
- `train_demo.py --include_goal --include_task` toggles both
- Input dim: 17 → **32** (state 17 + goal 3 + task one-hot 12)
- Same loss / optimizer / split / 200 epochs as before

### Results — scene_level2 (conv decoder; ablation)

| variant | best val MSE (destd) | best Spearman ρ | final train_loss | final val_loss |
|---|---|---|---|---|
| Stage 1 (conv, state only) | **2.83** | 0.41 | 0.32 | 0.58 |
| Stage 2 conv + task | 3.64 | 0.47 | 0.28 | 0.62 |
| Stage 2 conv + goal | 3.62 | 0.47 | 0.29 | 0.60 |
| Stage 2 conv + goal + task | 3.81 | 0.41 | 0.29 | 0.64 |

Run dirs:
- `runs/heatmap_scene_level2_conv+task_20260428-154506/`
- `runs/heatmap_scene_level2_conv+goal_20260428-154831/`
- `runs/heatmap_scene_level2_conv+goal+task_20260428-154136/`

### Post-stage thinking

The two metrics fully decoupled here, in a way that matters for *what we
optimise* and *what we save*.

#### Headline numbers

- Adding `task` or `goal` *hurts* pixel MSE by ~28% (best 2.83 → ~3.6)
- ...but *improves* best per-obstacle Spearman by ~+0.06 (0.41 → 0.47)
- Combining `goal + task` makes both worse (3.81 / 0.41)

#### When ρ peaks tells us *what kind* of overfitting it is

Same look at the per-epoch ρ trajectories that we did in Stages 0–1, now
with a clearer pattern:

| variant | best ρ | epoch of best ρ | ρ over last 20 ep |
|---|---|---|---|
| Stage 1 (conv, state) | 0.409 | **4** | 0.374 |
| Stage 2 (conv + task) | 0.469 | **29** | 0.432 |
| Stage 2 (conv + goal) | 0.468 | **123** | 0.444 |
| Stage 2 (conv + task + goal) | 0.415 | 138 | 0.406 |

Two things stand out:

1. **`task` peaks early (epoch 29).** The task one-hot delivers most of
   its rank-correlation lift in the first 15% of training, then continued
   pixel-MSE optimisation slowly degrades it (0.469 → 0.43). This is the
   "MSE-vs-ρ wedge" we saw in Stages 0–1, but starting from a higher
   ceiling. Stop the model early, you keep the win; train it to MSE
   convergence, you give some of it back.
2. **`goal` peaks late (epoch 123).** Unlike `task`, `goal_pos` keeps
   delivering ρ improvements deep into training, settling at a *higher*
   last-20-epoch mean (0.444) than `task` does (0.432). This is suspicious
   — and consistent with the per-traj-memorisation hypothesis below. The
   model finds late, narrow, traj-specific shortcuts; ρ on val rises only
   because those shortcuts happen to align with held-out trajs that share
   trajectory shape with training trajs.

#### Why goal_pos is a leaky feature

`goal_pos` is **constant across all configs that share a (task, traj_id)
tuple**. With ~12 tasks × 30 trajs/task, there are ~360 unique `goal_pos`
values in the dataset. The model can use `goal_pos` as a near-perfect
traj identifier and memorise per-traj outcome patterns. On a held-out
traj, the new goal_pos is unseen but lies near training goal_poss in 3D
space, so smooth interpolation gives partial generalisation — enough to
look like a ρ win.

Real test for this: per-task ρ on held-out trajs vs random (task,
traj_id) shuffles. Deferred — but the shape of the ρ curve (slow rise,
late peak) is consistent with memorisation.

#### Why goal+task is worse than either alone

Two compatible mechanisms:

1. **Capacity dilution.** Same model parameters, larger input vector
   (32 vs 29 vs 20). The encoder spreads its representation budget across
   more dimensions; dropout (0.1) is constant. The combined vector also
   contains *more redundant* signal (`goal_pos` and `task_id` both
   identify the trajectory context), so per-feature usefulness goes down.
2. **Shortcut + categorical interference.** With *both* task one-hot and
   `goal_pos` available, the model preferentially leans on `goal_pos` (it's
   continuous, more discriminative per traj) and underweights `task` —
   destroying the task-driven rank signal that gave +task its win.

Train/val gap evolution supports both: gap goes 1.85× (Stage 1) → 2.22×
(`+task`) → 2.09× (`+goal`) → 2.23× (`+goal+task`). More signal → more
overfit. The conv decoder makes this *visible* because it's expressive
enough to actually use the shortcut.

#### So what is `task` actually adding?

Best reading: `task` is a *categorical pointer to which obstacles matter*.
Different tasks send the arm to different goal regions, and different
goal regions privilege different obstacles for impact. Without `task`,
the model can only infer this from `qpos` (already noisy) and `qvel`
(noisier still). The +task model uses the one-hot to consult an
implicit per-task obstacle-importance prior. That's exactly what lifts
ρ — and exactly the kind of thing that should *not* improve pixel-MSE,
because it doesn't change *where* mass lands, only *which obstacles are
asked about*.

This also explains why MSE got worse: the encoder spends parameters on
the task one-hot mapping that don't directly produce pixel mass.
Standardising the task one-hot adds noise to those rare-task dims.

#### Action items

1. ~~Track best-on-ρ separately from best-on-val_loss~~ **Done.** Trainer now
   saves `best_mse.pt` and `best_rho.pt` independently and renders pred panels
   from both (`preds_best_mse.png`, `preds_best_rho.png`). `best.pt` kept as a
   legacy alias for `best_mse.pt` so existing notebooks keep loading.
2. **Add early-stopping / stronger regularisation** for higher-input variants —
   probably `dropout=0.2`, or weight decay on the new input dims specifically.
3. **`task` alone is the better single-feature add** — same Spearman win as
   `goal` but fewer over-fit symptoms (`goal` peaks at epoch 48 then degrades
   harder; `task` peaks at epoch 107 and stays closer to peak). Carry
   `+task` (no goal) into Stage 3 unless vision changes the picture.
4. **`goal_pos` is a leaky feature** in this dataset structure (constant per
   traj). To use it properly we'd want goal-from-frame (predict from
   `pre_rgb`/`pre_depth`) or relative `goal − ee_pos` to break the per-traj
   constancy. Park as a Stage 4+ idea.

So Stage 2 is a **conditional success**: it advances the metric we actually
care about, even though pixel MSE got worse. Worth the negative-result entry
in the report — it's exactly what staged complexity is for.

---

---

## Stage 3 — Add `pre_rgb` via small CNN encoder

### Pre-stage thinking

Clutter, free space, where objects are sitting — none of that is in `qpos`.
Stage 3 is where we expect the biggest qualitative jump, especially on
cluttered scenes. Carrying `+task` forward (best single-feature win from
Stage 2); deferring `goal_pos` until we can break per-traj constancy.

Watch out for:
- `front_cam` is fixed per scene → CNN can shortcut on background pixels.
  Within-scene that's fine; cross-scene transfer (Stage 5) will expose it.
- Train/val split must remain traj-key-level so visually near-duplicate frames
  don't leak across.

### Setup

- Dataset: `pre_rgb` resized to (96, 128) via `cv2.INTER_AREA`, scaled to
  [0, 1], returned as `(3, H, W)` float32.
- Model: `HeatmapVisionConvDecoder`
  - State path: same Linear(state_dim → 256 → 512) as Stage 1
  - Vision path: small CNN (4× Conv stride-2 + AdaptiveAvgPool 4×4 → Linear)
    → 128-dim embedding
  - Fuse by concat, then identical conv decoder + bilinear resize
- Input: state(17) + task one-hot(12) + RGB → state_dim = 29, plus image stream
- 200 epochs, same loss / split / seed as Stages 0–2

### Results — scene_level2 (200 epochs, seed 0)

Run dir: `runs/heatmap_scene_level2_vision+task+rgb_20260428-162828/`

| variant | best val MSE (destd) | best Spearman ρ | final train_loss |
|---|---|---|---|
| Stage 1 (conv, state only) | **2.83** | 0.41 | 0.32 |
| Stage 2 (conv + task) | 3.64 | **0.47** | 0.28 |
| **Stage 3 (vision + task)** | 3.32 | 0.46 | **0.18** |

(Both checkpoints saved: `best_mse.pt` ep 124 val_loss 0.59, `best_rho.pt`
ep 94 ρ 0.455.)

### Post-stage thinking

Vision **did not help** on the planner-relevant metric. Best Spearman ρ
moved from 0.47 (task-only) to 0.46 (task + vision) — flat within noise.
Pixel MSE landed between Stage 1 and Stage 2 (3.32).

#### Training dynamics — biggest train/val gap so far

- **Best val MSE 3.32 at epoch 124, ~converged at epoch 87.** Roughly the
  same convergence pace as Stage 1.
- **Final train_loss 0.18 vs val_loss 0.61 ⇒ gap = 3.38×.** This is the
  largest train/val gap of any stage. Stage 1 was 1.85×, Stage 2 (`+task`)
  was 2.22×, Stage 3 here is 3.38×. The CNN is fitting the training set
  *much* harder without that translating into val improvement.
- **Best ρ = 0.455 at epoch 94, last-20-ep mean 0.44.** Unlike Stages 0–2,
  ρ does *not* peak early and degrade — it climbs alongside MSE and stays
  near peak. So the *failure mode is different* here: not "MSE optimisation
  hurts ρ" but "the extra capacity doesn't translate into either".

#### What is the CNN actually doing?

The 3.38× gap with no ρ gain says the CNN is using its ~120k extra params
to memorise training-frame artefacts: shadow patterns, exact arm pose
silhouettes, fingertip pixel positions. None of those transfer to a
held-out traj because the held-out traj produces a *different* exact
arm-pose silhouette in a slightly different location.

We confirm this with the vision-only ablation (next subsection): without
`task` in the input, vision drives MSE to its lowest value of any
stage (2.67) but ρ collapses to ~0.31 stable. So the CNN is genuinely
learning the *spatial mass distribution* (where pixel density should
land) but not the *categorical signal* (which obstacle is which) that
task carries.

#### Why redundancy with `task` matters

With `task` already in the input, the front_cam image isn't bringing new
*invariant* information for held-out trajs. Across trajs of the same
task, the front_cam shows essentially the same scene with the arm at
slightly different configs — and `qpos` already encodes arm pose more
precisely than pixels can. So vision's marginal contribution to features
*not already in (state, task)* is small, while its overfitting cost is
large.

This is the cleanest single-stage refutation of "vision always helps."
Vision helps when it carries information not in your other inputs. Here
it doesn't (within scene_level2). Stage 5 will test the contrapositive:
when scenes vary, qpos no longer disambiguates the world, and vision
should finally have to do work.

Counter-evidence to "vision is useless here":
- We trained a tiny from-scratch CNN (~120k params) on ~2,500 images. That's
  often *not enough* signal-to-overfit to extract a useful representation.
- We only tested on scene_level2, which is the cleanest scene. On
  scene_cluttered the obstacle layout actually varies and vision should
  matter more.
- We combined RGB at 96×128 — small enough that fine-grained obstacle edges
  may be smoothed away by `INTER_AREA` resize.

What I'd try if we want to revisit Stage 3:
1. **Test on a busier scene** (scene_cluttered, scene_kitchen) where pre_rgb
   carries information that `qpos` does not.
2. **Use depth not RGB**, or both. The geometric prior in pre_depth is
   exactly what RGB has to reverse-engineer through shading.
3. **Use a pretrained backbone** (frozen ResNet-18 features) — bypasses the
   "small model on small data" problem.
4. **Drop `task`** when adding vision — there's clear redundancy. A clean A/B
   would be `vision-only` vs `state-only` to see if RGB carries genuine
   independent signal.

Decision: **keep Stage 2 (conv + task) as the working best for now**, since
adding RGB gave no Spearman gain and overfit harder. The cheapest next move
isn't more architecture — it's **multi-scene**, where vision should finally
be forced to do work because `qpos` no longer disambiguates the scene. That's
Stage 5.

### Stage 3 ablation — vision-only (no task)

Run dir: `runs/heatmap_scene_level2_vision+rgb_20260428-164327/`

| variant | best MSE | stable ρ (last 50 ep) |
|---|---|---|
| Stage 1 (state, conv) | 2.83 | ~0.40 |
| Stage 2 (state + task) | 3.64 | ~0.45 |
| Stage 3 (state + task + rgb) | 3.32 | ~0.43 |
| **Stage 3 ablation: rgb-only** | **2.67** | **~0.30** |

(The header `best rho = 0.40 at ep 4` for vision-only is an early-training
artifact — the model output is still close to the prior mean, which
correlates spuriously with target structure. The stable ρ over the last
50 epochs is 0.31.)

**The metrics fully decouple.** Vision-only achieves the lowest pixel MSE
across every variant tried, but the *worst* rank correlation. The picture:

- **Vision is the spatial-mass feature** — it learns *where* contact density
  should land, well enough to fit pixels closely, but doesn't carry the
  categorical signal of which obstacle is which.
- **`task` is the rank-information feature** — it lifts ρ but spends pixel-MSE
  capacity on overfitting (constant-per-traj feature interacts badly with
  per-traj training samples).
- The combined Stage 3 model muddies both — neither metric is best.

This is a clean attribution. The right way to combine the two is probably a
**dual-head loss** (one head supervises pixel reconstruction, another
supervises per-entity score directly) or **better regularisation on the task
head**. Neither is the right next step; we should first check whether vision
even matters across scenes — that's where `qpos` and `task` together can no
longer disambiguate the world.

Decision: **proceed to Stage 5 (multi-scene)**, carry only the conv decoder
(state-only) and the conv+task variant as the two baselines to beat.

---

---

## Stage 4 — Add depth (+ optionally ee_cam) — **multi-scene context**

### Pre-stage thinking

`pre_depth` is a strong geometric prior that the RGB CNN otherwise has to
reverse-engineer through shading. `ee_cam_*` gives a wrist-eye view of what
is directly under the gripper — important for cluttered tasks where
front_cam is occluded by the arm itself.

We deliberately ran Stage 4 **after** Stage 5 (multi-scene), not before.
Stage 3 within scene_level2 showed RGB was redundant once `task` was in
the input: `qpos` already encodes arm pose and the front_cam shows the
same scene every config. The vision-only ablation drove this home — the
CNN learned spatial mass distribution well (lowest pixel MSE of any
single-scene run, 2.67) but the rank-correlation signal for the planner
came entirely from `task`. Within one scene, vision had nothing
*invariant* to add.

Multi-scene flips that. Now `qpos` has to disambiguate 5 different
worlds, and `task` only carries the within-scene categorical pointer.
Vision *should* finally have to do work — and depth, which is the
cleanest geometric prior, should be where the win shows up.

Scope choice: implement only the `pre_depth` stream this round. Stack it
as a 4th channel on `pre_rgb` (`(4, H, W)` tensor through the same small
CNN with `in_ch=4`). `ee_cam_*` is deferred — it needs a parallel CNN
and a fusion layer, ~2× the code change for an unclear marginal gain.

### Setup

- Dataset: `HeatmapDataset` extended with `include_depth` flag. When set,
  `_read_rgb` also reads `pre_depth`, clips to [0.05, 2.0] m, scales to
  [0, 1], resizes to (96, 128), and stacks as a 4th channel.
- `MultiSceneHeatmapDataset` threads the flag through to per-scene subdatasets.
- `_CNNEncoder` first conv layer: `in_ch` configurable (3 → 4 here).
- Trainer: `scripts/train_multiscene.py --include_rgb --include_depth`.
- Same masked MSE loss, AdamW, cosine schedule, traj-key per-scene split,
  200 epochs, seed 0.
- Total params still ~1.95M (extra channel is just one Conv2d row).

### Results — multi-scene 200 ep, conv + task + rgb + depth

Run dir: `runs/heatmap_multi_vision+task+rgb+depth_20260428-181029/`

**Comparison to Stage 5 (multi-scene, conv + task, no vision):**

| scene | Stage 5 MSE | Stage 4 MSE | Δ | Stage 5 ρ | Stage 4 ρ |
|---|---|---|---|---|---|
| scene_level2     | 3.98 (−69%) | 4.66 (−64%) | **worse** | 0.27 | 0.23 |
| scene_kitchen    | 0.90 (−66%) | 1.05 (−61%) | slightly worse | 0.95 | 0.95 |
| scene_workshop   | 1.70 (−13%) | **1.45 (−26%)** | **better** | 0.82 | 0.82 |
| scene_grocery    | 1.75 (−10%) | 1.96 (−1%)  | worse | 0.94 | 0.94 |
| scene_cluttered  | 7.14 (−30%) | **6.39 (−37%)** | **better** | 0.98 | 0.98 |
| pooled ρ         | 0.94        | 0.93        | flat | — | — |
| ~converged epoch | 93          | **46**      | 2× faster | — | — |
| final train_loss | 0.187       | 0.129       | overfit harder | — | — |

### Post-stage thinking

#### Vision+depth helps where geometry varies, hurts where it doesn't

Per-scene MSE deltas split cleanly:

- **Wins**: workshop (−13% → −26%), cluttered (−30% → −37%). Both have
  varied 3-D obstacle heights and clutter density. Depth disambiguates
  *which obstacle is tall vs. short*, *where the arm has clearance*,
  *how far the goal corridor is from the wall*. The geometric prior pays
  off exactly where you'd expect.
- **Losses**: level2 (−69% → −64%), grocery (−10% → −1%). Level2 has
  obstacles within centimetres of each other; depth at 96×128 doesn't
  resolve them better than the arm-pose signal already does. Grocery
  has a fixed wall+shelf — the geometry is *constant per scene*, so
  depth carries no per-config signal beyond what `qpos` and `task` give.
- **Neutral**: kitchen (essentially unchanged). Kitchen geometry is rich
  but the model already gets ρ = 0.95 from `qpos + task`, so there's no
  room for depth to lift ranking.

Spearman ρ is **flat** on every scene. Reading: per-obstacle ranking is
governed by *which obstacle is targeted* (carried by `task`), and depth
moves *how much* mass falls on each — visible in MSE, not in ρ. This is
the same pattern from Stage 3's vision-only ablation: vision is a
*spatial-mass* feature, not a *rank-information* feature.

#### Convergence: 2× faster, but final train_loss drops further

- ~converged at epoch 46 vs Stage 5's 93. Depth gives strong, low-noise
  gradient signal that the encoder exploits early.
- final `train_loss = 0.129` vs Stage 5's `0.187` — the model fits train
  31% better than without depth, while val plateaus.
- Train/val gap ratio: harder to compare directly (val is in different
  units), but the train descent is steeper.

This is the same overfitting symptom as Stage 3 (combined). Depth gives
the model a powerful new feature to memorise *training-frame*
geometric details that don't repeat on held-out trajs of the same task.
Where the geometric variance is real and discriminative (workshop,
cluttered), the generalisation outweighs the overfit. Where it isn't,
the overfit dominates.

#### What this says about the staged plan

The "vision should help in multi-scene" prediction from Stage 5 was
**partially confirmed**: depth helps on 2 of 5 scenes — the ones with
genuinely varied 3-D geometry. It does not help on scenes where the
underlying geometry is fixed (grocery's wall + shelf, level2's compact
table) or where ρ is already saturated (kitchen).

The pooled / planner-relevant metric (ρ) is essentially unchanged.
That means **for downstream planner use, vision+depth is not the right
next axis** — it doesn't move the rank correlation that drives the
planner's choice between configs. We're still encoder-bound on level2,
and that's the lever to pull next.

#### What we would try next, in order of expected payoff

1. **Capacity bump for the multi-scene encoder.** Same conclusion as
   Stage 5's action item, now reinforced: depth gave us 2× faster
   convergence with no ρ gain, suggesting the bottleneck is
   representation capacity, not feature richness. Widen encoder to
   (256→512→1024) and double `feat_ch` from 64 → 128.
2. **Scene FiLM** — depth helped where geometry varies; FiLM lets the
   *encoder* condition on scene without spending input-dim capacity on a
   one-hot. May recover scene_level2 ρ specifically.
3. **Per-scene output heads** (shared trunk + per-scene 1×1 conv).
   Restores per-scene specialisation while keeping multi-scene training.
4. **Add ee_cam stream** — only worth doing if (1)–(3) plateau. ee_cam
   carries close-range geometry that depth at 96×128 misses, but the
   wins from the front-cam depth-only run suggest the marginal
   contribution will be small.

### Note on what we *did not* try

`ee_cam_*` (RGB + depth from the gripper-mounted camera) was descoped
for this stage. To add it later: parallel `_CNNEncoder` instance, concat
its embedding with the front-cam embedding before the state fusion. The
infrastructure already supports it (`HeatmapDataset` reads `ee_cam_rgb`
and `ee_cam_depth` from npz), it just needs the model wiring.

---

## Stage 5 — Multi-scene training

### Pre-stage thinking

5 scenes, different table sizes ⇒ different `(ny, nx)`. Picked the simplest
viable design that lets us run today:
1. **Output grid**: pad all targets to the max scene grid `(93, 133)`, mask
   the loss so padding contributes zero.
2. **Conditioning**: scene one-hot prepended to the input vector. Defer
   FiLM / scene-conditional BN until we know one-hot fails.
3. **Standardisation**: per-scene per-cell, applied within each scene's
   bounded region.
4. **Task vocab**: union across all scenes prefixed with scene name (54
   tasks total) to avoid `clean_nominal` collisions across scenes.

If Stages 1–4 already overfit per-scene, joint training will tank metrics
until we add capacity / regularisation. Don't conflate "multi-scene is hard"
with "model is broken."

### Setup

- Dataset: new `MultiSceneHeatmapDataset` (in `planner/risk/dataset.py`)
  composes 5 per-scene `HeatmapDataset` instances. Each item returns
  `(x, y_padded, mask, scene_idx, idx)` (or 6-tuple with rgb).
- Trainer: new `scripts/train_multiscene.py` with masked MSE loss, per-scene
  destd MSE / Spearman tracked every epoch. Selection metric switched to
  **per-scene-relative MSE** (mean over scenes of `mse / baseline_mse`)
  because the standardised val_loss is dominated by per-cell-std outliers
  on rarely-touched cells.
- Config: 16,532 configs total split per-scene (val_frac 0.10), so val keys
  are unseen trajectories within each scene.
- Model: `HeatmapConvDecoder`, output grid `(93, 133)`. Same param count
  as single-scene (~1.96M). Input dim = 76 (state 17 + task 54 + scene 5).
- 200 epochs, same seed, optimizer, schedule as previous stages.

### Results — multi-scene (200 ep, conv + task)

Run dir: `runs/heatmap_multi_conv+task_20260428-170453/`

**Per-scene metrics (best-MSE epoch 179, MSE in destandardised units):**

| scene | configs (train/val) | val MSE | baseline MSE | reduction | val ρ |
|---|---|---|---|---|---|
| scene_level2     | 2532 / 302 | 3.98  | 12.88 | **69%** | 0.27 |
| scene_kitchen    | 2475 / 275 | 0.90  | 2.66  | **66%** | 0.95 |
| scene_workshop   | 3318 / 407 | 1.70  | 1.96  | 13%     | 0.82 |
| scene_grocery    | 3417 / 383 | 1.75  | 1.94  | 10%     | 0.94 |
| scene_cluttered  | 3142 / 281 | 7.14  | 10.19 | 30%     | 0.98 |
| **pooled**       | 14884 / 1648 |  —  | —     | —       | **0.94** |

`final train_loss = 0.19`, `best val_destd_avg = 0.62` (pseudo-relative).

### Post-stage thinking

**The multi-scene story is more nuanced than a single number.** Pooled
Spearman ρ = 0.94 looks great, but it's misleading — it's pooling across
scenes whose obstacle scores live on different absolute magnitudes, which
gives huge inter-scene variance and inflates the rank correlation. The
*honest* number is per-scene ρ, which varies wildly (0.27 ↔ 0.98).

#### Training dynamics — when each scene's ρ peaked

| scene | best ρ | epoch | best MSE | epoch | ρ last-20 mean |
|---|---|---|---|---|---|
| scene_level2 | 0.339 | 55 | 3.86 | 130 | **0.271** |
| scene_kitchen | 0.950 | **12** | 0.76 | 52 | 0.947 |
| scene_workshop | 0.848 | 57 | 1.68 | 144 | 0.815 |
| scene_grocery | 0.942 | **16** | 1.74 | 167 | 0.940 |
| scene_cluttered | 0.976 | 136 | 6.97 | 138 | 0.976 |
| (global pooled) | 0.938 | 57 | — | — | 0.927 |

Three patterns worth noting:

1. **Two scenes hit peak ρ in the first 10% of training** (kitchen at
   ep 12, grocery at ep 16) and stay there. They have strong intrinsic
   obstacle asymmetries — a near-mean prediction already gets the rank
   right, and additional MSE optimisation only refines pixel placement.
2. **scene_level2 ρ peaks at epoch 55 (0.339) then degrades to 0.271** —
   same MSE-vs-ρ wedge we saw in single-scene Stages 0–2, now with the
   added pressure of capacity competition. The MSE checkpoint sits at
   epoch 130, where ρ has already lost 6 points from peak.
3. **scene_cluttered is the only scene where best ρ (ep 136) and best MSE
   (ep 138) coincide.** It's also the one with the highest absolute val
   MSE (7.14). Reading: the geometry is busy enough that fitting pixels
   *is* fitting obstacle ranks — there's no slack to overfit *within*
   footprints because every footprint matters.

The global ρ peak at epoch 57 is dominated by these scenes: it's the epoch
where level2 + workshop ρ are simultaneously at peak, and the others have
already plateaued. Saving `best_rho.pt` at ep 57 vs `best_mse.pt` at ep 179
captures meaningfully different models.

What multi-scene revealed:

1. **Multi-scene training works structurally** — a single shared model
   handles all 5 scenes, with no destabilisation. Padding + masking +
   per-scene standardisation is enough.

2. **scene_level2 ρ regressed** (single-scene Stage 2: 0.47 → multi-scene:
   0.27). Two plausible reasons:
   - **Capacity competition**: same param count now serves 5 scenes; level2
     loses out to scenes with stronger geometric signals.
   - **Targets are harder there**: level2 has the smallest table, densest
     obstacles, and the most symmetric pickable layout. Per-obstacle
     ranking is structurally harder than in workshop/grocery/cluttered
     where obstacle density/heights have clearer asymmetries.

3. **scene_workshop and scene_grocery underperform on MSE reduction** (10%,
   13%) but have very strong Spearman (0.82, 0.94). The MSE numbers are
   small absolutely (~1.7-1.8) and so are the baselines (~1.9-2.0), which
   means the model has little room to beat the per-cell mean. The targets
   in these scenes are concentrated near the carry corridor, so the
   "predict the mean heatmap" baseline is already pretty informative.

4. **Train_loss = 0.19** is much lower than val_destd_avg = 0.62, but the
   denominator scales differ — train_loss is in standardised units, val
   in relative-baseline units. They're not directly comparable. The
   per-epoch curves (`loss_curve.png`) show train continuing to drop while
   val plateaus around epoch 50, which is the usual mild overfit.

5. **scene_cluttered has highest ρ (0.98)** despite high absolute MSE.
   Reading: the model gets the *which* obstacles right (lots of them,
   varied heights, easy to rank) but not the *how much* (long-tail
   contact bursts during transport).

### Reading the Stage 5 results

The per-scene table has six numbers per row. Each is computed differently
and on a different denominator — they are *not* directly comparable across
scenes without the context below.

#### 1. `val MSE` — pixel error in original heatmap units

Mean of `(predicted − target)²` over all valid cells of a scene's grid,
averaged over val configs in that scene. **Lower is better.** "Original
units" means we *de-standardise* before computing it: predictions get
multiplied by per-cell std and per-cell mean is added back, so the number
is on the same scale as the raw contact-density values in `targets.npz`.

What it does **not** account for: scenes where target heatmaps are
intrinsically smaller will have lower absolute MSE even at equal *quality*.
That's why we report `reduction` separately.

#### 2. `baseline MSE` — `mean((y_target − y_train_mean)²)`

The per-cell error you'd get by predicting the train mean for every config.
**A trained model has to beat this to be worth anything.** It's a property
of the target distribution, not the model:

- **High baseline** (level2 = 12.88): contact patterns vary widely between
  configs. The train mean is a poor guess; lots of head room.
- **Low baseline** (workshop = 1.96, grocery = 1.94): contact patterns are
  similar across configs. The mean is already informative; the model has
  little room to improve.

#### 3. `reduction` — `1 − val_MSE / baseline_MSE`

How much pixel-level signal the model extracts beyond the mean predictor.
Reading the table:

- **−69% / −66% (level2, kitchen)**: model fits well; lots of head room
  AND the model uses it.
- **−30% (cluttered)**: middling. Contacts in cluttered are noisy
  (failure mode triggers many bumps); model captures high-mass regions
  but misses long-tail bursts.
- **−13% / −10% (workshop, grocery)**: small *not because the model is
  bad on these scenes* but because `baseline` is already informative.
  The metric has a low ceiling there.

Don't read workshop's "13%" as a failure. Read it together with
`baseline = 1.96` (small) and `val_ρ = 0.82` (high) — model is doing OK,
it just has nowhere to go.

#### 4. `val ρ` — per-obstacle Spearman, the planner-relevant metric

For each val config:
1. Integrate the *predicted* heatmap over each obstacle's 2-D footprint
   → scalar predicted score per obstacle.
2. Same for the *target* → scalar target score per obstacle.
3. Across all (config, obstacle) pairs in the scene, compute Spearman
   rank correlation of predicted vs target scores.

`ρ = 1.0` ⇒ model perfectly preserves which obstacles are riskiest in
each config. `ρ = 0` ⇒ random. The planner *ranks* configs by obstacle
risk, so this maps onto our use case more directly than MSE.

Per-scene reading:
- **0.95–0.98 (kitchen, grocery, cluttered)**: very high. Scene geometry
  has clear asymmetries (specific tall obstacles in specific zones); even
  a smooth-ish prediction picks the right obstacle even when the
  magnitudes are off.
- **0.82 (workshop)**: still very good.
- **0.27 (level2)**: weak. Level2 has the smallest table with 3–4
  obstacles within centimetres of each other; per-obstacle integrals are
  very sensitive to small shifts in predicted mass, and smoothed
  predictions don't differentiate them.

#### 5. Why "pooled ρ = 0.94" is misleading

The pooled Spearman pools (config, obstacle, scene) triples across all
scenes before computing one rank correlation. Different scenes have
different *absolute* obstacle-score ranges — a cluttered obstacle
integrates to ≈50, a workshop one to ≈5. Most of the pooled ranking
signal is "this is a cluttered-scene obstacle, those rank higher than
workshop ones." That structure exists even with random within-scene
predictions, which is why pooled ρ is inflated.

**The honest single-number summary across scenes is the val-config-weighted
mean of per-scene ρ:**

  `(0.27·302 + 0.95·275 + 0.82·407 + 0.94·383 + 0.98·281) / 1648 ≈ 0.79`

That's the number to compare against single-scene Stage 2's 0.47 on
level2. Across 5 scenes the model averages **ρ ≈ 0.79** — but the weakest
scene (level2) lost 0.20 ρ versus a dedicated single-scene model.

#### 6. Why scene_level2 ρ regressed: 0.47 → 0.27

Two compatible hypotheses, ranked by suspicion:

**(a) Capacity competition.** Same 1.95M params now serve 5 scenes'
spatial variance. The model implicitly trades capacity off; level2 (the
scene with the densest, most symmetric obstacle layout) loses out.
Diagnose by retraining on level2 alone with this exact codebase and
confirming we recover Stage 2's ρ ≈ 0.47.

**(b) Targets in level2 are intrinsically harder for ρ.** Obstacles in
level2 sit within centimetres of each other; their footprints overlap or
nearly do. A small spatial shift in predicted mass swaps which obstacle
"owns" a hot cell. Other scenes have larger separations, so integration
is forgiving. If (b) dominates, no amount of capacity will fix it
without sharper predictions (smaller σ in target generation, or a
per-obstacle loss directly).

(a) is testable cheaply (~10 min). Test (a) first.

#### 7. `train_loss = 0.19` vs `val_destd_avg = 0.62` — different units

Not directly comparable.

- `train_loss` is the *standardised* masked MSE used during optimisation.
  Units are z-score² (per-cell mean subtracted, per-cell std divided).
- `val_destd_avg` is the *de-standardised* MSE divided by per-scene
  baseline, averaged across scenes. `1.0` ≈ "as bad as predicting the
  mean"; `0.0` ≈ "perfect"; `0.62` ≈ "captures 38% of residual variance
  after the mean baseline."

The training loss dropping much lower than val is the usual mild overfit,
but the unit gap also makes the gap look larger than it really is.

#### 8. The `y_eps` story — why we bumped `1e-3 → 0.05`

Per-cell standardisation divides target values by per-cell std. Cells
far from any contact have train std ≈ 0; the original `eps = 1e-3` floor
meant any non-zero val sample at such a cell standardised to ~10³, and
masked MSE was dominated by ~100-200 outlier cells. The first multi-scene
run reported `val_loss = 76` while train was 0.7 — pure outlier
domination, not a real model failure.

`eps = 0.05` puts a floor at "5% of typical heatmap magnitude": a val
sample 1× the unit of original measurement now standardises to at most
~20, which the loss can absorb. This was a multi-scene-only issue because
more configs leave more cells effectively constant in train; on
single-scene level2 the problem was small enough that `1e-3` worked.

Logged as a subtlety, not a bug: per-cell standardisation is brittle in
low-density regions. A dual scheme (per-cell mean, *global* std) might be
more robust; defer until needed.

### Action items

1. **Capacity bump**: simplest knob is widening the encoder (e.g. 256→512,
   512→1024) and/or doubling decoder channels. Current 1.95M params is
   tiny for 5 scenes worth of variance.
2. **Scene FiLM** instead of input one-hot: lets the encoder produce
   different feature distributions per scene without the model having to
   learn that mapping inside the linear head.
3. **Per-scene output heads**: shared trunk + small per-scene 1×1 conv
   final layer. Restores per-scene specialisation without losing transfer.
4. **Re-add vision** now that scenes differ: `pre_rgb` should finally
   carry information that `qpos` can't supply (the actual scene layout).
   Stage 3's negative result was scene-specific.
5. **Investigate scene_level2 specifically** — why did its ρ drop more
   than other scenes? Maybe the smaller grid means proportionally fewer
   cells contribute to its loss in the masked sum.

### Decision

This is the most informative stage so far — it shows multi-scene is
*feasible* (no instability, sensible per-scene metrics) but exposes
per-scene heterogeneity that was hidden in single-scene runs. The
practical state of the predictor:

- **Best for downstream planner** (per-scene ρ): use single-scene Stage 2
  (`+task`) checkpoints per scene.
- **Best for unified deployment** (one model): this multi-scene model,
  accepting that scene_level2 underperforms.

Pick action item 1 or 2 next — probably **scene FiLM**, since it's the
cheapest change that targets the diagnosed problem (capacity competition
across scenes).

---

## Stage 5b — Capacity bump (R1 of next-round plan)

### Pre-stage thinking

Stage 5 / Stage 4 diagnosed *capacity competition across scenes*: same
1.95M-param encoder serving 5 scenes' worth of variance, with the
recurring weak link being scene_level2's ρ and scene_grocery's near-zero
MSE reduction. R1 attacks this directly — same architecture, more
parameters — to settle whether capacity is *the* bottleneck or just
one of several. Cheapest reconstruction-side lever, highest diagnostic
value.

### Setup

- `--capacity big` flag added to `scripts/train_multiscene.py`.
- `HeatmapConvDecoder` widened: `hidden=(256, 512)` → `(512, 1024)`,
  `feat_ch=64` → `128`, `decoder_channels=(32, 16, 8)` → `(64, 32, 16, 8)`.
  Spatial path becomes `6×9 → 12×18 → 24×36 → 48×72 → 96×144` →
  bilinear-resize to `(93, 133)`.
- `HeatmapVisionConvDecoder` gets the same widening of its state path;
  the small CNN stays 120k params (don't conflate axes).
- Param count: `1.95M → 7.80M` (no-vision); `2.19M → 8.05M` (vision+depth).
- Two runs: multi-scene + task (no vision), and multi-scene + task + rgb +
  depth. 200 epochs, seed 0, identical pipeline to Stage 4 / 5 otherwise.

### Results — multi-scene (200 epochs, seed 0)

Run dirs:
- `runs/heatmap_multi_convBIG+task_20260429-004457/`
- `runs/heatmap_multi_visionBIG+task+rgb+depth_20260429-010658/`

| metric | Stage 5 (small) | Stage 4 (small + vision) | R1 (big) | **R1 + vision (big)** |
|---|---|---|---|---|
| params | 1.95M | 2.19M | 7.80M | 8.05M |
| **val_destd_avg** | 0.623 | 0.626 | 0.595 | **0.542** |
| best epoch | 179 | 98 | 110 | 115 |
| ~converged epoch | 93 | 46 | 62 | 109 |
| final train_loss | 0.187 | 0.129 | 0.117 | 0.102 |

| scene | Stage 5 | Stage 4 | R1 (no vision) | **R1 + vision** |
|---|---|---|---|---|
| level2     | −69% | −64% | **−74%** | −73% |
| kitchen    | −66% | −61% | **−75%** | −67% |
| workshop   | −13% | −26% | −18% | **−26%** |
| grocery    | −10% | −1%  | −5% | **−18%** |
| cluttered  | −30% | −37% | −30% | **−45%** |

### Post-stage thinking

#### Capacity *unlocks* vision; vision alone never did

The headline: small + vision (Stage 4) beat small + no-vision (Stage 5)
on workshop / cluttered only. Big + vision beats *every* baseline on
*every* scene except level2 (where big-no-vision is fractionally
better). The pattern is clear:

- **Big without vision** improves the scenes that were *already easy*
  (level2 −69→−74%, kitchen −66→−75%) but doesn't help workshop /
  grocery / cluttered. More state-encoder capacity = better fit on
  scenes where state already carries the signal.
- **Big + vision** is where the per-scene story changes. Every
  geometrically-rich scene improves: workshop −18→−26%, grocery −5→**−18%**
  (the biggest delta of the round), cluttered −30→**−45%**.

The Stage 4 result that "depth helped only on busy scenes, ρ flat
everywhere" *was a capacity story*: the small encoder couldn't extract
enough from RGB+depth to lift ρ-relevant features, so it landed as
spatial-mass-only. Bigger encoder + same vision pipeline = vision
finally pays off across the board.

#### scene_grocery rescued

Stage 5 / Stage 4 had grocery basically tied with the predict-mean
baseline (−10% / −1%). R1 + vision drives it to **−18%** — the largest
relative gain of any single intervention this round. Reading: grocery's
fixed wall + shelf geometry *does* carry per-config signal once the
model has both (a) enough capacity and (b) a vision stream that sees
the 3D structure. Neither alone was sufficient; the combination is.

#### scene_level2 improved on MSE — but was it ever a *reconstruction* problem?

Level2 was our recurring weak link in the Stage 5 analysis. R1 dropped
its MSE to 3.31 (−74%) without vision, 3.51 (−73%) with vision. That's
the strongest level2 reconstruction we've seen — but it's *not* what
the original ρ regression was about. The ρ ceiling on level2 was
diagnosed as a structural footprint-overlap problem (Stage 5 §6), not
a reconstruction problem. We'll re-check ρ offline; if it's still
stuck, that confirms the structural diagnosis and Stage 9a (obstacle-
integral aux loss) becomes the right next move.

#### Training dynamics — vision now slows convergence

R1 no-vision converges at epoch 62, R1 + vision at epoch 109 — bigger
gap than between Stage 4 (vision, ep 46) and Stage 5 (no-vision, ep 93).
The bigger CNN-driven gradient signal that made Stage 4 converge fast
(ep 46) doesn't carry over here; the wider state path needs more time
to balance against the vision branch. This is fine — the longer
schedule pays off in final quality (val_destd_avg 0.542 vs 0.626).

Final train_loss 0.102 (vs Stage 4's 0.129, Stage 5's 0.187): stronger
fit on train, but val also improved, so this isn't pure overfit. Some
of the new train descent translated into actual generalisation. Healthy.

#### Verdict

R1 + vision is the **new best multi-scene baseline**. `val_destd_avg
= 0.542` is a 13.5% improvement on Stage 5 and a 13.4% improvement on
Stage 4. Per-scene wins are distributed across all 5 scenes (no
regressions when combined with vision).

Per the round plan, this counts as a clear PASS on R1's criterion
(≥10% reduction on ≥2 scenes — actually 4 of 5, all but level2 where
it's neutral). Proceeding to **R3 (DINOv2)** next as the orthogonal
"is feature richness still a lever?" test, against this new R1+vision
baseline.

---

## Stage 6 (optional) — Temporal context

### Pre-stage thinking

Each trial is currently a single `(state, heatmap)` pair where `state =
(pre_qpos, pre_qvel, pre_ee_pos)`. `pre_qvel` already gives the model
*first-order* temporal information (velocity at the failure instant), which
covers most of what naive temporal context would add. The marginal
hypothesis for Stage 6 is that *higher-order* temporal context — the K
preceding configs — adds value primarily for transport-phase failures
where the arm has built up swing, and for cluttered scenes where the
trajectory shape disambiguates "which way the arm was sweeping."

### Status: **Not run — concrete roadmap below**

Lower priority than the multi-scene action items because:
- `pre_qvel` already encodes first-order temporal info
- Stage 4's results show capacity / scene conditioning is the
  near-term ρ bottleneck, not feature richness
- Implementation requires a data-pipeline change (~1-2 hour task on
  its own)

#### What "running it" would actually require

The data we have:
- `datasets/v10/<scene>/<task>/exp_*.npz` — captures `pre_qpos`,
  `pre_qvel`, `pre_ee_pos` at the *failure instant* (one snapshot per
  trial), plus `traj_progress ∈ [0, 1]`.
- `scenes/<scene>/trajs/*.pkl` — segmented full-mission trajectories
  (5 segments, dense waypoints) used to *generate* the npz captures.

Two implementation paths, in increasing fidelity / cost:

**Path A — back-derive from pkl + traj_progress (cheap, lossy).**

1. New helper: given `(scene, task, traj_id, traj_progress)`, replay the
   segmented trajectory's waypoints up to `traj_progress`, return the
   trailing K waypoints as `(K, 7)` qpos and `(K, 7)` qvel arrays.
2. Update `HeatmapDataset.__getitem__` (behind an `include_window=K` flag)
   to call this helper and return the window alongside the existing
   single-state vec.
3. Model: small 1-D conv or MLP over the K×features window, concat
   embedding into the existing state path.

**Caveats** of Path A: the npz `pre_qpos` was captured *post-physics-
settle* with grasp-lock applied, while the pkl has the *commanded*
trajectory. The two differ slightly due to controller tracking error.
For most configs the deviation is <1 cm but it isn't zero. For Stage 6
to be a clean test, we'd want either (a) accept the discrepancy and
treat the window as approximate, or (b) Path B.

**Path B — re-run capture with windowed state (expensive, exact).**

1. Modify `ExperimentRunner` to record a rolling buffer of the last K
   `qpos`/`qvel` post-physics each step.
2. Write `pre_qpos_window (K, 7)` and `pre_qvel_window (K, 7)` into the
   npz alongside the existing keys.
3. Re-run the v10 dataset. ~6-8 hours on a single machine.

Path B is the right thing to do if temporal context is going to be a
load-bearing stage. Don't do it before pulling the multi-scene
capacity/FiLM levers — those are cheaper and more directly target the
demonstrated bottleneck.

#### What we'd expect to see

If Stage 6 ran cleanly:
- ρ moves most on **scene_workshop** (long transport corridor where
  arm swing builds up) and **scene_kitchen** (deeper reach distances).
- ρ moves least on **scene_level2** (short trajectories, dominated by
  obstacle clustering not motion direction) — the same scene that's
  been our recurring weak link.
- Pixel MSE could go either way; the recurring pattern across stages
  is that *richer features → harder train fit, weaker val transfer*
  unless the new feature carries genuinely invariant info.

So Stage 6 is a candidate for *workshop/kitchen win, level2 flat*. If
that's the result, it confirms the diagnosis that level2 needs
sharper-not-richer signal (target σ change, per-obstacle loss, scene
FiLM) rather than more features.

This stays as a roadmap entry until the multi-scene action items
(capacity bump, FiLM, per-scene heads) have been tested.

---

## DINOv2 consideration (proposed Stage 7)

The "Things explicitly held off" entry below originally rejected pretrained
backbones with the reasoning "16k configs, not 16M." That argument is sound
*for training a transformer from scratch* and over-applies to using a
*frozen* pretrained backbone. Two threads of evidence make DINOv2
worth reconsidering:

### Why reconsider pretrained vision

**Stage 3 evidence**: our from-scratch CNN (~120k params) on ~2,500
single-scene images had `final train_loss = 0.18` vs val 0.61 — a 3.38×
train/val gap, the largest of any stage. The CNN learned spatial-mass
distribution well (MSE-best variant in the vision-only ablation, 2.67) but
contributed nothing to ρ on top of `task`.

**Stage 4 evidence**: depth helped MSE on 2 of 5 scenes (workshop,
cluttered) and did not help ρ anywhere. Across multi-scene runs,
**ρ on scene_level2 has been the recurring weak link** (0.27 in Stage 5,
0.23 in Stage 4) — it stayed stuck even when we added the cleanest
geometric signal we have.

Both findings point at the same diagnosis: pixel-derived feature richness
is a real lever, but our small encoder can't extract it without
overfitting. A **frozen** pretrained backbone targets exactly this — it
imports features learned on ~140M images and bypasses the
small-model-on-small-data problem entirely.

### Design (frozen features, precomputed)

- **Variant**: DINOv2 ViT-S/14 (~22M params, 384-dim CLS embedding,
  256 patch tokens at 224×224 input). Smallest available; fast enough to
  precompute features for all 16,532 configs in <10 min on the RTX 3070.
- **Mode**: frozen, no fine-tuning. Avoids the wall-clock + overfitting
  problems of training a 22M backbone on 14k configs.
- **Feature cache**: one-time pass over `pre_rgb` for all configs →
  `cache/dinov2/<scene>/<task>/<exp_id>.npy`. Storage with CLS-only:
  ~25 MB total. With patch grid (256 × 384): ~6 GB — borderline; start
  with CLS-only.
- **Model wiring**: replace `_CNNEncoder` in `HeatmapVisionConvDecoder`
  with `Linear(384 → rgb_emb_dim=128)` consuming the cached CLS vector.
  Everything downstream (state fusion, conv decoder, masked loss) stays
  identical.
- **Dataset wiring**: add `include_dinov2` flag to `HeatmapDataset`;
  `_read_rgb` is replaced by an `_read_dinov2_feature` cache lookup
  when the flag is set.

### Expected payoff

- **Stage 4 was a partial win** — DINOv2 should make it a fuller win.
  Geometric variance that helped workshop / cluttered with raw depth
  should also help level2 / grocery once the encoder has the *capacity*
  to use the signal.
- **scene_level2 ρ specifically** is the most likely scene to break the
  multi-scene ceiling. DINOv2 has the right inductive bias for cluttered
  small objects (trained on natural-image diversity); level2's tightly
  packed obstacle layout is exactly the regime where richer pixel
  features should help disambiguate per-obstacle ranks.

### Costs and risks

- **Domain mismatch**: DINOv2 is trained on natural images; our scenes
  are MuJoCo-rendered. Real risk, but DINOv2 is notoriously domain-robust
  on segmentation / depth / matching. Worth a one-stage test before
  deeper investment.
- **Resize**: `pre_rgb` is 480×640; DINOv2 wants 224×224 (16 × 14
  patches). Letterbox preferred over centre-crop to keep the table edges.
- **Implementation cost**: ~2 hours.
  - Hour 1: `scripts/precompute_dinov2.py` + dataset flag + cache layout.
  - Hour 2: model wiring + smoke test + 200-epoch run.
- **No-go signal**: if frozen DINOv2 + the existing decoder doesn't beat
  Stage 5 on level2 ρ, the bottleneck isn't feature richness — it's the
  structural problem flagged in "Reading the Stage 5 results §6"
  (level2 obstacles too close together for per-obstacle integrals to
  discriminate, regardless of feature quality).

### Decision

DINOv2 is the **highest-value next experiment after the multi-scene
capacity / FiLM levers** if those plateau. Should be Stage 7 — keep the
ordering: capacity bump → FiLM → per-scene heads → DINOv2.

---

## Stage 9 (proposed) — Loss redesign (Sinkhorn / OT auxiliary)

A direct response to the framing pivot: *now* that we're optimising MSE
only, what's the cheapest principled way to bring back planner-relevant
signal *without* abandoning reconstruction? Three loss-family candidates
were considered (BCE, Soft Dice, Sinkhorn / OT). Sinkhorn is the one
worth scheduling.

### Why Sinkhorn (and not BCE / Dice)

Our targets are continuous Gaussian-blurred contact density heatmaps,
not binary masks. That immediately constrains the choices:

- **BCE** treats each cell as a Bernoulli classification — wrong fit for
  continuous density. Soft-BCE works mathematically but adds nothing
  beyond MSE on the same normalised target. Skip.
- **Soft Dice** is scale-invariant — captures *where* mass should land
  without caring about absolute density. Useful only as `MSE + λ·Dice`
  where Dice gates spatial support and MSE polishes magnitude. Standard
  segmentation cocktail. Second priority.
- **Sinkhorn / OT** directly fixes the diagnosed MSE/ρ decoupling:
  MSE scores per-cell errors *independently of distance*, so predicting
  mass 1 cell off costs the same as predicting it 50 cells off.
  Optimal transport penalises moving mass across cells weighted by
  spatial distance. That is exactly the right inductive bias for a
  downstream cost that integrates over obstacle footprints — predicting
  the *right region* matters more than predicting the *right per-cell
  value*.

### Two implementations, ordered cheapest-first

#### 9a. Per-obstacle integral auxiliary loss (~30 min) — **first try**

Cheapest principled option. We already compute
`integrate_per_entity(pred, fps)` and `integrate_per_entity(target, fps)`
during eval. Add an L1/L2 term on the per-obstacle score *vector
difference* during training:

```
L_total = masked_MSE(pred, target)
        + λ_obs * || obstacle_scores(pred) − obstacle_scores(target) ||
```

This is "Sinkhorn where the support is the obstacle footprints" — a
much cheaper, exactly-on-target version. Tunes the precise number the
planner consumes (per-entity integrated risk).

Implementation: lift `entity_footprints(...)` and
`integrate_per_entity(...)` into a torch-friendly batched op (currently
numpy / per-sample). About 30 minutes. Try `λ_obs = 0.1, 0.5, 1.0`
sweep on multi-scene + task baseline.

#### 9b. Coarse-pooled Sinkhorn (~3 hours) — fall-back if 9a plateaus

If the cheap obstacle-integral loss helps but plateaus, the fuller
Sinkhorn variant catches *spatial structure between* obstacles too
(which obstacle-integrals discard).

- **Library**: `geomloss` (PyTorch, GPU-friendly, supports Sinkhorn
  approximation with ε regularisation). Add as dep, no implementation
  required.
- **Coarse-pool first**: full-resolution Sinkhorn at 12k cells × batch
  128 is too expensive. Avg-pool predictions and targets 8× (e.g.
  workshop's `(93, 133)` → `(12, 17)` ≈ 200 cells). That's the
  obstacle-region scale anyway — sub-cm precision is MSE's job.
- **Loss combination**: `L = masked_MSE + λ_ot · Sinkhorn(pool(pred),
  pool(target))` with `ε = 0.01` Sinkhorn regularisation.
- **Dataset**: keep current targets unchanged.

### Costs and risks

- **9a (obstacle-integral)**: ~30 min implementation. Risk: if MSE/ρ
  decoupling is intrinsic (level2 obstacle overlap, see Stage 5 §6),
  obstacle-integral loss won't fix the structural problem either —
  we'll find that out cheaply.
- **9b (Sinkhorn)**: ~3 hours including ε / λ_ot tuning. Sinkhorn is
  numerically tricky at small ε; at large ε it collapses to MSE. Three
  λ_ot values × two ε values = 6 short runs to sweep.
- **Both**: the framing-pivot decision says we don't *want* to optimise
  ρ during training right now. These additions only get scheduled
  *after* MSE plateaus across the multi-scene capacity / FiLM / DINOv2
  levers; otherwise we're optimising downstream signal before
  reconstruction is done.

### Decision

**Defer 9a/9b until:**
- Multi-scene capacity bump (action item from Stage 5) is done
- Scene FiLM is tested
- DINOv2 (Stage 7) has been evaluated
- The reconstruction story has a clear plateau

When that plateau hits, **9a is the right experiment first** — almost
free, exactly aligned with the planner cost, and a clean A/B against the
pure-MSE baseline. 9b only if 9a delivers but the ceiling is still
loose.

---

## Things explicitly held off

- **Training a vision transformer from scratch** — dataset is 16k configs,
  not 16M. Small CNN + MLP is the right size class for trained-from-scratch
  components. *Frozen* pretrained backbones (e.g. DINOv2 features
  precomputed once) are a separate question — see "DINOv2 consideration
  (proposed Stage 7)" above.
- **BCE / Soft Dice losses** — both considered as Stage 9 alternatives;
  see that section. BCE is the wrong fit for continuous density targets;
  Dice is only useful in combination with MSE and is lower-priority than
  the obstacle-integral / Sinkhorn options.
- **Per-entity output heads** — we already have `integrate_per_entity`
  post-hoc; collapsing in the model would lose spatial info we can't recover.
- **End-to-end planner training** — predictor first; the planner cost
  `Σ score·S` doesn't need gradients through it.
