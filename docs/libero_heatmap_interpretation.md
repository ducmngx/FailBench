# Interpreting the LIBERO contact-heatmap model

Companion to `docs/libero_heatmap_model.md` (which covers the architecture).
This document explains, in detail, **what the model is predicting**, **what the
inputs encode**, **how each input shapes the output**, and **how to read the
qualitative gallery in `notebooks/eval_libero_model.ipynb`**.

If you only read one paragraph: the model's `mass` output is "if you applied a
random failure from our failure distribution to the robot in this scene at
this pose, **how much severity-weighted contact would land on each pixel of
the agentview image, on expectation?**" — measured in log1p-compressed
"force × prior-probability" units, smoothed by a 2D Gaussian. High = lots of
expected impact at this pixel; zero = no expected impact.

---

## 1. What "mass" actually is — full chain from physics to label

Every pixel value in the predicted (or target) mass map is the end of a long
chain of transformations. Walking through it forward:

### 1.1 Raw contacts during a settle (in `LiberoRunner`)

Per failure trial, MuJoCo's contact solver reports a set of contact points
during the post-failure settle. Each contact has:

- `pos` — 3D world-frame point where the contact occurred.
- `force` — 6-vector (3 linear + 3 torque) of the contact wrench.
- `geom1, geom2` — which two geoms touched.

These are filtered to "robot-impact" contacts (one geom on the robot, one
not) at force-magnitude ≥ 1 N, and saved into the trial npz as
`contact_positions (N, 3)` and `contact_forces (N, 6)`.

### 1.2 Severity weighting — force magnitude

A 100 N collision matters more than a 2 N graze. Each contact's weight starts
as:

```
severity_i = ‖force_xyz_i‖     # magnitude of the 3D linear-force component, in Newtons
```

### 1.3 Failure-prior weighting — the "expectation" part

A single trial samples *one* failure config (e.g. `GRIPPER_OPEN` with prior
prob 0.25, or `SINGLE_JOINT joint4` with prior 0.10). The planner cares
about the **expected** contact distribution given the *full* failure
distribution. So each contact is weighted by the prior probability of the
failure that produced it:

```
weight_i = severity_i · failure_probs[contact_failure_id_i]
```

(`failure_probs` is stored per-trial in the npz; this is the prior probability of
the sampled failure config in the dataset's failure-sampling distribution.)

Implementation: `planner/risk/projection_labels.py::contact_weights_force_prior`.

### 1.4 Projection to the agentview image plane

Each 3D contact position is projected through the agentview camera into a 2D
pixel coordinate `(u, v)` and a depth value (metres in front of the camera).
Off-screen / behind-camera contacts are dropped via `in_frame()`.

This step is **deterministic and lossy**: the geometry of the projection is
fully known (we have the camera's world-frame pose and intrinsics), but
multiple 3D points along the same camera ray collapse onto the same pixel.
The depth channel (Section 1.6) is what recovers some of this information.

Implementation: `planner/risk/projection_labels.py::_DynamicCameraProjector`.

### 1.5 Rasterisation + Gaussian smoothing — the mass map

Each surviving contact's weight is splatted into the pixel grid it lands in:

```
mass_sum[v_i, u_i] += weight_i
```

Then the whole grid is convolved with a 2D Gaussian (σ = 8 px) to spread the
splat over a few-pixel neighbourhood. Smoothing serves two purposes:
1. Makes the regression target dense (an MSE-friendly signal across pixels)
   rather than a delta-function-like sparse target.
2. Captures positional uncertainty (the exact pixel a contact lands on is
   noisier than the rough region).

Mass is preserved by the Gaussian: `sum(after) ≈ sum(before)` modulo boundary
truncation. So at this point, each pixel value carries the units of
**"severity × prior, smoothed"** — call this raw mass.

### 1.6 Depth channel — pixel-wise mean depth

In parallel with the mass sum, we also accumulate `depth_sum[v_i, u_i] +=
weight_i · depth_i`. After Gaussian-smoothing both numerator and denominator,
the per-pixel mean depth is

```
depth_map[v, u] = gauss(depth_sum) / gauss(mass_sum)     where mass_sum > 0
                = 0                                       elsewhere
```

So `depth_map[v, u]` is the **weighted-mean depth (in metres) of the contacts
that contributed mass to this pixel**. It disambiguates occlusion: at a pixel
where the agentview ray passes through both the table and the cabinet, the
depth value tells you which one was actually struck.

### 1.7 Aggregation across sibling failure trials

Each `(task, demo_key, bin_idx)` group in the LIBERO v1 dataset has three
sibling trials — same pre-failure state, three different sampled failures.
The training target is the weighted mean across siblings, weighted by their
failure priors:

```
target_mass = (Σ_i mass_i · failure_prior_i) / Σ_i failure_prior_i
```

This is the empirical estimate of `E[mass | failure distribution]`. The
aggregation reduces per-failure variance (so the same input doesn't have
three wildly different targets) and matches the paper's risk formulation.

Implementation: `planner/risk/projection_labels.py::aggregate_labels`.

### 1.8 `log1p` compression

Raw aggregated mass values span ~0 to 10⁴ (some trials have many high-force
contacts, others almost none). MSE on raw values would be dominated by a few
extreme pixels in a few extreme trials. We apply `log1p` so the storage and
training target are in a ~[0, 5] range:

```
target_mass_log1p = log(1 + target_mass_raw)
```

`log1p` is **invertible** via `expm1`. The label files (`labels.npz`) store
this log1p-compressed version as `target_mass`. **Every reference to "mass"
in the model output is in log1p units** unless explicitly noted otherwise.

### Summary chain (one pixel)

```
contact force in N (e.g. 35 N)
  × failure prior (e.g. 0.25)                     → severity × prior, e.g. 8.75
  splat into a pixel of (480, 640) agentview      → raw mass per pixel
  smooth with σ=8 px Gaussian                     → still raw mass per pixel, blurred
  average across 3 sibling trials (prob-weighted) → expected mass per pixel
  log1p                                           → target_mass value, e.g. 2.27
```

Any predicted mass pixel of 2.27 means: "the model thinks `log1p(expected
force-prior-weighted contact mass at this pixel) = 2.27`". Undo: about 8.7
units of severity-weighted mass.

---

## 2. What the model takes in

Per item (after `LiberoLabelDataset.__getitem__`):

| Input | Shape | Source | What it encodes |
|---|---|---|---|
| `rgb` | (3, H, W) | `pre_rgb` from npz, ImageNet-normalised | Agentview RGB at the pre-failure instant. Shows objects, table, cabinet, ramekin, etc. — the *scene configuration* and the *robot's pose in that scene*. |
| `depth` | (1, H, W) | `pre_depth` from npz | Per-pixel agentview depth in metres. Encodes 3D geometry directly: object surfaces, table height, robot occlusion. |
| `state` (opt.) | (14,) | `pre_qpos` + `pre_qvel` | Joint angles + joint velocities at the pre-failure instant. Redundant-but-not-identical to what's in the RGB (the arm pose is visible there, but state gives it without the visual translation). |
| `is_holding` | () | `<split>/holding.csv` sidecar (built by `scripts/libero/compute_holding_flag.py`) | Binary flag: 1 iff the gripper is in commanded-close mode at `fail_idx`. Disambiguates pre-failure regimes that look similar in RGB but produce very different contact distributions — e.g., gripper-open failure during transport (holding=1, drops object) vs during approach (holding=0, near-zero new contacts). |

### Why the `is_holding` bit matters

A 1-D binary feature is information-dense for this task. Concrete failure-distribution differences it captures:

- **`GRIPPER_OPEN` × holding=1** → dropped object → table contacts directly below the gripper.
- **`GRIPPER_OPEN` × holding=0** → near-zero new contacts (failure is invisible).
- **`SINGLE_JOINT` × holding=1** → arm + held object both become contact sources, often hitting separate surfaces.
- **`SINGLE_JOINT` × holding=0** → only arm-link contacts.

The signal is derived from the LIBERO demo HDF5 `actions[:, -1]` at the pre-failure step (positive ≈ commanded-close), with a ±2-frame majority vote to handle action-flip transients. See `scripts/libero/compute_holding_flag.py` for the exact rule.

All three describe the **same instant** — just before the failure is injected.
The model is asked to predict what *would* happen if a random failure from
the failure distribution were applied right after.

`pre_rgb` and `pre_depth` are pixel-aligned (same camera pose, same H×W), and
the target `target_mass` is pixel-aligned with them. This is the load-bearing
property: the model never has to translate between coordinate systems.

---

## 3. What the model outputs

Forward returns a dict with three tensors:

| Output | Shape | Units | Meaning |
|---|---|---|---|
| `mass` | (B, 1, H, W) | log1p of severity-prior | Predicted log1p of expected severity-weighted contact density at each pixel, under the failure distribution. |
| `depth` | (B, 1, H, W) | metres | Predicted mean depth of the contacts that contribute to each pixel. Only meaningful where `mass` is non-negligible (where mass≈0, depth is undefined). |
| `mass_total` | (B,) | log1p, scaled by 1/1000 | Predicted scalar — the sum of log1p mass across the entire image, divided by 1000 (so targets stay O(1)). An auxiliary "how much will go wrong overall" signal that gives the bottleneck features a global loss to optimise. |

### What these are NOT

- **NOT a binary mask.** Each pixel is a real-valued regression target, not
  "contact here / no contact here". A value of 0.5 doesn't mean a 50%
  probability of contact; it means the log1p of expected severity-weighted
  mass is 0.5, which inverts to ~0.65 units of force × prior.
- **NOT a per-failure prediction.** The model predicts the expectation over
  the entire failure distribution; it does not distinguish "gripper-open
  contacts" from "joint-4 contacts".
- **NOT in physical force units directly.** It's force × prior weight ×
  Gaussian-smoothed × log1p-compressed × aggregated. To recover something
  close to "expected total Newton-seconds of contact at this pixel", you'd
  need `expm1(pred_mass)` and treat the result as severity × prior.
- **NOT a 3D quantity.** Mass is the agentview projection — it loses height
  information (which the voxel head would have preserved).

### Output value ranges in practice

Empirically from the trained model:
- `mass` peak per image: 0.5 – 5 (peak corresponds to ~0.6 – 150 units of
  raw severity × prior).
- `mass` across most pixels: 0 – 0.1 (the heatmap is sparse).
- `depth`: 0.6 – 1.5 m where mass is non-negligible (the LIBERO workspace
  is roughly that range from the agentview camera).
- `mass_total`: 0.1 – 3 (after the ×1000 scaling — undo by ×1000 to recover
  the raw integrated log1p mass).

---

## 4. Input → output: which input drives which output, and how?

### 4.1 RGB → mass

The RGB stream is the primary source of **scene-configuration information**.
Two LIBERO demos within the same task have different starting positions for
the objects (bowl-on-stove vs bowl-on-cabinet, etc.), and that's only
visible in pixels.

What the model learns from RGB:
- **Object identity and position** — "there's a black bowl near the centre
  of the table at this pose, with a wooden cabinet behind it".
- **Robot pose, visible portion** — the agentview shows the arm; the arm's
  configuration influences which obstacles the failure will sweep toward.
- **Scene affordances** — heuristically, contacts cluster around graspable
  objects, tabletop surfaces, and the cabinet front. RGB pixels covering
  those regions tend to predict elevated mass.

The first conv is 4-channel (RGB + depth fused immediately), so RGB and
depth jointly drive every subsequent feature map. You can't strictly
attribute "this output pixel came from RGB only" — but ablating RGB
(zero-fill) would lose object semantics, leaving the model to guess from
geometry alone.

### 4.2 Depth → mass and depth output

Depth contributes:
- **3D structure independent of texture.** Two visually-similar scenes
  (e.g. matte black objects on a matte black table) can have very different
  contact distributions if their 3D layout differs; depth gives the model a
  reliable shape signal.
- **Distance-aware reasoning.** Pixels at larger depth are physically
  farther; the model can use this to weight where the arm is likely to swing
  into objects.
- **A target the model can copy.** The depth-output head's target is the
  *contact-mean depth*, which on object surfaces will closely track the
  observed `pre_depth` value. The model can essentially route observed depth
  to predicted depth at high-mass pixels.

### 4.3 State (when enabled) → bottleneck

`pre_qpos` and `pre_qvel` are concatenated, passed through a small MLP, and
broadcast-tiled to the encoder's bottleneck (stride-32). This means the
state feature only influences *global* decisions: which large region of the
image gets elevated mass. It cannot encode pixel-level distinctions.

Round-4 learnability analysis showed that **state alone cannot beat the
constant baseline** — its job in this architecture is to disambiguate
visually-similar scenes that have different arm configurations, *if* there
are any. In practice, the visible portion of the arm in RGB already conveys
most of the qpos signal.

### 4.4 Bottleneck → spatial output

The encoder bottleneck (`s4`, shape `(B, 512, H/32, W/32)`) is the most
compressed representation of the input. By the time information reaches
here, the input image has been reduced from 240×320 to 7×10 features. So:

- A single bottleneck feature corresponds to a ~32×32 patch of the input.
- The decoder's job is to upsample this back to (H, W) while injecting
  high-resolution detail from the encoder skip connections.
- The output mass at a pixel is therefore a function of **both** the
  bottleneck features at the surrounding ~32×32 patch (semantic / global
  information about "which obstacles are nearby") **and** the encoder
  skip-connection features at this exact pixel (local edge / colour /
  depth detail).

### 4.5 Effective receptive field

For ResNet-18 the theoretical receptive field is the whole image by the
bottleneck. In practice the effective receptive field is smaller (a few
hundred pixels around each output location). This means:

- A predicted hot spot at pixel (200, 300) is informed by RGB+depth from
  roughly a 150-px-radius neighbourhood.
- A contact that should land at (200, 300) but whose causal "evidence" is a
  cabinet edge 400 pixels away may be missed.

Cross-task evidence transfer (e.g. "I learned what cabinets look like in
task A; apply it to task B") happens through the shared ResNet weights —
not through long-range attention within a single forward pass.

---

## 5. Reading the qualitative gallery

The gallery cell in `notebooks/eval_libero_model.ipynb` shows three groups —
best, median, worst — based on per-trial MSE. Each row has four columns.

### 5.1 The four columns

| Column | What it shows | How to read it |
|---|---|---|
| **RGB** | The de-normalised `pre_rgb` from the trial's npz | Sanity check on what the scene looked like. Note the object positions and the visible arm pose. |
| **target** | `pre_rgb` with the target heatmap overlaid in `cmap="hot"` | Where contacts *actually* landed (force-prior weighted, smoothed, aggregated, log1p). Bright = high mass, dark = none. The colormap is per-trial-scaled, so brightness across rows is not comparable. |
| **prediction** | `pre_rgb` with the predicted heatmap overlaid in `cmap="hot"` | What the model thinks the target should be. Same colourmap and scaling as the target column for direct visual comparison. |
| **pred − target** | Signed difference map in `cmap="RdBu_r"` | **Red** = pred > target (model over-predicted). **Blue** = pred < target (model under-predicted). White = perfect match. The dynamic range is symmetric per trial so the centre is always 0. |

### 5.2 What "good" looks like

A successful prediction (low per-trial MSE, high Pearson):
- Target and prediction heatmaps have hot spots in the same locations.
- The diff panel is mostly white with mild red/blue speckles.
- High target mass corresponds to a physically plausible region in the RGB
  (e.g., over a graspable object, over the table directly under the arm,
  over a cabinet door the arm is about to swing into).

### 5.3 What "bad" looks like

Common failure patterns in the worst-case rows:

| Pattern | Diff panel | What it means |
|---|---|---|
| Bright target, dark prediction | Strong **blue** patch where the target is hot | Model under-predicts a real contact region. Often happens when the failure produces an unusual or sparse contact pattern the model hasn't seen. |
| Dark target, bright prediction | Strong **red** patch | Model predicts contacts that shouldn't happen. Often the model defaults to a "mean prior" hot spot regardless of trial. This is the **Simpson's paradox** failure mode (Section 6). |
| Spatially-offset hot spots | **Blue** at the true location + **red** nearby | Model has the right *kind* of contact but wrong location. Often a few pixels off due to coarse decoder upsampling or limited spatial precision. |
| Both target and prediction dark | Diff panel is mostly white | A quiet trial — neither has much going on. Often the model wins these by accident even when its prediction is wrong because both values are near zero. |

### 5.4 Reading individual cases

For a specific row, ask:
1. **Where in the RGB is the bright target region?** Is it over an object,
   the table edge, the cabinet front?
2. **Does the model's hot spot match?** Yes → the model has learned that
   scene's contact pattern. No → where is the model's hot spot, and what's
   there in the RGB?
3. **What does the diff panel tell you?** Concentrated blue = systematic
   miss. Concentrated red = systematic over-prediction. Diffuse =
   stochastic noise.
4. **Compare to siblings.** Each trial belongs to a 3-trial group; the
   target is averaged. The "real" failure that produced this trial might
   look very different from the smoothed-target shown here.

---

## 6. The Simpson's paradox finding — why aggregate MSE looks better than per-trial

The eval notebook reports:

- **Aggregate**: model MSE / baseline MSE = **0.687** (model is 31% better)
- **Per-trial mean reduction**: **−27.6%** (model is 28% *worse* on average)

These are not contradictory; they tell different stories about the same
predictions. The math:

- Aggregate ratio: `mean(model_mse) / mean(baseline_mse)`.
- Per-trial mean reduction: `mean(1 - model_mse / baseline_mse)`.

The two would agree only if the per-trial MSE ratios were constant across
trials. They diverge whenever the model **gains a lot on a few trials and
loses a little on many trials**, because the gainers dominate the
aggregate but each loser pulls the per-trial mean down equally.

Empirically, in our data:

- **High-mass trials** (trials with lots of failure-induced contacts) have
  large absolute baseline MSE values. The model often nails them — its
  spatial structure is correct, so its MSE on these trials is much smaller
  than baseline. These dominate the aggregate ratio.
- **Low-mass trials** (sparse or near-zero contacts) have small baseline
  MSE values. The constant-mean predictor wins by default — predicting the
  global average is closer to "near-zero" than the model's typical
  over-prediction. So the per-trial reduction on these is negative.

This is the classic "average of ratios vs ratio of averages" effect. Both
numbers are correct; they answer different questions:

- "How much do I save on a typical high-impact trial?" → aggregate (model
  wins).
- "On a randomly sampled trial, does the model help me?" → per-trial mean
  (model often hurts).

For a planner that weighs trials by their potential impact, the aggregate
view is the relevant one. For a "should I even use this model?" sanity
check on quiet trials, the per-trial view matters.

**Implication**: the model has implicitly learned a "predict the mean
distribution" prior and applies it everywhere. The next architectural
improvement should help it distinguish quiet trials from busy ones — either
via a learned global gate, or by training the `mass_total` head harder so
the model knows when to suppress the spatial head.

---

## 7. Worked example

Suppose you open the eval notebook, look at a worst-case row, and see:

- **RGB**: A LIBERO `libero_spatial` scene with a black bowl on the stove,
  a plate to the right, and the robot arm overhead.
- **Target**: Bright hot spot over the stove (centre-left of the image) and
  a smaller spot over the plate.
- **Prediction**: Bright hot spot over the table directly below the arm
  (centre of the image), nothing over the stove.
- **Diff**: Strong **blue** over the stove (target was hot, pred wasn't),
  strong **red** over the table-centre (pred was hot, target wasn't).

**Interpretation**:

The model is predicting that contacts will happen "wherever the arm is",
not "wherever the failure will swing the arm into objects". For this
specific trial, the failure caused the bowl on the stove to be knocked,
plus a glance off the plate. The model's prior is dominated by "drops over
the gripper centre" trials and hasn't learned this scene-specific failure
pattern.

What to do about it:
- Add wrist-cam head — the wrist camera sees the bowl directly and would
  give a localised signal.
- Train longer (or with augmentation) to overcome overfitting that
  collapsed at epoch 6.
- Train a per-trial variant that takes `failure_mode` as a learned
  embedding, since this row's outcome strongly depends on which joint
  failed.

---

## 8. Caveats and pitfalls

- **The colormap is per-trial-scaled.** Brightness between rows is not
  comparable. A bright spot in row 1 might be 10× weaker than a bright
  spot in row 2.
- **`mass` is log1p-compressed.** A target of 2 corresponds to raw
  severity-prior of `expm1(2) ≈ 6.4`, not 2. The visual contrast is
  compressed, which makes faint spots more visible at the cost of
  flattening the dynamic range.
- **`depth` is undefined at zero-mass pixels.** Don't interpret the depth
  channel where the mass channel is near zero — the value is just whatever
  the network happened to produce.
- **The model is trained at 240×320, not native 480×640.** Predictions are
  inherently coarser than the target's native resolution; pixel-level
  comparisons are at the half-resolution scale.
- **Aggregation hides per-failure detail.** The target you see in the
  gallery is the prior-weighted average of 3 sibling trials. The actual
  per-failure heatmaps differ; if you want to see them, load the trial
  npzs directly and project per-trial (see
  `notebooks/inspect_libero_labels.ipynb`).
- **The cam-projected heatmap can't see contacts behind opaque surfaces.**
  A contact inside a closed cabinet projects to a cabinet-front pixel; the
  depth channel will report the cabinet depth (not the deeper contact
  depth). For occlusion-aware analysis, the voxel head is the right tool —
  but that head isn't trained yet.

---

## Round 7 update — per-trial training with `failure_mode` input

The Round 3/5/6 model is trained against **aggregated** labels (one target
per (demo, bin_idx) group, averaging 3 sibling failure trials). Section 6's
Simpson's-paradox finding turned out to be a symptom of a deeper issue:

**The aggregated label was hiding a near-deterministic rule.** A
gripper-class failure (`GRIPPER_OPEN`, `SLIPPERY_GRIP`) with no held object
produces *almost zero contacts*. The arm doesn't fall, the gripper just
opens on nothing; no surfaces are touched.

Empirically (libero_spatial):

| failure_mode × is_holding | zero-contact rate |
|---|---|
| `GRIPPER_OPEN`, h=0 | **94 %** |
| `SLIPPERY_GRIP`, h=0 | **92 %** |
| `MULTI_JOINT`, h=0 | 17 % |
| `SINGLE_JOINT`, h=0 | 37 % |
| `ALL_JOINTS`, h=0 | 0 % |
| any failure, h=1 | 0–7 % |

When you aggregate `(GRIPPER_OPEN, h=0)` (target ≈ 0) with `(SINGLE_JOINT,
h=0)` (target ≈ heavy joint contacts) and `(MULTI_JOINT, h=0)` (similar) into
one label, the heavy contacts pull the aggregate away from zero. The trained
model never gets to learn the rule.

### 7.1 What changed

Round 7 retrains the same architecture on **per-trial labels** — one label
per of the 45 000 trials, indexed by trial. Adds a new model input:
`failure_mode_id`, encoded as a 5-D one-hot and concatenated to the FiLM
conditioning vector alongside `is_holding`. The FiLM `cond_dim` grows from
1 to 6 but identity initialisation is preserved (γ=1, β=0 at start).

Per-trial labels live in `datasets/libero/v1/<split>/labels_per_trial.npz`
(produced by `scripts/libero/build_per_trial_labels.py`). Same projection
+ smoothing pipeline as the aggregated labels, just no averaging step.
Stored at half resolution (240×320, float16) so total size is ~100 MB per
split rather than ~6 GB per split for full-res per-trial labels.

The training script gains `--per_trial` and `--use_failure_mode` flags;
`use_failure_mode` requires `per_trial` because the failure-mode input is
incoherent with the aggregated target.

### 7.2 What the model is now predicting

The aggregated model predicted `E[contact | failure distribution]`. The
per-trial-with-failure_mode model predicts **`E[contact | failure_mode,
is_holding, pre-state]`** — a much finer conditional.

This changes the semantic of `mass[v, u]` slightly:

| Round | target meaning | model output meaning |
|---|---|---|
| 3 / 5 / 6 | expected log1p mass under the full failure distribution | predicted expectation |
| 7 | log1p mass for the specific sampled failure mode (per-trial) | predicted log1p mass given the failure mode the input one-hot encodes |

To get the original aggregated quantity from R7 at inference time, you'd
run the model 5 times (once per `failure_mode_id`) and weight-sum by the
failure prior.

### 7.3 The deterministic rule learned

After 50 epochs, per-(failure_mode, is_holding) val MSE looks like this:

| failure_mode | is_holding | n | model MSE | target total | target zero-rate |
|---|---|---|---|---|---|
| GRIPPER_OPEN | 0 | 159 | **0.00005** | 5.1 | 96 % |
| GRIPPER_OPEN | 1 | 212 | 0.00035 | 45.8 | 19 % |
| SLIPPERY_GRIP | 0 | 98 | **0.00004** | 6.5 | 93 % |
| SLIPPERY_GRIP | 1 | 135 | 0.00021 | 31.3 | 24 % |
| SINGLE_JOINT | 0 | 264 | 0.00778 | 633 | 37 % |
| SINGLE_JOINT | 1 | 379 | 0.00942 | 908 | 0 % |
| MULTI_JOINT | 0 | 59 | 0.00374 | 505 | 12 % |
| MULTI_JOINT | 1 | 80 | 0.00378 | 676 | 0 % |
| ALL_JOINTS | 0 | 44 | 0.00602 | 777 | 0 % |
| ALL_JOINTS | 1 | 70 | 0.00489 | 1054 | 0 % |

The two near-deterministic classes (`GRIPPER_OPEN × h=0` and `SLIPPERY_GRIP
× h=0`) have val MSE around **0.00005** — two orders of magnitude lower
than the R3 model's MSE on the same trials. The model has clearly learned
"gripper-class failure + not holding → predict zero everywhere".

The harder classes (joint failures, especially the 33 % of `SINGLE_JOINT ×
h=0` trials where the arm does find something to hit) remain the dominant
contributors to overall val MSE.

### 7.4 Conditional flip-sensitivity

| Flip | mean L2 change in prediction | what it tells us |
|---|---|---|
| `is_holding` (0↔1), R5 (bottleneck) | 0.008 | model ignores the bit |
| `is_holding` (0↔1), R6 (FiLM) | 0.008 | same — FiLM didn't help |
| **`is_holding` (0↔1), R7 (per-trial + failure_mode)** | **0.004** | bit is partially used |
| **`failure_mode` (GRIPPER_OPEN ↔ ALL_JOINTS), R7** | **0.078** | model is genuinely using the failure mode |

R7 produces ~20× larger prediction changes from flipping `failure_mode`
than from flipping `is_holding`. The failure-mode input is doing real
work; `is_holding` matters mostly through its interaction with
`failure_mode` (in particular, lowering the mass on gripper-class
failures).

### 7.5 Implications for reading the gallery

When inspecting an R7 prediction:

- If `failure_mode ∈ {GRIPPER_OPEN, SLIPPERY_GRIP}` and `is_holding=0`,
  the prediction should be near-zero everywhere. A non-zero hot spot is
  a real anomaly worth investigating.
- For joint failures, the prediction reflects what physically happens
  with that *specific* failure mode, not the mode-mixture average. The
  spatial pattern can differ substantially across modes — see the
  "failure-mode sweep" gallery cell in the eval notebook.
- The target's mass total varies an order of magnitude across rows of
  the per-class table above. The colourmap is per-trial-scaled, so a
  bright hot spot in a `GRIPPER_OPEN × h=1` row corresponds to ~45 units
  of total mass, whereas the same brightness in an `ALL_JOINTS × h=1`
  row corresponds to ~1050 units. Always check the printed peak / total
  next to the panel.

### 7.6 What R7 did NOT solve

- Joint-failure trials with high contact mass still have val MSE
  comparable to the aggregated model (~0.005–0.009). The conditioning
  signal helps where the rule is near-deterministic; it doesn't unlock
  any new information on the genuinely-hard joint-collision cases.
- The model still doesn't see wrist-cam input. Many fine-grained contact
  patterns inside a cabinet would benefit from it.
- Multi-split (libero_goal, libero_object) training still pending.

---

## TL;DR cheatsheet

For **R3/R5/R6** (aggregated training):

- `mass[v, u]` = predicted log1p(expected severity × prior contact mass)
  at pixel (v, u) of the agentview image, **under the full failure
  distribution**.
- Inputs: RGB + depth (+ optional `is_holding` for R5/R6).
- Architecture: ResNet-18 encoder + U-Net decoder; R6 adds FiLM modulation
  at every decoder stage.
- Aggregate val mass-MSE ≈ 0.0048 across all three rounds.

For **R7** (per-trial + failure_mode):

- `mass[v, u]` = predicted log1p(severity × prior contact mass) at pixel
  (v, u) **for the specific failure mode the model is conditioned on**.
- Inputs: RGB + depth + `is_holding` + `failure_mode_id` (5-D one-hot).
- Same architecture; FiLM `cond_dim` widened from 1 → 6.
- Trained on 45 000 per-trial labels (vs 15 000 aggregated groups).
- Val mass-MSE **0.0046**; on near-deterministic classes (gripper × no
  holding), MSE drops to **0.00005** — two orders of magnitude better than
  R3.
- To recover the aggregated quantity at inference time: run the model
  once per `failure_mode_id` and weight-sum by the failure prior.

Universal rules:

- **`depth[v, u]`** = predicted mean depth (m) of the contacts
  contributing to that pixel. Defined only where mass is non-zero.
- **`mass_total`** = predicted total log1p mass over the whole image,
  scaled by 1/1000. Auxiliary signal.
- **Gallery diff column**: red = over-prediction; blue = under-prediction.
- **High target** = many high-force impacts × high failure prior.
- **Colourmap is per-trial-scaled**; brightness across rows is not
  comparable. Read the printed peak/total annotation.
