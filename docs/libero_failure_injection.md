# Failure Injection on LIBERO Demonstrations

This document describes how FailBench injects hardware-failure events into LIBERO demonstrations and why the implementation differs from the position-controlled pipeline used for our own MJCF scenes. The key technical contribution is a hybrid replay scheme that combines kinematic playback of teleop trajectories with active torque-level control of the surviving joints, so the *post-failure* dynamics are physically meaningful even though the underlying robot is torque-actuated and we never reimplement LIBERO's operational-space controller.

The intended audience is the paper write-up; everything is grounded in the actual implementation under `planner/experiments/libero/` and the verification numbers reproduced from our diagnostic.


## 1. Background

FailBench studies how a Franka Panda arm fails when a hardware fault is injected mid-mission. The output of each trial is a labelled record consisting of a pre-failure RGB-D scene observation, the robot's joint state at the failure instant, and the 3-D contact cloud induced during the post-failure settle. The original pipeline drives our own MJCF scenes with IK + RRT trajectories and writes one HDF5/NPZ per trial.

Adding LIBERO as a second trajectory source is attractive because it ships ~130 teleop manipulation tasks with completed demos that we cannot author by hand. The failure-injection logic itself is independent of where the trajectory comes from — failure modes act on the live MuJoCo state machine. The catch is that LIBERO and our own scenes differ in how the robot is actuated:

| | Our MJCF (`franka_emika_panda/panda.xml`) | LIBERO (robosuite `MountedPanda`) |
| --- | --- | --- |
| Actuator type | Position (`ctrlrange = jnt_range`) | Torque (`robot0_torq_j{1..7}`, ±80 / ±12 N·m) |
| Per-step `ctrl[:7]` | Set every replay step to the dense waypoint | Never written by us — robosuite's OSC is bypassed |
| Native pose-holding | Yes — write a position, integrator tracks it | No — actuators output exactly the commanded torque |
| MJCF source | Hand-authored, single file | Per-demo, robosuite-templated, baked-in absolute paths |

These differences propagate everywhere downstream: failure-injector name lookups, asset-path resolution, post-failure dynamics, and ultimately what the resulting videos and contact clouds look like.


## 2. The naive port — and why it fails visually

The first end-to-end LIBERO trial worked: HDF5 → kinematic replay of `obs/joint_states` → checkpoint → injector flips actuator gains → settle → contacts. The schema matched the existing pipeline and contacts were captured. But the videos were nearly indistinguishable across failure modes:

- `GRIPPER_OPEN` looked like a full arm collapse with the bowl falling somewhere in the wreckage.
- `SINGLE_JOINT(j2)`, `SINGLE_JOINT(j4)`, `SINGLE_JOINT(j6)`, and `ALL_JOINTS` produced near-identical silhouettes.

The cause is the interaction between our replay scheme and robosuite's actuator type. During pre-failure replay we drive the demo trajectory by writing `qpos` directly (`_kinematic_step` in `planner/experiments/libero/runner.py`):

```python
for adr, q in zip(self.handles.arm_qpos_adrs, self.demo.arm_qpos[t]):
    self.data.qpos[adr] = q
...
mujoco.mj_forward(self.model, self.data)
```

We deliberately bypass the OSC controller because reimplementing it would make the runner depend on robosuite at trial time and would also break the demo-faithfulness guarantee — kinematic replay reproduces the recorded trajectory exactly. The unavoidable consequence is that `data.ctrl[arm_actuators]` is never written, so it stays at zero throughout pre-failure.

When failure fires and we transition from kinematic replay to free physics (`mj_step` inside `_capture_and_fork`), the arm has zero commanded torque. On a torque-actuated Panda this means *no joint is actively holding pose*, regardless of which joint we marked as failed. Combined with the model-default damping the arm sags as a whole, and the failure mode's specific signature (e.g. the elbow droop that should distinguish `SINGLE_JOINT(j4)`) is buried under global passive collapse.

This contradicts the behaviour on our own scenes, where the arm uses position actuators and `ctrl[:7]` is written at every replay step from `dense_traj[pt_idx]`. There, the surviving joints continue tracking their last commanded position when failure fires, so the signature is legible. Migrating that behaviour to the torque-controlled robosuite arm requires a torque-level controller for the *post-failure* phase — that is the rest of this document.


## 3. Hybrid replay: kinematic pre-failure, torque-level resistance post-failure

### 3.1 Design decision

We considered three options and committed to the third:

1. **Reimplement OSC throughout the trial.** Maximally faithful to LIBERO's runtime, but pulls robosuite into the failbench env, complicates determinism, and obscures the demo trajectory.
2. **Switch the entire arm to position actuators in the per-demo MJCF.** Mechanically simple but invasive: changing actuator types invalidates the controller assumptions baked into LIBERO's HDF5 (the `actions` field, gain tuning, ctrlranges). It also asymmetrically changes pre- and post-failure dynamics in ways that are hard to defend.
3. **Keep kinematic replay pre-failure, add a torque-level holding controller post-failure on healthy joints only.** The pre-failure trajectory is still exactly the LIBERO demo. Post-failure, the surviving joints actively reject disturbance, while failed joints remain in the same "no actuator authority + no passive resistance" state the injector imposes. This is the smallest change that fixes the qualitative problem.

Option 3 is what we implemented. The conceptual model is: at the failure instant, the robot's last commanded pose is the demo's qpos at that timestep; surviving joints are commanded to hold that pose under a gravity-compensated PD law; failed joints have already had their actuator gains zeroed and thus see no torque regardless of `ctrl`.

### 3.2 The control law

For each healthy arm joint *i* with actuator id *a* and DoF address *d*:

$$\tau_i = b_i + K_{p,i}\,(q^*_i - q_i) - K_{d,i}\,\dot q_i$$

where
- $q^*_i = \mathrm{demo.arm\_qpos}[\mathrm{fail\_idx}, i]$ is the *last commanded pose* — the demo's joint position at the moment failure was injected;
- $b_i = \mathrm{data.qfrc\_bias}[d]$ is MuJoCo's bias term, equal to gravity + Coriolis at the current state;
- $K_{p,i}, K_{d,i}$ are joint-scaled position / damping gains;
- the result is clipped to the actuator's `ctrlrange` and written into `data.ctrl[a]`.

The bias term plays the role of a gravity-compensation feed-forward. Without it, the PD controller would have to fight the full weight of the arm and joints visibly droop before the position error grows enough to compensate. With it, the PD term only needs to reject *disturbances* from the failed joint dragging on the chain, so the surviving arm holds essentially rigid.

`qfrc_bias` is recomputed by `mj_forward` and refreshed inside `mj_step`. We call `mj_forward` at the top of `_apply_resistance` so the bias term reflects the current state, then write `ctrl`, then let `mj_step` integrate one timestep.

### 3.3 Gain selection

The Panda's actuator ranges differ across joints — the wrist actuators are limited to ±12 N·m where the shoulder is ±80 N·m. We picked per-joint Kp scaled to those limits and let Kd = 2·√Kp (critical damping):

```python
pd_kp: tuple = (600.0, 600.0, 600.0, 600.0, 300.0, 120.0, 120.0)
pd_kd: Optional[tuple] = None  # → 2*sqrt(kp)
```

These values are conservative: at the home pose with no disturbance, the bias term alone is sufficient to hold and the PD term contributes negligibly, so saturation is not a concern. Under adversarial loading (e.g. ALL_JOINTS-but-one, where one joint must support six dragging chain links) the PD term can saturate the wrist actuators. In practice this is a non-issue because we never drive a single wrist joint as the sole resistance — multi-joint cascades that include the wrist joints also include the upstream pitch joints.

### 3.4 Healthy-joint masking

The injector tracks failed joints by id:

```python
class LiberoFailureInjector:
    def __init__(self, model, data, handles):
        ...
        self.failed_joint_ids: set = set()
    def _kill_joint(self, joint_id):
        ... self.failed_joint_ids.add(joint_id)
```

Inside `_apply_resistance` we skip any joint whose id is in `failed_joint_ids`. This is a clarity invariant only: `_kill_joint` already zeros that joint's `actuator_gainprm`, so even if we wrote a non-zero `ctrl[a]` for a failed joint, MuJoCo would compute zero torque. The masking makes the dataflow obvious to readers and saves a few floating-point ops per step.

### 3.5 Sequencing of one trial

The end-to-end sequence in `LiberoRunner.run` is:

```
1. Build mj_model from per-demo MJCF (asset paths rewritten on first load,
   then cached at datasets/libero/mjcf_cache/<sha>.xml).
2. Resolve robosuite ↔ FailBench names via ModelHandles (arm joint IDs,
   actuator IDs, finger joints/actuators, EE site, agentview / eye-in-hand
   cameras, robosuite robot-body geom set).
3. Optionally seed full sim state from data/demo_N.attrs["init_state"].
4. For t in 0..fail_idx:
     _kinematic_step(t): qpos[arm] = demo.arm_qpos[t]
                        qpos[fingers] = demo.finger_qpos[t]
                        qvel zeroed on those DoFs
                        mj_forward
5. Capture pre-failure RGB-D + RobotState, save SimStateCheckpoint.
6. last_qpos_cmd = demo.arm_qpos[fail_idx].copy()
7. For each FailureConfig:
     SimStateCheckpoint.restore(...)
     _inject_failure(fc):
       GRIPPER_OPEN/SLIPPERY_GRIP → write to gripper actuator(s)
       SINGLE/MULTI/ALL_JOINTS    → injector.fail_*(joint_idx[s])
     For step in 0..post_failure_settle_steps:
       if resistance_mode == "gravcomp_pd":
           _apply_resistance(last_qpos_cmd):
               for each healthy arm joint i:
                   tau = qfrc_bias[d] + Kp_i*(q*_i - q_i) - Kd_i*qd_i
                   ctrl[a] = clip(tau, ctrlrange[a])
       mj_step
       collect contacts
     injector.restore_all()
8. Aggregate contacts, write npz.
```

Step 6 is the critical one: `last_qpos_cmd` is captured *before* the failure is injected, so the hold target is the trajectory the robot was tracking at that instant — not whatever pose the failure has dragged the arm to.

### 3.6 ALL_JOINTS as a degenerate case

When every arm joint is in `failed_joint_ids`, `_apply_resistance` is a no-op. This is correct: there is no surviving joint to actively control. The arm collapses identically with or without the resistance flag. We keep both videos in the gallery deliberately so the reader can see that the resistance code is *not* a global stiffness override — it touches only what it should.


## 4. Failure-mode semantics on the torque-controlled arm

`FailureMode` is shared with the position-controlled pipeline (`planner/experiments/config.py`), but the implementations route through `LiberoFailureInjector` so they work on the robosuite Panda's actuator graph. The summary:

| FailureMode | Mechanism on robosuite arm |
| --- | --- |
| `GRIPPER_OPEN` | Drive the per-finger actuators to their open ends (`gripper0_gripper_finger_joint{1,2}`, ranges `[0, 0.04]` and `[-0.04, 0]` respectively — "open" is whichever end is closer to zero). Falls back to a single scalar gripper actuator if the model exposes one. Arm actuators are untouched. |
| `SLIPPERY_GRIP` | Same as `GRIPPER_OPEN` but the normalised target is `clip(grip_value/255, 0, 1)`, mapping FailBench's [0, 255] convention onto the local actuator range. |
| `SINGLE_JOINT(j_i)` | `_kill_joint(joint_id)`: zero `actuator_gainprm` rows whose `actuator_trnid[:,0] == joint_id`; set the corresponding `gaintype = biastype = 0`; zero `jnt_stiffness[joint_id]`, `dof_damping[d]`, `dof_frictionloss[d]`; widen `jnt_range[joint_id]` to ±2× the original span around its centre. The joint becomes a free DoF. |
| `MULTI_JOINT([j_a, j_b, ...])` | `_kill_joint` applied independently to each listed joint id. |
| `ALL_JOINTS` | `_kill_joint` applied to all seven arm joint ids. |

Restoration uses cached arrays from the model: `original_gainprm`, `original_biastype`, `original_gaintype`, `original_stiffness`, `original_damping`, `original_ranges`, `original_frictionloss`. We snapshot at injector construction and overwrite the live model on `restore_all()`. This is cheaper than tracking per-joint deltas and avoids drift across multi-failure forks within a single trial.

### 4.1 Joint coverage and the dataset weights

A joint failure is "interesting" for the planner only insofar as it produces structured contact patterns. The Panda's seven joints split cleanly into two groups by whether their axis is gravity-loaded:

| | role | gravity-loaded? | failure visual |
| --- | --- | --- | --- |
| j1 | base yaw | no | arm hangs at current azimuth, mostly still |
| **j2** | **shoulder pitch** | **yes — heavy** | whole arm collapses forward/down |
| j3 | upper-arm twist | no | very subtle |
| **j4** | **elbow pitch** | **yes — moderate** | forearm + EE swing down |
| j5 | forearm twist | no | EE rolls slightly |
| **j6** | **wrist pitch** | **yes — light** | EE pitches down |
| j7 | wrist twist | no | EE rotates in place |

The pitch joints (j2, j4, j6) carry the planner-relevant signal. The rotation joints (j1, j7) contribute mostly to dataset diversity. j3 and j5 are deliberately omitted from the default sampler — their failure signatures are too close to "no failure" to be worth the trial budget.

The default `failure_configs` reflect this priority:

```python
GRIPPER_OPEN                                      0.25
SLIPPERY_GRIP        grip_value=180.0             0.15
SINGLE_JOINT(j2)     shoulder pitch               0.18  ┐
SINGLE_JOINT(j4)     elbow pitch                  0.13  │ pitch group
SINGLE_JOINT(j6)     wrist pitch                  0.07  ┘
SINGLE_JOINT(j1)     base yaw                     0.03  ┐
SINGLE_JOINT(j7)     wrist twist                  0.02  ┘ rotation diversity
MULTI_JOINT(j2, j4)                               0.05  ┐
MULTI_JOINT(j4, j6)                               0.04  ┘ pitch cascades
ALL_JOINTS                                        0.08
```

Probabilities sum to 1.0. Under `failure_sample_mode="sample"` the sampler draws one mode per trial from this distribution; under `failure_sample_mode="all"` (used for ablations) every mode is run on every demo.


## 5. Asset-path remapping (orthogonal but unavoidable)

Each LIBERO HDF5 demo includes a `model_file` attribute containing the per-demo MJCF as a string. When robosuite generated the dataset it baked absolute paths from the recorder's machine into the XML — every `<mesh file=...>` and `<texture file=...>` references e.g. `/Users/yifengz/workspace/robosuite-master/robosuite/...` or `/Users/yifengz/workspace/libero-dev/chiliocosm/...`. MuJoCo cannot load these on any other machine.

Our adapter (`planner/experiments/libero/adapter.py`) resolves this in `_rewrite_xml`: parse the XML, walk `<asset>` mesh/texture children, locate the `robosuite-master/robosuite/`, `chiliocosm/`, or `libero/` token in each `file=` attribute, and rewrite the prefix to the local installation under either `external/LIBERO/.venv/lib/.../site-packages/robosuite/` or `external/LIBERO/libero/libero/`. The rewritten XML is hashed (SHA-1) and cached at `datasets/libero/mjcf_cache/<sha>.xml`. Subsequent loads of demos that share an XML hit the cache without parsing.

We pinned `robosuite==1.4.0` in the sidecar venv specifically because the dataset references `mounts/meshes/rethink_mount/pedestal.stl`, which exists in 1.4.0 but was renamed to `bases/meshes/...` in 1.5.x. Pinning to 1.4 keeps the asset layout consistent with what the dataset expects and avoids per-demo retargeting.

Once a demo's MJCF is materialised, robosuite is no longer needed at trial time — `mujoco.MjModel.from_xml_path` consumes the cached XML directly. The sidecar venv is touched only for (a) the original dataset download and (b) any future re-materialisation runs.


## 5a. Visual-only rendering

LIBERO MJCFs ship two parallel geom sets per body: collision primitives (capsules / boxes / convex hulls) at `geom_group=0` and textured visual meshes at `geom_group=1`. MuJoCo's default `MjvOption` enables every geom group, which renders the green collision capsules on top of the visual mesh — visible as a green-tinted gripper and arm in raw RGB output. That contamination would propagate into any model trained on these images.

`OffscreenRenderer.__init__` in `planner/experiments/data_capture.py` auto-detects this convention. If both `geom_group==0` and `geom_group==1` are present in `model.geom_group`, it constructs an `MjvOption` with `geomgroup[0]=0` (collision off) and `geomgroup[1]=1` (visual on), then threads that option through every `update_scene(...)` call (`render`, `render_depth`, `render_rgbd`, `render_all_cameras`). Models that don't use a 0-vs-1 split — our hand-authored `franka_emika_panda/panda.xml` uses groups 2 and 3 with a few group-0 floor/world geoms — keep the default option, so the existing pipeline renders unchanged.

The toggle applies to both RGB and depth. If a future use case needs the *collision* depth (e.g. ground-truth occupancy or mesh-collision distance fields), that's a separate render path with the option flipped — not an override of the default.


## 6. Verification

The implementation is validated three ways: by-joint dynamics, end-to-end NPZ schema, and qualitative video review.

### 6.1 Per-joint drift after 600 settle steps

Same demo, mid-trajectory pose, settle for 600 steps under each resistance mode:

| failure | resistance | j1 | j2 | j3 | j4 | j5 | j6 | j7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| healthy | `none` | +0.04 | +0.24 | +0.00 | −0.11 | −0.05 | +0.09 | +0.09 |
| healthy | `gravcomp_pd` | **+0.00** | **+0.00** | **+0.00** | **−0.00** | **+0.00** | **+0.00** | **+0.00** |
| j2 | `gravcomp_pd` | +0.00 | **+0.30** | +0.00 | +0.03 | +0.00 | +0.02 | −0.00 |
| j4 | `gravcomp_pd` | +0.00 | −0.00 | +0.00 | **−0.61** | +0.00 | +0.01 | +0.00 |
| j6 | `gravcomp_pd` | +0.00 | −0.00 | +0.00 | +0.00 | −0.00 | **−0.44** | +0.00 |
| all 1..7 | `gravcomp_pd` | +0.05 | +0.23 | −0.00 | −0.15 | −0.12 | +0.15 | +0.20 |

Under `gravcomp_pd`, healthy joints lock at ≤0.03 rad drift; the broken joint exhibits a clean failure-specific droop (j2 +0.30, j4 −0.61, j6 −0.44). With `none`, every healthy joint drifts as much as the failed one — 0.5–4 rad over 800 settle steps in earlier diagnostics — masking which joint actually broke. The `all 1..7` row demonstrates the degenerate case: every joint is failed, both modes are equivalent.

### 6.2 NPZ schema

The smoke test (`scripts/libero/test_libero_pipeline.py`) verifies that the npz output of a LIBERO trial is identical in schema to the existing pipeline's output: same required arrays (`pre_rgb`, `pre_qpos`, `pre_qvel`, `pre_ee_pos`, `contact_positions`, `contact_forces`, `contact_geom_pairs`, `contact_failure_id`, `failure_modes`, `failure_probs`, `impacted_geom_ids`, `task_id`, `traj_id`, `traj_progress`, `seed`, `pre_qvel_norm`), same dtypes, plus per-extra-camera RGB/depth and `post_rgb`. This means downstream tooling — the heatmap regressor, the inspect notebooks, the manifest writer — consumes LIBERO trials without modification.

### 6.3 Qualitative gallery

`notebooks/libero_failure_gallery.ipynb` renders 18 mp4s side-by-side under `none` and `gravcomp_pd` for every distinct failure scenario. The most visually informative pair is `GRIPPER_OPEN`: under `none` the arm sags into the table while the bowl falls, indistinguishable from `ALL_JOINTS`; under `gravcomp_pd` the arm holds rigid at its last commanded pose and only the bowl drops out of the opening fingers. This is the correct visual for that failure mode.


## 7. Limitations and honest caveats

Several aspects of the implementation are deliberately simplified and are worth flagging in the paper:

1. **No active resistance during pre-failure replay.** The pre-failure trajectory is purely kinematic, so the arm has no velocity profile that a real torque controller would produce, and the failure boundary is a one-step transition from "perfect tracking" to "PD hold." We absorb the resulting micro-jolt in the 500-step settle. A future version could ramp Kp from zero to target over ~50 steps to remove the jolt entirely.

2. **`qfrc_bias` includes Coriolis.** When the failed joint induces motion in the chain, the bias term fights that motion in addition to gravity. This is desirable for pose-holding but means our "gravity compensation" is technically gravity-plus-Coriolis compensation. For datasets used in quantitative downstream tasks (force prediction, energy budgets) this distinction may matter.

3. **No locked-joint or partial-degradation failure modes.** The injector models *limp* failures only — total loss of authority and resistance. A real motor failure can also produce a stuck or seized joint (controller commands ignored, but the joint is held by mechanical friction at its current angle), or a partially degraded one (reduced torque limit, intermittent dropouts). Adding these would be a `fail_lock(joint)` and `fail_degraded(joint, factor)` alongside `fail_single(joint)`. Out of scope for this milestone but a natural extension.

4. **Loosened joint limits during failure.** `_kill_joint` widens `jnt_range[joint_id]` to ±2× the original span. Without this, the hard stop would clamp the failed joint mid-fall and the failure would look like "stuck" instead of "free." A more faithful version would simulate the joint hitting its true mechanical stop, but in practice the post-failure settle rarely reaches even the original limits.

5. **No cross-validation against real-robot teleop.** The active-resistance Kp/Kd values are tuned for visual plausibility, not measured against a real Panda's compliance. If the contact clouds are used to predict outcomes on physical hardware, these gains are an obvious place to investigate sensitivity.


## 7a. Contact analysis: entity-level baseline filtering

For visual analysis we want to separate *failure-induced* contacts from *baseline* contacts that already existed at the failure instant — gripper-finger holding the bowl, two cabinet drawer faces touching, plate-on-table, and so on. The naive filter is to record all `(geom1, geom2)` pairs in contact during a pre-injection settle and drop any post-failure contact whose pair is in that baseline set.

That filter has a subtle failure mode in the LIBERO scenes. The wooden_cabinet body has many internal collision geoms representing its drawer mechanism — top panel, side panels, drawer faces. At rest the drawer touches one specific face of the cabinet frame, e.g. `wooden_cabinet_1_g5 ↔ wooden_cabinet_1_g18`. After failure the arm or its held object disturbs the table by a sub-millimeter, the drawer shifts, and the *same* drawer-frame contact transfers from `g5` to a different frame face like `g1` — producing a "new" pair `g1 ↔ g18` that the geom-pair filter classifies as failure-induced. It then projects to a red dot apparently sitting on top of the cabinet, which is misleading: the contact is internal cabinet articulation that flickered between adjacent geoms, not a robot-cabinet interaction.

The fix used in `notebooks/inspect_libero.ipynb` is to filter at the *scene-entity* level. For each geom we compute its top-level scene entity by walking up the body tree (`model.body_parentid`) until we hit a child of the world body, with one special case: every body whose name starts with `robot0_`, `gripper0_`, or `mount0_` is collapsed to a single synthetic "robot" entity (-1), so robot self-contacts don't multiply across the joint chain. The baseline is then a set of `(entity_a, entity_b)` tuples plus *every* `(e, e)` self-pair unconditionally — internal articulation noise within any one entity (cabinet, stove, robot) is always classified as baseline.

Concretely on the test scene this collapses 271 geoms down to 10 scene entities (robot, world, table, two bowls, cookies, ramekin, plate, cabinet, stove). The phantom cabinet contact disappears because both `g1 ↔ g18` and `g5 ↔ g18` map to the same `(cabinet, cabinet)` self-pair, which is in the baseline. Top induced entities for `GRIPPER_OPEN` then look like: `world` (floor, 59 hits), `cookies` (24), `table` (24), `bowl_1` (24), `bowl_2` (23), `plate` (21) — i.e. exactly what the bowl falls onto when the gripper opens. Joint-failure severity ordering also resolves cleanly: `SINGLE_JOINT(j2)` and `ALL_JOINTS` produce ~4× more induced contacts than the lighter elbow/wrist failures, because the heavier collapse drags more of the chain into the scene.

This filter currently lives only in the inspection notebook. The on-disk NPZs from `manager.save_sample_npz` still record raw `(geom1, geom2)` pairs in `contact_geom_pairs`, so any downstream tool that wants the same separation must redo the entity-mapping on the captured `model`. A future refactor could move the filter into the runner or into a post-processing pass over the NPZ corpus.


## 8. File map

For reviewers diving into the code:

```
planner/experiments/libero/
  adapter.py     load_demo + materialise_mjcf + asset-path rewriting
  naming.py      ModelHandles: robosuite ↔ FailBench joint/actuator name resolution
  failure.py     LiberoFailureInjector + parse_joint_spec
  runner.py      LiberoRunner, LiberoTrialConfig, _apply_resistance, _default_failures

scripts/libero/
  download_libero.py       wrapper around LIBERO's HF downloader
  play_libero_demo.py      MuJoCo viewer playback (no failure)
  run_libero_trial.py      one trial → npz, with --resistance flag
  record_failure_video.py  same trial → annotated mp4
  test_libero_pipeline.py  end-to-end smoke + npz schema check

notebooks/
  inspect_libero.ipynb         per-trial inspection (state, all images, contacts)
  libero_failure_gallery.ipynb 18-video A/B gallery of failure modes
```

The shared infrastructure that LIBERO trials *do not* duplicate: `planner/experiments/data_capture.py` (ContactExtractor, OffscreenRenderer, ContactProjector, SimStateCheckpoint, RobotStateCollector), `planner/experiments/config.py` (FailureMode, FailureConfig), `planner/experiments/manager.py` (npz writer, manifest). LIBERO trials live alongside our own under a separate dataset root (`datasets/libero/v<N>/`) and never touch `scenes/`.
