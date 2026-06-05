# RoboCasa integration — Phase 0 complete, Phase 1 pending

**Status (2026-06-01):** Schema confirmed compatible, install green, smoke replay passing. No adapter edits yet. Two open design decisions before Phase 1 (see §6).

This document is the durable record of what's installed, what works, and what's still to decide. Pick up from §6 when resuming.

---

## 1. Why

LIBERO v1 (45k trials, 3 kitchen scenes) is reaching diminishing returns for the heatmap regressor (val Spearman 0.42 on scene_level2). RoboCasa adds 24 procedurally-generated kitchen tasks with substantially more visual + scene diversity than LIBERO. Two research lines need the data:

1. **Contact / interaction prediction** (current FailBench line). The heatmap regressor predicts per-config 2D contact density. More scene diversity should help generalisation, especially when paired with a pretrained visual encoder.
2. **Action + failure-conditioned world model** (next FailBench line — see `docs/world_model_design.md`). Predicts the post-failure rollout, not just the final contact pattern. Needs per-step sim states and frames during the post-failure settle.

The generation schema must support both lines from day one to avoid a costly v2 regen.

---

## 2. Data source — which artifact, why

Three candidate HuggingFace artifacts were evaluated; only one is usable for failure injection.

| Artifact | Schema fit | Verdict |
|---|---|---|
| `nvidia/RoboCasa-Cosmos-Policy` | strips `model_file`/`init_state` from HDF5 | Useless for replay. Useful only as a visual-encoder pretraining corpus (RGB only). |
| `binhng/robocasa-100demos-5chosen-tasks` (user's curated mirror) | upstream RoboCasa robomimic schema preserved | **Right artifact.** 5.38 GB, 5 task HDF5s, 100 demos each. |
| Original upstream RoboCasa release | same schema, more tasks | Reachable if the binhng subset gets exhausted. ~50 demos × 24 tasks. |

The user's binhng/* collection currently covers ~10–12 unique tasks across multiple subsets (`-100demos-5chosen-tasks`, `-30demos-5chosen-tasks`, `-30and100demos-7chosen-tasks-for-Binh`, etc.). De-duplicated estimate: 600–1,200 demos → ~9k–18k contact trials at the LIBERO v1 multiplier (15 trials/demo).

### 2.1 Provenance of the binhng HDF5s

The MJCFs reference fixtures like `stovetops/pack_1_top_gas/` and embed absolute paths from `/home/soroush/code/{robosuite,robocasa}-dev/`. The fixture naming and Soroush attribution match **UT-Austin-RPL/mimicdroid-robocasa**, not vanilla `robocasa/robocasa`. Confirmed by checking the fork's fixture directory: `pack_1_top_gas` exists in the fork, does not exist in vanilla (where stovetops are named `Stovetop002`/`Stovetop004`/...).

**Don't reinstall vanilla robocasa.** The MJCFs were generated against the mimicdroid fork and only work with it.

---

## 3. Install state

### 3.1 Repositories

```
external/robocasa/                  UT-Austin-RPL/mimicdroid-robocasa @ latest
external/robosuite_for_robocasa/    ShahRutav/robosuite             @ abs_robot
```

Both cloned `--depth 1`. Total source ~80 MB; the asset tree at
`external/robocasa/robocasa/models/assets/` is 21 GB after asset download.

### 3.2 Sidecar conda env `robocasa`

- Python **3.10** (mimicdroid pins `numba==0.56.4` which requires ≤3.10; system only had 3.12, so a fresh conda env was the path of least resistance).
- `robosuite==1.5.0` (ShahRutav abs_robot fork, editable)
- `robocasa==0.2.0` (mimicdroid, editable)
- `mujoco==3.2.6` (pinned by robocasa; older than failbench_env's 3.3.4 but works for replay)
- `matplotlib`, `h5py`, `numpy==1.23.3`, `numba==0.56.4`

This env is **only** needed for one-time MJCF asset path resolution. The runtime replay pipeline (Phase 1) can run from `failbench_env` using bare MuJoCo, mirroring the LIBERO sidecar pattern.

### 3.3 Install gotcha — `robocasa.__path__`

Running Python from `/home/aaron/workspace/FailBench/` causes the editable-install finder to be shadowed: `robocasa.__path__` resolves to a `_NamespacePath(['external/robocasa'])` (the repo root) instead of `external/robocasa/robocasa/` (the package). Symptom: scripts using `robocasa.__path__[0]` fail to find files. **Workaround**: `cd /tmp` (or any path outside the FailBench tree) before running robocasa CLI scripts. The editable finder's `MAPPING` is correct — only cwd-based loading misroutes.

### 3.4 Asset download

```
cd /tmp
echo y | /home/aaron/miniconda3/envs/robocasa/bin/python \
  external/robocasa/robocasa/scripts/download_kitchen_assets.py
```

Downloads 6 packs (textures, generative_textures, fixtures, objaverse objects, AI-generated objects, lightwheel objects) totalling ~21 GB into `external/robocasa/robocasa/models/assets/`. ~30 min over decent bandwidth.

`echo y | …` is required because the script uses `input()` and `conda run` eats stdin. Calling python directly (full path to env's binary) preserves the pipe.

---

## 4. Schema findings (TurnOffStove.hdf5/demo_1037)

```
data/
  attrs:
    env_args   {"env_name": "TurnOffStove", "env_version": "1.5.0",
                "type": 1, "env_kwargs": {"robots": "PandaMobile",
                                          "controller_configs": {"type": "OSC_POSE", ...}}}
    total      544815
  demo_*/                                100 demos per HDF5
    attrs:
      model_file    MJCF XML string, ~380 KB, self-contained after path remap
      ep_meta       scene generation seed/layout/style metadata (ignored — model_file is sufficient)
      num_samples   T
    actions          (T, 12)  float64   OSC_POSE, 6-DoF EEF + base controls + gripper
    actions_abs      (T, 12)  float64   absolute-frame variant
    dones            (T,)     int64
    rewards          (T,)     int64
    states           (T, 112) float64   [time | qpos(56) | qvel(55)]
    obs/
      object                       (T, 14)         object pose state
      robot0_agentview_left_image  (T, 128, 128, 3) uint8
      robot0_agentview_right_image (T, 128, 128, 3) uint8
      robot0_eye_in_hand_image     (T, 128, 128, 3) uint8
      robot0_base_pos              (T, 3)
      robot0_base_quat             (T, 4)
      robot0_eef_pos               (T, 3)
      robot0_eef_quat              (T, 4)
      robot0_joint_pos             (T, 7)
      robot0_joint_vel             (T, 7)
      robot0_gripper_qpos          (T, 2)
      robot0_gripper_qvel          (T, 2)
```

### 4.1 Robot — `PandaMobile`, not stock Panda

Built model reports `nq=56, nv=55`. `states.shape[1] == 1 + nq + nv == 112` — matches the `[time | qpos | qvel]` flattening LIBERO uses, so `LiberoRunner._set_full_state` accepts it unchanged.

The 7-DoF arm is still there, but mounted on an **Omron mobile base** with extra joints:

```
robot joints (13 total):
  base0_joint_mobile_forward
  base0_joint_mobile_side
  base0_joint_mobile_yaw
  base0_joint_torso_height
  robot0_joint1 … robot0_joint7         ← arm; failure-injection candidates
  gripper0_right_finger_joint1
  gripper0_right_finger_joint2
```

Body prefixes in the model: `robot0_`, `base0_`, `gripper0_`. (Plus the kitchen fixtures, which are not robot-prefixed.)

### 4.2 Cameras (model has 6, all `robot0_*`-prefixed)

```
robot0_robotview                view from robot's onboard sensor
robot0_agentview_center         third-person workspace (recommended primary)
robot0_agentview_left           stereo left
robot0_agentview_right          stereo right
robot0_frontview                front-of-kitchen wide angle
robot0_eye_in_hand              gripper-mounted
```

Stereo `_left`/`_right` are recorded in the HDF5 obs streams at 128×128. We render any of these offscreen at arbitrary resolution.

### 4.3 Embedded paths the adapter must rewrite

MJCFs were generated on **multiple** dataset author machines and bake in different absolute paths per machine. Observed so far across the 5 binhng tasks:

```
/home/soroush/code/robosuite-dev/robosuite/...   (TurnOffStove)
/home/soroush/code/robocasa-dev/robocasa/...     (TurnOffStove)
/data1/aaronl/rpl-robocasa/robosuite-dev/...     (TurnOnMicrowave, PnPCabToCounter, CoffeeSetupMug)
/data1/aaronl/rpl-robocasa/robocasa-dev/...      (same)
/home/abhim/robocasa/robosuite/robosuite/...     (PnPCounterToCab)
/home/abhim/robocasa/robocasa/...                (same)
```

A simple `str.replace` per author won't generalise — use a regex on the trailing component instead. The implementation pattern that works:

```python
_ROBOSUITE_PAT = re.compile(r"/[^\"<>\s]+?/robosuite/models/")
_ROBOCASA_PAT  = re.compile(r"/[^\"<>\s]+?/robocasa/models/")

xml = _ROBOSUITE_PAT.sub(f"{ROBOSUITE_ROOT}/models/", xml)
xml = _ROBOCASA_PAT.sub(f"{ROBOCASA_ROOT}/models/",  xml)
xml = xml.replace('meshdir="meshes/"', f'meshdir="{ROBOCASA_ROOT}/models/assets/"')
```

The `meshdir="meshes/"` rewrite is still needed because the MJCF compiler prepends `meshdir/` to any relative `file=...` attribute the regex didn't catch (a few remain in stovetop / fixture references).

---

## 5. Smoke replay — what's been verified

`scripts/robocasa/smoke_replay_demo.py` loads TurnOffStove/demo_1037, patches the MJCF, steps through `states[]` at progress [0, 0.25, 0.5, 0.75, 1.0], renders the `robot0_agentview_center` camera, and extracts contacts.

```
loaded TurnOffStove.hdf5/demo_1037: T=179 state_dim=112 action_dim=12
model: nq=56 nv=55 ngeom=1238 ncam=6
robot/base/gripper geoms: 79 of 1238
  t=  0/178 (p=0.00): non-robot contacts (>1N)=16  max_force=  24.30 N
  t= 44/178 (p=0.25): non-robot contacts (>1N)=16  max_force=  11.90 N
  t= 89/178 (p=0.50): non-robot contacts (>1N)=18  max_force=  54.16 N
  t=134/178 (p=0.75): non-robot contacts (>1N)=12  max_force=  39.93 N
  t=178/178 (p=1.00): non-robot contacts (>1N)=8   max_force=   2.25 N
```

Output PNG at `out/robocasa_smoke_replay.png`. Replay end-to-end works without any failure injection.

### 5.1 Failure-injection videos (2026-06-02)

`scripts/robocasa/record_failure_video.py` is a self-contained record script (no LIBERO adapter dependency). For each task it:
1. Reads the demo HDF5, regex-remaps the MJCF paths (§4.3), builds `MjModel`.
2. Resolves PandaMobile handles by name (`robot0_joint{1..7}`, `robot0_torq_j{1..7}`, `base0_*`, `gripper0_right_finger_joint{1,2}`).
3. Kinematic-replays `states[:fail_idx]`, then injects the failure, then physics-steps `--post_steps` (default 600) with gravity-comp + PD active resistance on every healthy joint (arm, base, torso) so the mobile base doesn't drift.
4. Renders `robot0_agentview_center` and writes an annotated mp4.

PD gains tuned for PandaMobile torque actuators: arm `Kp = [60, 60, 60, 40, 20, 10, 10]` (matches torque ctrlrange ±80/±12), base `Kp = [300, 300, 100, 5000]` (torso needs high stiffness because torso_height has a wide ctrlrange).

Five videos in `out/robocasa_failures/`, summary frame grid at `_summary_grid.png`:

| Task | Failure | Visible result |
|---|---|---|
| TurnOffStove | SINGLE_JOINT j2 (shoulder) | arm collapses over induction cooktop |
| TurnOnMicrowave | SINGLE_JOINT j4 (elbow) | elbow collapse in front of microwave |
| PnPCounterToCab | GRIPPER_OPEN | object dropped on counter mid-transport |
| PnPCabToCounter | SINGLE_JOINT j6 (wrist) | wrist droops over granite counter |
| CoffeeSetupMug | ALL_JOINTS | full arm collapse next to coffee machine |

This validates that the failure-injection mechanism transfers from LIBERO to RoboCasa with no functional changes — only data-path / naming differences (which the videos absorb inline).

### 5.2 The static-contact problem

**RoboCasa kitchens have ~16 static-scene contacts at rest** — pots sitting in cabinets, drawer fronts touching cabinet frames, etc. LIBERO's current `ContactExtractor` filters only "skip if BOTH geoms are robot", so on RoboCasa it would record these baseline contacts as if they were failure-induced. This is the load-bearing design decision before Phase 1 generation.

---

## 6. Open design decisions — resolve before Phase 1 generation

### 6.1 Contact filtering for RoboCasa

LIBERO's current filter: skip only robot-vs-robot contacts. On RoboCasa this captures kitchen-fixture-on-fixture contacts (~16 per state) that have nothing to do with the robot or the failure. Three options:

**Option A — Restrict to robot-involving contacts.** Accept a contact iff at least one of (geom1, geom2) is in `robot_geom_ids`. Misses any "object thrown by failed gripper hits another object" downstream effects. Cleanest semantics.

**Option B — Baseline subtraction.** Save a "baseline contact set" snapshot from `pre_qpos` state (the contacts present before failure), and at `post_qpos` time subtract any contact whose (geom1, geom2, ~pos) was already in baseline. Captures both robot contacts and failure-induced object-object contacts. More expensive to compute per trial; more correct for downstream world-model use (line 2).

**Option C — Hybrid.** Record both `all_contacts` (current logic) and `robot_contacts_only` in the npz. Defer filtering to the training-time loader. Doubles contact-array size; cheapest engineering, most future-flexible.

Recommendation: **B**, because line 2 (world model) is going to want to know about object-object contacts caused by the failure (e.g. dropped mug rolls into another mug), and A throws that information away. The expense of computing a baseline contact set is small — it's done once per trial during `pre_*` snapshotting.

### 6.2 Rollout frame capture for line 2

LIBERO v1 records `pre_rgb` (one frame at pre-failure state) and `post_rgb` (one frame at end-of-settle). The world model needs the **whole post-failure trajectory** of frames, not just endpoints.

Proposal: add `post_rgb_seq` of shape `(N_keyframes, H, W, 3)` to the npz, capturing N evenly-spaced keyframes during `_settle_with_resistance` (e.g. N=8 for a 100-step settle → keyframe every 12 steps). Same for `post_depth_seq` and `post_qpos_seq` / `post_qvel_seq` (the qpos/qvel sequence is already mentioned as needed in `docs/world_model_design.md`).

Storage cost: at 256×256, N=8, this is ~1.5 MB per trial in float32 depth + uint8 rgb. For 18k trials this is ~27 GB extra on top of the per-trial contact data. Acceptable.

This is a strict superset of LIBERO v1's schema, so old code continues to read `post_rgb` (last keyframe) without changes.

### 6.3 Failure-injection joint restriction

PandaMobile has 13 robot joints. Only the 7 arm joints (`robot0_joint1..7`) are valid failure candidates for FailBench's research goal. The base joints are intentional — failing them simulates "robot falls over" or "base stops working" which is out of scope. The naming module must enumerate failure candidates from a 7-joint allowlist, not from all joints with the `robot0_` prefix.

### 6.4 Two robosuite installs side-by-side

`planner/experiments/libero/adapter.py::_find_robosuite_root` currently globs `external/LIBERO/.venv/lib/python*/site-packages/robosuite` and returns the first match. For RoboCasa we need it to find `external/robosuite_for_robocasa/robosuite` instead. The cleanest fix is a `dataset` parameter on the adapter (`"libero"` → LIBERO sidecar; `"robocasa"` → ShahRutav fork checkout) rather than auto-detection from MJCF content.

---

## 7. Phase 1 — adapter + scripts extension (DONE 2026-06-03)

Schema is locked at **LIBERO v2** (see `docs/libero_v2_dataset.md`); RoboCasa
writes the same 54 trial-level fields + 18 attrs to per-task HDF5s. The
"v3" framing from earlier sections is withdrawn — v2's
`settle_qpos`/`settle_qvel` + `pre_target_qpos` cover the world-model line
already; settle RGB is rerendered on demand.

What landed:

| File | Change |
|---|---|
| `planner/experiments/robocasa/adapter.py` (new) | Reads RoboCasa HDF5 → produces a `LiberoDemo`. Regex-remaps three known author paths (soroush/aaronl/abhim) plus the relative `meshdir`. Injects `inertia="shell"` on `<mesh>` assets and strips `shellinertia="true"` from geoms so MuJoCo 3.3.4 accepts the visual-only fixtures (utensil holders, microwaves) that 3.3 rejects by default. |
| `planner/experiments/robocasa/scene.py` (new) | `build_scene_overrides(model, data, ep_meta)` → returns a `SceneOverrides` with the manipulated-object allowlist (from `ep_meta["object_cfgs"]`), a counter-aware `scene_table_z`, and a tight per-entity AABB. Reduces `obj_names` from 117 (walls, floor, fixtures) to 1 (the cookware). |
| `planner/experiments/libero/runner.py` | Constructor accepts `mjcf_path=` (bypasses LIBERO's path-rewriter that would corrupt RoboCasa absolute paths) and `scene_overrides=`. `run_v2()` reads from these overrides when present; LIBERO defaults are unchanged. Also: graceful fallback when `obs/ee_states` is absent — synthesizes from `obs/robot0_eef_pos`. |
| `planner/experiments/libero/naming.py` | Candidate lists extended with RoboCasa names (`gripper0_right_*`, `robot0_agentview_center/left/right`, `gripper0_right_grip_site`) and `base0_` body prefix. |
| `scripts/robocasa/_smoke_v2_write.py` (new) | 20-trial smoke against `TurnOffStove`. Confirms 54/54 field parity with LIBERO v2. |
| `scripts/robocasa/build_v2_dataset.py` (new) | Driver: enumerate raw HDF5s, expand each demo into 15 stratified failure trials (5 progress × 3 modes), spawn 4 workers, write per-task HDF5 + per-split `manifest.csv`. Mirrors `scripts/libero/build_v2_dataset.py` lessons (EGL backend at module scope, `HDF5_USE_FILE_LOCKING=FALSE`, fstype pre-flight). |
| `planner/risk/dataset_v2.py` | New `V2Source` dataclass + `PooledV2Dataset` for mixed LIBERO+RoboCasa training. Adds `source` field to sample dicts. RoboCasa's flat layout (`<root>/<task>.h5`) is supported via `h5_layout="flat"` (default `"split"` keeps LIBERO unchanged). |
| `planner/risk/benchmark_dataset.py` | `BenchmarkDataset(sources=[...])` constructor for pooled training. `v2_root=` path stays as the single-corpus default. |
| `scripts/benchmark/train_one.py` | New `--robocasa_v2_root` CLI arg. When set, training pools LIBERO + RoboCasa transparently; per-source eval split available via the new `"source"` sample field. |

Notes on what was DROPPED from the original §6 plan:

- §6.1 baseline contact subtraction: **skipped**. Diagnostic showed v2's existing `≥1N + robot-vs-robot exclusion` already gates out static kitchen contacts (all 129 settle contacts on TurnOffStove smoke were robot-involved). No code change to `ContactExtractor`.
- §6.2 `post_rgb_seq` capture: **withdrawn**. v2 stores `settle_qpos / settle_obj_pos / settle_obj_quat` per Caveat 1 → rerender on demand.

## 8. Tier 1 generation status (running 2026-06-03)

Dry-run (2 demos × 5 tasks × 15 trials = 150 trials, 4 workers) completed in 3.6 min with 0 errors. Per-task throughput 0.20–0.48 trials/s; aggregate ~0.69 trials/s on 4 workers, ~3.6 MB/trial output.

Tier 1 full build (`scripts/robocasa/build_v2_dataset.py`, 5 tasks × 100 demos × 15 trials = 7,500 trials) launched 2026-06-03 17:26 to `/media/aaron/F/failbench/robocasa/v2/`. ETA ~2 h based on observed throughput.

Output layout:

```
/media/aaron/F/failbench/robocasa/v2/
  manifest.csv                                                # one row per trial
  CoffeeSetupMug.h5
  PnPCabToCounter.h5
  PnPCounterToCab.h5
  TurnOffStove.h5
  TurnOnMicrowave.h5
```

Expected final size: ~27 GB.

## 9. Phase 2 — training (when Tier 1 completes)

Pooled training command (matches the LIBERO v2 benchmark conventions, with the new `--robocasa_v2_root` flag):

```bash
PYTHONPATH=. /path/to/conda/envs/failbench_env/bin/python -u \
  -m scripts.benchmark.train_one \
  --v2_root /media/aaron/F/failbench/libero/v2 \
  --robocasa_v2_root /media/aaron/F/failbench/robocasa/v2 \
  --model mlp --modalities state \
  --epochs 10 --batch_size 64
```

Sample dicts carry a `"source"` field — split val by source at eval time to measure cross-dataset generalisation (the second-order question we're answering with RoboCasa in the first place).

---

## 8. Files and dirs at end of Phase 0

```
external/robocasa/                                       mimicdroid-robocasa @ latest (~21 GB after assets)
external/robosuite_for_robocasa/                         ShahRutav/robosuite @ abs_robot
datasets/robocasa/raw/TurnOffStove.hdf5                  735 MB, one demo HDF5 from binhng
scripts/robocasa/smoke_replay_demo.py                    smoke test (this doc § 5)
out/robocasa_smoke_replay.png                            smoke test output
docs/robocasa_integration.md                             this doc
~/.claude/projects/-home-aaron-workspace-FailBench/memory/project_robocasa_integration.md   memory pointer
```

No edits to `planner/experiments/libero/*` yet. Phase 1 is the first commit to that tree.

---

## 9. Resume checklist

When resuming:

1. Re-read §6 (the four open design decisions). Decide on (A/B/C) for §6.1 and confirm §6.2 / §6.3 / §6.4. None of these are reversible after generating v1 — pick now.
2. Implement Phase 1 edits per §7 in the order listed.
3. Smoke test on `TurnOffStove.hdf5/demo_1037` with `--fail_progress 0.5 --joints joint4`.
4. Download the other 4 binhng tasks if §3 smoke passes:
   ```
   curl -fL -O https://huggingface.co/datasets/binhng/robocasa-100demos-5chosen-tasks/resolve/main/{CoffeeSetupMug,PnPCabToCounter,PnPCounterToCab,TurnOnMicrowave}.hdf5
   ```
   ~4.6 GB total. Each into `datasets/robocasa/raw/`.
5. Run dataset regen: 5 tasks × 100 demos × stratified failures ≈ ~7.5k trials → `datasets/robocasa/v1/`.
6. Decision point on Tier 2 (download remaining ~19 upstream RoboCasa tasks) depends on whether RoboCasa v1's val MSE / per-entity Spearman justifies more data, or whether we're architecture-bound.
