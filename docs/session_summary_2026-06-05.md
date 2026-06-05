# Session summary — RoboCasa Phase 1 + Tier 1 build (2026-06-02 → 2026-06-05)

End-to-end work to bring RoboCasa Panda demos into the FailBench failure-injection pipeline at the LIBERO v2 schema. Outcome: pooled LIBERO + RoboCasa v2 corpus on disk, ready for contact-prediction training.

## Pipeline state at end of session

```
datasets/robocasa/raw/      5 task HDF5s from binhng/* HF mirror (5.0 GB total)
external/robocasa/          mimicdroid-robocasa @ latest + 21 GB asset tree
external/robosuite_for_robocasa/   ShahRutav/robosuite @ abs_robot
/media/aaron/F/failbench/robocasa/v2/
  manifest.csv              7,500 trials
  {5 tasks}.h5              26 GB total, baseline_contact_* arrays, scene metadata fixed
```

Both research lines (contact prediction + world model) can train on this corpus pooled with the existing `libero/v2` 177 GB.

## What landed this session

### Adapter + runner integration (Phase 1)

- **`planner/experiments/robocasa/adapter.py`** (new) — reads RoboCasa HDF5 into a `LiberoDemo` so `LiberoRunner.run_v2()` consumes both sources. Regex-remaps three known author absolute paths (soroush / aaronl / abhim). Patches MJCFs so MuJoCo 3.3.4 accepts visual-only meshes: strips `shellinertia="true"` from `<geom>` + injects `inertia="shell"` on `<mesh>` assets (verified necessary for ~30 % of demos that include microwave / utensil-holder fixtures).
- **`planner/experiments/robocasa/scene.py`** (new) — `build_scene_overrides(model, data, ep_meta)` returns a `SceneOverrides` containing the manipulated-object allowlist (from `ep_meta["object_cfgs"]`), a counter-aware `scene_table_z`, and tight per-entity AABB. `_body_aabb` walks the body **subtree** (RoboCasa composes objects as `*_main → *_main_group → *_g{N}` hierarchies — the named parent has zero direct geoms; an earlier shallow walk fell back to LIBERO's 0.91 m default on 20–67 % of trials per task).
- **`planner/experiments/libero/runner.py`** — added `mjcf_path=` constructor param (bypasses LIBERO's path-rewriter that would corrupt RoboCasa absolute paths), `scene_overrides=` constructor param, and a fallback when `obs/ee_states` is absent (synthesizes from `obs/robot0_eef_pos`).
- **`planner/experiments/libero/naming.py`** — candidate lists extended with RoboCasa names (`gripper0_right_*`, `robot0_agentview_center/left/right`, `gripper0_right_grip_site`) and `base0_` body prefix. Backward-compatible — LIBERO still uses the first match.

### Data generation

- **`scripts/robocasa/build_v2_dataset.py`** (new) — multiprocess driver. Per-demo stratified failure sampling (5 progress × 3 modes = 15 trials/demo). Bakes in the LIBERO v2 build-notes gotchas: `MUJOCO_GL=egl` + `HDF5_USE_FILE_LOCKING=FALSE` at module scope, fstype pre-flight, **3 workers** as the safe default for a 32 GiB host (4 workers caused swap thrash in our first attempt).
- **`scripts/robocasa/add_baseline_contacts.py`** (new) — post-process pass that, for each trial, seeds the sim to pre-failure state, steps 50 physics frames with PD holding all arm joints (no failure), and writes `baseline_contact_*` arrays alongside the existing `contact_*`. Loaders subtract by `(geom_pair, ‖pos − baseline_pos‖ < 3 cm)`. ~170 min wall on Tier 1.
- **`scripts/robocasa/fix_scene_metadata.py`** (new) — in-place rewriter for `scene_table_z` / `scene_aabb_*` after the subtree-AABB fix in `scene.py`. Skips already-correct trials; deliberately does NOT overwrite the variable-size `scene_entities_json` string attr (its growth fragments HDF5 storage and stalled the first attempt on PnPCounterToCab for 39 min).
- **`scripts/robocasa/_smoke_v2_write.py`** (new) — 20-trial smoke used during Phase 1 to confirm 54/54 field parity with LIBERO v2.

### Trainer surface

- **`planner/risk/dataset_v2.py`** — added `V2Source` dataclass + `PooledV2Dataset` for mixed corpora. RoboCasa's flat layout (`<root>/<task>.h5`) is supported via `h5_layout="flat"`; LIBERO's `<root>/<split>/<task>.h5` stays the default. Sample dicts gain a `"source"` field for per-source val splits at training time.
- **`planner/risk/benchmark_dataset.py`** — `BenchmarkDataset(sources=[V2Source.libero(...), V2Source.robocasa(...)])` constructor. Single-corpus path (`v2_root=`) unchanged.
- **`scripts/benchmark/train_one.py`** — new `--robocasa_v2_root` arg. When set, pools the two corpora through the new constructor.

### Verification + visualization notebooks

- **`notebooks/verify_robocasa_v2.ipynb`** — gate-style numerical checks: schema parity vs LIBERO v2, baseline-subtraction effectiveness, state / object / camera sanity, failure-mode coverage matrix, RGB panels per task. Regenerator at `notebooks/_build_verify_robocasa_v2.py`.
- **`notebooks/visualize_robocasa_v2.ipynb`** — visual inspection: failure contacts projected onto agentview RGB, per-task overview, same-demo-different-failure comparison, progress sweep, top-down spatial distribution, outlier inspection (catastrophic vs zero-contact failures), per-task heatmap label preview, count distribution histogram. Regenerator at `notebooks/_build_visualize_robocasa_v2.py`.

## Key numbers from the verified data

- **7,500 trials** generated, 0 errors, 26 GB on disk (~3.5 MB/trial).
- **Baseline subtraction**: mean **n_raw 8,719 → mean n_failure 1,029 (16.1 %)**. Robot-involved fraction 5 % → 24 % post-subtraction; the remaining 76 % of failure-induced contacts are object-on-object cascades (still failure-conditioned signal, just not directly involving the robot).
- **Scene metadata fix**: **0/7,500 trials** at the `scene_table_z = 0.91` fallback (was 20–67 % per task). Workspace AABBs now ≤ 1 m³ around manipulated objects vs the ~2 m³ LIBERO default.
- **Per-task failure-contact density** (median over 300-trial sample): PnPCabToCounter 1,284 / PnPCounterToCab 1,233 / CoffeeSetupMug 999 / TurnOffStove 447 / **TurnOnMicrowave 15 (sparse outlier)**. Worth weighting at training time.

## Gotchas resolved (worth keeping in memory)

1. **MuJoCo 3.3+ strictness on mesh inertia.** RoboCasa MJCFs use `shellinertia="true"` on visual-only geoms with sub-mm meshes. MuJoCo 3.3.4 rejects these unless the mesh asset declares `inertia="shell"` AND the geom-level `shellinertia` is stripped. The adapter does both. mujoco 3.2.6 (in the robocasa conda env) doesn't accept `inertia="shell"` at all — they're version-mutually-exclusive.
2. **`h5clear -s` is destructive ONLY mid-write.** Killing a writer mid-flush leaves chunk indices incomplete; `h5clear -s` then truncates EOA → data unrecoverable. Killing a writer that has been idle for >5 min is safe — chunks are flushed; `h5clear -s` only clears the lock flag.
3. **Variable-size HDF5 string attrs fragment storage.** Overwriting `scene_entities_json` (a JSON string that grew from `"[]"` to ~250 B) caused 39-minute stalls. Don't rewrite varlen attrs in-place; rebuild from primitive attrs at load time.
4. **3 workers, not 4, for RoboCasa generation on a 32 GiB host.** RoboCasa scenes are ~5–8 GB RSS per worker (richer kitchens than LIBERO). 4 workers triggered `BrokenProcessPool` from swap thrash in our first attempt.
5. **`base0_` body prefix.** PandaMobile's Omron base contributes 4 extra joints. They must be in the robot-geom prefix set (for contact filtering) but excluded from failure-injection candidates.

## Open

- **Task #14 — train pooled contact prediction.** Data is ready; user has paused start until after this commit.
- **`scene_entities_json` is stale** on the ~20 % of trials whose `obj_main` body needed the subtree walk. Acceptable for contact-prediction training (which doesn't use the field) but should be recomputed at load time if per-entity risk labels are needed.
- **Tier 2 expansion (~19 more upstream RoboCasa tasks).** Defer until Tier 1 evals say generalization is the bottleneck.

## Files to commit

```
A docs/robocasa_integration.md
A docs/session_summary_2026-06-05.md
M planner/experiments/libero/naming.py
M planner/experiments/libero/runner.py
A planner/experiments/robocasa/__init__.py
A planner/experiments/robocasa/adapter.py
A planner/experiments/robocasa/scene.py
M planner/risk/benchmark_dataset.py
M planner/risk/dataset_v2.py
M scripts/benchmark/train_one.py
A scripts/robocasa/__init__.py
A scripts/robocasa/_smoke_v2_write.py
A scripts/robocasa/add_baseline_contacts.py
A scripts/robocasa/build_v2_dataset.py
A scripts/robocasa/fix_scene_metadata.py
A scripts/robocasa/record_failure_video.py
A scripts/robocasa/smoke_replay_demo.py
A notebooks/_build_verify_robocasa_v2.py
A notebooks/_build_visualize_robocasa_v2.py
A notebooks/verify_robocasa_v2.ipynb
A notebooks/visualize_robocasa_v2.ipynb
```

Plus a `.gitignore` update for `external/robocasa/` and `external/robosuite_for_robocasa/` (clone trees, not part of the repo).

The unrelated modification to `scripts/cluster/README.md` is from prior work and not part of this commit.
