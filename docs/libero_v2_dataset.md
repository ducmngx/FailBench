# LIBERO v2 dataset

Standalone HDF5 contact-prediction dataset that augments v1 with: a T-frame
pre-failure window of observations + state, K-step goal/intent feature,
time-resolved contacts, world-frame contact forces, dense settle state
trajectory, object poses pre/post, and camera + scene calibration so any
downstream label form is constructible without re-simulating.

v1 stays on disk untouched. v2 is self-contained — a single training loader
reads it without joining against v1.

See `docs/libero_v1_dataset.md` for v1; this document follows the same
structure for easy comparison.

## Storage

```
<output_root>/
  <split>/
    <task>.h5            # one HDF5 file per task
    manifest.csv         # one row per trial
```

Default `<output_root>`: `/media/aaron/F/failbench/libero/v2` (empty 932 GB
external volume; isolates v2 from system disk pressure).

30 HDF5 files total (10 tasks × 3 splits). Per-task file size depends on the
contact-cloud density and ranges roughly 1–10 GB. Total ≈ 200 GB expected
for the full 45 k trials.

Compression: blosc:lz4 via `hdf5plugin` when installed (recommended), with a
fallback to h5py's built-in `lzf` filter. `lzf` is ~30% worse but always
present.

## Per-trial schema (HDF5 group at `/trials/<trial_id>`)

`trial_id` is `<demo_key>_s<seed_idx>_b<bin_idx>` (e.g. `demo_3_s1_b7`) so
ordering is deterministic and the manifest row → group navigation is trivial.

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `window_frame_idx` | (T,) | i32 | Demo step indices for the window |
| `window_qpos` | (T, 7) | f32 | Arm joint positions at each window frame |
| `window_qvel` | (T, 7) | f32 | Real (finite-diff) joint velocities — non-zero, unlike v1 |
| `window_ee_pos` | (T, 3) | f32 | End-effector world position |
| `window_gripper_ctrl` | (T, 1) | f32 | Mean finger qpos summary |
| `window_agentview_rgb` | (T, 240, 320, 3) | u8 | Agentview RGB at each window frame |
| `window_agentview_depth` | (T, 240, 320) | f16 | Agentview depth |
| `window_wrist_rgb` | (T, 240, 320, 3) | u8 | Wrist-cam RGB |
| `window_wrist_depth` | (T, 240, 320) | f16 | Wrist-cam depth |
| `goal_qpos` | (K, 7) | f32 | Demo's commanded qpos at fail_idx + offsets |
| `goal_qvel` | (K, 7) | f32 | Demo's qvel at same offsets |
| `goal_ee_pos` | (K, 3) | f32 | Demo's EE pos at same offsets |
| `goal_gripper_ctrl` | (K, 1) | f32 | Demo's gripper ctrl at same offsets |
| `goal_offsets` | (K,) | i32 | The future-step offsets, default `[5, 15, 30]` |
| `pre_qpos` | (7,) | f64 | Single-frame state at fail_idx (= `window_qpos[-1]`) |
| `pre_qvel` | (7,) | f64 |  |
| `pre_ee_pos` | (3,) | f64 |  |
| `pre_gripper_ctrl` | (1,) | f64 |  |
| `pre_target_qpos` | (7,) | f64 | Demo's commanded qpos — used as resistance PD hold target during settle |
| `pre_rgb` | (240, 320, 3) | u8 | Agentview RGB at fail_idx |
| `pre_depth` | (240, 320) | f16 | Agentview depth at fail_idx |
| `robot0_eye_in_hand_rgb` | (240, 320, 3) | u8 | Wrist-cam RGB at fail_idx |
| `robot0_eye_in_hand_depth` | (240, 320) | f16 |  |
| `contact_positions` | (N, 3) | f32 | World-frame contact points, accumulated over 500 settle steps |
| `contact_forces` | (N, 6) | f32 | Wrench in MuJoCo contact frame (3 linear + 3 torque) |
| `contact_force_world` | (N, 3) | f32 | Linear force rotated to world frame |
| `contact_time` | (N,) | i32 | Settle-step index of each contact, ∈ [0, 500) |
| `contact_geom_pairs` | (N, 2) | i32 | MuJoCo geom IDs |
| `contact_failure_id` | (N,) | i32 | Always 0 (one failure / trial); kept for v1 compat |
| `impacted_geom_ids` | (M,) | i32 | Sorted union of touched geoms |
| `post_agentview_rgb` | (240, 320, 3) | u8 | Agentview after settle |
| `post_agentview_depth` | (240, 320) | f16 |  |
| `post_wrist_rgb` | (240, 320, 3) | u8 | Wrist after settle |
| `post_wrist_depth` | (240, 320) | f16 |  |
| `cam_agentview_pos` | (3,) | f64 | Camera world position (agentview is static during settle) |
| `cam_agentview_mat0` | (3, 3) | f64 | Camera rotation (MuJoCo raw — apply Y-flip for image convention) |
| `cam_agentview_fovy` | () | f64 | Vertical FOV in degrees |
| `cam_agentview_size` | (2,) | i32 | (W, H) of stored RGB |
| `cam_wrist_pos_window` | (T, 3) | f64 | Wrist-cam pos at each window frame (it moves with the arm) |
| `cam_wrist_mat0_window` | (T, 3, 3) | f64 |  |
| `cam_wrist_fovy` | () | f64 |  |
| `cam_wrist_size` | (2,) | i32 |  |
| `failure_joints` | (J,) | i32 | 1-based joint indices that failed (length 0 for non-joint modes) |
| `obj_names` | (n_obj,) | str | Ordered list of non-robot, non-table scene bodies |
| `obj_pos_pre` | (n_obj, 3) | f32 | Object world positions at fail_idx |
| `obj_quat_pre` | (n_obj, 4) | f32 | (w, x, y, z) |
| `obj_pos_post` | (n_obj, 3) | f32 | After settle |
| `obj_quat_post` | (n_obj, 4) | f32 |  |
| `settle_step_idx` | (S,) | i32 | Settle-step indices for snapshots, e.g. `[10, 20, ..., 500]` |
| `settle_qpos` | (S, 7) | f32 | Robot arm trajectory during settle |
| `settle_qvel` | (S, 7) | f32 |  |
| `settle_gripper_qpos` | (S, 2) | f32 |  |
| `settle_obj_pos` | (S, n_obj, 3) | f32 | Object positions during settle |
| `settle_obj_quat` | (S, n_obj, 4) | f32 |  |

### Per-trial attrs

`trial_id, split, task, demo_key, seed, seed_idx, bin_idx, fail_idx,
traj_progress, failure_mode, failure_prob, is_holding, force_frame,
scene_table_z, scene_aabb_min, scene_aabb_max, scene_entities_json,
robot_geom_ids`.

`force_frame` is always the string `"contact"` for `contact_forces`;
`contact_force_world` is in world frame.

### File-level attrs

`schema_version, split, task, window_T, window_stride, goal_offsets,
settle_S`. Schema version is `2` for this format; readers should gate
behavior on this if v3 introduces breaking changes.

## Downstream label forms

Because v2 stores world-frame contacts + calibration + scene metadata, any of
these label forms is constructible from a single trial without re-simulating:

| Label form | Recipe |
|---|---|
| v1-style agentview heatmap | Project `contact_positions` through `cam_agentview_*` (Y-flip the camera rotation for image convention; see `planner/risk/projection_labels.py`) |
| World top-down 2D heatmap | Bin XY contacts above `scene_table_z`; see `planner/risk/spatial.py` |
| Wrist-cam image-plane heatmap | Project through `cam_wrist_*_window[-1]` |
| 3D voxel grid | Bin contacts in the box `[scene_aabb_min, scene_aabb_max]` |
| Per-entity risk vector | Mask contacts by entries in `scene_entities_json` |
| Time-resolved heatmaps | Split by `contact_time` bins (e.g. first 100 ms vs last 100 ms) |
| First-touch heatmap | Restrict to the smallest `contact_time` per `contact_geom_pairs` |

## Generation

```bash
MUJOCO_GL=egl python -m scripts.libero.build_v2_dataset \
    --splits libero_spatial libero_object libero_goal \
    --v1_root datasets/libero/v1 \
    --output_dir /media/aaron/F/failbench/libero/v2 \
    --workers 8 --resume
```

The script reads each split's v1 `manifest.csv`, replays the exact same
`(demo, seed, bin_idx, failure_mode, failure_joints)` configs through
`LiberoRunner.run_v2`, and writes one HDF5 per task. v1 is read-only here;
nothing under `datasets/libero/v1` is modified.

Wall-time: ~10–12 h on 8 EGL workers (physics-bound at 500-step settle, same
order as the v1 regen). One-time, kick-off-overnight.

### Resume

`--resume` skips any `(task, trial_id)` group that already exists in its
per-task HDF5. Safe to interrupt and re-run.

### Dry-run

For a 1500-trial single-task smoke (~5 GB, ~10 min on 8 workers):

```bash
MUJOCO_GL=egl python -m scripts.libero.build_v2_dataset \
    --splits libero_spatial --output_dir /tmp/v2_dry \
    --workers 8 --limit_tasks 1 --resume
python -m scripts.libero.test_v2_pipeline --output_dir /tmp/v2_smoke --limit 4
```

The smoke test asserts:

* `window_qpos[-1]` ≈ `pre_qpos` (last window frame = fail moment, by construction).
* `window_frame_idx[-1]` == trial-group `fail_idx` attr.
* `window_qvel[-1]` non-zero (vs v1's `pre_qvel ≈ 0`).
* `contact_time` monotonic-nondecreasing and within `[0, settle_steps)`.
* `||contact_force_world||` equals `||contact_forces[:, :3]||` (rotation preserves norm).
* Depth values within plausible range for all rendered frames.

## Caveats

1. **Settle RGB is not stored.** Contact prediction doesn't need it; for a
   future pixel-space world model, re-render from `settle_qpos` +
   `settle_obj_pos` + `settle_obj_quat` via the cached MJCF (no physics
   replay needed) at ~10 Hz playback (S=50 over a 1-second settle).
2. **`contact_failure_id` is degenerate.** One failure per trial means it's
   always 0. Kept for v1-schema compatibility only.
3. **`obj_names` order is body-id stable per demo MJCF** but **not** shared
   across tasks. Always read it per trial.
4. **`cam_*_mat0` is raw MuJoCo** — apply the Y-flip negate-row-1
   convention from `ContactProjector._validate_projection()` to map into
   image-pixel convention.
5. **`contact_forces[:, :3]` is in the MuJoCo contact frame**, not world.
   `contact_force_world` is the world-frame counterpart (`force_frame`
   attribute documents this).
6. **`scene_table_z` for LIBERO** is detected from the largest horizontal
   box geom in any body whose name mentions "table". Defaults to 0.91 m
   (LIBERO standard) if detection fails.

## Build operational notes

Gotchas hit during the first full v2 build (2026-05-15, ~5h30m wall time,
45 000 trials, 0 errors after these were fixed):

1. **FAT32 / exfat single-file 4 GiB cap silently corrupts HDF5.**
   The first build ran on a FAT32-formatted external drive and stalled
   after several hours when worker .h5 files hit exactly 4 294 967 295
   bytes — the 2³² − 1 byte ceiling FAT32 enforces. HDF5 doesn't surface
   this as a clean error; workers just deadlock on the next write. The
   v2 driver now runs `findmnt` at startup and aborts with a clear
   message if the output volume is `vfat`/`msdos`/`exfat`. Format the
   output volume as **ext4 / xfs / btrfs / zfs** before running.

2. **`kill -KILL <main_pid>` does NOT reap multiprocessing spawn
   workers.** Python's `concurrent.futures.ProcessPoolExecutor` with the
   spawn context creates child processes that are NOT in the same
   process group as the driver. Killing only the driver leaves the
   workers orphaned, where they continue holding ~2 GB RSS each. Across
   a few aborted runs they accumulated to >20 GB and triggered an OOM
   kill on the next attempt. Use `pkill -9 -f multiprocessing.spawn`
   (or kill the process group: `kill -KILL -<pgid>`) to clean up
   properly. Verify with `ps -eo pid,rss,command --sort=-rss | head`
   before relaunching.

3. **`conda run` buffers stdout** until the wrapped command exits.
   The very first launch used `nohup conda run -n env python ...` and
   produced an *empty* log file for hours, hiding the trial-failure
   tracebacks that were actually occurring. **Always invoke the env's
   Python directly** for long-running builds:
   ```bash
   /path/to/conda/envs/failbench_env/bin/python -u -m scripts.libero.build_v2_dataset ...
   ```
   The `-u` flag combined with direct invocation gives line-buffered
   real-time logging.

4. **ext4 enforces HDF5 advisory locks** where FAT32 silently ignored
   them. After an unclean process exit, partial HDF5 files retain the
   lock state from the dead writer; fresh workers then fail to open
   them with
   `BlockingIOError: Resource temporarily unavailable` or
   `OSError: file is already open for write/SWMR write`. The driver now
   sets `HDF5_USE_FILE_LOCKING=FALSE` at module scope (safe because v2
   is one-writer-per-file by design) so spawn workers inherit it. If
   you ever recover a partial file manually, run `h5clear -s <file>`
   to clear stale superblock flags.

5. **v1 manifest uses comma-separated, CSV-quoted joint lists** for
   `MULTI_JOINT` rows (e.g. `"joint4,joint6"`) but the field looks
   semicolon-separated at first glance. `_parse_joints` in the v2
   driver accepts both `,` and `;` separators. If you ever extend the
   schema, keep this parser permissive.

6. **8 workers exceeded the 31 GiB RAM budget** on the build host;
   each worker peaked at ~4 GB RSS (renderer state + HDF5 chunk
   cache + MuJoCo data) — double the original estimate. **4 workers is
   the safe default for a 32 GB machine**; bump to 6 only if you've
   verified peak RSS empirically. The `--workers` flag is the knob.

7. **Worker re-spawning across split boundaries is normal.** Between
   splits, `ProcessPoolExecutor` may recycle worker processes, so
   `STIME` in `ps` will reset and the `etime` column briefly misleads.
   Use the manifest row count or the per-task `[done=...]` log lines
   for true progress, not worker process age.

8. **libero_goal tasks are ~30 % slower than libero_spatial / object**
   (≈40 min/task vs ≈30 min/task at 4 workers). The difference is more
   contacts per trial during the settle. Not a bug; just budget for it
   in ETA estimates.

The driver code (`scripts/libero/build_v2_dataset.py`) now bakes in
guards for (1) and (4); (2), (3), (6), (7), (8) are operational
practices to keep in mind for future runs.
