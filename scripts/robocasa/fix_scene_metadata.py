#!/usr/bin/env python3
"""Rewrite scene_table_z / scene_aabb_* / scene_entities_json on RoboCasa v2 trials.

Trial attrs written by the original Tier 1 build are wrong on ~20% of trials
because the old ``_body_aabb`` only inspected direct children of the
manipulated-object body. RoboCasa composes objects as ``*_main → *_main_group
→ *_g{N}`` hierarchies, so the named parent has zero direct geoms and its AABB
came out as ``(inf, -inf)`` → ``scene_table_z`` fell back to the 0.91 LIBERO
default.

The fix in ``planner/experiments/robocasa/scene.py`` walks the body subtree.
This script re-runs ``build_scene_overrides`` against the cached MJCF + ep_meta
for every trial and overwrites the four scene attrs in place. No physics,
no contact extraction — just metadata.

Run::

    /home/aaron/miniconda3/envs/failbench_env/bin/python -u \\
        -m scripts.robocasa.fix_scene_metadata \\
        --v2_root /media/aaron/F/failbench/robocasa/v2

Idempotent — same input MJCF + ep_meta → same output.
"""
from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planner.experiments.robocasa.adapter import (
    load_demo, materialise_mjcf, read_ep_meta,
)
from planner.experiments.robocasa.scene import build_scene_overrides


def process_task(h5_path: Path, raw_root: Path, max_trials: int | None = None) -> dict:
    src = raw_root / f"{h5_path.stem}.hdf5"
    if not src.exists():
        return {"task": h5_path.stem, "n_done": 0, "n_err": 0,
                "error": f"raw missing: {src}"}

    t0 = time.time()
    n_done = n_err = 0
    demo_cache: dict = {}

    with h5py.File(h5_path, "r+") as h:
        tids = list(h["trials"].keys())
        if max_trials is not None:
            tids = tids[:max_trials]
        n_skip = 0
        for ti, tid in enumerate(tids):
            g = h[f"trials/{tid}"]
            # Skip trials whose scene_table_z is already non-fallback.
            try:
                tz = float(g.attrs.get("scene_table_z", 0.91))
                if tz != 0.91:
                    n_skip += 1
                    continue
            except Exception:
                pass
            try:
                demo_key = g.attrs.get("demo_key", tid.rsplit("_s", 1)[0])
                if isinstance(demo_key, bytes):
                    demo_key = demo_key.decode()
                fail_idx = int(g.attrs["fail_idx"])

                key = (demo_key,)
                if key not in demo_cache:
                    demo = load_demo(str(src), demo_key)
                    xml_path = materialise_mjcf(demo.model_xml)
                    model = mujoco.MjModel.from_xml_path(xml_path)
                    data = mujoco.MjData(model)
                    ep_meta = read_ep_meta(str(src), demo_key)
                    demo_cache[key] = (demo, model, data, ep_meta)
                    if len(demo_cache) > 4:
                        demo_cache.pop(next(iter(demo_cache)))
                demo, model, data, ep_meta = demo_cache[key]

                # Re-seed to the trial's failure boundary so AABBs reflect the
                # actual pre-failure config (object may be held mid-air etc.).
                flat = demo.full_states[fail_idx]
                nq, nv = model.nq, model.nv
                off = 1 if flat.shape[0] == 1 + nq + nv else 0
                data.qpos[:] = flat[off:off + nq]
                data.qvel[:] = flat[off + nq:off + nq + nv]
                mujoco.mj_forward(model, data)

                ov = build_scene_overrides(model, data, ep_meta)
                meta = ov.scene_metadata

                # In-place attr overwrite. h5py raises if the new value's dtype
                # /shape differs — for float64 vectors / scalars and the JSON
                # string this matches exactly, so simple assignment works.
                g.attrs["scene_table_z"] = float(meta["scene_table_z"])
                g.attrs["scene_aabb_min"] = np.asarray(
                    meta["scene_aabb_min"], dtype=np.float64)
                g.attrs["scene_aabb_max"] = np.asarray(
                    meta["scene_aabb_max"], dtype=np.float64)
                # NOTE: scene_entities_json deliberately not overwritten —
                # variable-size str attrs trigger expensive HDF5 reallocation
                # that stalled the prior run on PnPCounterToCab. Loaders that
                # need the corrected entity list can recompute on demand from
                # the AABB + obj_pos_pre.
                n_done += 1
            except Exception as e:
                print(f"  [{h5_path.stem}/{tid}] failed: {e}", flush=True)
                n_err += 1
            if (ti + 1) % 500 == 0:
                dt = time.time() - t0
                print(f"  {h5_path.stem}: {ti + 1}/{len(tids)} "
                      f"({n_done} ok / {n_skip} skip / {n_err} err)  "
                      f"{(ti + 1) / max(dt, 1e-6):.1f}/s", flush=True)
        h.flush()

    return {"task": h5_path.stem, "n_done": n_done, "n_err": n_err,
            "elapsed_s": time.time() - t0}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v2_root", required=True, type=Path)
    p.add_argument("--raw_root", default=str(REPO_ROOT / "datasets" / "robocasa" / "raw"),
                   type=Path)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--max_trials", type=int, default=None)
    args = p.parse_args()

    h5_files = sorted(args.v2_root.glob("*.h5"))
    if args.tasks:
        h5_files = [p for p in h5_files if p.stem in args.tasks]
    if not h5_files:
        print(f"No HDF5s in {args.v2_root}", file=sys.stderr)
        return 2

    t0 = time.time()
    for h5_path in h5_files:
        print(f"=== {h5_path.name} ===", flush=True)
        res = process_task(h5_path, args.raw_root, max_trials=args.max_trials)
        if "error" in res:
            print(f"  ERROR: {res['error']}", flush=True)
            continue
        print(f"  done: {res['n_done']} ok / {res['n_err']} err  in "
              f"{res['elapsed_s']:.1f}s "
              f"({res['n_done'] / max(res['elapsed_s'], 1e-6):.1f}/s)",
              flush=True)

    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
