#!/usr/bin/env python3
"""Precompute per-trial agentview contact projections for fast training.

Walks both v2 corpora and projects each trial's contacts to in-frame pixel
coordinates, writing per-task HDF5s under ``target_cache/``. Cache schema:

    target_cache/
      libero_spatial/<task>.h5
        /<trial_id>/projection         (N_in_frame, 3) float32 [u, v, force_mag]
        /<trial_id>/failure_prob       () float32
      libero_object/<task>.h5
      libero_goal/<task>.h5
      robocasa/<task>.h5

At training time, ``BenchmarkDataset(..., target_cache_root=...)`` reads these
projections and the trainer applies the Gaussian + failure_prob multiply on
GPU via ``planner.risk.v2_targets.build_target_from_projection``.

The cache stores neither the Gaussian-smoothed heatmap nor the failure-prob
multiply — both happen on GPU at runtime. That lets you tune ``sigma_px`` and
the per-trial prior without rebuilding the cache.

Run::

    /home/aaron/miniconda3/envs/failbench_env/bin/python -u -m scripts.data.precompute_targets \\
        --cache_root /media/aaron/F/failbench/target_cache

Idempotent — trials whose projection is already cached are skipped. Outputs
~750 MB total across both corpora.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import hdf5plugin  # noqa: F401 — register filters
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planner.risk.dataset_v2 import V2Source, _load_manifest  # noqa: E402
from planner.risk.v2_store import V2Reader  # noqa: E402
from planner.risk.v2_targets import project_contacts_for_cache  # noqa: E402


LIBERO_SRC = Path(os.environ.get(
    "FAILBENCH_LIBERO_V2", "/media/aaron/F/failbench/libero/v2"))
ROBOCASA_SRC = Path(os.environ.get(
    "FAILBENCH_ROBOCASA_V2", "/media/aaron/F/failbench/robocasa/v2"))
# Per-trial fields needed to compute the projection — keep IO tiny.
KEEP_KEYS = {
    "contact_positions", "contact_forces", "contact_force_world",
    "cam_agentview_pos", "cam_agentview_mat0",
    "cam_agentview_fovy", "cam_agentview_size",
}


def process_task(src_root: Path, split: str, h5_path: Path,
                 cache_dir: Path,
                 quarantined: set,
                 task_name: str | None = None) -> dict:
    """Walk one task HDF5; write per-trial projections to the cache."""
    if task_name is None:
        task_name = h5_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{task_name}.h5"

    n_done = n_skip = n_err = n_empty = 0
    t0 = time.time()

    reader = V2Reader(str(h5_path))
    try:
        with h5py.File(cache_path, "a") as cache_h:
            existing = set(cache_h.keys())
            with h5py.File(h5_path, "r") as src:
                tids = list(src["trials"].keys())
            for tid in tids:
                if (split, task_name, tid) in quarantined:
                    n_skip += 1
                    continue
                if tid in existing and "projection" in cache_h[tid]:
                    n_skip += 1
                    continue
                try:
                    trial = reader.read_trial(tid, keys=KEEP_KEYS)
                    proj, _H, _W = project_contacts_for_cache(trial)
                    failure_prob = float(trial.get("failure_prob", 1.0))

                    if tid in cache_h:
                        del cache_h[tid]
                    g = cache_h.create_group(tid)
                    # Pure float32 cache — no compression for these tiny arrays
                    g.create_dataset("projection", data=proj.astype(np.float32))
                    g.attrs["failure_prob"] = np.float32(failure_prob)
                    if proj.shape[0] == 0:
                        n_empty += 1
                    n_done += 1
                except (OSError, RuntimeError, KeyError) as e:
                    n_err += 1
                    if n_err <= 5:
                        print(f"  [{task_name}/{tid}] error: {str(e)[:120]}",
                              flush=True)
                if (n_done + n_skip) % 250 == 0 and n_done > 0:
                    dt = time.time() - t0
                    rate = n_done / max(dt, 1e-6)
                    print(f"  {task_name}: done={n_done} skip={n_skip} err={n_err} "
                          f"({rate:.1f}/s)", flush=True)
    finally:
        reader.close() if hasattr(reader, "close") else None

    dt = time.time() - t0
    return {"task": task_name, "n_done": n_done, "n_skip": n_skip,
            "n_err": n_err, "n_empty": n_empty, "elapsed_s": dt,
            "cache_path": str(cache_path)}


def load_quarantine(path: Path) -> dict:
    """Return {source: set((split, task, trial_id))}."""
    if not path.exists():
        return {}
    import csv
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out.setdefault(r["source"], set()).add(
                (r["split"], r["task"], r["trial_id"]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root", default="/media/aaron/F/failbench/target_cache",
                   type=Path)
    p.add_argument("--quarantine", default=str(REPO_ROOT / "out" / "data_verify"
                                                / "quarantine.csv"),
                   type=Path)
    p.add_argument("--sources", nargs="+", default=["libero", "robocasa"])
    p.add_argument("--limit_files", type=int, default=None)
    args = p.parse_args()

    args.cache_root.mkdir(parents=True, exist_ok=True)
    quarantine = load_quarantine(args.quarantine)
    print(f"loaded quarantine: "
          f"{sum(len(v) for v in quarantine.values())} trials across sources",
          flush=True)

    overall = []
    t_all = time.time()
    if "libero" in args.sources:
        q = quarantine.get("libero", set())
        for split in ("libero_spatial", "libero_object", "libero_goal"):
            src_dir = LIBERO_SRC / split
            if not src_dir.exists():
                continue
            cache_split = args.cache_root / split
            h5_files = sorted(src_dir.glob("*.h5"))
            if args.limit_files is not None:
                h5_files = h5_files[: args.limit_files]
            for h5_path in h5_files:
                print(f"\n=== libero/{split}/{h5_path.name} ===", flush=True)
                res = process_task(LIBERO_SRC, split, h5_path, cache_split, q)
                print(f"  done: {res['n_done']} new, {res['n_skip']} skip, "
                      f"{res['n_err']} err, {res['n_empty']} empty in "
                      f"{res['elapsed_s']:.1f}s", flush=True)
                overall.append(res)

    if "robocasa" in args.sources:
        q = quarantine.get("robocasa", set())
        cache_split = args.cache_root / "robocasa"
        h5_files = sorted(ROBOCASA_SRC.glob("*.h5"))
        if args.limit_files is not None:
            h5_files = h5_files[: args.limit_files]
        for h5_path in h5_files:
            print(f"\n=== robocasa/{h5_path.name} ===", flush=True)
            res = process_task(ROBOCASA_SRC, "robocasa", h5_path, cache_split, q)
            print(f"  done: {res['n_done']} new, {res['n_skip']} skip, "
                  f"{res['n_err']} err, {res['n_empty']} empty in "
                  f"{res['elapsed_s']:.1f}s", flush=True)
            overall.append(res)

    n_done = sum(r["n_done"] for r in overall)
    n_skip = sum(r["n_skip"] for r in overall)
    n_err  = sum(r["n_err"]  for r in overall)
    n_empty = sum(r["n_empty"] for r in overall)
    print(f"\nALL DONE in {(time.time() - t_all) / 60:.1f} min  "
          f"new={n_done} skip={n_skip} err={n_err} empty={n_empty}", flush=True)

    # Report cache size
    total_bytes = sum(
        (args.cache_root / split).glob("*.h5").__class__
        and sum(p.stat().st_size for p in (args.cache_root / split).glob("*.h5"))
        for split in ("libero_spatial", "libero_object", "libero_goal", "robocasa")
        if (args.cache_root / split).exists()
    )
    print(f"cache size: {total_bytes / 2**30:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
