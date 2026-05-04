#!/usr/bin/env python3
"""Generate a LIBERO failure-injection dataset.

For each split × task × demo × seed × progress-bin, run one
:class:`LiberoRunner` trial and write a per-trial npz + per-split manifest.

Example::

    python scripts/libero/run_libero_dataset.py \
        --splits libero_spatial libero_object libero_goal \
        --output_dir datasets/libero/v1 \
        --seeds 3 --progress_bins 10 --workers 8
"""

from __future__ import annotations

import os

# Force headless GL backend BEFORE importing mujoco anywhere. Without this,
# mujoco.Renderer falls back to GLFW which deadlocks in glfw.init() on
# headless machines (no X display). Must be set in workers too — set here
# at module scope so spawn'd workers re-execute the import and inherit it.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import concurrent.futures
import glob
import logging
import multiprocessing
import sys
import time
from typing import List, Optional

from planner.experiments.libero.adapter import load_demo
from planner.experiments.libero.dataset import (
    LIBERO_MANIFEST_COLUMNS,
    LiberoTrialSpec,
    append_libero_manifest,
    libero_manifest_row,
    make_trial_specs,
    save_libero_sample_npz,
)
from planner.experiments.libero.runner import LiberoRunner, LiberoTrialConfig

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RAW_ROOT = os.path.join(REPO_ROOT, "datasets", "libero", "raw")


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


def _run_one_trial(args) -> Optional[dict]:
    spec, output_dir, resistance_mode, settle_steps = args
    npz_path = os.path.join(output_dir, spec.npz_filename)
    try:
        demo = load_demo(spec.hdf5, spec.demo_key)
        cfg = LiberoTrialConfig(
            fail_progress=spec.progress,
            failure_configs=[spec.failure],
            seed=spec.seed,
            resistance_mode=resistance_mode,
            post_failure_settle_steps=settle_steps,
        )
        runner = LiberoRunner(demo, cfg)
        try:
            sample = runner.run(experiment_id=spec.experiment_id)
        finally:
            runner.close()
        if sample is None:
            return None
        # Override task_id with the split-prefixed form.
        sample.task_id = spec.task_id
        save_libero_sample_npz(sample, npz_path)
        return libero_manifest_row(spec, sample, spec.npz_filename)
    except Exception:
        logger.exception("Trial %s failed", spec.experiment_id)
        return None


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _run_chunk(specs: List[LiberoTrialSpec], output_dir: str,
                workers: int, resistance: str, settle_steps: int,
                per_trial_timeout_s: float) -> List[Optional[dict]]:
    args_list = [(s, output_dir, resistance, settle_steps) for s in specs]
    if workers <= 1:
        return [_run_one_trial(a) for a in args_list]

    results: List[Optional[dict]] = [None] * len(args_list)
    ctx = multiprocessing.get_context("spawn")
    i = 0
    while i < len(args_list):
        end = min(i + workers, len(args_list))
        batch = args_list[i:end]
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=ctx
            ) as ex:
                futures = {ex.submit(_run_one_trial, a): (i + k)
                           for k, a in enumerate(batch)}
                for fut in concurrent.futures.as_completed(
                    futures, timeout=per_trial_timeout_s * 2,
                ):
                    idx = futures[fut]
                    cfg_id = batch[idx - i][0].experiment_id
                    try:
                        results[idx] = fut.result(timeout=per_trial_timeout_s)
                    except concurrent.futures.TimeoutError:
                        logger.warning("Trial %s timed out (>%.0fs)",
                                       cfg_id, per_trial_timeout_s)
                    except concurrent.futures.process.BrokenProcessPool:
                        logger.warning("Trial %s worker crashed", cfg_id)
                    except Exception as e:
                        logger.warning("Trial %s raised: %s", cfg_id, e)
        except concurrent.futures.process.BrokenProcessPool:
            logger.warning("Pool broken in batch %d-%d", i, end)
        except concurrent.futures.TimeoutError:
            logger.warning("Batch %d-%d overall timeout", i, end)
        i = end
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", nargs="+", required=True,
                   choices=["libero_spatial", "libero_object", "libero_goal"])
    p.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    p.add_argument("--output_dir", default=os.path.join(
        REPO_ROOT, "datasets", "libero", "v1"))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--progress_bins", type=int, default=10)
    p.add_argument("--progress_lo", type=float, default=0.05)
    p.add_argument("--progress_hi", type=float, default=0.9)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resistance", choices=["none", "gravcomp_pd"],
                   default="gravcomp_pd")
    p.add_argument("--settle_steps", type=int, default=500)
    p.add_argument("--chunk_size", type=int, default=64,
                   help="Manifest is appended after each chunk")
    p.add_argument("--per_trial_timeout_s", type=float, default=120.0)
    p.add_argument("--limit_tasks", type=int, default=None)
    p.add_argument("--limit_demos", type=int, default=None)
    p.add_argument("--limit_total", type=int, default=None,
                   help="Cap the total number of trials (after spec build)")
    p.add_argument("--resume", action="store_true",
                   help="Skip trials whose npz already exists")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    overall_t0 = time.time()
    overall_done = 0

    for split in args.splits:
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        manifest_path = os.path.join(split_dir, "manifest.csv")

        hdf5_files = sorted(glob.glob(
            os.path.join(args.raw_root, split, "*.hdf5")))
        if args.limit_tasks is not None:
            hdf5_files = hdf5_files[:args.limit_tasks]
        if not hdf5_files:
            logger.warning("No HDF5 files under %s/%s — skipping",
                           args.raw_root, split)
            continue

        logger.info("Split %s: %d task hdf5 files", split, len(hdf5_files))

        specs = make_trial_specs(
            hdf5_files=hdf5_files,
            split=split,
            seeds=args.seeds,
            progress_bins=args.progress_bins,
            base_seed=args.base_seed,
            progress_lo=args.progress_lo,
            progress_hi=args.progress_hi,
            limit_demos=args.limit_demos,
        )
        logger.info("Split %s: built %d trial specs", split, len(specs))

        if args.resume:
            before = len(specs)
            specs = [s for s in specs
                     if not os.path.exists(os.path.join(split_dir, s.npz_filename))]
            logger.info("Resume: %d/%d remaining for %s",
                        len(specs), before, split)

        if args.limit_total is not None:
            specs = specs[:args.limit_total]

        if not specs:
            continue

        split_t0 = time.time()
        for chunk_start in range(0, len(specs), args.chunk_size):
            chunk = specs[chunk_start:chunk_start + args.chunk_size]
            rows_raw = _run_chunk(
                chunk, split_dir, args.workers, args.resistance,
                args.settle_steps, args.per_trial_timeout_s,
            )
            rows = [r for r in rows_raw if r is not None]
            if rows:
                append_libero_manifest(rows, manifest_path)
            elapsed = time.time() - split_t0
            done = chunk_start + len(chunk)
            rate = done / max(elapsed, 1e-3)
            eta = (len(specs) - done) / max(rate, 1e-3)
            logger.info(
                "[%s] chunk %d/%d done=%d ok=%d rate=%.2f trials/s eta=%.1fs",
                split, chunk_start // args.chunk_size + 1,
                (len(specs) + args.chunk_size - 1) // args.chunk_size,
                done, len(rows), rate, eta,
            )
            overall_done += len(rows)

    total_elapsed = time.time() - overall_t0
    logger.info("Done: %d successful trials in %.1fs",
                overall_done, total_elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
