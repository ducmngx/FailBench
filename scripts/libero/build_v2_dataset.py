#!/usr/bin/env python3
"""Build the LIBERO v2 contact-prediction dataset (HDF5 per task).

Reads the v1 manifest per split to replay the same (demo, seed, bin_idx,
failure) configs through :meth:`LiberoRunner.run_v2`, producing enriched
trials with: pre-failure window of frames per camera, K-step goal full
state, time-resolved contacts, world-frame forces, dense settle state
trajectory, object poses pre/post, and camera + scene calibration.

Output layout::

    <output_dir>/<split>/<task>.h5
    <output_dir>/<split>/manifest.csv

v1 itself is never modified. v2 is fully standalone — `LiberoV2Dataset`
joins everything it needs from the v2 HDF5s alone.

Example::

    python scripts/libero/build_v2_dataset.py \\
        --splits libero_spatial \\
        --v1_root datasets/libero/v1 \\
        --output_dir /media/aaron/F/failbench/libero/v2 \\
        --workers 8 --resume
"""
from __future__ import annotations

import os

# Force headless GL backend BEFORE importing mujoco. Without this,
# mujoco.Renderer falls back to GLFW which deadlocks in glfw.init() on
# headless workers. Must be set at module scope so spawn'd workers
# inherit it.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# Disable HDF5 file locking. Our design is one writer per per-task HDF5,
# so locks are unnecessary; with stale lock state from prior crashed
# processes they actively block legitimate appends. Must be set BEFORE
# h5py imports anywhere — module-scope here propagates to spawn workers.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import concurrent.futures
import csv
import glob
import logging
import multiprocessing
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = REPO_ROOT / "datasets" / "libero" / "raw"
DEFAULT_V1_ROOT = REPO_ROOT / "datasets" / "libero" / "v1"


V2_MANIFEST_COLUMNS = (
    "trial_id", "split", "task", "demo_key", "seed", "seed_idx", "bin_idx",
    "fail_idx", "traj_progress", "failure_mode", "failure_joints",
    "failure_prob", "is_holding", "n_contacts", "h5_path",
)


@dataclass(frozen=True)
class V2TrialSpec:
    """One v2 trial to build, derived from a v1 manifest row."""
    split: str
    task: str
    hdf5_path: str          # source LIBERO HDF5
    demo_key: str
    seed: int
    seed_idx: int
    bin_idx: int
    progress: float
    failure_mode: str
    failure_joints: tuple
    failure_prob: float
    experiment_id: str

    @property
    def trial_id(self) -> str:
        return f"{self.demo_key}_s{self.seed_idx}_b{self.bin_idx}"


# --------------------------------------------------------------------------
# v1 manifest -> v2 specs
# --------------------------------------------------------------------------


def _parse_joints(s: str) -> tuple:
    """Parse v1's `failure_joints` manifest field.

    v1 writes MULTI_JOINT entries as comma-separated, CSV-quoted strings
    (e.g. ``"joint4,joint6"``). Some older rows may use ``;`` instead.
    Accept either separator and ignore empty fragments.
    """
    s = (s or "").strip()
    if not s:
        return tuple()
    parts = [n.strip() for n in s.replace(";", ",").split(",")]
    return tuple(p for p in parts if p)


def load_v2_specs(v1_manifest: Path, raw_root: Path,
                  split: str) -> List[V2TrialSpec]:
    specs: List[V2TrialSpec] = []
    with open(v1_manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["task"]
            hdf5 = raw_root / split / f"{task}_demo.hdf5"
            # Some v1 rows don't carry seed_idx explicitly; derive from
            # experiment_id pattern "..._s{seed_idx}_b{bin_idx}".
            exp = row["experiment_id"]
            seed_idx = int(exp.rsplit("_s", 1)[1].split("_")[0])
            specs.append(V2TrialSpec(
                split=split,
                task=task,
                hdf5_path=str(hdf5),
                demo_key=row["demo_key"],
                seed=int(row["seed"]),
                seed_idx=seed_idx,
                bin_idx=int(row["bin_idx"]),
                progress=float(row["traj_progress"]),
                failure_mode=row["failure_mode"],
                failure_joints=_parse_joints(row["failure_joints"]),
                failure_prob=float(row["failure_prob"]),
                experiment_id=exp,
            ))
    return specs


# --------------------------------------------------------------------------
# Worker: build one task's HDF5
# --------------------------------------------------------------------------


def _build_task_worker(args) -> dict:
    """One worker owns one (split, task) HDF5 file end-to-end."""
    (split, task, specs, output_dir, settle_steps, resistance,
     image_w, image_h, resume) = args
    # Late imports so they happen inside the worker process (post-spawn).
    from planner.experiments.libero.adapter import load_demo
    from planner.experiments.libero.runner import (
        LiberoRunner, LiberoTrialConfig, V2CaptureSpec,
    )
    from planner.experiments.config import FailureConfig, FailureMode
    from planner.risk.v2_store import V2Writer

    out_h5 = Path(output_dir) / split / f"{task}.h5"
    n_done, n_err, t0 = 0, 0, time.time()
    rows: List[dict] = []

    capture = V2CaptureSpec(image_w=image_w, image_h=image_h)

    with V2Writer(out_h5, split=split, task=task) as writer:
        existing = writer.existing_trial_ids() if resume else set()

        # Group specs by demo so we reuse the LiberoRunner per demo.
        by_demo: dict = defaultdict(list)
        for s in specs:
            by_demo[(s.hdf5_path, s.demo_key)].append(s)

        for (hdf5_path, demo_key), demo_specs in by_demo.items():
            try:
                demo = load_demo(hdf5_path, demo_key)
            except Exception:
                logger.exception("Failed to load %s/%s", hdf5_path, demo_key)
                n_err += len(demo_specs)
                continue
            for spec in demo_specs:
                if spec.trial_id in existing:
                    continue
                try:
                    failure = FailureConfig(
                        mode=FailureMode[spec.failure_mode],
                        joint_names=list(spec.failure_joints),
                        probability=spec.failure_prob,
                    )
                    cfg = LiberoTrialConfig(
                        fail_progress=spec.progress,
                        seed=spec.seed,
                        resistance_mode=resistance,
                        post_failure_settle_steps=settle_steps,
                        image_width=image_w,
                        image_height=image_h,
                    )
                    runner = LiberoRunner(demo, cfg)
                    try:
                        payload = runner.run_v2(capture, failure,
                                                experiment_id=spec.experiment_id)
                    finally:
                        runner.close()
                    if payload is None:
                        n_err += 1
                        continue
                    payload.update({
                        "trial_id": spec.trial_id,
                        "split": split,
                        "task": task,
                        "demo_key": demo_key,
                        "seed": int(spec.seed),
                        "seed_idx": int(spec.seed_idx),
                        "bin_idx": int(spec.bin_idx),
                    })
                    writer.write_trial(spec.trial_id, payload)
                    rows.append({
                        "trial_id": spec.trial_id,
                        "split": split,
                        "task": task,
                        "demo_key": demo_key,
                        "seed": spec.seed,
                        "seed_idx": spec.seed_idx,
                        "bin_idx": spec.bin_idx,
                        "fail_idx": int(payload["fail_idx"]),
                        "traj_progress": float(payload["traj_progress"]),
                        "failure_mode": payload["failure_mode"],
                        "failure_joints": ";".join(spec.failure_joints),
                        "failure_prob": float(payload["failure_prob"]),
                        "is_holding": bool(payload["is_holding"]),
                        "n_contacts": int(payload["contact_positions"].shape[0]),
                        "h5_path": str(out_h5),
                    })
                    n_done += 1
                except Exception:
                    logger.exception("Trial %s failed", spec.trial_id)
                    n_err += 1

    dt = time.time() - t0
    return {
        "split": split,
        "task": task,
        "n_done": n_done,
        "n_err": n_err,
        "elapsed_s": dt,
        "rows": rows,
        "h5_path": str(out_h5),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _append_manifest(rows: List[dict], path: Path) -> None:
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        w = csv.DictWriter(f, fieldnames=V2_MANIFEST_COLUMNS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in V2_MANIFEST_COLUMNS})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", nargs="+", required=True,
                   choices=["libero_spatial", "libero_object", "libero_goal"])
    p.add_argument("--v1_root", default=str(DEFAULT_V1_ROOT),
                   help="Where v1 manifest.csv lives (one per split)")
    p.add_argument("--raw_root", default=str(DEFAULT_RAW_ROOT),
                   help="LIBERO HDF5 root (split subdirs of task .hdf5 files)")
    p.add_argument("--output_dir", required=True,
                   help="v2 output root, e.g. /media/aaron/F/failbench/libero/v2")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resistance", choices=["none", "gravcomp_pd"],
                   default="gravcomp_pd")
    p.add_argument("--settle_steps", type=int, default=500)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--resume", action="store_true",
                   help="Skip trials whose group already exists")
    p.add_argument("--limit_tasks", type=int, default=None)
    p.add_argument("--limit_per_task", type=int, default=None,
                   help="Cap trials per task (for dry-run)")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    raw_root = Path(args.raw_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Filesystem pre-flight: vfat/exfat cap files at 4 GiB which our per-task
    # HDF5s will exceed. ext4 / xfs / btrfs / zfs are all fine. Bail loudly
    # rather than silently corrupt the build.
    try:
        out = subprocess.check_output(
            ["findmnt", "-no", "FSTYPE", "--target", str(out_root)],
            text=True,
        ).strip()
        if out in ("vfat", "msdos", "exfat"):
            logger.error(
                "Output volume %s is %s — has a 4 GiB per-file limit that will "
                "corrupt per-task HDF5s. Reformat to ext4 (or pick a different "
                "--output_dir).", out_root, out,
            )
            return 2
        logger.info("Output volume %s is %s — OK", out_root, out)
    except Exception as e:  # pragma: no cover
        logger.warning("Filesystem pre-flight skipped: %s", e)

    overall_t0 = time.time()

    for split in args.splits:
        v1_manifest = Path(args.v1_root) / split / "manifest.csv"
        if not v1_manifest.exists():
            logger.warning("Missing v1 manifest: %s — skipping split", v1_manifest)
            continue
        specs = load_v2_specs(v1_manifest, raw_root, split)
        logger.info("Split %s: %d v1 manifest rows", split, len(specs))

        # Group by task and dispatch one worker per task.
        by_task: dict = defaultdict(list)
        for s in specs:
            by_task[s.task].append(s)
        tasks = sorted(by_task.keys())
        if args.limit_tasks is not None:
            tasks = tasks[:args.limit_tasks]
        if args.limit_per_task is not None:
            for t in tasks:
                by_task[t] = by_task[t][:args.limit_per_task]

        manifest_path = out_root / split / "manifest.csv"

        worker_args = [
            (split, t, by_task[t], str(out_root),
             args.settle_steps, args.resistance,
             args.image_w, args.image_h, args.resume)
            for t in tasks
        ]

        results: List[dict] = []
        if args.workers <= 1:
            for wa in worker_args:
                results.append(_build_task_worker(wa))
                if results[-1]["rows"]:
                    _append_manifest(results[-1]["rows"], manifest_path)
        else:
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=ctx
            ) as ex:
                futures = {ex.submit(_build_task_worker, wa): wa
                           for wa in worker_args}
                for fut in concurrent.futures.as_completed(futures):
                    wa = futures[fut]
                    try:
                        res = fut.result()
                        results.append(res)
                        if res["rows"]:
                            _append_manifest(res["rows"], manifest_path)
                        logger.info(
                            "[%s/%s] done=%d err=%d in %.1fs (%.2f trials/s)",
                            res["split"], res["task"], res["n_done"],
                            res["n_err"], res["elapsed_s"],
                            res["n_done"] / max(res["elapsed_s"], 1e-6),
                        )
                    except Exception:
                        logger.exception("Worker for %s/%s crashed", wa[0], wa[1])

        n_done = sum(r["n_done"] for r in results)
        n_err = sum(r["n_err"] for r in results)
        elapsed = time.time() - overall_t0
        logger.info("Split %s: %d done, %d errors, %.1f min total elapsed",
                    split, n_done, n_err, elapsed / 60.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
