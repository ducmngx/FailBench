#!/usr/bin/env python3
"""Build the RoboCasa v2 contact-prediction dataset (HDF5 per task).

Sibling to ``scripts/libero/build_v2_dataset.py`` — writes the same v2 schema
into ``<output_dir>/<task>.h5`` files plus a manifest.csv. RoboCasa has no
upstream v1 manifest to mirror, so this script generates the (demo × failure)
trial specs from scratch using a fixed stratified sampler (see
``TIER1_TRIALS_PER_DEMO`` below — 15 trials/demo, balanced across modes and
progress bins).

The output schema is bit-identical to LIBERO v2 — pooled training loaders
consume both source roots through a single manifest.

Example::

    /path/to/conda/envs/failbench_env/bin/python -u -m scripts.robocasa.build_v2_dataset \\
        --raw_root datasets/robocasa/raw \\
        --output_dir /media/aaron/F/failbench/robocasa/v2 \\
        --workers 4 --resume

Per the LIBERO v2 build notes: invoke the env's python directly (not via
``conda run``) so stdout stays line-buffered for the long run.

**Recovery after worker crash:** if the process pool collapses mid-build,
the per-task HDF5s on disk still have the data flushed to that point.
**DO NOT run ``h5clear -s`` on them** — in our h5py 3.x environment this
truncates the metadata's end-of-allocation pointer and leaves the data
unreachable (verified 2026-06-03). Instead: ``rm`` the partial HDF5s
(+ the partial manifest.csv), re-launch with ``--resume``. The 4-worker
first attempt hit swap thrash at ~3 GB written; **3 workers is the safe
default for a 32 GiB host** with RoboCasa-density scenes.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import concurrent.futures
import csv
import logging
import multiprocessing
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


V2_MANIFEST_COLUMNS = (
    "trial_id", "split", "task", "demo_key", "seed", "seed_idx", "bin_idx",
    "fail_idx", "traj_progress", "failure_mode", "failure_joints",
    "failure_prob", "is_holding", "n_contacts", "h5_path",
)


# Stratified failure-spec table — 15 trials/demo, balanced over progress bins
# and failure modes. Tier 1 (5 tasks × 100 demos) → 7,500 trials.
TIER1_TRIALS_PER_DEMO: List[Tuple[float, str, Tuple[str, ...]]] = [
    (0.20, "SINGLE_JOINT", ("joint2",)),
    (0.20, "SINGLE_JOINT", ("joint4",)),
    (0.20, "GRIPPER_OPEN",  ()),
    (0.35, "SINGLE_JOINT", ("joint2",)),
    (0.35, "SINGLE_JOINT", ("joint4",)),
    (0.35, "SINGLE_JOINT", ("joint6",)),
    (0.50, "SINGLE_JOINT", ("joint4",)),
    (0.50, "GRIPPER_OPEN",  ()),
    (0.50, "ALL_JOINTS",    ()),
    (0.65, "SINGLE_JOINT", ("joint2",)),
    (0.65, "SINGLE_JOINT", ("joint6",)),
    (0.65, "GRIPPER_OPEN",  ()),
    (0.80, "SINGLE_JOINT", ("joint4",)),
    (0.80, "GRIPPER_OPEN",  ()),
    (0.80, "ALL_JOINTS",    ()),
]
# Failure-mode prior probability (uniform across modes for tier 1).
_MODE_PROB = 1.0 / len(TIER1_TRIALS_PER_DEMO)


SPLIT_NAME = "robocasa"  # single split — RoboCasa doesn't split by suite


@dataclass(frozen=True)
class RobocasaTrialSpec:
    """One v2 trial: identifies the demo and the failure to inject."""
    task: str
    hdf5_path: str
    demo_key: str
    seed: int
    seed_idx: int
    bin_idx: int
    progress: float
    failure_mode: str
    failure_joints: Tuple[str, ...]

    @property
    def trial_id(self) -> str:
        return f"{self.demo_key}_s{self.seed_idx}_b{self.bin_idx}"

    @property
    def experiment_id(self) -> str:
        return f"robocasa_{self.task}_{self.demo_key}_s{self.seed_idx}_b{self.bin_idx}"


def build_task_specs(hdf5_path: Path, limit_demos: int = None) -> List[RobocasaTrialSpec]:
    """Enumerate the stratified spec set for one task HDF5."""
    from planner.experiments.robocasa.adapter import list_demos
    demos = list_demos(str(hdf5_path))
    if limit_demos is not None:
        demos = demos[:limit_demos]
    task = hdf5_path.stem
    specs: List[RobocasaTrialSpec] = []
    for demo_key in demos:
        for bin_idx, (prog, mode, joints) in enumerate(TIER1_TRIALS_PER_DEMO):
            specs.append(RobocasaTrialSpec(
                task=task,
                hdf5_path=str(hdf5_path),
                demo_key=demo_key,
                seed=42 + bin_idx,
                seed_idx=0,
                bin_idx=bin_idx,
                progress=prog,
                failure_mode=mode,
                failure_joints=joints,
            ))
    return specs


# --------------------------------------------------------------------------
# Worker: build one task's HDF5
# --------------------------------------------------------------------------


def _build_task_worker(args) -> dict:
    """One worker owns one task HDF5 file end-to-end."""
    (task, specs, output_dir, settle_steps, resistance,
     image_w, image_h, resume) = args

    # Imports inside the worker so they happen post-spawn.
    import mujoco
    from planner.experiments.config import FailureConfig, FailureMode
    from planner.experiments.robocasa import (
        load_demo, materialise_mjcf, read_ep_meta, build_scene_overrides,
    )
    from planner.experiments.libero.runner import (
        LiberoRunner, LiberoTrialConfig, V2CaptureSpec,
    )
    from planner.risk.v2_store import V2Writer

    out_h5 = Path(output_dir) / f"{task}.h5"
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    capture = V2CaptureSpec(image_w=image_w, image_h=image_h)
    rows: List[dict] = []
    n_done = n_err = 0
    t0 = time.time()

    with V2Writer(out_h5, split=SPLIT_NAME, task=task) as writer:
        existing = writer.existing_trial_ids() if resume else set()

        by_demo: dict = defaultdict(list)
        for s in specs:
            by_demo[(s.hdf5_path, s.demo_key)].append(s)

        for (hdf5_path, demo_key), demo_specs in by_demo.items():
            # Skip the demo entirely if every trial is already written.
            pending = [s for s in demo_specs if s.trial_id not in existing]
            if not pending:
                continue

            try:
                demo = load_demo(hdf5_path, demo_key)
                xml_path = materialise_mjcf(demo.model_xml)
                ep_meta = read_ep_meta(hdf5_path, demo_key)
                # Seed an MjData at states[0] to derive scene overrides.
                m_tmp = mujoco.MjModel.from_xml_path(xml_path)
                d_tmp = mujoco.MjData(m_tmp)
                nq, nv = m_tmp.nq, m_tmp.nv
                flat = demo.full_states[0]
                offset = 1 if flat.shape[0] == 1 + nq + nv else 0
                d_tmp.qpos[:] = flat[offset:offset + nq]
                d_tmp.qvel[:] = flat[offset + nq:offset + nq + nv]
                mujoco.mj_forward(m_tmp, d_tmp)
                overrides = build_scene_overrides(m_tmp, d_tmp, ep_meta)
                del m_tmp, d_tmp
            except Exception:
                logger.exception("[%s] load/build failed for %s", task, demo_key)
                n_err += len(pending)
                continue

            runner = None
            try:
                cfg = LiberoTrialConfig(
                    resistance_mode=resistance,
                    post_failure_settle_steps=settle_steps,
                    image_width=image_w,
                    image_height=image_h,
                    seed_from_init_state=False,
                )
                runner = LiberoRunner(demo, cfg,
                                      mjcf_path=xml_path,
                                      scene_overrides=overrides)
                for spec in pending:
                    runner.config.fail_progress = spec.progress
                    runner.config.seed = spec.seed
                    failure = FailureConfig(
                        mode=FailureMode[spec.failure_mode],
                        joint_names=list(spec.failure_joints),
                        probability=_MODE_PROB,
                    )
                    try:
                        payload = runner.run_v2(
                            capture, failure, experiment_id=spec.experiment_id)
                        if payload is None:
                            n_err += 1
                            continue
                        payload.update({
                            "trial_id": spec.trial_id,
                            "split": SPLIT_NAME,
                            "task": task,
                            "demo_key": demo_key,
                            "seed": int(spec.seed),
                            "seed_idx": int(spec.seed_idx),
                            "bin_idx": int(spec.bin_idx),
                        })
                        writer.write_trial(spec.trial_id, payload)
                        rows.append({
                            "trial_id": spec.trial_id,
                            "split": SPLIT_NAME,
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
                        logger.exception("[%s/%s] trial %s failed",
                                         task, demo_key, spec.trial_id)
                        n_err += 1
            finally:
                if runner is not None:
                    try:
                        runner.close()
                    except Exception:
                        pass

    dt = time.time() - t0
    return {
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
    p.add_argument("--raw_root", default=str(REPO_ROOT / "datasets" / "robocasa" / "raw"),
                   help="RoboCasa HDF5 root (flat: <raw_root>/<task>.hdf5)")
    p.add_argument("--output_dir", required=True,
                   help="v2 output root, e.g. /media/aaron/F/failbench/robocasa/v2")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Subset of task names (without .hdf5). Default: all in raw_root.")
    p.add_argument("--workers", type=int, default=3,
                   help="Default 3; 4+ causes swap thrash on 32 GiB hosts "
                        "with RoboCasa-density scenes (~5–6 GB RSS/worker).")
    p.add_argument("--resistance", choices=["none", "gravcomp_pd"],
                   default="gravcomp_pd")
    p.add_argument("--settle_steps", type=int, default=500)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit_demos", type=int, default=None,
                   help="Cap demos per task (for dry-run)")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    raw_root = Path(args.raw_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Filesystem pre-flight: vfat/exfat → silent HDF5 corruption.
    try:
        fstype = subprocess.check_output(
            ["findmnt", "-no", "FSTYPE", "--target", str(out_root)],
            text=True,
        ).strip()
        if fstype in ("vfat", "msdos", "exfat"):
            logger.error("Output volume %s is %s — has a 4 GiB per-file limit. "
                         "Reformat to ext4 or pick a different --output_dir.",
                         out_root, fstype)
            return 2
        logger.info("Output volume %s is %s — OK", out_root, fstype)
    except Exception as e:
        logger.warning("Filesystem pre-flight skipped: %s", e)

    # Find task HDF5s.
    all_tasks = sorted([p.stem for p in raw_root.glob("*.hdf5")])
    if not all_tasks:
        logger.error("No .hdf5 files under %s", raw_root)
        return 2
    if args.tasks:
        missing = set(args.tasks) - set(all_tasks)
        if missing:
            logger.error("Unknown tasks: %s. Available: %s", missing, all_tasks)
            return 2
        tasks = list(args.tasks)
    else:
        tasks = all_tasks
    logger.info("Tasks: %s", tasks)

    # Build specs per task.
    task_specs: dict = {}
    n_total = 0
    for task in tasks:
        specs = build_task_specs(raw_root / f"{task}.hdf5", limit_demos=args.limit_demos)
        task_specs[task] = specs
        n_total += len(specs)
        logger.info("  %s: %d demos × %d trials/demo = %d specs",
                    task, len(specs) // len(TIER1_TRIALS_PER_DEMO),
                    len(TIER1_TRIALS_PER_DEMO), len(specs))
    logger.info("Total: %d trials across %d tasks", n_total, len(tasks))

    manifest_path = out_root / "manifest.csv"
    worker_args = [
        (t, task_specs[t], str(out_root),
         args.settle_steps, args.resistance,
         args.image_w, args.image_h, args.resume)
        for t in tasks
    ]

    overall_t0 = time.time()
    results: List[dict] = []
    if args.workers <= 1:
        for wa in worker_args:
            res = _build_task_worker(wa)
            results.append(res)
            if res["rows"]:
                _append_manifest(res["rows"], manifest_path)
            logger.info("[%s] done=%d err=%d in %.1fs (%.2f/s)",
                        res["task"], res["n_done"], res["n_err"],
                        res["elapsed_s"],
                        res["n_done"] / max(res["elapsed_s"], 1e-6))
    else:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=ctx) as ex:
            futures = {ex.submit(_build_task_worker, wa): wa for wa in worker_args}
            for fut in concurrent.futures.as_completed(futures):
                wa = futures[fut]
                try:
                    res = fut.result()
                    results.append(res)
                    if res["rows"]:
                        _append_manifest(res["rows"], manifest_path)
                    logger.info("[%s] done=%d err=%d in %.1fs (%.2f/s)",
                                res["task"], res["n_done"], res["n_err"],
                                res["elapsed_s"],
                                res["n_done"] / max(res["elapsed_s"], 1e-6))
                except Exception:
                    logger.exception("Worker for %s crashed", wa[0])

    n_done = sum(r["n_done"] for r in results)
    n_err = sum(r["n_err"] for r in results)
    elapsed = time.time() - overall_t0
    logger.info("OVERALL: %d done, %d errors, %.1f min total",
                n_done, n_err, elapsed / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
