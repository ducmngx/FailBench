"""LIBERO dataset bookkeeping: trial enumeration, manifest schema, NPZ saver.

This module is intentionally robosuite-free — it reads HDF5 keys with h5py and
constructs deterministic trial specs. Worker processes use these specs to drive
``LiberoRunner`` and write per-trial npzs.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import h5py
import numpy as np

from planner.experiments.config import FailureConfig, FailureMode
from planner.experiments.libero.runner import _default_failures
from planner.experiments.manager import save_sample_npz
from planner.experiments.runner import DataSample


# --------------------------------------------------------------------------
# HDF5 enumeration
# --------------------------------------------------------------------------


def iter_demo_keys(hdf5_path: str) -> List[str]:
    """Return sorted demo keys (`demo_0`, `demo_1`, ...) without robosuite."""
    with h5py.File(hdf5_path, "r") as f:
        return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))


def task_name_from_hdf5(hdf5_path: str) -> str:
    """Strip the `_demo` suffix LIBERO appends to task hdf5 filenames."""
    stem = os.path.splitext(os.path.basename(hdf5_path))[0]
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem


# --------------------------------------------------------------------------
# Trial spec
# --------------------------------------------------------------------------


@dataclass
class LiberoTrialSpec:
    """One unit of work for the LIBERO dataset driver."""
    hdf5: str
    demo_key: str
    split: str
    task: str
    seed: int
    bin_idx: int
    progress: float
    failure: FailureConfig
    experiment_id: str

    @property
    def task_id(self) -> str:
        return f"{self.split}__{self.task}"

    @property
    def npz_filename(self) -> str:
        return f"{self.experiment_id}.npz"


def _spec_seed(base_seed: int, split: str, task: str, demo_key: str,
               seed_idx: int, bin_idx: int) -> int:
    """Deterministic per-trial seed independent of iteration order."""
    h = hashlib.sha1(
        f"{base_seed}|{split}|{task}|{demo_key}|{seed_idx}|{bin_idx}".encode("utf-8")
    ).hexdigest()
    # 31-bit value fits in numpy int32 (the dataset's manifest/npz schema).
    return int(h[:8], 16) & 0x7FFFFFFF


def _sample_failure(rng: random.Random,
                    failures: Sequence[FailureConfig]) -> FailureConfig:
    weights = [fc.probability for fc in failures]
    return rng.choices(list(failures), weights=weights, k=1)[0]


def make_trial_specs(
    hdf5_files: Sequence[str],
    split: str,
    seeds: int,
    progress_bins: int,
    base_seed: int,
    progress_lo: float = 0.05,
    progress_hi: float = 0.9,
    failures: Optional[Sequence[FailureConfig]] = None,
    limit_demos: Optional[int] = None,
) -> List[LiberoTrialSpec]:
    """Build a deterministic list of trial specs for one split.

    For each (hdf5, demo, seed_idx, bin_idx) we draw one continuous
    ``progress`` from the bin's sub-range and one ``FailureConfig`` from the
    weighted failure mix, both via a per-spec RNG keyed by ``base_seed``.
    """
    if failures is None:
        failures = _default_failures()

    bin_edges = np.linspace(progress_lo, progress_hi, progress_bins + 1)
    specs: List[LiberoTrialSpec] = []

    for hdf5 in hdf5_files:
        task = task_name_from_hdf5(hdf5)
        demo_keys = iter_demo_keys(hdf5)
        if limit_demos is not None:
            demo_keys = demo_keys[:limit_demos]
        for demo_key in demo_keys:
            for seed_idx in range(seeds):
                for bin_idx in range(progress_bins):
                    seed = _spec_seed(base_seed, split, task, demo_key,
                                       seed_idx, bin_idx)
                    rng = random.Random(seed)
                    lo, hi = bin_edges[bin_idx], bin_edges[bin_idx + 1]
                    progress = rng.uniform(lo, hi)
                    fc = _sample_failure(rng, failures)
                    eid = f"exp_{split}_{task}_{demo_key}_s{seed_idx}_b{bin_idx}"
                    specs.append(LiberoTrialSpec(
                        hdf5=hdf5, demo_key=demo_key, split=split, task=task,
                        seed=seed, bin_idx=bin_idx, progress=progress,
                        failure=fc, experiment_id=eid,
                    ))
    return specs


# --------------------------------------------------------------------------
# NPZ + manifest
# --------------------------------------------------------------------------


def save_libero_sample_npz(sample: DataSample, path: str) -> None:
    """Save via the shared ``save_sample_npz`` (it already writes pre_target_qpos)."""
    save_sample_npz(sample, path)


LIBERO_MANIFEST_COLUMNS = [
    "experiment_id", "split", "task_id", "task", "demo_key", "traj_id",
    "seed", "bin_idx", "traj_progress",
    "failure_mode", "failure_joints", "failure_prob",
    "num_contacts", "had_any_collision", "impacted_geom_ids",
    "pre_qvel_norm", "npz_file",
]


def libero_manifest_row(spec: LiberoTrialSpec, sample: DataSample,
                         npz_filename: str) -> dict:
    fc = spec.failure
    return {
        "experiment_id": sample.experiment_id,
        "split": spec.split,
        "task_id": sample.task_id,
        "task": spec.task,
        "demo_key": spec.demo_key,
        "traj_id": sample.traj_id,
        "seed": sample.seed,
        "bin_idx": spec.bin_idx,
        "traj_progress": round(float(sample.traj_progress), 4),
        "failure_mode": fc.mode.name,
        "failure_joints": ",".join(fc.joint_names) if fc.joint_names else "",
        "failure_prob": round(float(fc.probability), 4),
        "num_contacts": len(sample.aggregate_contacts),
        "had_any_collision": any(fr.had_collision
                                  for fr in sample.failure_results),
        "impacted_geom_ids": ";".join(str(g)
                                       for g in sample.all_impacted_geom_ids),
        "pre_qvel_norm": round(float(np.linalg.norm(
            sample.pre_failure_robot.qvel)), 4),
        "npz_file": npz_filename,
    }


def append_libero_manifest(rows: List[dict], path: str) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LIBERO_MANIFEST_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
