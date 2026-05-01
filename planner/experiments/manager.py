"""Batch experiment orchestration, npz I/O, and CSV manifest generation."""

import csv
import concurrent.futures
import glob as globmod
import logging
import multiprocessing
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from planner.experiments.config import ExperimentConfig, FailureConfig
from planner.experiments.data_capture import ContactPoint
from planner.experiments.runner import DataSample, ExperimentRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NPZ I/O  (Option C: flat arrays with per-contact failure ID tags)
# ---------------------------------------------------------------------------


def save_sample_npz(sample: DataSample, path: str) -> None:
    """Serialize a DataSample to a single .npz file."""
    # Flatten contacts
    contacts = sample.aggregate_contacts
    n_contacts = len(contacts)

    if n_contacts > 0:
        contact_positions = np.array([c.pos for c in contacts], dtype=np.float32)
        contact_forces = np.array([c.force for c in contacts], dtype=np.float32)
        contact_geom_pairs = np.array([[c.geom1, c.geom2] for c in contacts], dtype=np.int32)
    else:
        contact_positions = np.zeros((0, 3), dtype=np.float32)
        contact_forces = np.zeros((0, 6), dtype=np.float32)
        contact_geom_pairs = np.zeros((0, 2), dtype=np.int32)

    # Build per-contact failure_id mapping
    contact_failure_id = np.zeros(n_contacts, dtype=np.int32)
    offset = 0
    for fid, fr in enumerate(sample.failure_results):
        count = len(fr.contacts)
        contact_failure_id[offset : offset + count] = fid
        offset += count

    # Failure mode metadata
    failure_modes = np.array(
        [fr.failure_config.mode.value for fr in sample.failure_results],
    )
    failure_probs = np.array(
        [fr.failure_config.probability for fr in sample.failure_results],
        dtype=np.float32,
    )

    arrays = {
        "pre_rgb": sample.pre_failure_rgb,
        "pre_qpos": sample.pre_failure_robot.qpos.astype(np.float64),
        "pre_qvel": sample.pre_failure_robot.qvel.astype(np.float64),
        "pre_ee_pos": sample.pre_failure_robot.ee_pos.astype(np.float64),
        "pre_gripper_ctrl": np.array([sample.pre_failure_robot.gripper_ctrl], dtype=np.float64),
        "contact_positions": contact_positions,
        "contact_forces": contact_forces,
        "contact_geom_pairs": contact_geom_pairs,
        "contact_failure_id": contact_failure_id,
        "failure_modes": failure_modes,
        "failure_probs": failure_probs,
        "impacted_geom_ids": np.array(sample.all_impacted_geom_ids, dtype=np.int32),
        "task_id": np.array([sample.task_id]),
        "traj_id": np.array([sample.traj_id], dtype=np.int32),
        "traj_progress": np.array([sample.traj_progress], dtype=np.float32),
        "seed": np.array([sample.seed], dtype=np.int32),
        "pre_qvel_norm": np.array(
            [np.linalg.norm(sample.pre_failure_robot.qvel)], dtype=np.float64
        ),
    }

    if sample.pre_failure_depth is not None:
        arrays["pre_depth"] = sample.pre_failure_depth

    if sample.pre_target_qpos is not None:
        arrays["pre_target_qpos"] = np.asarray(sample.pre_target_qpos, dtype=np.float64)

    # Extra camera views (e.g., ee_cam)
    if sample.extra_camera_views:
        for cam_name, (rgb, depth) in sample.extra_camera_views.items():
            arrays[f"{cam_name}_rgb"] = rgb
            arrays[f"{cam_name}_depth"] = depth

    if sample.post_failure_rgb is not None:
        arrays["post_rgb"] = sample.post_failure_rgb

    np.savez_compressed(path, **arrays)


def load_sample_npz(path: str) -> dict:
    """Load an npz file and return the raw dict of arrays."""
    return dict(np.load(path, allow_pickle=True))


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _manifest_row(sample: DataSample, npz_filename: str) -> dict:
    return {
        "experiment_id": sample.experiment_id,
        "task_id": sample.task_id,
        "traj_id": sample.traj_id,
        "trajectory_file": os.path.basename(sample.trajectory_file),
        "seed": sample.seed,
        "traj_progress": round(float(sample.traj_progress), 4),
        "num_contacts": len(sample.aggregate_contacts),
        "num_failure_modes": len(sample.failure_results),
        "had_any_collision": any(fr.had_collision for fr in sample.failure_results),
        "impacted_geom_ids": ";".join(str(g) for g in sample.all_impacted_geom_ids),
        "pre_qvel_norm": round(float(np.linalg.norm(sample.pre_failure_robot.qvel)), 4),
        "npz_file": npz_filename,
    }


_MANIFEST_COLUMNS = [
    "experiment_id", "task_id", "traj_id", "trajectory_file", "seed",
    "traj_progress", "num_contacts", "num_failure_modes", "had_any_collision",
    "impacted_geom_ids", "pre_qvel_norm", "npz_file",
]


def write_manifest(rows: List[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def append_manifest(rows: List[dict], path: str) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Worker function (top-level for pickling)
# ---------------------------------------------------------------------------


def _run_and_save(args) -> Optional[dict]:
    """Run one trial and save its npz inside the worker.

    Returns only the small manifest-row dict (pickle-safe) to the parent.
    The previous design returned the full DataSample (images + 10k-100k
    contact rows, several MB) via the Pool's reply pipe; that deadlocked
    multiprocessing under load. Saving in-worker sidesteps it.

    `args` is a (ExperimentConfig, output_dir) tuple so this function stays
    a single-arg callable for pool.imap_unordered.
    """
    config, output_dir = args
    runner = ExperimentRunner(config)
    try:
        sample = runner.run()
        if sample is None:
            return None
        npz_name = f"{sample.experiment_id}.npz"
        npz_path = os.path.join(output_dir, npz_name)
        save_sample_npz(sample, npz_path)
        return _manifest_row(sample, npz_name)
    except Exception:
        logger.exception("Trial %s failed", config.experiment_id)
        return None
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# BatchExperimentManager
# ---------------------------------------------------------------------------


class BatchExperimentManager:
    """Generates configs, runs experiments in parallel, writes dataset to disk."""

    def __init__(self, output_dir: str, num_workers: int = 1):
        self.output_dir = output_dir
        self.num_workers = num_workers
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def generate_configs(
        self,
        scene_xml_path: str,
        robot_xml_path: str,
        trajectory_files: List[str],
        num_trials_per_trajectory: int = 50,
        failure_configs: Optional[List[FailureConfig]] = None,
        base_seed: int = 0,
        **extra_kwargs,
    ) -> List[ExperimentConfig]:
        """Generate ExperimentConfig instances across trajectories and seeds."""
        configs: List[ExperimentConfig] = []
        idx = 0
        for traj_file in trajectory_files:
            for trial in range(num_trials_per_trajectory):
                seed = base_seed + idx
                cfg = ExperimentConfig(
                    scene_xml_path=scene_xml_path,
                    robot_xml_path=robot_xml_path,
                    trajectory_file=traj_file,
                    seed=seed,
                    experiment_id=f"exp_{idx:05d}",
                    **extra_kwargs,
                )
                if failure_configs is not None:
                    cfg.failure_configs = failure_configs
                configs.append(cfg)
                idx += 1
        logger.info("Generated %d experiment configs", len(configs))
        return configs

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def run_batch(self, configs: List[ExperimentConfig]) -> str:
        """Run all configs (parallel or sequential) and write dataset.

        Returns path to manifest.csv.
        """
        return self.run_batch_chunked(configs, chunk_size=len(configs))

    def run_batch_chunked(
        self, configs: List[ExperimentConfig], chunk_size: int = 100,
        per_trial_timeout_s: float = 300.0,
    ) -> str:
        """Run in chunks, writing manifest after each chunk for crash resilience.

        Uses ``concurrent.futures.ProcessPoolExecutor`` rather than
        ``multiprocessing.Pool`` because the Pool API deadlocks the whole batch
        when any worker dies mid-task (MuJoCo crashes, OOM kills). Executor
        exposes per-future timeouts and a ``BrokenProcessPool`` signal, so we
        can skip a stuck trial and keep the batch moving.
        """
        manifest_path = os.path.join(self.output_dir, "manifest.csv")
        total = len(configs)

        for chunk_start in range(0, total, chunk_size):
            chunk = configs[chunk_start : chunk_start + chunk_size]
            logger.info(
                "Running chunk %d-%d / %d",
                chunk_start, chunk_start + len(chunk), total,
            )

            args = [(c, self.output_dir) for c in chunk]
            if self.num_workers <= 1:
                rows_raw = [_run_and_save(a) for a in args]
            else:
                rows_raw = _run_chunk_executor(
                    args, self.num_workers, per_trial_timeout_s)

            rows = [r for r in rows_raw if r is not None]
            if rows:
                append_manifest(rows, manifest_path)
                logger.info(
                    "Saved %d/%d samples in chunk, manifest updated",
                    len(rows), len(chunk),
                )

        return manifest_path


def _run_chunk_executor(args_list, num_workers, per_trial_timeout_s):
    """Run a chunk with a process-pool executor, resilient to worker death.

    Each trial is submitted as a separate future with ``per_trial_timeout_s``.
    If a worker dies or a trial exceeds the timeout, we log and move on — the
    executor is then torn down, a fresh one is created for the remaining
    trials. This matches the ``maxtasksperchild=1`` spirit (one fresh process
    per trial) while giving us robust failure handling.
    """
    results = [None] * len(args_list)
    ctx = multiprocessing.get_context("spawn")

    # Process trials in mini-batches of num_workers; each mini-batch uses a
    # fresh Executor so BrokenProcessPool from one dead worker doesn't poison
    # subsequent trials.
    i = 0
    while i < len(args_list):
        batch_end = min(i + num_workers, len(args_list))
        batch = args_list[i:batch_end]
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers, mp_context=ctx
            ) as ex:
                futures = {ex.submit(_run_and_save, a): (i + k)
                           for k, a in enumerate(batch)}
                for fut in concurrent.futures.as_completed(
                    futures, timeout=per_trial_timeout_s * 2,
                ):
                    idx = futures[fut]
                    cfg_id = batch[idx - i][0].experiment_id
                    try:
                        results[idx] = fut.result(timeout=per_trial_timeout_s)
                    except concurrent.futures.TimeoutError:
                        logger.warning("Trial %s timed out (>%.0fs) — skipped",
                                       cfg_id, per_trial_timeout_s)
                        results[idx] = None
                    except concurrent.futures.process.BrokenProcessPool:
                        logger.warning("Trial %s worker crashed — skipped", cfg_id)
                        results[idx] = None
                    except Exception as e:
                        logger.warning("Trial %s raised: %s", cfg_id, e)
                        results[idx] = None
        except concurrent.futures.process.BrokenProcessPool:
            # One worker died, executor is unusable — results for this batch
            # that haven't returned yet are lost. Continue with next batch.
            logger.warning("Pool broken during mini-batch %d-%d; skipping "
                           "unfinished trials in this batch", i, batch_end)
        except concurrent.futures.TimeoutError:
            logger.warning("Mini-batch %d-%d overall timeout; skipping rest",
                           i, batch_end)
        i = batch_end
    return results
