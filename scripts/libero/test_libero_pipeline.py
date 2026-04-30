#!/usr/bin/env python3
"""End-to-end smoke test for the LIBERO replay workflow.

Loads one demo from a given HDF5, runs a single trial with
``FailureMode.SINGLE_JOINT`` at progress=0.5, writes an npz, then validates
that the npz schema matches what the existing manager writes.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from planner.experiments.config import FailureConfig, FailureMode
from planner.experiments.libero.adapter import list_demos, load_demo
from planner.experiments.libero.runner import LiberoRunner, LiberoTrialConfig
from planner.experiments.manager import load_sample_npz, save_sample_npz


_REQUIRED_NPZ_KEYS = {
    "pre_rgb", "pre_qpos", "pre_qvel", "pre_ee_pos", "pre_gripper_ctrl",
    "contact_positions", "contact_forces", "contact_geom_pairs",
    "contact_failure_id", "failure_modes", "failure_probs",
    "impacted_geom_ids", "task_id", "traj_id", "traj_progress", "seed",
    "pre_qvel_norm",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--demo", default=None,
                   help="Demo key (default: first demo in the file)")
    p.add_argument("--fail_progress", type=float, default=0.5)
    args = p.parse_args()

    demo_key = args.demo or list_demos(args.hdf5)[0]
    demo = load_demo(args.hdf5, demo_key)
    print(f"Loaded demo {demo_key}: T={demo.arm_qpos.shape[0]} "
          f"task_id={demo.task_id}")

    cfg = LiberoTrialConfig(
        fail_progress=args.fail_progress,
        failure_configs=[FailureConfig(
            mode=FailureMode.SINGLE_JOINT, joint_names=["joint4"])],
        seed=0,
    )

    runner = LiberoRunner(demo, cfg)
    try:
        sample = runner.run(experiment_id="libero_smoke")
    finally:
        runner.close()

    assert sample is not None, "Runner returned no sample"
    assert len(sample.failure_results) == 1
    print(f"Trial OK: {len(sample.aggregate_contacts)} contacts, "
          f"impacted geoms = {sample.all_impacted_geom_ids}")

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "smoke.npz")
        save_sample_npz(sample, out)
        d = load_sample_npz(out)

    missing = _REQUIRED_NPZ_KEYS - set(d.keys())
    assert not missing, f"npz missing required keys: {missing}"
    assert d["pre_rgb"].ndim == 3
    assert d["pre_qpos"].shape == (7,)
    assert d["pre_qvel"].shape == (7,)
    assert d["pre_ee_pos"].shape == (3,)
    assert d["contact_positions"].ndim == 2 and d["contact_positions"].shape[1] == 3
    assert np.isclose(float(d["traj_progress"][0]), args.fail_progress, atol=0.01)
    print("npz schema OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
