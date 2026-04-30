#!/usr/bin/env python3
"""Run a single LIBERO demo as one FailBench trial → npz.

Example::

    python scripts/libero/run_libero_trial.py \
        --hdf5 datasets/libero/raw/libero_spatial/<task>.hdf5 \
        --demo demo_0 --fail_progress 0.5 --mode SINGLE_JOINT \
        --joints joint4 \
        --output datasets/libero/v1/exp_libero_0.npz
"""

from __future__ import annotations

import argparse
import os

from planner.experiments.config import FailureConfig, FailureMode
from planner.experiments.libero.adapter import load_demo
from planner.experiments.libero.runner import LiberoRunner, LiberoTrialConfig
from planner.experiments.manager import save_sample_npz


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--demo", default="demo_0")
    p.add_argument("--fail_progress", type=float, default=0.5)
    p.add_argument("--mode", default="SINGLE_JOINT",
                   choices=[m.name for m in FailureMode])
    p.add_argument("--joints", default="joint4",
                   help="Comma-separated joint names for SINGLE/MULTI_JOINT modes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resistance", choices=["none", "gravcomp_pd"], default="none",
                   help="Active resistance for healthy joints during settle "
                        "(default: none)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    demo = load_demo(args.hdf5, args.demo)

    joint_names = [s.strip() for s in args.joints.split(",") if s.strip()]
    fc = FailureConfig(mode=FailureMode[args.mode], joint_names=joint_names)

    cfg = LiberoTrialConfig(
        fail_progress=args.fail_progress,
        failure_configs=[fc],
        seed=args.seed,
        resistance_mode=args.resistance,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    runner = LiberoRunner(demo, cfg)
    try:
        sample = runner.run(experiment_id=os.path.splitext(os.path.basename(args.output))[0])
    finally:
        runner.close()

    if sample is None:
        print("Trial returned no sample.")
        return 1
    save_sample_npz(sample, args.output)
    print(f"Wrote {args.output} — "
          f"{len(sample.aggregate_contacts)} contacts, "
          f"{len(sample.failure_results)} failure modes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
