#!/usr/bin/env python3
"""Patch a LIBERO split manifest by appending rows for npzs that are missing.

Useful after a kill-time loss where some trials' npzs landed on disk but their
manifest-append never ran. Reconstructs the missing rows from:

    1. The deterministic spec list (``make_trial_specs`` with the same params
       used for generation), keyed by ``experiment_id``.
    2. The npz file itself, for runtime fields (contact counts, traj_progress,
       traj_id, seed, pre_qvel_norm, impacted_geom_ids).

Usage::

    python -m scripts.libero.rebuild_manifest --split libero_spatial \
        --output_dir datasets/libero/v1
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

from planner.experiments.libero.dataset import (
    LIBERO_MANIFEST_COLUMNS,
    append_libero_manifest,
    make_trial_specs,
    task_name_from_hdf5,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RAW_ROOT = os.path.join(REPO_ROOT, "datasets", "libero", "raw")


def _load_existing_npz_files(manifest_path: str) -> set:
    if not os.path.exists(manifest_path):
        return set()
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        return {row["npz_file"] for row in reader}


def _row_from_npz(spec, npz_path: str) -> dict:
    """Build a manifest row by reading minimal fields from the npz."""
    with np.load(npz_path, allow_pickle=True) as z:
        n_contacts = int(z["contact_geom_pairs"].shape[0])
        traj_progress = float(z["traj_progress"][0])
        traj_id = int(z["traj_id"][0])
        seed = int(z["seed"][0])
        pre_qvel_norm = float(z["pre_qvel_norm"][0])
        impacted = z["impacted_geom_ids"].tolist()
    fc = spec.failure
    return {
        "experiment_id": spec.experiment_id,
        "split": spec.split,
        "task_id": spec.task_id,
        "task": spec.task,
        "demo_key": spec.demo_key,
        "traj_id": traj_id,
        "seed": seed,
        "bin_idx": spec.bin_idx,
        "traj_progress": round(traj_progress, 4),
        "failure_mode": fc.mode.name,
        "failure_joints": ",".join(fc.joint_names) if fc.joint_names else "",
        "failure_prob": round(float(fc.probability), 4),
        "num_contacts": n_contacts,
        "had_any_collision": n_contacts > 0,
        "impacted_geom_ids": ";".join(str(g) for g in impacted),
        "pre_qvel_norm": round(pre_qvel_norm, 4),
        "npz_file": os.path.basename(npz_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True,
                   choices=["libero_spatial", "libero_object", "libero_goal"])
    p.add_argument("--raw_root", default=DEFAULT_RAW_ROOT)
    p.add_argument("--output_dir", default=os.path.join(
        REPO_ROOT, "datasets", "libero", "v1"))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--progress_bins", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--progress_lo", type=float, default=0.05)
    p.add_argument("--progress_hi", type=float, default=0.9)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    split_dir = os.path.join(args.output_dir, args.split)
    manifest_path = os.path.join(split_dir, "manifest.csv")

    hdf5_files = sorted(glob.glob(
        os.path.join(args.raw_root, args.split, "*.hdf5")))
    if not hdf5_files:
        print(f"No HDF5 files under {args.raw_root}/{args.split}", file=sys.stderr)
        return 1

    specs = make_trial_specs(
        hdf5_files=hdf5_files,
        split=args.split,
        seeds=args.seeds,
        progress_bins=args.progress_bins,
        base_seed=args.base_seed,
        progress_lo=args.progress_lo,
        progress_hi=args.progress_hi,
    )
    spec_by_id = {s.experiment_id: s for s in specs}
    print(f"Built {len(specs)} expected specs for {args.split}")

    on_disk = {os.path.basename(p) for p in
                glob.glob(os.path.join(split_dir, "*.npz"))}
    in_manifest = _load_existing_npz_files(manifest_path)
    missing = sorted(on_disk - in_manifest)
    extra = sorted(in_manifest - on_disk)

    print(f"npzs on disk:    {len(on_disk)}")
    print(f"manifest rows:   {len(in_manifest)}")
    print(f"missing rows:    {len(missing)}")
    print(f"manifest-only:   {len(extra)} (rows pointing at non-existent npzs)")

    if not missing:
        print("Nothing to do.")
        return 0

    rows = []
    for npz_name in missing:
        eid = os.path.splitext(npz_name)[0]
        spec = spec_by_id.get(eid)
        if spec is None:
            print(f"WARN: no spec for {eid} — skipping", file=sys.stderr)
            continue
        rows.append(_row_from_npz(spec, os.path.join(split_dir, npz_name)))

    print(f"Built {len(rows)} new manifest rows.")
    if args.dry_run:
        print("Dry run — not writing.")
        return 0

    append_libero_manifest(rows, manifest_path)
    print(f"Appended {len(rows)} rows to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
