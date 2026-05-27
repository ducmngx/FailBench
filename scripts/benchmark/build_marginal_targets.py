"""Precompute per-(demo, bin) marginal contact heatmaps for the realistic benchmark.

Each (split, task, demo_key, bin_idx) group has N=3 sibling trials at the same
pre-failure moment with different sampled failure modes. The marginal heatmap
is the probability-weighted sum across siblings::

    marginal[group] = sum_i  failure_prob_i * heatmap_i

where ``heatmap_i = build_agentview_target(sibling_i)`` (mass-preserving
Gaussian-splat agentview projection).

This expresses "the expected contact mass landing in the agentview, under
failure-mode uncertainty" — the quantity a deployed planner actually wants
to predict.

By default the result is *un-normalised* (sum of prob × heatmap, where the
probs sum to ~0.4 in libero_spatial). With ``--normalize`` it becomes the
conditional expectation given that one of the sampled modes fires (probs
re-normalised to sum to 1.0 within each group).

Output: one HDF5 file per task::

    <out_dir>/<split>/<task>__marginals.h5
        /group_keys         (N_groups,) str   "demo_<k>__b<bin>"
        /target             (N_groups, H, W) f32
        /total_prob         (N_groups,)  f32  sum of failure_probs in group
        /n_siblings         (N_groups,)  i32
        /representative     (N_groups,) str   trial_id of the sibling whose
                                              pre-failure inputs are used at
                                              train time (all siblings share
                                              pre-failure state; this picks
                                              the lexically first one).

Usage::

    PYTHONPATH=. python -m scripts.benchmark.build_marginal_targets \\
        --v2_root /home/aaron/scratch/v2_ssd \\
        --splits libero_spatial \\
        --out_dir cache/marginal_targets_v2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import hdf5plugin  # noqa: F401
import h5py        # noqa: E402
import pandas as pd  # noqa: E402

from planner.risk.v2_store import V2Reader  # noqa: E402
from planner.risk.v2_targets import build_agentview_target  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_root", type=Path,
                    default=Path(os.environ.get("FAILBENCH_V2_ROOT",
                                                "/home/aaron/scratch/v2_ssd")))
    ap.add_argument("--splits", nargs="+", default=["libero_spatial"])
    ap.add_argument("--out_dir", type=Path,
                    default=REPO_ROOT / "cache" / "marginal_targets_v2")
    ap.add_argument("--normalize", action="store_true",
                    help="renormalise probs to sum to 1.0 within each group")
    ap.add_argument("--sigma_px", type=float, default=4.0)
    return ap.parse_args()


def build_one_task(manifest_subset: pd.DataFrame, h5_path: str,
                   *, normalize: bool, sigma_px: float):
    """Walk one task's siblings and return arrays for HDF5 output."""
    reader = V2Reader(h5_path)
    groups = manifest_subset.groupby(["demo_key", "bin_idx"], sort=True)

    out_keys, out_target, out_tp, out_n, out_repr = [], [], [], [], []
    for (demo_key, bin_idx), rows in groups:
        # Sum prob × heatmap across siblings.
        accum = None
        total_prob = 0.0
        siblings = sorted(rows["trial_id"].tolist())
        probs = dict(zip(rows["trial_id"], rows["failure_prob"]))
        for tid in siblings:
            trial = reader.read_trial(tid, keys={
                "contact_positions", "contact_force_world", "contact_forces",
                "cam_agentview_pos", "cam_agentview_mat0",
                "cam_agentview_fovy", "cam_agentview_size",
            })
            heatmap = build_agentview_target(trial, sigma_px=sigma_px,
                                             weighting="force").heatmap
            p = float(probs[tid])
            total_prob += p
            if accum is None:
                accum = p * heatmap
            else:
                accum += p * heatmap
        if normalize and total_prob > 0:
            accum = accum / total_prob
        out_keys.append(f"{demo_key}__b{int(bin_idx)}")
        out_target.append(accum.astype(np.float32))
        out_tp.append(np.float32(total_prob))
        out_n.append(np.int32(len(siblings)))
        out_repr.append(siblings[0])

    reader.close()
    return (
        np.asarray(out_keys, dtype=object),
        np.stack(out_target, axis=0),
        np.asarray(out_tp, dtype=np.float32),
        np.asarray(out_n, dtype=np.int32),
        np.asarray(out_repr, dtype=object),
    )


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"v2_root={args.v2_root}  out={args.out_dir}  "
          f"splits={args.splits}  normalize={args.normalize}")

    for split in args.splits:
        manifest_path = args.v2_root / split / "manifest.csv"
        if not manifest_path.exists():
            print(f"  skip {split}: no manifest at {manifest_path}")
            continue
        manifest = pd.read_csv(manifest_path)
        out_split = args.out_dir / split
        out_split.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {split}: {len(manifest)} trials, "
              f"{manifest.groupby(['task', 'demo_key', 'bin_idx']).ngroups} groups")

        for task, msub in manifest.groupby("task"):
            h5_path = msub["h5_path"].iloc[0]
            t0 = time.perf_counter()
            keys, target, total_prob, n_sib, repr_tid = build_one_task(
                msub.reset_index(drop=True), h5_path,
                normalize=args.normalize, sigma_px=args.sigma_px)
            dt = time.perf_counter() - t0

            out_file = out_split / f"{task}__marginals.h5"
            with h5py.File(out_file, "w") as f:
                f.create_dataset("group_keys",
                                 data=np.array([k.encode("utf-8") for k in keys]))
                f.create_dataset("target", data=target,
                                 **hdf5plugin.Blosc(cname="lz4"))
                f.create_dataset("total_prob", data=total_prob)
                f.create_dataset("n_siblings", data=n_sib)
                f.create_dataset("representative",
                                 data=np.array([s.encode("utf-8") for s in repr_tid]))
                f.attrs["split"] = split
                f.attrs["task"] = task
                f.attrs["normalize"] = bool(args.normalize)
                f.attrs["sigma_px"] = float(args.sigma_px)
            print(f"  {task[:60]:60s}  {len(keys):4d} groups  "
                  f"{dt:.1f}s  -> {out_file.name}")

    print("\ndone.")


if __name__ == "__main__":
    main()
