"""Per-trial ``is_holding`` sidecar for the LIBERO v1 dataset.

For each trial in a split, derive a binary flag indicating whether the
robot's gripper is in *commanded-close* mode at the pre-failure instant
(``fail_idx`` = ``int(traj_progress * (T-1))``). This proxies "policy intends
to be holding an object" — sufficient for the model to disambiguate
gripper-open-while-holding vs gripper-open-empty failures.

Writes ``datasets/libero/v1/<split>/holding.csv`` with columns:

    experiment_id, is_holding, action_gripper, spread

The original trial npzs, ``labels.npz``, and ``manifest.csv`` are NEVER
modified. The sidecar is loaded by ``LiberoLabelDataset`` if present;
when absent the dataset falls back to ``is_holding = 0``.

Signal definition
-----------------
At the pre-failure step:
  * ``action_gripper`` = robosuite action's last component
    (positive ≈ commanded close, negative ≈ commanded open)
  * Take a ±2-frame window around ``fail_idx`` and is_holding=1 iff a
    majority of those frames have ``action_gripper > 0``.
  * ``spread`` = ``obs/gripper_states[fail_idx, 0] - obs/gripper_states[fail_idx, 1]``
    is recorded as a diagnostic but does NOT participate in the rule:
    LIBERO uses thin objects (bowls, cookies) for which the held-spread
    overlaps the closed-empty range, so spread is unreliable.

Usage
-----
    python -m scripts.libero.compute_holding_flag --split all
    python -m scripts.libero.compute_holding_flag --split libero_spatial --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = REPO_ROOT / "datasets" / "libero" / "v1"
RAW_ROOT = REPO_ROOT / "datasets" / "libero" / "raw"

WINDOW_HALF = 2   # ±2 frames around fail_idx for the vote


def _hdf5_path(split: str, task: str) -> Path:
    return RAW_ROOT / split / f"{task}_demo.hdf5"


def _fail_idx(T: int, traj_progress: float) -> int:
    return max(1, min(int(traj_progress * (T - 1)), T - 1))


def process_split(split: str, force: bool = False) -> Path | None:
    manifest_path = V1_ROOT / split / "manifest.csv"
    if not manifest_path.exists():
        print(f"  skip {split}: no manifest")
        return None
    out_path = V1_ROOT / split / "holding.csv"
    if out_path.exists() and not force:
        print(f"  skip {split}: {out_path.name} exists (use --force)")
        return out_path

    df = pd.read_csv(manifest_path)
    print(f"  {split}: {len(df)} trials, "
          f"{df.groupby(['task', 'demo_key']).ngroups} unique demos")

    rows = []
    # Group rows by (task, demo_key) so we open each HDF5 once.
    by_demo = df.groupby(["task", "demo_key"])
    n_done = 0
    for (task, demo_key), group_df in by_demo:
        h5 = _hdf5_path(split, task)
        if not h5.exists():
            print(f"    WARN: missing {h5}")
            continue
        with h5py.File(h5, "r") as f:
            actions = f[f"data/{demo_key}/actions"][:]                  # (T, 7)
            grip_states = f[f"data/{demo_key}/obs/gripper_states"][:]   # (T, 2)
        T = actions.shape[0]
        action_g = actions[:, -1]
        spread = grip_states[:, 0] - grip_states[:, 1]

        for r in group_df.itertuples():
            fi = _fail_idx(T, float(r.traj_progress))
            lo = max(0, fi - WINDOW_HALF)
            hi = min(T, fi + WINDOW_HALF + 1)
            close_frac = float((action_g[lo:hi] > 0.0).mean())
            is_holding = int(close_frac > 0.5)
            rows.append({
                "experiment_id": r.experiment_id,
                "is_holding": is_holding,
                "action_gripper": float(action_g[fi]),
                "spread": float(spread[fi]),
                "fail_idx": fi,
                "T": T,
            })
        n_done += 1
        if n_done % 50 == 0 or n_done == by_demo.ngroups:
            print(f"    demos done: {n_done}/{by_demo.ngroups}")

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"  → wrote {out_path} ({len(out)} rows)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all",
                    help="libero_spatial | libero_goal | libero_object | all")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if holding.csv exists.")
    args = ap.parse_args()

    splits = (["libero_spatial", "libero_goal", "libero_object"]
              if args.split == "all" else [args.split])
    for s in splits:
        print(f"\n=== {s} ===")
        process_split(s, force=args.force)


if __name__ == "__main__":
    main()
