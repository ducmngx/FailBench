"""Build per-config 2D contact-density targets for FailBench risk modeling.

Walks `<dataset_root>/<scene>/<task>/manifest.csv`, computes a smoothed
prior-weighted contact heatmap per trial, and writes:
  - `<dataset_root>/<scene>/grid.json`           -- per-scene grid metadata
  - `<dataset_root>/<scene>/<task>/targets.npz`  -- per-task batched targets

Storage uses npz (not parquet) to avoid a pyarrow dependency. One npz per
task dir holds (N, ny, nx) float32 heatmaps stacked along axis 0 plus
parallel scalar arrays.

Usage:
  python scripts/build_density_targets.py --dataset datasets/v10
  python scripts/build_density_targets.py --dataset datasets/v10 \\
      --sigma_cm 2.0 --bin_cm 1.0 --pad_cm 5.0 --force-weighted
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import mujoco

from planner.risk.spatial import (
    SceneGrid, derive_scene_grid, compute_target, save_grid)


def _process_task(args) -> tuple[str, int, str]:
    """Build targets.npz for one (scene, task) directory. Returns (task_dir, n, status)."""
    scene, task_dir, grid_dict, sigma_cm, force_weighted, margin, force = args
    grid = SceneGrid.from_dict(grid_dict)
    task_dir = Path(task_dir)
    out = task_dir / "targets.npz"
    manifest_csv = task_dir / "manifest.csv"
    if not manifest_csv.exists():
        return (str(task_dir), 0, "skip: no manifest")

    if out.exists() and not force:
        # Idempotent: skip if every input npz is older than the output.
        out_mtime = out.stat().st_mtime
        df = pd.read_csv(manifest_csv)
        all_older = all((task_dir / row.npz_file).stat().st_mtime <= out_mtime
                        for row in df.itertuples()
                        if (task_dir / row.npz_file).exists())
        if all_older:
            return (str(task_dir), len(df), "skip: up-to-date")

    df = pd.read_csv(manifest_csv)
    n = len(df)
    H_stack = np.zeros((n, grid.ny, grid.nx), dtype=np.float32)
    H_force_stack = np.zeros_like(H_stack) if force_weighted else None
    n_above = np.zeros(n, dtype=np.int32)
    total_w = np.zeros(n, dtype=np.float32)
    exp_ids = []
    missing = 0

    for i, row in enumerate(df.itertuples()):
        npz_path = task_dir / row.npz_file
        exp_ids.append(row.experiment_id)
        if not npz_path.exists():
            missing += 1
            continue
        with np.load(npz_path) as d:
            H, na, wsum = compute_target(d, scene, grid,
                                         sigma_cm=sigma_cm,
                                         force_weighted=False,
                                         margin=margin)
            H_stack[i] = H
            n_above[i] = na
            total_w[i] = wsum
            if force_weighted:
                Hf, _, _ = compute_target(d, scene, grid,
                                          sigma_cm=sigma_cm,
                                          force_weighted=True,
                                          margin=margin)
                H_force_stack[i] = Hf

    save_kwargs = dict(
        experiment_id=np.array(exp_ids, dtype=np.str_),
        target_heatmap=H_stack,
        n_contacts_above_table=n_above,
        total_weight=total_w,
        sigma_cm=np.float32(sigma_cm),
        margin=np.float32(margin),
    )
    if force_weighted:
        save_kwargs["target_heatmap_force"] = H_force_stack

    np.savez_compressed(out, **save_kwargs)
    return (str(task_dir), n, f"wrote {out.name} (missing={missing})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("datasets/v10"))
    ap.add_argument("--scenes_dir", type=Path, default=Path("scenes"))
    ap.add_argument("--sigma_cm", type=float, default=2.0)
    ap.add_argument("--bin_cm", type=float, default=1.0)
    ap.add_argument("--pad_cm", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=-0.01)
    ap.add_argument("--force-weighted", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--scene", type=str, default=None,
                    help="Limit to one scene (defaults to all under --dataset).")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if targets.npz is up-to-date.")
    args = ap.parse_args()

    if not args.dataset.exists():
        sys.exit(f"dataset root not found: {args.dataset}")

    scenes = sorted(d.name for d in args.dataset.iterdir() if d.is_dir())
    if args.scene:
        scenes = [s for s in scenes if s == args.scene]
        if not scenes:
            sys.exit(f"scene not found under {args.dataset}: {args.scene}")

    # Derive + persist per-scene grids first (single-threaded, fast).
    grids: dict[str, SceneGrid] = {}
    for scene in scenes:
        scene_xml = args.scenes_dir / scene / "scene.xml"
        if not scene_xml.exists():
            print(f"[{scene}] skip: no {scene_xml}")
            continue
        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        grid = derive_scene_grid(model, scene, bin_cm=args.bin_cm, pad_cm=args.pad_cm)
        grids[scene] = grid
        save_grid(grid, args.dataset / scene / "grid.json")
        print(f"[{scene}] grid {grid.shape} extent x[{grid.x_min:.2f},{grid.x_max:.2f}] "
              f"y[{grid.y_min:.2f},{grid.y_max:.2f}]")

    # Build a job list: one (scene, task_dir) per task directory.
    jobs = []
    for scene, grid in grids.items():
        scene_dir = args.dataset / scene
        for task_dir in sorted(scene_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            if not (task_dir / "manifest.csv").exists():
                continue
            jobs.append((scene, str(task_dir), grid.to_dict(),
                         args.sigma_cm, args.force_weighted, args.margin, args.force))

    print(f"\nbuilding {len(jobs)} tasks across {len(grids)} scenes "
          f"(workers={args.workers}, sigma={args.sigma_cm}cm, bin={args.bin_cm}cm)\n")

    if args.workers <= 1:
        results = [_process_task(j) for j in jobs]
    else:
        with mp.Pool(args.workers) as pool:
            results = pool.map(_process_task, jobs)

    for path, n, status in results:
        rel = Path(path).relative_to(args.dataset.parent)
        print(f"  {rel}  n={n:<5d}  {status}")


if __name__ == "__main__":
    main()
