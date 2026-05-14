"""Scale aggregated agentview cam_heatmap labels to ALL groups in a LIBERO split.

For each (task, demo_key, bin_idx) group in the manifest, compute the
failure-prob-weighted mean of the agentview-projected contact heatmap (mass +
depth channels) across the group's sibling trials, then store log1p(mass) +
depth as float16 arrays.

Output: ``datasets/libero/v1/<split>/labels.npz``. The original trial npzs
and prototype label files are NEVER modified.

Schema of labels.npz (per split):

    group_keys              (G,) U200    "<split>/<task>/<demo_key>/bin<bin_idx>"
    task                    (G,) U200
    demo_key                (G,) U20
    bin_idx                 (G,) int32
    target_mass             (G, H, W) float16   log1p of aggregated mass
    target_depth            (G, H, W) float16   aggregated mean depth (metres)
    target_mass_total       (G,) float32        sum of target_mass per group
    group_n_trials          (G,) int32
    group_total_failure_prob (G,) float32
    sigma_px, height, width (scalars)

Usage:
    python -m scripts.libero.build_full_labels --split all --workers 8
    python -m scripts.libero.build_full_labels --split libero_spatial --limit 2  # smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402

from planner.experiments.libero.adapter import load_demo, materialise_mjcf  # noqa: E402
from planner.experiments.libero.naming import resolve_model_handles  # noqa: E402
from planner.risk.projection_labels import (  # noqa: E402
    aggregate_labels,
    build_camera_heatmap,
    contact_weights_force_prior,
)

V1_ROOT = REPO_ROOT / "datasets" / "libero" / "v1"
RAW_ROOT = REPO_ROOT / "datasets" / "libero" / "raw"
DEFAULT_STAGING = V1_ROOT / "_labels_staging"

H_DEFAULT = 480
W_DEFAULT = 640
SIGMA_PX_DEFAULT = 8.0


def _process_demo(args):
    """Aggregate labels for all bins of one (split, task, demo_key).

    Returns (out_path, n_groups) or (None, reason_str) on skip.
    Per-demo staging file is written atomically (.tmp → rename).
    """
    (split, task, demo_key, manifest_path, raw_root,
     staging_dir, height, width, sigma_px) = args

    out_path = Path(staging_dir) / f"{task}__{demo_key}.npz"
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path), -1  # cached

    df_all = pd.read_csv(manifest_path)
    df = df_all[(df_all["task"] == task) & (df_all["demo_key"] == demo_key)]
    if df.empty:
        return None, "no rows"

    hdf5_path = Path(raw_root) / split / f"{task}_demo.hdf5"
    if not hdf5_path.exists():
        return None, f"hdf5 missing: {hdf5_path}"

    demo = load_demo(str(hdf5_path), demo_key=demo_key)
    mjcf_path = materialise_mjcf(demo.model_xml)
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    handles = resolve_model_handles(model)
    cam_a = handles.agentview_cam
    if cam_a is None:
        return None, f"no agentview cam for {demo_key}"

    data = mujoco.MjData(model)
    if demo.full_states is not None:
        data.qpos[:] = demo.full_states[0, 1:1 + model.nq]
    arm_adrs = handles.arm_qpos_adrs

    manifest_dir = Path(manifest_path).parent
    groups_out = []
    for bin_idx, bin_df in df.groupby("bin_idx"):
        masses, depths, weights = [], [], []
        for _, row in bin_df.iterrows():
            npz_path = manifest_dir / row["npz_file"]
            if not npz_path.exists():
                continue
            with np.load(npz_path) as d:
                cp = d["contact_positions"].astype(np.float32)
                cf = d["contact_forces"].astype(np.float32)
                cfid = d["contact_failure_id"].astype(np.int32)
                fprobs = d["failure_probs"].astype(np.float32)
                pre_qpos = d["pre_qpos"].astype(np.float64)
            cw = contact_weights_force_prior(cf, cfid, fprobs)

            for k, adr in enumerate(arm_adrs):
                data.qpos[adr] = pre_qpos[k]
            mujoco.mj_forward(model, data)

            heatmap, _, _ = build_camera_heatmap(
                model, data, cp, cw, camera_name=cam_a,
                width=width, height=height, sigma_px=sigma_px,
                with_depth=True,
            )
            masses.append(heatmap[..., 0])
            depths.append(heatmap[..., 1])
            weights.append(float(row["failure_prob"]))

        if not masses:
            continue
        masses_arr = np.stack(masses, axis=0)
        depths_arr = np.stack(depths, axis=0)
        wts = np.asarray(weights, dtype=np.float32)
        agg_mass = aggregate_labels(masses_arr, wts)
        agg_depth = aggregate_labels(depths_arr, wts)
        log_mass = np.log1p(agg_mass).astype(np.float16)
        depth16 = agg_depth.astype(np.float16)

        groups_out.append(dict(
            key=f"{split}/{task}/{demo_key}/bin{int(bin_idx)}",
            task=task, demo_key=demo_key, bin_idx=int(bin_idx),
            mass=log_mass, depth=depth16,
            mass_total=float(log_mass.astype(np.float32).sum()),
            n_trials=int(len(bin_df)),
            total_fp=float(wts.sum()),
        ))

    if not groups_out:
        return None, "no groups produced"

    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        group_keys=np.array([g["key"] for g in groups_out], dtype=np.str_),
        task=np.array([g["task"] for g in groups_out], dtype=np.str_),
        demo_key=np.array([g["demo_key"] for g in groups_out], dtype=np.str_),
        bin_idx=np.array([g["bin_idx"] for g in groups_out], dtype=np.int32),
        target_mass=np.stack([g["mass"] for g in groups_out], axis=0),
        target_depth=np.stack([g["depth"] for g in groups_out], axis=0),
        target_mass_total=np.array([g["mass_total"] for g in groups_out], dtype=np.float32),
        group_n_trials=np.array([g["n_trials"] for g in groups_out], dtype=np.int32),
        group_total_failure_prob=np.array([g["total_fp"] for g in groups_out], dtype=np.float32),
    )
    os.replace(tmp_path, out_path)
    return str(out_path), len(groups_out)


def _concat_partials(partial_files, out_path, sigma_px, height, width):
    """Concatenate per-demo partial npzs into a single labels.npz."""
    print(f"  concatenating {len(partial_files)} partial files...")
    parts = {k: [] for k in [
        "group_keys", "task", "demo_key", "bin_idx",
        "target_mass", "target_depth",
        "target_mass_total", "group_n_trials", "group_total_failure_prob",
    ]}
    for pf in partial_files:
        with np.load(pf) as p:
            for k in parts:
                parts[k].append(p[k])
    final = {}
    for k, v in parts.items():
        if v[0].ndim >= 2:
            final[k] = np.concatenate(v, axis=0)
        else:
            final[k] = np.concatenate(v)
    final["sigma_px"] = np.float32(sigma_px)
    final["height"] = np.int32(height)
    final["width"] = np.int32(width)
    np.savez_compressed(out_path, **final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all",
                    help="libero_spatial | libero_goal | libero_object | all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process the first N demos (debug).")
    ap.add_argument("--staging_dir", type=Path, default=DEFAULT_STAGING)
    ap.add_argument("--width", type=int, default=W_DEFAULT)
    ap.add_argument("--height", type=int, default=H_DEFAULT)
    ap.add_argument("--sigma_px", type=float, default=SIGMA_PX_DEFAULT)
    ap.add_argument("--force", action="store_true",
                    help="Re-process even if final labels.npz exists.")
    ap.add_argument("--keep_staging", action="store_true",
                    help="Keep per-demo staging npzs after concat (default: delete).")
    args = ap.parse_args()

    splits = (["libero_spatial", "libero_goal", "libero_object"]
              if args.split == "all" else [args.split])

    for split in splits:
        print(f"\n=== split: {split} ===")
        manifest_path = V1_ROOT / split / "manifest.csv"
        if not manifest_path.exists():
            print(f"  skip: no manifest at {manifest_path}")
            continue

        out_path = V1_ROOT / split / "labels.npz"
        if out_path.exists() and not args.force:
            print(f"  skip: {out_path} already exists (use --force)")
            continue

        df = pd.read_csv(manifest_path)
        demos = df[["task", "demo_key"]].drop_duplicates().values.tolist()
        if args.limit > 0:
            demos = demos[: args.limit]
        n_groups_total = df.groupby(["task", "demo_key", "bin_idx"]).ngroups
        print(f"  {len(demos)} unique demos, {len(df)} trials, "
              f"{n_groups_total} groups to aggregate")

        staging_split = args.staging_dir / split
        staging_split.mkdir(parents=True, exist_ok=True)

        jobs = [(split, task, demo, str(manifest_path), str(RAW_ROOT),
                 str(staging_split), args.height, args.width, args.sigma_px)
                for task, demo in demos]

        t0 = time.time()
        ok = 0
        cached = 0
        n_groups_seen = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_process_demo, j) for j in jobs]
            for fut in as_completed(futures):
                path, n = fut.result()
                if path is None:
                    print(f"  WARN: skip ({n})")
                    continue
                ok += 1
                if n == -1:
                    cached += 1
                else:
                    n_groups_seen += n
                if ok % 25 == 0 or ok == len(jobs):
                    el = time.time() - t0
                    rate = ok / max(el, 1e-6)
                    eta = (len(jobs) - ok) / max(rate, 1e-6)
                    print(f"    {ok}/{len(jobs)} demos done  "
                          f"({rate:.2f}/s, ETA {eta/60:.1f}min, "
                          f"cached={cached}, groups_seen={n_groups_seen})")

        partial_files = sorted(staging_split.glob("*.npz"))
        _concat_partials(partial_files, out_path,
                         args.sigma_px, args.height, args.width)
        size_mb = out_path.stat().st_size / 1e6
        elapsed = time.time() - t0
        print(f"  → wrote {out_path} ({size_mb:.0f} MB, {elapsed/60:.1f}min total)")

        if not args.keep_staging:
            for pf in partial_files:
                pf.unlink()
            try:
                staging_split.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    main()
