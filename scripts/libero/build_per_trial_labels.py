"""Build per-trial (not aggregated) labels at half resolution.

Sibling of ``build_full_labels.py``. Difference:

* ``build_full_labels.py`` aggregates 3 sibling trials per
  ``(task, demo_key, bin_idx)`` group into one label.
* This script keeps **one label per trial** (45 000 per split) and stores
  ``failure_mode_id`` + ``is_holding`` as per-trial metadata so the model
  can condition on them. Storage at half-resolution (240×320, float16) to
  keep total disk usage tractable.

Output: ``datasets/libero/v1/<split>/labels_per_trial.npz``. Schema:

    experiment_id            (N,)         U-string
    task                     (N,)         U-string
    demo_key                 (N,)         U-string
    bin_idx                  (N,)         int32
    target_mass              (N, h, w)    float16   log1p of per-trial mass
    target_depth             (N, h, w)    float16   per-trial mean depth (m)
    target_mass_total        (N,)         float32   sum of target_mass per trial
    failure_mode_id          (N,)         int32     index into failure_mode_names
    failure_mode_names       (5,)         U-string  vocab
    is_holding               (N,)         int32     from holding.csv sidecar
    group_id                 (N,)         int32     index into a group dedup table
    group_keys               (G,)         U-string  for cross-reference with aggregated labels

The original trial npzs, ``labels.npz``, and ``holding.csv`` are NEVER
modified.

Usage:
    python -m scripts.libero.build_per_trial_labels --split libero_spatial --workers 12
    python -m scripts.libero.build_per_trial_labels --split all --workers 12
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
    build_camera_heatmap,
    contact_weights_force_prior,
)

V1_ROOT = REPO_ROOT / "datasets" / "libero" / "v1"
RAW_ROOT = REPO_ROOT / "datasets" / "libero" / "raw"
DEFAULT_STAGING = V1_ROOT / "_per_trial_labels_staging"

# Build labels at native 480x640 then downsample to 240x320 by 2x2 box-mean.
NATIVE_H, NATIVE_W = 480, 640
H_HALF, W_HALF = 240, 320
SIGMA_PX_DEFAULT = 8.0

FAILURE_MODE_NAMES = np.array(
    ["GRIPPER_OPEN", "SINGLE_JOINT", "MULTI_JOINT", "ALL_JOINTS", "SLIPPERY_GRIP"],
    dtype=np.str_,
)
_FM_LOOKUP = {n: i for i, n in enumerate(FAILURE_MODE_NAMES)}


def _box_mean_half(arr: np.ndarray) -> np.ndarray:
    """2x2 box mean from (..., 480, 640) → (..., 240, 320)."""
    h, w = arr.shape[-2], arr.shape[-1]
    assert h == NATIVE_H and w == NATIVE_W, (h, w)
    out = arr.reshape(*arr.shape[:-2], H_HALF, 2, W_HALF, 2).mean(axis=(-3, -1))
    return out


def _process_demo(args):
    """Per-trial labels for all trials of one (split, task, demo_key)."""
    (split, task, demo_key, manifest_path, holding_csv, raw_root,
     staging_dir, sigma_px) = args

    out_path = Path(staging_dir) / f"{task}__{demo_key}.npz"
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path), -1

    df_all = pd.read_csv(manifest_path)
    df = df_all[(df_all["task"] == task) & (df_all["demo_key"] == demo_key)]
    if df.empty:
        return None, "no rows"

    h_all = pd.read_csv(holding_csv).set_index("experiment_id")["is_holding"]

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
    rows_out = []
    for _, row in df.iterrows():
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
            width=NATIVE_W, height=NATIVE_H, sigma_px=sigma_px,
            with_depth=True,
        )
        mass_full = heatmap[..., 0]
        depth_full = heatmap[..., 1]
        mass_half = _box_mean_half(mass_full)
        depth_half = _box_mean_half(depth_full)
        log_mass = np.log1p(mass_half).astype(np.float16)
        depth16 = depth_half.astype(np.float16)

        fm_name = str(row["failure_mode"])
        fm_id = _FM_LOOKUP.get(fm_name, -1)

        rows_out.append(dict(
            experiment_id=row["experiment_id"],
            task=task,
            demo_key=demo_key,
            bin_idx=int(row["bin_idx"]),
            mass=log_mass,
            depth=depth16,
            mass_total=float(log_mass.astype(np.float32).sum()),
            failure_mode_id=int(fm_id),
            is_holding=int(h_all.get(row["experiment_id"], 0)),
            group_key=f"{split}/{task}/{demo_key}/bin{int(row['bin_idx'])}",
        ))

    if not rows_out:
        return None, "no trials produced"

    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        experiment_id=np.array([r["experiment_id"] for r in rows_out], dtype=np.str_),
        task=np.array([r["task"] for r in rows_out], dtype=np.str_),
        demo_key=np.array([r["demo_key"] for r in rows_out], dtype=np.str_),
        bin_idx=np.array([r["bin_idx"] for r in rows_out], dtype=np.int32),
        target_mass=np.stack([r["mass"] for r in rows_out], axis=0),
        target_depth=np.stack([r["depth"] for r in rows_out], axis=0),
        target_mass_total=np.array([r["mass_total"] for r in rows_out], dtype=np.float32),
        failure_mode_id=np.array([r["failure_mode_id"] for r in rows_out], dtype=np.int32),
        is_holding=np.array([r["is_holding"] for r in rows_out], dtype=np.int32),
        group_key=np.array([r["group_key"] for r in rows_out], dtype=np.str_),
    )
    os.replace(tmp_path, out_path)
    return str(out_path), len(rows_out)


def _concat_partials(partial_files, out_path, sigma_px):
    """Concatenate per-demo partial npzs into one labels_per_trial.npz."""
    print(f"  concatenating {len(partial_files)} partial files...")
    keys = ["experiment_id", "task", "demo_key", "bin_idx",
            "target_mass", "target_depth", "target_mass_total",
            "failure_mode_id", "is_holding", "group_key"]
    parts = {k: [] for k in keys}
    for pf in partial_files:
        with np.load(pf) as p:
            for k in keys:
                parts[k].append(p[k])
    final = {}
    for k, v in parts.items():
        if v[0].ndim >= 2:
            final[k] = np.concatenate(v, axis=0)
        else:
            final[k] = np.concatenate(v)
    # Build group_id from unique group_keys.
    unique_keys, group_id = np.unique(final["group_key"], return_inverse=True)
    final["group_id"] = group_id.astype(np.int32)
    final["group_keys"] = unique_keys.astype(np.str_)
    del final["group_key"]
    final["failure_mode_names"] = FAILURE_MODE_NAMES
    final["sigma_px"] = np.float32(sigma_px)
    final["height"] = np.int32(H_HALF)
    final["width"] = np.int32(W_HALF)
    np.savez_compressed(out_path, **final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all",
                    help="libero_spatial | libero_goal | libero_object | all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process the first N demos (debug).")
    ap.add_argument("--staging_dir", type=Path, default=DEFAULT_STAGING)
    ap.add_argument("--sigma_px", type=float, default=SIGMA_PX_DEFAULT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep_staging", action="store_true")
    args = ap.parse_args()

    splits = (["libero_spatial", "libero_goal", "libero_object"]
              if args.split == "all" else [args.split])

    for split in splits:
        print(f"\n=== split: {split} ===")
        manifest_path = V1_ROOT / split / "manifest.csv"
        holding_csv = V1_ROOT / split / "holding.csv"
        if not manifest_path.exists() or not holding_csv.exists():
            print(f"  skip: missing manifest.csv or holding.csv in {V1_ROOT/split}")
            continue
        out_path = V1_ROOT / split / "labels_per_trial.npz"
        if out_path.exists() and not args.force:
            print(f"  skip: {out_path} exists (use --force)")
            continue

        df = pd.read_csv(manifest_path)
        demos = df[["task", "demo_key"]].drop_duplicates().values.tolist()
        if args.limit > 0:
            demos = demos[: args.limit]
        n_trials_total = len(df) if args.limit == 0 else None
        print(f"  {len(demos)} unique demos, {len(df)} total trials")

        staging_split = args.staging_dir / split
        staging_split.mkdir(parents=True, exist_ok=True)

        jobs = [(split, task, demo, str(manifest_path), str(holding_csv),
                 str(RAW_ROOT), str(staging_split), args.sigma_px)
                for task, demo in demos]

        t0 = time.time()
        ok = 0
        cached = 0
        n_trials_seen = 0
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
                    n_trials_seen += n
                if ok % 25 == 0 or ok == len(jobs):
                    el = time.time() - t0
                    rate = ok / max(el, 1e-6)
                    eta = (len(jobs) - ok) / max(rate, 1e-6)
                    print(f"    {ok}/{len(jobs)} demos  "
                          f"({rate:.2f}/s, ETA {eta/60:.1f}min, "
                          f"cached={cached}, trials_seen={n_trials_seen})")

        partial_files = sorted(staging_split.glob("*.npz"))
        _concat_partials(partial_files, out_path, args.sigma_px)
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
