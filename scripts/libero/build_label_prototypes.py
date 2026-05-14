"""Prototype LIBERO contact-distribution labels on a small subset.

Round 2: computes per-trial labels (cam heatmap with depth channel + voxel
density), their log1p and sum-normalised variants, and per-pre-state
aggregates (mass-channel weighted mean over sibling failure trials).

All labels are force-weighted with the per-contact failure-mode prior
(``‖force_xyz‖ * failure_probs[contact_failure_id]``).

Sampling: picks ``--n_groups`` random (split, task, demo_key, bin_idx) groups
and loads every trial in each group. This is required for the aggregate
labels — each group's sibling trials are different sampled failure modes at
the same pre-failure state.

Usage:
  python -m scripts.libero.build_label_prototypes \\
      --manifest datasets/libero/v1/libero_spatial/manifest.csv \\
      --n_groups 8 \\
      --output datasets/libero/v1/label_prototypes.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Force EGL so headless runs (no display) work — same convention as the dataset
# generator. Must be set before mujoco is imported.
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
    build_voxel_density,
    contact_weights_force_prior,
    derive_bounds_from_contacts,
    log1p_label,
    sum_normalize_label,
)

DEFAULT_RAW_ROOT = REPO_ROOT / "datasets" / "libero" / "raw"


def _hdf5_path_for_row(row, raw_root: Path) -> Path:
    return raw_root / row["split"] / f"{row['task']}_demo.hdf5"


def _load_trial(npz_path: Path):
    d = np.load(npz_path)
    out = {
        "contact_positions": d["contact_positions"].astype(np.float32),
        "contact_forces": d["contact_forces"].astype(np.float32),
        "contact_failure_id": d["contact_failure_id"].astype(np.int32),
        "failure_probs": d["failure_probs"].astype(np.float32),
        "pre_qpos": d["pre_qpos"].astype(np.float64),
    }
    d.close()
    return out


def _select_rows(df: pd.DataFrame, n_groups: int, seed: int) -> pd.DataFrame:
    """Pick n_groups random (split, task, demo_key, bin_idx) groups and return
    every row belonging to those groups."""
    candidates = df[df["num_contacts"] > 0]
    if len(candidates) == 0:
        candidates = df
    keys = candidates[["split", "task", "demo_key", "bin_idx"]].drop_duplicates()
    if len(keys) < n_groups:
        n_groups = len(keys)
    sampled_keys = keys.sample(n=n_groups, random_state=seed).reset_index(drop=True)
    out = candidates.merge(sampled_keys, on=["split", "task", "demo_key", "bin_idx"])
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, nargs="+", required=True,
                    help="One or more manifest.csv paths. n_groups groups are "
                         "sampled from EACH manifest (so total groups = "
                         "n_manifests * n_groups).")
    ap.add_argument("--n_groups", type=int, default=8,
                    help="Number of (demo, bin_idx) groups to sample PER manifest; "
                         "every trial in each group is loaded so the aggregate "
                         "label has siblings to average over.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample_seed", type=int, default=0)
    ap.add_argument("--raw_root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--sigma_px", type=float, default=8.0)
    ap.add_argument("--voxel_cm", type=float, default=3.0)
    ap.add_argument("--sigma_vox", type=float, default=2.0)
    args = ap.parse_args()

    manifests = [p for p in args.manifest]
    for m in manifests:
        if not m.exists():
            sys.exit(f"manifest not found: {m}")

    # Sample n_groups groups from each manifest, concatenate, remember each row's
    # source manifest dir so we know where the trial npzs live.
    dfs = []
    for i, m in enumerate(manifests):
        df_full = pd.read_csv(m)
        df_part = _select_rows(df_full, n_groups=args.n_groups,
                               seed=args.sample_seed + i)
        df_part["_manifest_dir"] = str(m.parent)
        dfs.append(df_part)
    df = pd.concat(dfs, ignore_index=True)
    n = len(df)
    # Stable group order: groups appear in the order their first row was seen.
    group_keys_str = (df["split"] + "/" + df["task"] + "/" + df["demo_key"]
                      + "/bin" + df["bin_idx"].astype(str)).tolist()
    seen: dict[str, int] = {}
    group_ids = np.zeros(n, dtype=np.int32)
    for i, gk in enumerate(group_keys_str):
        if gk not in seen:
            seen[gk] = len(seen)
        group_ids[i] = seen[gk]
    n_groups = len(seen)
    group_keys = np.array(sorted(seen.keys(), key=lambda k: seen[k]), dtype=np.str_)
    print(f"selected {n} trials across {n_groups} groups")

    # Pass 1: load per-trial contact arrays.
    trials = []
    for _, row in df.iterrows():
        npz_path = Path(row["_manifest_dir"]) / row["npz_file"]
        if not npz_path.exists():
            sys.exit(f"missing npz: {npz_path}")
        t = _load_trial(npz_path)
        t["weights"] = contact_weights_force_prior(
            t["contact_forces"], t["contact_failure_id"], t["failure_probs"])
        # failure_probs is a (1,) array per the npz schema.
        t["failure_prob_scalar"] = float(row["failure_prob"])
        t["experiment_id"] = row["experiment_id"]
        t["task"] = row["task"]
        t["demo_key"] = row["demo_key"]
        t["split"] = row["split"]
        trials.append(t)
    print(f"loaded contact arrays; total contacts = "
          f"{sum(t['contact_positions'].shape[0] for t in trials)}")

    bounds = derive_bounds_from_contacts(
        [t["contact_positions"] for t in trials])
    print(f"voxel bounds (m): x={bounds[0]} y={bounds[1]} z={bounds[2]}")
    voxel_m = args.voxel_cm / 100.0
    nx = int(np.ceil((bounds[0, 1] - bounds[0, 0]) / voxel_m))
    ny = int(np.ceil((bounds[1, 1] - bounds[1, 0]) / voxel_m))
    nz = int(np.ceil((bounds[2, 1] - bounds[2, 0]) / voxel_m))
    print(f"voxel grid: ({nz}, {ny}, {nx})")

    H, W = args.height, args.width
    cam_agent = np.zeros((n, H, W, 2), dtype=np.float32)
    cam_eye = np.zeros((n, H, W, 2), dtype=np.float32)
    voxel = np.zeros((n, nz, ny, nx), dtype=np.float32)
    n_agent_in = np.zeros(n, dtype=np.int32)
    n_eye_in = np.zeros(n, dtype=np.int32)
    n_vox_in = np.zeros(n, dtype=np.int32)
    w_agent_in = np.zeros(n, dtype=np.float32)
    w_eye_in = np.zeros(n, dtype=np.float32)
    w_vox_in = np.zeros(n, dtype=np.float32)
    total_w = np.zeros(n, dtype=np.float32)

    t_start = time.time()
    by_demo: dict[tuple[str, str, str], list[int]] = {}
    for i, t in enumerate(trials):
        by_demo.setdefault((t["split"], t["task"], t["demo_key"]), []).append(i)

    for (split, task, demo_key), idxs in by_demo.items():
        row = df.iloc[idxs[0]]
        hdf5_path = _hdf5_path_for_row(row, args.raw_root)
        if not hdf5_path.exists():
            sys.exit(f"missing hdf5: {hdf5_path}")
        demo = load_demo(str(hdf5_path), demo_key=demo_key)
        mjcf_path = materialise_mjcf(demo.model_xml)
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        handles = resolve_model_handles(model)
        cam_a = handles.agentview_cam
        cam_e = handles.ee_cam
        if cam_a is None or cam_e is None:
            sys.exit(f"camera resolution failed for {demo_key}")

        data = mujoco.MjData(model)
        if demo.full_states is not None:
            data.qpos[:] = demo.full_states[0, 1:1 + model.nq]
        arm_adrs = handles.arm_qpos_adrs

        for i in idxs:
            t = trials[i]
            cp = t["contact_positions"]
            w = t["weights"]
            total_w[i] = float(w.sum())

            for k, adr in enumerate(arm_adrs):
                data.qpos[adr] = t["pre_qpos"][k]
            mujoco.mj_forward(model, data)

            ha, na, wa = build_camera_heatmap(
                model, data, cp, w, camera_name=cam_a,
                width=W, height=H, sigma_px=args.sigma_px, with_depth=True)
            cam_agent[i] = ha
            n_agent_in[i] = na
            w_agent_in[i] = wa

            he, ne, we = build_camera_heatmap(
                model, data, cp, w, camera_name=cam_e,
                width=W, height=H, sigma_px=args.sigma_px, with_depth=True)
            cam_eye[i] = he
            n_eye_in[i] = ne
            w_eye_in[i] = we

            vd, nv, wv = build_voxel_density(
                cp, w, bounds=bounds,
                voxel_cm=args.voxel_cm, sigma_vox=args.sigma_vox)
            voxel[i] = vd
            n_vox_in[i] = nv
            w_vox_in[i] = wv

        elapsed = time.time() - t_start
        print(f"  done demo {split}/{task}/{demo_key}  ({len(idxs)} trials, "
              f"{elapsed:.1f}s elapsed)")

    # Per-trial processed variants. Only mass channel for cam heatmaps.
    cam_agent_mass = cam_agent[..., 0]
    cam_eye_mass = cam_eye[..., 0]

    cam_agent_log1p = log1p_label(cam_agent_mass)
    cam_eye_log1p = log1p_label(cam_eye_mass)
    voxel_log1p = log1p_label(voxel)

    cam_agent_norm = np.zeros_like(cam_agent_mass)
    cam_eye_norm = np.zeros_like(cam_eye_mass)
    voxel_norm = np.zeros_like(voxel)
    for i in range(n):
        cam_agent_norm[i], _ = sum_normalize_label(cam_agent_mass[i])
        cam_eye_norm[i], _ = sum_normalize_label(cam_eye_mass[i])
        voxel_norm[i], _ = sum_normalize_label(voxel[i])

    # Per-group aggregates (mass channel only).
    group_n_trials = np.zeros(n_groups, dtype=np.int32)
    group_total_fp = np.zeros(n_groups, dtype=np.float32)
    group_cam_agent = np.zeros((n_groups, H, W), dtype=np.float32)
    group_cam_eye = np.zeros((n_groups, H, W), dtype=np.float32)
    group_voxel = np.zeros((n_groups, nz, ny, nx), dtype=np.float32)
    for g in range(n_groups):
        sel = np.where(group_ids == g)[0]
        if sel.size == 0:
            continue
        group_n_trials[g] = sel.size
        wts = np.array([trials[i]["failure_prob_scalar"] for i in sel], dtype=np.float32)
        group_total_fp[g] = float(wts.sum())
        group_cam_agent[g] = aggregate_labels(cam_agent_mass[sel], wts)
        group_cam_eye[g] = aggregate_labels(cam_eye_mass[sel], wts)
        group_voxel[g] = aggregate_labels(voxel[sel], wts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        experiment_id=np.array([t["experiment_id"] for t in trials], dtype=np.str_),
        # Per-trial raw labels (cam heatmaps now have depth channel).
        cam_heatmap_agentview=cam_agent,
        cam_heatmap_eye_in_hand=cam_eye,
        voxel_density=voxel,
        # Per-trial processed variants (mass-only for cams).
        cam_heatmap_agentview_log1p=cam_agent_log1p,
        cam_heatmap_eye_in_hand_log1p=cam_eye_log1p,
        voxel_density_log1p=voxel_log1p,
        cam_heatmap_agentview_normalized=cam_agent_norm,
        cam_heatmap_eye_in_hand_normalized=cam_eye_norm,
        voxel_density_normalized=voxel_norm,
        # Per-pre-state aggregates.
        group_ids=group_ids,
        group_keys=group_keys,
        group_n_trials=group_n_trials,
        group_total_failure_prob=group_total_fp,
        group_cam_heatmap_agentview=group_cam_agent,
        group_cam_heatmap_eye_in_hand=group_cam_eye,
        group_voxel_density=group_voxel,
        # Metadata + counters.
        voxel_bounds=bounds.astype(np.float32),
        voxel_cm=np.float32(args.voxel_cm),
        sigma_px=np.float32(args.sigma_px),
        sigma_vox=np.float32(args.sigma_vox),
        n_in_frame_agentview=n_agent_in,
        n_in_frame_eye_in_hand=n_eye_in,
        n_in_bounds_voxel=n_vox_in,
        weight_in_frame_agentview=w_agent_in,
        weight_in_frame_eye_in_hand=w_eye_in,
        weight_in_bounds_voxel=w_vox_in,
        total_weight=total_w,
        failure_prob_per_trial=np.array(
            [t["failure_prob_scalar"] for t in trials], dtype=np.float32),
    )
    print(f"\nwrote {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
