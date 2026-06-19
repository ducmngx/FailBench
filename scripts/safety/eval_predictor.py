#!/usr/bin/env python3
"""Per-checkpoint quantitative eval — produces Table 1 numbers for the paper.

Iterates every ``ContactPredictor``-loadable checkpoint, runs each over the
same held-out val subset, and reports:

    median / mean / p95 argmax pixel distance
    ≤3 px hit rate
    ≥50 px fail rate
    gatekeeper precision / recall / F1 (gated models only)
    per-source (libero/robocasa) breakdown
    per-failure-mode breakdown

Outputs ``out/predictor_eval/results.parquet`` (one row per
(checkpoint, source, mode, trial)) and prints a markdown summary table.

The val split is reproduced deterministically by calling
:func:`planner.risk.benchmark_dataset.demo_stratified_split` with ``seed=0``
on a freshly-built ``BenchmarkDataset`` — matches what the trainer used.

Run::

    /home/aaron/miniconda3/envs/failbench_env/bin/python -u \\
        -m scripts.safety.eval_predictor \\
        --libero_root /media/aaron/FAILBENCH/failbench/libero/v2 \\
        --ckpt_root  notebooks/model_playground/cluster_download \\
        --n_trials   500 \\
        --out_root   out/predictor_eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Checkpoints + arch hints to scan. Add new ones here when a fresh
# variant comes in.
#
# ``corpus`` records what the model was trained on. RoboCasa eval rows for
# LIBERO-only-trained models surface OOD performance, NOT in-distribution
# spatial quality — they're reported separately so the headline ablation
# remains apples-to-apples.
KNOWN_CKPTS = [
    # (relative path, optional arch_override, corpus)
    ("06162026/unet_state_rgb/epoch5.pt",
        "HeatmapUNetLegacy(state_dim=156, base_ch=16)", "pooled"),
    ("06162026/ufilm_state_rgb/epoch10.pt", None, "pooled"),
    ("06162026/ucoord_state_rgb/epoch10.pt", None, "pooled"),
    ("06162026/unet_state_rgb_gated/epoch10_heat0.0650.pt", None, "libero"),
    ("06172026/dualgated_state_rgb/best_ep08_val0.0648.pt", None, "libero"),
]


@dataclass
class TrialResult:
    ckpt_name: str
    corpus: str          # "pooled" or "libero" — training corpus of this ckpt
    source: str          # "libero" or "robocasa" — eval-trial source
    task: str
    trial_id: str
    mode: str
    n_contacts: int
    argmax_d_px: float
    pred_max: float
    gate_prob: float
    has_contact: bool
    pred_has_contact: bool
    in_distribution: bool   # True iff (corpus == "pooled") or (corpus == source)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--libero_root", type=Path,
                    default=Path("/media/aaron/FAILBENCH/failbench/libero/v2"))
    ap.add_argument("--robocasa_root", type=Path,
                    default=Path("/media/aaron/FAILBENCH/failbench/robocasa/v2"),
                    help="Optional. Set to None to skip RoboCasa")
    ap.add_argument("--ckpt_root", type=Path,
                    default=Path("notebooks/model_playground/cluster_download"))
    ap.add_argument("--n_trials", type=int, default=500,
                    help="Total val trials to evaluate per checkpoint")
    ap.add_argument("--out_root", type=Path, default=Path("out/predictor_eval"))
    ap.add_argument("--device", default=None,
                    help="cpu / cuda / cuda:0 (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import torch

    from planner.risk.inference import ContactPredictor
    from planner.risk.v2_targets import build_agentview_target

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # ------------------------------------------------------------------
    # 1. Build the val index by reproducing demo_stratified_split via
    #    BenchmarkDataset. Pulls in h5py + dataset_v2 machinery.
    # ------------------------------------------------------------------
    print("\nbuilding val index (demo-stratified split)...", flush=True)
    from planner.risk.benchmark_dataset import (
        BenchmarkDataset, ModalityConfig, TargetConfig, demo_stratified_split)
    from planner.risk.dataset_v2 import V2Source

    sources = [V2Source.libero(args.libero_root)]
    if args.robocasa_root and Path(args.robocasa_root).exists():
        sources.append(V2Source.robocasa(args.robocasa_root))

    dataset = BenchmarkDataset(
        sources=sources,
        modalities=ModalityConfig(
            state=True, goal=False, rgb=True, depth=False,
            failure_mode=True, failure_joints=True,
        ),
        target_cfg=TargetConfig(sigma_px=4.0, log1p=True,
                                 filter_baseline=False),
        use_window=True,
    )
    print(f"  dataset size: {len(dataset):,}")
    train_idx, val_idx = demo_stratified_split(dataset, val_frac=0.10,
                                                seed=args.seed)
    print(f"  val: {len(val_idx):,} trials")

    # Random subsample
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(val_idx, size=min(args.n_trials, len(val_idx)),
                      replace=False)
    pick.sort()
    print(f"  evaluating {len(pick):,} sampled val trials")

    # ------------------------------------------------------------------
    # 2. For each checkpoint: load, evaluate over the picked val indices,
    #    aggregate metrics.
    # ------------------------------------------------------------------
    rows: list[TrialResult] = []
    summary_rows = []
    for rel, override, corpus in KNOWN_CKPTS:
        full = args.ckpt_root / rel
        if not full.exists():
            print(f"\nSKIP {rel}: missing")
            continue
        print(f"\n=== {rel}  [corpus={corpus}] ===")
        t_load = time.time()
        cp = ContactPredictor.from_checkpoint(full, device=device,
                                               arch_override=override)
        print(f"  loaded ({time.time()-t_load:.1f}s) arch={cp.meta.arch} "
              f"ep={cp.meta.epoch} val_heat={cp.meta.val_heat}")

        t0 = time.time()
        per_ckpt_rows: list[TrialResult] = []
        for i, idx in enumerate(pick):
            sample = dataset[int(idx)]
            mode_oh = sample["failure_mode"]
            mode_idx = int(np.argmax(mode_oh))
            from planner.risk.inference import _FAILURE_MODES
            mode_name = _FAILURE_MODES[mode_idx]
            joints_oh = sample.get("failure_joints", np.zeros(7, np.float32))
            joints = [j + 1 for j, v in enumerate(joints_oh) if v > 0.5]

            # rgb_window comes out of BenchmarkDataset as float32 in [0, 1]
            # (the dataset's _rgb_window_chw divides by 255). The model was
            # trained on this exact format — build_inputs does
            # batch.float() which leaves [0,1] unchanged. So pass through.
            rgb_window = sample["rgb_window"]
            state_window = sample["state_window"]

            heat, gprob = cp.predict(rgb_window, state_window, mode_name, joints)

            # GT target: rebuild from contacts. Use target.heatmap not target_log1p
            # so we can compare to predictor's log-space output.
            target_np = sample.get("target_log1p", sample.get("target"))
            if target_np is None:
                # Build on the fly
                trial_for_target = {}
                continue
            target_np = np.asarray(target_np)

            # argmax distance
            if target_np.max() > 1e-6 and heat.max() > 1e-6:
                H, W = target_np.shape
                ty, tx = np.unravel_index(target_np.argmax(), target_np.shape)
                py, px = np.unravel_index(heat.argmax(), heat.shape)
                d = float(np.hypot(ty - py, tx - px))
            else:
                d = float("nan")

            has_contact = target_np.max() > 1e-6
            pred_has = gprob > 0.5 if not np.isnan(gprob) else heat.max() > 0.3

            src_name = sample.get("source", "libero")
            in_dist = (corpus == "pooled") or (corpus == src_name)
            row = TrialResult(
                ckpt_name=rel,
                corpus=corpus,
                source=src_name,
                task=sample["task"],
                trial_id=sample["trial_id"],
                mode=mode_name,
                n_contacts=int(target_np.sum() > 0),  # binary "has any contact"
                argmax_d_px=d,
                pred_max=float(heat.max()),
                gate_prob=float(gprob) if not np.isnan(gprob) else float("nan"),
                has_contact=bool(has_contact),
                pred_has_contact=bool(pred_has),
                in_distribution=in_dist,
            )
            rows.append(row)
            per_ckpt_rows.append(row)

            if (i + 1) % 50 == 0 or i == len(pick) - 1:
                dt = time.time() - t0
                print(f"  {i+1:4d}/{len(pick)}  "
                      f"({(i+1)/dt:.1f} trial/s  ETA "
                      f"{(len(pick)-i-1)/((i+1)/dt)/60:.1f} min)",
                      flush=True)

        # HEADLINE comparison set: LIBERO val trials only. Every checkpoint
        # was trained on LIBERO (whether alone or pooled), so this is the
        # cleanest apples-to-apples ablation across architectures.
        libero_rows = [r for r in per_ckpt_rows if r.source == "libero"]
        ds_positive = [r for r in libero_rows if r.has_contact]
        argmax_d = np.array([r.argmax_d_px for r in ds_positive
                              if not np.isnan(r.argmax_d_px)])
        if argmax_d.size == 0:
            print("  WARNING: no LIBERO positive trials — skip")
            continue

        # Gatekeeper metrics (on LIBERO val rows)
        y_true = np.array([r.has_contact for r in libero_rows])
        y_pred = np.array([r.pred_has_contact for r in libero_rows])
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        f1   = 2 * prec * rec / max(1e-9, prec + rec)
        acc  = (tp + tn) / max(1, tp + tn + fp + fn)

        # RoboCasa metrics. For pooled models this is in-distribution;
        # for LIBERO-only models it's OOD — distinguish on report.
        rc_rows = [r for r in per_ckpt_rows if r.source == "robocasa"]
        rc_positive = [r for r in rc_rows
                        if r.has_contact and not np.isnan(r.argmax_d_px)]
        rc_d = np.array([r.argmax_d_px for r in rc_positive])
        rc_med = float(np.median(rc_d)) if rc_d.size else float("nan")
        rc_mean = float(rc_d.mean()) if rc_d.size else float("nan")
        rc_fail = 100 * float((rc_d >= 50).mean()) if rc_d.size else float("nan")

        summary_rows.append({
            "ckpt": rel,
            "arch": cp.meta.arch,
            "corpus": corpus,
            "val_heat_ckpt": cp.meta.val_heat,
            "n_libero": len(libero_rows),
            "n_libero_pos": len(ds_positive),
            "libero_median": float(np.median(argmax_d)),
            "libero_mean": float(argmax_d.mean()),
            "libero_p95": float(np.percentile(argmax_d, 95)),
            "libero_hit_3px_%": 100 * float((argmax_d <= 3).mean()),
            "libero_fail_50px_%": 100 * float((argmax_d >= 50).mean()),
            "gate_acc": float(acc),
            "gate_prec": float(prec),
            "gate_recall": float(rec),
            "gate_f1": float(f1),
            "n_robocasa": len(rc_rows),
            "n_robocasa_pos": len(rc_positive),
            "robocasa_median": rc_med,
            "robocasa_mean": rc_mean,
            "robocasa_fail_50px_%": rc_fail,
            "robocasa_status": (
                "in-distribution" if corpus == "pooled" else "OOD"),
        })

    # ------------------------------------------------------------------
    # 3. Print + save summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TABLE 1 — ARCHITECTURE ABLATION on LIBERO val (apples-to-apples)")
    print("Every checkpoint was trained on LIBERO (alone or pooled with")
    print("RoboCasa). LIBERO-val numbers compare architectures on equal")
    print("footing.")
    print("=" * 100)
    print(f"{'Checkpoint':50s} {'corp':6s} {'n':>4s} {'med':>5s} {'mean':>5s} "
          f"{'p95':>5s} {'≤3%':>5s} {'≥50%':>5s} {'gateF1':>6s}")
    for s in summary_rows:
        print(f"{s['ckpt'][-50:]:50s} {s['corpus']:6s} "
              f"{s['n_libero_pos']:4d} "
              f"{s['libero_median']:5.1f} {s['libero_mean']:5.1f} "
              f"{s['libero_p95']:5.0f} "
              f"{s['libero_hit_3px_%']:5.1f} {s['libero_fail_50px_%']:5.1f} "
              f"{s['gate_f1']:6.3f}")

    print("\n" + "=" * 100)
    print("TABLE 2 — CROSS-CORPUS GENERALISATION (RoboCasa val)")
    print("Pooled models report in-distribution RoboCasa performance.")
    print("LIBERO-only models report OOD RoboCasa performance (not a fair")
    print("comparison against pooled — included for the robustness")
    print("discussion only).")
    print("=" * 100)
    print(f"{'Checkpoint':50s} {'corp':6s} {'status':15s} {'n':>4s} "
          f"{'med':>5s} {'mean':>5s} {'≥50%':>5s}")
    for s in summary_rows:
        if s["n_robocasa_pos"] == 0:
            continue
        print(f"{s['ckpt'][-50:]:50s} {s['corpus']:6s} "
              f"{s['robocasa_status']:15s} {s['n_robocasa_pos']:4d} "
              f"{s['robocasa_median']:5.1f} {s['robocasa_mean']:5.1f} "
              f"{s['robocasa_fail_50px_%']:5.1f}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Per-trial parquet (or csv if pyarrow absent)
    try:
        import pandas as pd
        df = pd.DataFrame([asdict(r) for r in rows])
        try:
            import pyarrow  # noqa: F401
            df.to_parquet(out_root / "results.parquet")
            print(f"\nwrote {out_root / 'results.parquet'}  ({len(df):,} rows)")
        except ImportError:
            df.to_csv(out_root / "results.csv", index=False)
            print(f"\nwrote {out_root / 'results.csv'}  ({len(df):,} rows)")

        # Per-source breakdown
        src = (df.groupby(['ckpt_name', 'source'])
                 .apply(lambda g: pd.Series({
                     'n': len(g),
                     'med': g[g['has_contact']]['argmax_d_px'].median(),
                     'mean': g[g['has_contact']]['argmax_d_px'].mean(),
                     'fail50%': 100 * (g[g['has_contact']]['argmax_d_px'] >= 50).mean(),
                 }))
                 .reset_index())
        src.to_csv(out_root / "per_source.csv", index=False)
        print(f"wrote {out_root / 'per_source.csv'}")
        print("\nper-source breakdown:")
        print(src.to_string(index=False))

        # Per-mode breakdown
        mod = (df[df['has_contact']]
                 .groupby(['ckpt_name', 'mode'])
                 .apply(lambda g: pd.Series({
                     'n': len(g),
                     'med': g['argmax_d_px'].median(),
                     'mean': g['argmax_d_px'].mean(),
                     'fail50%': 100 * (g['argmax_d_px'] >= 50).mean(),
                 }))
                 .reset_index())
        mod.to_csv(out_root / "per_mode.csv", index=False)
        print(f"\nwrote {out_root / 'per_mode.csv'}")
    except ImportError:
        with open(out_root / "results.json", "w") as f:
            json.dump([asdict(r) for r in rows], f, default=float)
        print(f"\nwrote {out_root / 'results.json'} (no pandas/pyarrow)")

    # Headline summary
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"wrote {out_root / 'summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
