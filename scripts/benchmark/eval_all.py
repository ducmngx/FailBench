"""Evaluate every benchmark checkpoint with the full metric suite.

Walks ``runs/bench/<run>/best.pt``, reconstructs each model + dataset from the
checkpoint's saved args, recomputes the demo-stratified val split, then
accumulates the metrics in :mod:`planner.risk.benchmark_metrics` across the
val set. Writes a single comparison table to ``runs/bench/_eval/``.

Usage:
    PYTHONPATH=. python -m scripts.benchmark.eval_all \\
        --v2_root /home/aaron/scratch/v2_ssd \\
        --dino_cache_root cache/dinov2_v2

Per-failure-mode breakdowns are written alongside the overall table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import hdf5plugin  # noqa: F401

from planner.risk.benchmark_dataset import (  # noqa: E402
    BenchmarkDataset, ModalityConfig, TargetConfig, demo_stratified_split,
)
from planner.risk import benchmark_metrics as M  # noqa: E402
from planner.risk.models import make_model  # noqa: E402
from scripts.benchmark.train_one import _collate, _to_device  # noqa: E402


FAILURE_MODES = ("GRIPPER_OPEN", "SLIPPERY_GRIP", "SINGLE_JOINT",
                 "MULTI_JOINT", "ALL_JOINTS")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=Path,
                    default=REPO_ROOT / "runs" / "bench",
                    help="walk for <run>/best.pt files under this directory")
    ap.add_argument("--out_dir", type=Path,
                    default=REPO_ROOT / "runs" / "bench" / "_eval")
    ap.add_argument("--v2_root", type=Path,
                    default=Path(os.environ.get("FAILBENCH_V2_ROOT",
                                                "/home/aaron/scratch/v2_ssd")))
    ap.add_argument("--dino_cache_root", type=Path,
                    default=REPO_ROOT / "cache" / "dinov2_v2")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", nargs="*", default=None,
                    help="optional subset of run-dir basenames to evaluate")
    ap.add_argument("--min_epoch", type=int, default=10,
                    help="skip checkpoints whose best_epoch < this (filters early-killed runs)")
    ap.add_argument("--with_latency", action="store_true",
                    help="also measure inference latency (small extra cost)")
    return ap.parse_args()


def _load_checkpoint(run_dir: Path) -> dict:
    """Read best.pt + metrics.json + args.json into a single descriptor."""
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no best.pt in {run_dir}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics_path = run_dir / "metrics.json"
    saved_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return {"ckpt": ckpt, "saved_metrics": saved_metrics, "run_dir": run_dir}


def _rebuild_dataset(args, ckpt_args: dict) -> BenchmarkDataset:
    """Recreate the BenchmarkDataset that this checkpoint trained on."""
    mods_dict = ckpt_args["modalities"] if "modalities" in ckpt_args else \
        ckpt_args.get("__mod_cfg_dict__", {})
    if not mods_dict:
        # Older runs store the dict under the canonical key.
        mods_dict = {}
        for k in ("state", "goal", "rgb", "depth", "dino", "failure_mode"):
            mods_dict[k] = bool(ckpt_args.get(k, False))
    mod_cfg = ModalityConfig(**mods_dict)
    T = int(ckpt_args.get("T", 8))
    splits = tuple(ckpt_args.get("splits", ("libero_spatial",)))
    return BenchmarkDataset(
        args.v2_root,
        modalities=mod_cfg,
        target_cfg=TargetConfig(sigma_px=ckpt_args.get("sigma_px", 4.0), log1p=True),
        splits=splits,
        dino_cache_root=args.dino_cache_root if mod_cfg.dino else None,
        use_window=(T == 8),
    ), mod_cfg, T, splits


def _rebuild_model(ckpt: dict, mod_cfg: ModalityConfig,
                   grid_hw: tuple, T: int, device: str) -> torch.nn.Module:
    cargs = ckpt["args"]
    name = cargs["model"]
    extra = {}
    if name == "unet":
        extra["temporal_mode"] = cargs.get("unet_temporal", "mean")
    model = make_model(name, modalities=mod_cfg, grid_hw=grid_hw, T=T, **extra)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def _eval_one(run_dir: Path, args) -> dict | None:
    print(f"\n--- {run_dir.name}")
    info = _load_checkpoint(run_dir)
    ckpt = info["ckpt"]; saved = info["saved_metrics"]
    cargs = ckpt["args"]

    best_ep = saved.get("best_epoch") or ckpt.get("epoch", 0)
    if best_ep < args.min_epoch:
        print(f"  skip (best_epoch={best_ep} < min_epoch={args.min_epoch})")
        return None

    ds, mod_cfg, T, splits = _rebuild_dataset(args, {
        "modalities": ckpt.get("modalities", {}),
        "T": cargs.get("T", 8),
        "splits": cargs.get("splits", ["libero_spatial"]),
        "sigma_px": cargs.get("sigma_px", 4.0),
    })
    print(f"  model={cargs['model']}  T={T}  splits={splits}  modalities={mod_cfg}")

    # Re-derive split to match training (same seed, val_frac).
    _, val_idx = demo_stratified_split(ds, val_frac=cargs.get("val_frac", 0.10),
                                       seed=cargs.get("seed", 0))
    print(f"  val_idx: {len(val_idx)} trials")

    grid_hw = tuple(ckpt["grid_hw"])
    model = _rebuild_model(ckpt, mod_cfg, grid_hw, T, args.device)
    n_params = sum(p.numel() for p in model.parameters())

    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, val_idx),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=_collate,
    )

    # Accumulate per-batch metrics & failure-mode-stratified predictions.
    # For mode breakdown we group trial-level metrics by manifest failure_mode.
    overall_preds, overall_targets = [], []
    per_mode_preds = {m: [] for m in FAILURE_MODES}
    per_mode_targets = {m: [] for m in FAILURE_MODES}
    # Use the dataset's underlying manifest to get failure_mode + suite per trial.
    failure_modes_per_trial = [ds._base._index[i].failure_mode for i in val_idx]
    splits_per_trial = [ds._base._index[i].split for i in val_idx]
    uniq_splits = sorted(set(splits_per_trial))
    per_split_preds = {s: [] for s in uniq_splits}
    per_split_targets = {s: [] for s in uniq_splits}

    t0 = time.perf_counter()
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, args.device)
            out = model(batch)
            pred = out["pred"].cpu().numpy()
            target = batch["target_log1p"].cpu().numpy()
            overall_preds.append(pred)
            overall_targets.append(target)
            # Group by failure mode + suite within this batch.
            for j in range(pred.shape[0]):
                fm = failure_modes_per_trial[seen + j]
                if fm in per_mode_preds:
                    per_mode_preds[fm].append(pred[j])
                    per_mode_targets[fm].append(target[j])
                sp = splits_per_trial[seen + j]
                per_split_preds[sp].append(pred[j])
                per_split_targets[sp].append(target[j])
            seen += pred.shape[0]
    dt = time.perf_counter() - t0

    P = np.concatenate(overall_preds, axis=0)
    T_ = np.concatenate(overall_targets, axis=0)
    overall = M.compute_all(P, T_)
    overall["n_val"] = int(seen)
    overall["n_params"] = int(n_params)
    overall["best_epoch"] = int(best_ep)
    overall["wall_seconds"] = round(dt, 1)
    overall["model"] = cargs["model"]
    overall["modality_tag"] = "+".join(
        k for k in ("state", "goal", "rgb", "depth", "dino", "failure_mode")
        if getattr(mod_cfg, k))
    overall["T"] = T
    overall["unet_temporal"] = cargs.get("unet_temporal", "")
    overall["run"] = run_dir.name

    # Per-failure-mode breakdown.
    per_mode = {}
    for fm in FAILURE_MODES:
        if not per_mode_preds[fm]:
            continue
        Pm = np.stack(per_mode_preds[fm], axis=0)
        Tm = np.stack(per_mode_targets[fm], axis=0)
        per_mode[fm] = M.compute_all(Pm, Tm) | {"n": int(Pm.shape[0])}

    # Per-suite breakdown (one populated cell for single-suite runs, all three
    # for pooled runs — recovers per-suite numbers from a single pooled model).
    per_split = {}
    for sp in uniq_splits:
        if not per_split_preds[sp]:
            continue
        Ps = np.stack(per_split_preds[sp], axis=0)
        Ts = np.stack(per_split_targets[sp], axis=0)
        per_split[sp] = M.compute_all(Ps, Ts) | {"n": int(Ps.shape[0])}

    print(f"  overall: mse={overall['weighted_mse_log1p']:.4f}  "
          f"iou={overall['soft_iou']:.3f}  kl={overall['symmetric_kl']:.3f}  "
          f"mass_ratio={overall['mass_total_ratio']:.3f}")

    if args.with_latency:
        sample = {k: v[:1] if torch.is_tensor(v) else [v[0]]
                  for k, v in next(iter(loader)).items()}
        sample = _to_device(sample, args.device)
        overall["latency_ms_b1"] = round(
            M.inference_latency_ms(model, sample, device=args.device), 2)

    if per_split:
        print("  per-suite: " + "  ".join(
            f"{sp.replace('libero_', '')}={per_split[sp]['weighted_mse_log1p']:.4f}"
            f"(n={per_split[sp]['n']})" for sp in sorted(per_split)))

    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return {"overall": overall, "per_mode": per_mode, "per_split": per_split}


def _write_tables(results: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not results:
        print("no checkpoints evaluated.")
        return

    # Overall CSV
    import csv
    cols = ["run", "model", "modality_tag", "T", "unet_temporal", "n_params", "n_val",
            "best_epoch", "weighted_mse_log1p", "rmse_raw", "soft_iou", "auprc_mass",
            "mass_total_ratio", "symmetric_kl", "wall_seconds"]
    if any("latency_ms_b1" in r["overall"] for r in results):
        cols.append("latency_ms_b1")
    # Sort *results* by overall MSE so per-mode and overall rows stay aligned.
    results = sorted(results, key=lambda r: r["overall"]["weighted_mse_log1p"])
    rows = [r["overall"] for r in results]
    with open(out_dir / "bench_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_dir / 'bench_table.csv'}")

    # Overall Markdown
    with open(out_dir / "bench_table.md", "w") as f:
        f.write("# Benchmark eval table\n\n")
        f.write("Sorted by `weighted_mse_log1p` ascending.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    cells.append(f"{v:.4f}" if c != "wall_seconds" else f"{v:.1f}")
                else:
                    cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")

        # Per-failure-mode section.
        f.write("\n\n## Per-failure-mode breakdown\n\n")
        f.write("`weighted_mse_log1p` only (full per-metric table in CSV).\n\n")
        all_modes = sorted({m for r in results for m in r["per_mode"].keys()})
        header = ["run"] + all_modes
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in results:
            row = [r["overall"]["run"]]
            for m in all_modes:
                if m in r["per_mode"]:
                    row.append(f"{r['per_mode'][m]['weighted_mse_log1p']:.4f}")
                else:
                    row.append("—")
            f.write("| " + " | ".join(row) + " |\n")

        # Per-suite section. Single-suite runs show one populated column; pooled
        # runs (trained on all three) show a fair per-suite breakdown for one model.
        f.write("\n\n## Per-suite breakdown\n\n")
        f.write("`weighted_mse_log1p` (val n in parens). Single-suite runs populate one "
                "column; pooled runs populate all three.\n\n")
        all_splits = sorted({s for r in results for s in r.get("per_split", {})})
        sheader = ["run"] + [s.replace("libero_", "") for s in all_splits]
        f.write("| " + " | ".join(sheader) + " |\n")
        f.write("|" + "|".join(["---"] * len(sheader)) + "|\n")
        for r in results:
            row = [r["overall"]["run"]]
            for s in all_splits:
                ps = r.get("per_split", {})
                if s in ps:
                    row.append(f"{ps[s]['weighted_mse_log1p']:.4f} (n={ps[s]['n']})")
                else:
                    row.append("—")
            f.write("| " + " | ".join(row) + " |\n")
    print(f"wrote {out_dir / 'bench_table.md'}")

    # Per-mode full JSON for downstream analysis.
    with open(out_dir / "bench_per_mode.json", "w") as f:
        json.dump([{"run": r["overall"]["run"], "per_mode": r["per_mode"]}
                   for r in results], f, indent=2)
    print(f"wrote {out_dir / 'bench_per_mode.json'}")

    # Per-suite full JSON for downstream analysis.
    with open(out_dir / "bench_per_split.json", "w") as f:
        json.dump([{"run": r["overall"]["run"], "per_split": r.get("per_split", {})}
                   for r in results], f, indent=2)
    print(f"wrote {out_dir / 'bench_per_split.json'}")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(d for d in args.runs_root.iterdir()
                      if d.is_dir() and d.name != "_eval" and (d / "best.pt").exists())
    if args.only:
        keep = set(args.only)
        run_dirs = [d for d in run_dirs if d.name in keep]
    print(f"discovered {len(run_dirs)} candidate runs")

    results = []
    for d in run_dirs:
        try:
            r = _eval_one(d, args)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  ERROR on {d.name}: {type(e).__name__}: {e}")
            continue

    _write_tables(results, args.out_dir)
    print(f"\ndone. evaluated {len(results)} / {len(run_dirs)} run dirs.")


if __name__ == "__main__":
    main()
