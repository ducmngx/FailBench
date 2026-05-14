"""Train ``LiberoHeatmapModel`` on the aggregated LIBERO labels.

Single-GPU training loop. Wires
``planner/risk/libero_dataset.py::LiberoLabelDataset`` to
``planner/risk/libero_model.py::LiberoHeatmapModel`` with the standard
composite loss documented in ``docs/libero_heatmap_model.md``.

Outputs (per run):
  runs/libero_<timestamp>/
    config.json   — every CLI flag + dataset stats + constant-baseline MSE
    metrics.csv   — per-epoch train/val loss components
    best.pt       — checkpoint with lowest val mass-MSE
    last.pt       — final epoch checkpoint

Smoke (1 epoch, 200 trials):
    python -m scripts.libero.train_libero_heatmap --limit_trials 200 --epochs 1

Full first run (libero_spatial only):
    python -m scripts.libero.train_libero_heatmap --epochs 50

Multi-split (requires memmap sidecars on disk):
    python -m scripts.libero.train_libero_heatmap \\
        --splits libero_spatial libero_goal libero_object --cache_memmap
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from planner.risk.libero_dataset import LiberoLabelDataset  # noqa: E402
from planner.risk.libero_model import LiberoHeatmapModel, count_parameters  # noqa: E402


# -------------------------------------------------------------------- helpers


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(batch, device, non_blocking=True):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def constant_baseline_mse(loader, device, key="target_mass"):
    """Constant-baseline MSE on a loader: predict the train mean.

    Returns (baseline_mse, baseline_mean_image_shape) measured against the
    same loader. Used as a reference floor in metrics.csv.
    """
    sum_img, count = None, 0
    for batch in loader:
        x = batch[key]
        if sum_img is None:
            sum_img = x.sum(dim=0)
        else:
            sum_img = sum_img + x.sum(dim=0)
        count += x.size(0)
    mean_img = (sum_img / count).to(device)
    sq, n = 0.0, 0
    for batch in loader:
        x = batch[key].to(device)
        sq += ((x - mean_img) ** 2).sum().item()
        n += x.numel()
    return sq / n


def run_epoch(model, loader, device, *, use_state, use_holding, use_failure_mode,
              optimizer=None, scaler=None,
              depth_w=0.1, total_w=0.01, log_every=0):
    """Run one epoch. If optimizer is None, evaluation mode."""
    train = optimizer is not None
    model.train(train)
    sums = {}
    n_items = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        batch = move(batch, device)
        state = batch["state"] if use_state else None
        is_holding = batch["is_holding"] if use_holding else None
        fail_oh = batch["failure_onehot"] if use_failure_mode else None
        with torch.set_grad_enabled(train):
            if scaler is not None and train:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(batch["rgb"], batch["depth"],
                                 state=state, is_holding=is_holding,
                                 failure_onehot=fail_oh)
                    loss, comps = LiberoHeatmapModel.loss(
                        pred, batch, depth_weight=depth_w,
                        mass_total_weight=total_w)
            else:
                pred = model(batch["rgb"], batch["depth"],
                             state=state, is_holding=is_holding,
                             failure_onehot=fail_oh)
                loss, comps = LiberoHeatmapModel.loss(
                    pred, batch, depth_weight=depth_w,
                    mass_total_weight=total_w)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bs = batch["rgb"].size(0)
        for k, v in comps.items():
            sums[k] = sums.get(k, 0.0) + v * bs
        n_items += bs

        if log_every and (i + 1) % log_every == 0:
            rate = n_items / max(time.time() - t0, 1e-6)
            avg = {k: v / n_items for k, v in sums.items()}
            print(f"      step {i+1}/{len(loader)} ({rate:.1f} it/s)  "
                  f"loss={avg['loss']:.4f} mass={avg['mass_loss']:.4f} "
                  f"depth={avg['depth_loss']:.4f}")
    return {k: v / max(n_items, 1) for k, v in sums.items()}


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    # Dataset
    ap.add_argument("--splits", nargs="+", default=["libero_spatial"])
    ap.add_argument("--cache_memmap", action="store_true",
                    help="Required when --splits has >1 entry.")
    ap.add_argument("--image_size", nargs=2, type=int, default=[240, 320],
                    help="(H W) for input + target resize.")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--limit_trials", type=int, default=0,
                    help="If >0, truncate train+val to this many trials (smoke).")
    # Model
    ap.add_argument("--use_state", action="store_true")
    ap.add_argument("--no_holding", action="store_true",
                    help="Disable the is_holding input (ablation control).")
    ap.add_argument("--holding_mode", default="bottleneck",
                    choices=["bottleneck", "film"],
                    help="How to inject is_holding into the network.")
    ap.add_argument("--per_trial", action="store_true",
                    help="Use per-trial labels (one item per trial, not per group).")
    ap.add_argument("--use_failure_mode", action="store_true",
                    help="Inject failure_mode one-hot as model input (requires --per_trial).")
    ap.add_argument("--no_pretrained", action="store_true",
                    help="Skip ImageNet weights (random init).")
    # Loss
    ap.add_argument("--depth_weight", type=float, default=0.1)
    ap.add_argument("--mass_total_weight", type=float, default=0.01)
    # Optim
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true",
                    help="Enable mixed-precision training (CUDA only).")
    # Misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--log_every", type=int, default=0,
                    help="Print intra-epoch progress every N steps (0=off).")
    args = ap.parse_args()

    if len(args.splits) > 1 and not args.cache_memmap:
        sys.exit("multi-split training requires --cache_memmap")

    seed_everything(args.seed)
    device = torch.device(args.device)

    # ----- dataset -----
    print(f"loading dataset (splits={args.splits} memmap={args.cache_memmap}) ...")
    t0 = time.time()
    use_holding = not args.no_holding
    if args.use_failure_mode and not args.per_trial:
        sys.exit("--use_failure_mode requires --per_trial (per-failure label needed)")
    ds = LiberoLabelDataset(
        splits=args.splits,
        image_size=tuple(args.image_size),
        cache_memmap=args.cache_memmap,
        return_meta=False,
        use_holding=use_holding,
        per_trial=args.per_trial,
    )
    print(f"  {len(ds)} trials, {ds.n_groups} groups, native={ds.native_hw}, "
          f"resized to {tuple(args.image_size)}  ({time.time()-t0:.1f}s)")

    train_idx, val_idx = ds.train_val_split(val_frac=args.val_frac, seed=args.seed)
    if args.limit_trials > 0:
        # Smoke mode: take an even slice across both splits for shape coverage.
        train_idx = train_idx[: int(args.limit_trials * (1 - args.val_frac))]
        val_idx = val_idx[: max(1, int(args.limit_trials * args.val_frac))]
    print(f"  train={len(train_idx)}  val={len(val_idx)}")

    train_loader = DataLoader(
        Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(
        Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0)

    # ----- constant baseline for reference -----
    print("computing constant-baseline val MSE (reference floor) ...")
    baseline_mse = constant_baseline_mse(val_loader, device, key="target_mass")
    print(f"  baseline val mass-MSE = {baseline_mse:.4f}")

    # ----- model -----
    model = LiberoHeatmapModel(
        use_state=args.use_state,
        use_holding=use_holding,
        holding_mode=args.holding_mode,
        use_failure_mode=args.use_failure_mode,
        n_failure_modes=getattr(ds, "n_failure_modes", 5),
        pretrained=not args.no_pretrained,
    ).to(device)
    n_trn, n_tot = count_parameters(model)
    print(f"model: {n_trn/1e6:.2f}M trainable / {n_tot/1e6:.2f}M total  "
          f"(use_state={args.use_state}, use_holding={use_holding}, "
          f"holding_mode={args.holding_mode}, "
          f"pretrained={not args.no_pretrained})")

    # ----- optim -----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    # ----- output dir -----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (REPO_ROOT / "runs" / f"libero_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir}")
    config = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_groups": int(ds.n_groups),
        "native_hw": list(ds.native_hw),
        "baseline_val_mass_mse": baseline_mse,
        "param_count_trainable": int(n_trn),
        "started_at": ts,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "lr",
            "train_loss", "train_mass", "train_depth", "train_total",
            "val_loss", "val_mass", "val_depth", "val_total",
            "vs_baseline",
        ])

    # ----- training loop -----
    best_val_mass = float("inf")
    for epoch in range(args.epochs):
        ep_start = time.time()
        print(f"\nepoch {epoch+1}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}")
        tr = run_epoch(model, train_loader, device,
                       use_state=args.use_state, use_holding=use_holding,
                       use_failure_mode=args.use_failure_mode,
                       optimizer=optimizer, scaler=scaler,
                       depth_w=args.depth_weight, total_w=args.mass_total_weight,
                       log_every=args.log_every)
        va = run_epoch(model, val_loader, device,
                       use_state=args.use_state, use_holding=use_holding,
                       use_failure_mode=args.use_failure_mode,
                       optimizer=None, scaler=None,
                       depth_w=args.depth_weight, total_w=args.mass_total_weight)
        scheduler.step()

        vs_base = va["mass_loss"] / max(baseline_mse, 1e-9)
        print(f"  train: loss={tr['loss']:.4f} mass={tr['mass_loss']:.4f} "
              f"depth={tr['depth_loss']:.4f}")
        print(f"  val:   loss={va['loss']:.4f} mass={va['mass_loss']:.4f} "
              f"depth={va['depth_loss']:.4f}  ({vs_base:.2%} of baseline) "
              f"[{time.time()-ep_start:.1f}s]")

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, scheduler.get_last_lr()[0],
                tr["loss"], tr["mass_loss"], tr["depth_loss"], tr["mass_total_loss"],
                va["loss"], va["mass_loss"], va["depth_loss"], va["mass_total_loss"],
                vs_base,
            ])

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_mass_loss": va["mass_loss"],
            "config": config,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if va["mass_loss"] < best_val_mass:
            best_val_mass = va["mass_loss"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  → new best (val mass={va['mass_loss']:.4f})")

    print(f"\ntraining done. best val mass-MSE = {best_val_mass:.4f}  "
          f"({best_val_mass/max(baseline_mse,1e-9):.2%} of baseline)")
    print(f"checkpoints: {out_dir}/best.pt, last.pt")


if __name__ == "__main__":
    main()
