"""Train one (model, modalities, seed) cell of the v2 benchmark.

Usage:
    PYTHONPATH=. python -m scripts.benchmark.train_one \
        --model mlp --modalities state \
        --epochs 10 --batch_size 64

Writes ``runs/bench/<model>__<mods>__seed<N>__<ts>/{best.pt, metrics.json, args.json}``.
Designed to be model-agnostic so the same trainer drives MLP → ConvDec →
UNet → Transformer → Diffusion (the diffusion case will add its own loss
adapter; deterministic models share this entry point).

The target is the log1p of the agentview heatmap from
:func:`planner.risk.v2_targets.build_agentview_target`. Training loss is
masked MSE with foreground re-weighting ``w_pix = 1 + alpha * (target > 0)``
to counter the ~99% zero-pixel imbalance.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import hdf5plugin  # noqa: F401, must register filter before h5py opens files

from planner.risk.benchmark_dataset import (  # noqa: E402
    BenchmarkDataset, MarginalBenchmarkDataset, ModalityConfig, TargetConfig,
    demo_stratified_split, task_held_out_split,
)
from planner.risk.models import make_model  # noqa: E402

ALL_MODALITIES = ("state", "goal", "rgb", "depth", "dino", "failure_mode", "failure_joints")


def parse_modalities(s: str) -> ModalityConfig:
    """Comma-separated → ModalityConfig.

    The returned config has *only* the listed modalities enabled; every flag
    defaults to False so callers get exactly what they asked for (state must
    be listed explicitly to be on).
    """
    if not s:
        return ModalityConfig(state=False)
    keys = [k.strip() for k in s.split(",") if k.strip()]
    unknown = [k for k in keys if k not in ALL_MODALITIES]
    if unknown:
        raise ValueError(f"unknown modalities {unknown}; options: {ALL_MODALITIES}")
    kwargs = {m: False for m in ALL_MODALITIES}
    for k in keys:
        kwargs[k] = True
    return ModalityConfig(**kwargs)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_root", type=Path,
                    default=Path(os.environ.get("FAILBENCH_V2_ROOT",
                                                "/media/aaron/F/failbench/libero/v2")))
    ap.add_argument("--splits", nargs="+",
                    default=["libero_spatial", "libero_object", "libero_goal"])
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--modalities", default="state",
                    help=f"comma-separated subset of {ALL_MODALITIES}")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=10.0,
                    help="foreground-pixel weight multiplier in masked MSE")
    ap.add_argument("--mass_total_weight", type=float, default=0.0,
                    help="weight of MSE(sum(expm1(pred)), sum(target)) auxiliary loss; "
                         "0 disables (default). 0.01 is a reasonable start.")
    ap.add_argument("--sigma_px", type=float, default=4.0)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--warmup_epochs", type=int, default=1,
                    help="linear LR warmup epochs before cosine decay")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after N epochs without val improvement (0=off)")
    ap.add_argument("--unet_temporal", default="mean",
                    choices=["mean", "conv3d", "last", "late_fusion"],
                    help="UNet wrapper temporal mode (ignored for non-unet models)")
    ap.add_argument("--T", type=int, default=8, choices=[1, 8],
                    help="window length; T=1 uses the single pre-failure frame, "
                         "T=8 uses the full window (v2 default)")
    ap.add_argument("--dino_cache_root", type=Path, default=Path("cache/dinov2_v2"),
                    help="root dir for precomputed DINOv2 features (used when 'dino' "
                         "is in --modalities)")
    ap.add_argument("--target_form", default="per_trial",
                    choices=["per_trial", "marginal"],
                    help="per_trial = predict this trial's failure heatmap (oracle setting); "
                         "marginal = predict the mode-prior-weighted marginal target per "
                         "(demo, bin) group (realistic deploy setting)")
    ap.add_argument("--marginal_root", type=Path, default=Path("cache/marginal_targets_v2"),
                    help="root dir for precomputed marginal targets (used when --target_form=marginal)")
    ap.add_argument("--split_by", default="demo", choices=["demo", "task"],
                    help="demo: per-demo-stratified 90/10 split; "
                         "task: hold out --n_val_tasks whole tasks for val "
                         "(tests cross-task generalisation)")
    ap.add_argument("--n_val_tasks", type=int, default=3,
                    help="when --split_by=task, number of held-out tasks (default 3)")
    ap.add_argument("--max_trials", type=int, default=None,
                    help="subsample dataset to this many trials (smoke tests)")
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Dataloader collation
# ---------------------------------------------------------------------------

def _collate(batch):
    """Stack numpy fields into torch tensors; pass through scalars/strings."""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, np.ndarray):
            out[k] = torch.from_numpy(np.stack([b[k] for b in batch]))
        elif isinstance(v0, (np.floating, np.integer, float, int)):
            out[k] = torch.tensor([float(b[k]) for b in batch], dtype=torch.float32)
        else:
            out[k] = [b[k] for b in batch]
    return out


def _to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def weighted_mse_log1p(pred: torch.Tensor, target_log1p: torch.Tensor,
                       *, alpha: float = 10.0) -> torch.Tensor:
    """Per-pixel weighted MSE with foreground reweighting.

    ``w = 1 + alpha * (target_log1p > 0)``. With α=10 the ~1% non-zero pixels
    contribute ~10× more to the loss, so the model can't just predict zero.
    """
    fg = (target_log1p > 0).float()
    w = 1.0 + alpha * fg
    sq = (pred - target_log1p) ** 2
    return (w * sq).sum() / w.sum()


def mass_total_mse(pred_log1p: torch.Tensor,
                   target_mass_total: torch.Tensor) -> torch.Tensor:
    """Per-trial MSE on log1p(total mass).

    Mass totals are O(10²-10³), so MSE on raw totals would dominate the main
    loss (which is O(0.1)) by ~6 orders of magnitude. log1p both totals to
    bring them onto a scale comparable to the per-pixel log1p target.
    Penalises both over- and under-prediction of total mass roughly
    multiplicatively (in raw units).
    """
    pred_total = torch.expm1(pred_log1p.clamp(min=0)).flatten(start_dim=1).sum(dim=1)
    return torch.nn.functional.mse_loss(
        torch.log1p(pred_total), torch.log1p(target_mass_total))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    mod_cfg = parse_modalities(args.modalities)
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        mod_tag = "+".join(k for k in ALL_MODALITIES if getattr(mod_cfg, k))
        tf_tag = "marg" if args.target_form == "marginal" else "pertrial"
        split_tag = f"splitT{args.n_val_tasks}" if args.split_by == "task" else "splitD"
        args.output_dir = (REPO_ROOT / "runs" / "bench" /
                           f"{args.model}__{mod_tag}__T{args.T}__{tf_tag}__{split_tag}__seed{args.seed}__{ts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing -> {args.output_dir}")

    # ---- Dataset + split ----
    if args.target_form == "marginal":
        if mod_cfg.failure_mode or mod_cfg.failure_joints:
            print("[warn] marginal target_form ignores failure_mode/failure_joints "
                  "modalities — disabling them for the realistic setting.")
        ds = MarginalBenchmarkDataset(
            args.v2_root,
            marginal_root=args.marginal_root,
            modalities=mod_cfg,
            splits=tuple(args.splits),
            use_window=(args.T == 8),
            dino_cache_root=args.dino_cache_root if mod_cfg.dino else None,
        )
        mod_cfg = ds.modalities   # MarginalBenchmarkDataset may have stripped oracle flags
    else:
        ds = BenchmarkDataset(
            args.v2_root,
            modalities=mod_cfg,
            target_cfg=TargetConfig(sigma_px=args.sigma_px, log1p=True),
            splits=tuple(args.splits),
            dino_cache_root=args.dino_cache_root if mod_cfg.dino else None,
            use_window=(args.T == 8),
        )
    val_task_names = None
    if args.split_by == "task":
        train_idx, val_idx, val_task_names = task_held_out_split(
            ds, n_val_tasks=args.n_val_tasks, seed=args.seed)
        print(f"split_by=task: holding out {args.n_val_tasks} tasks:")
        for t in val_task_names:
            print(f"  - {t}")
    else:
        train_idx, val_idx = demo_stratified_split(ds, val_frac=args.val_frac, seed=args.seed)
    if args.max_trials is not None:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.choice(train_idx, size=min(args.max_trials, len(train_idx)),
                               replace=False)
        val_idx = rng.choice(val_idx, size=min(args.max_trials // 9 or 16, len(val_idx)),
                             replace=False)
    print(f"dataset: {len(ds)} trials  train={len(train_idx)}  val={len(val_idx)}")

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, collate_fn=_collate)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=_collate)

    # ---- Determine grid (HxW) from one sample ----
    sample = ds[int(train_idx[0])]
    grid_hw = sample["target"].shape
    print(f"target grid: {grid_hw}  modalities={mod_cfg}")

    # ---- Model ----
    model_kwargs = {}
    if args.model == "unet":
        model_kwargs["temporal_mode"] = args.unet_temporal
    model = make_model(args.model, modalities=mod_cfg, grid_hw=grid_hw,
                       T=args.T, **model_kwargs).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}  params={n_params/1e6:.2f}M  device={args.device}"
          + (f"  unet_temporal={args.unet_temporal}" if args.model == "unet" else ""))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    if args.warmup_epochs > 0 and args.warmup_epochs < args.epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1.0 / max(args.warmup_epochs * 100, 1),
            end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.epochs - args.warmup_epochs)
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # ---- Baseline: predict per-pixel train mean of log1p target.
    # Reported under both the model's training loss (weighted MSE) and an
    # unweighted MSE so we can compare both ways.
    print("computing train-mean baseline...")
    sum_y = torch.zeros(grid_hw)
    n = 0
    with torch.no_grad():
        for batch in train_loader:
            y = batch["target_log1p"]
            sum_y += y.sum(dim=0)
            n += y.shape[0]
    mean_log1p = (sum_y / max(n, 1)).to(args.device)
    base_weighted = []; base_unw_sum = 0.0; base_unw_n = 0
    with torch.no_grad():
        for batch in val_loader:
            y = batch["target_log1p"].to(args.device)
            pred = mean_log1p.unsqueeze(0).expand_as(y)
            base_weighted.append(weighted_mse_log1p(pred, y, alpha=args.alpha).item())
            base_unw_sum += float(((pred - y) ** 2).sum())
            base_unw_n += y.numel()
    baseline_mse = float(np.mean(base_weighted))
    baseline_mse_unw = base_unw_sum / max(base_unw_n, 1)
    print(f"baseline weighted MSE (alpha={args.alpha}) = {baseline_mse:.4f}   "
          f"unweighted = {baseline_mse_unw:.4f}")

    # ---- Training loop ----
    history = {"train_loss": [], "val_loss": [], "val_mse_raw": []}
    best_val = float("inf")
    best_path = args.output_dir / "best.pt"
    epochs_since_best = 0

    for ep in range(args.epochs):
        model.train()
        tr = []
        t0 = time.perf_counter()
        for batch in train_loader:
            batch = _to_device(batch, args.device)
            out = model(batch)
            loss = weighted_mse_log1p(out["pred"], batch["target_log1p"],
                                      alpha=args.alpha)
            if args.mass_total_weight > 0:
                loss = loss + args.mass_total_weight * mass_total_mse(
                    out["pred"], batch["target_mass"])
            optim.zero_grad(); loss.backward(); optim.step()
            tr.append(loss.item())
        sched.step()
        ep_time = time.perf_counter() - t0

        model.eval()
        vl = []; sq_raw_sum = 0.0; raw_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, args.device)
                out = model(batch)
                vl.append(weighted_mse_log1p(out["pred"], batch["target_log1p"],
                                             alpha=args.alpha).item())
                # raw-unit MSE for interpretability (expm1 the log1p prediction)
                pred_raw = torch.expm1(out["pred"].clamp(min=0))
                tgt_raw = batch["target"]
                sq_raw_sum += float(((pred_raw - tgt_raw) ** 2).sum())
                raw_n += tgt_raw.numel()

        tr_loss = float(np.mean(tr)); vl_loss = float(np.mean(vl))
        vl_raw = sq_raw_sum / max(raw_n, 1)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_mse_raw"].append(vl_raw)

        improved = vl_loss < best_val
        mark = "*" if improved else " "
        print(f" ep {ep+1:>3d}/{args.epochs}  train={tr_loss:.4f}  val={vl_loss:.4f}  "
              f"val_raw_mse={vl_raw:.4f}  baseline={baseline_mse:.4f}  "
              f"({ep_time:.1f}s) {mark}")

        if improved:
            best_val = vl_loss
            epochs_since_best = 0
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args) | {"output_dir": str(args.output_dir),
                                       "v2_root": str(args.v2_root)},
                "modalities": mod_cfg.__dict__,
                "grid_hw": grid_hw,
                "epoch": ep + 1,
                "history": history,
                "baseline_mse_log1p": baseline_mse,
            }, best_path)
        else:
            epochs_since_best += 1
            if args.patience > 0 and epochs_since_best >= args.patience:
                print(f"early stop: no improvement for {args.patience} epochs "
                      f"(best val={best_val:.4f} at ep {ep+1-epochs_since_best})")
                break

    metrics = {
        "best_val_mse_log1p": best_val,
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "baseline_val_mse_log1p_weighted": baseline_mse,
        "baseline_val_mse_log1p_unweighted": baseline_mse_unw,
        "final_val_mse_raw": history["val_mse_raw"][-1],
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_params": n_params,
        "model": args.model,
        "modalities": mod_cfg.__dict__,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    # Loss curve for quick triage.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        eps = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(eps, history["train_loss"], label="train")
        ax.plot(eps, history["val_loss"], label="val")
        ax.axhline(baseline_mse, color="gray", linestyle="--",
                   label=f"baseline={baseline_mse:.3f}")
        ax.set(xlabel="epoch", ylabel="weighted MSE_log1p",
               title=f"{args.model} | {','.join(k for k in ALL_MODALITIES if getattr(mod_cfg, k))}")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(args.output_dir / "val_curve.png", dpi=110)
        plt.close()
    except Exception as e:
        print(f"val_curve.png failed: {e}")

    print(f"\nbest val MSE_log1p = {best_val:.4f} (vs baseline {baseline_mse:.4f}) "
          f"at ep {metrics['best_epoch']}")


if __name__ == "__main__":
    main()
