"""Train the FailBench heatmap regressor demo on one scene.

Usage:
    python scripts/train_demo.py --scene scene_level2 --epochs 50

Writes model + plots under runs/heatmap_<scene>_<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import mujoco

from planner.risk.dataset import HeatmapDataset, DatasetStats, split_traj_keys
from planner.risk.model import HeatmapMLP
from planner.risk.spatial import (
    load_grid, entity_footprints, integrate_per_entity)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("datasets/v10"))
    ap.add_argument("--scenes_dir", type=Path, default=Path("scenes"))
    ap.add_argument("--scene", type=str, default="scene_level2")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = Path("runs") / f"heatmap_{args.scene}_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing artifacts to {args.output_dir}")

    # ---- Build the full dataset, split by traj_id, fit stats on train ----
    full = HeatmapDataset(args.dataset, args.scene)
    print(f"loaded {len(full)} configs, grid={full.grid_shape}")
    train_keys, val_keys = split_traj_keys(full.traj_keys, val_frac=args.val_frac, seed=args.seed)
    assert set(train_keys).isdisjoint(val_keys), "traj split leakage!"
    print(f"split: {len(train_keys)} train (task,traj) keys, {len(val_keys)} val keys")

    train_ds = HeatmapDataset(args.dataset, args.scene, traj_keys=train_keys)
    val_ds   = HeatmapDataset(args.dataset, args.scene, traj_keys=val_keys)
    print(f"  train configs: {len(train_ds)}   val configs: {len(val_ds)}")

    print("computing standardisation stats on train split...")
    stats = DatasetStats.fit(train_ds)
    train_ds.stats = stats
    val_ds.stats = stats

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ---- Model ----
    model = HeatmapMLP(in_dim=HeatmapDataset.INPUT_DIM,
                       grid_shape=full.grid_shape).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params; device={args.device}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # ---- Per-pixel mean baseline (for context) ----
    y_mean_t = torch.from_numpy(stats.y_mean).to(args.device)
    y_std_t  = torch.from_numpy(stats.y_std).to(args.device)
    baseline_mse_destd = float(((stats.y_std ** 2).mean()))   # MSE if predicting train mean (in original units)

    # ---- For Spearman entity check ----
    grid = load_grid(args.dataset / args.scene / "grid.json")
    model_xml = args.scenes_dir / args.scene / "scene.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(model_xml))
    fps = entity_footprints(mj_model, grid)
    obstacle_names = [n for n in fps if "obstacle" in n.lower()]
    print(f"entity Spearman tracked over {len(obstacle_names)} obstacles")

    # ---- Loop ----
    history = {"train_loss": [], "val_loss": [], "val_mse_destd": [], "val_spearman": []}
    best_val = float("inf")
    best_path = args.output_dir / "best.pt"

    for ep in range(args.epochs):
        model.train()
        train_losses = []
        for x, y, _ in train_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        sched.step()

        # ---- Validation ----
        model.eval()
        val_losses = []
        # accumulate predictions for entity Spearman (only on val obstacles)
        ent_pred_acc = {n: [] for n in obstacle_names}
        ent_targ_acc = {n: [] for n in obstacle_names}
        sq_err_destd_sum = 0.0
        n_cells = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)
                pred = model(x)
                val_losses.append(F.mse_loss(pred, y).item())
                # de-standardise
                pred_d = pred * y_std_t + y_mean_t
                y_d    = y    * y_std_t + y_mean_t
                sq_err_destd_sum += float(((pred_d - y_d) ** 2).sum())
                n_cells += y_d.numel()
                # entity scores per sample (move to CPU for numpy footprints)
                pred_np = pred_d.cpu().numpy()
                y_np    = y_d.cpu().numpy()
                for k in range(pred_np.shape[0]):
                    ep_scores = integrate_per_entity(pred_np[k], fps)
                    et_scores = integrate_per_entity(y_np[k], fps)
                    for n in obstacle_names:
                        ent_pred_acc[n].append(ep_scores.get(n, 0.0))
                        ent_targ_acc[n].append(et_scores.get(n, 0.0))

        flat_pred = np.concatenate([np.asarray(v) for v in ent_pred_acc.values()])
        flat_targ = np.concatenate([np.asarray(v) for v in ent_targ_acc.values()])
        if flat_pred.std() > 0 and flat_targ.std() > 0:
            rho, _ = spearmanr(flat_pred, flat_targ)
        else:
            rho = float("nan")
        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        val_mse_destd = sq_err_destd_sum / n_cells
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse_destd"].append(val_mse_destd)
        history["val_spearman"].append(float(rho))

        improved = val_loss < best_val
        marker = "*" if improved else " "
        print(f" ep {ep+1:>3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"val_mse_destd={val_mse_destd:.3f}  "
              f"baseline_mse={baseline_mse_destd:.3f}  "
              f"rho={rho:.3f}  {marker}")
        if improved:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "stats": stats.to_dict(),
                "grid_shape": full.grid_shape,
                "scene": args.scene,
                "epoch": ep + 1,
                "history": history,
                "args": vars(args) | {"output_dir": str(args.output_dir),
                                       "dataset": str(args.dataset),
                                       "scenes_dir": str(args.scenes_dir)},
            }, best_path)

    print(f"\nbest val_loss = {best_val:.4f} (saved to {best_path})")
    print(f"baseline (predict train mean) MSE in original units: {baseline_mse_destd:.3f}")
    print(f"final val_mse_destd: {history['val_mse_destd'][-1]:.3f}")

    # ---- Loss curve plot ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    epochs = np.arange(1, args.epochs + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set(title="loss (standardised MSE)", xlabel="epoch", ylabel="MSE")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["val_mse_destd"], color="crimson")
    axes[1].axhline(baseline_mse_destd, color="gray", linestyle="--",
                    label=f"baseline (predict mean) = {baseline_mse_destd:.3f}")
    axes[1].set(title="val MSE (original units)", xlabel="epoch", ylabel="MSE")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(epochs, history["val_spearman"], color="forestgreen")
    axes[2].set(title="Spearman ρ (predicted vs target obstacle scores)",
                xlabel="epoch", ylabel="ρ")
    axes[2].axhline(0.85, color="gray", linestyle="--", label="0.85 floor")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(args.output_dir / "loss_curve.png", dpi=110)
    plt.close()

    # ---- Predicted vs actual panel for 5 held-out configs ----
    print("rendering preds.png on 5 held-out configs...")
    ckpt = torch.load(best_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(val_ds), size=5, replace=False)
    fig, axes = plt.subplots(5, 3, figsize=(11, 14))
    for k, idx in enumerate(sel):
        x, y, _ = val_ds[idx]
        with torch.no_grad():
            p = model(x.unsqueeze(0).to(args.device)).cpu().numpy()[0]
        # de-standardise
        target_d = y.numpy() * stats.y_std + stats.y_mean
        pred_d   = p          * stats.y_std + stats.y_mean
        err = np.abs(target_d - pred_d)
        vmax = max(target_d.max(), pred_d.max(), 1e-6)
        for c, (img, title) in enumerate([(target_d, "target"), (pred_d, "pred"), (err, "|err|")]):
            ax = axes[k, c]
            ax.imshow(img, origin="lower", extent=grid.extent, cmap="magma",
                      vmin=0, vmax=vmax if c < 2 else err.max() + 1e-6)
            ax.set_xticks([]); ax.set_yticks([])
        row = val_ds.rows[idx]
        axes[k, 0].set_ylabel(f"{row.task_id}\ntraj{row.traj_id}", fontsize=8)
        if k == 0:
            for c, t in enumerate(["target", "pred", "|err|"]):
                axes[k, c].set_title(t)
    plt.tight_layout(); plt.savefig(args.output_dir / "preds.png", dpi=110)
    plt.close()

    # ---- Save history json for downstream use ----
    (args.output_dir / "history.json").write_text(json.dumps({
        "history": history, "best_val": best_val,
        "baseline_mse_destd": baseline_mse_destd,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "n_train_keys": len(train_keys), "n_val_keys": len(val_keys),
    }, indent=2) + "\n")

    print(f"done. open {args.output_dir}/preds.png and loss_curve.png")


if __name__ == "__main__":
    main()
