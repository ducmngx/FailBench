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

from planner.risk.dataset import HeatmapDataset, DatasetStats, split_traj_keys
from planner.risk.model import HeatmapMLP, HeatmapConvDecoder, HeatmapVisionConvDecoder
from planner.risk.spatial import load_grid

# Per-obstacle Spearman has been demoted to offline eval — see
# notebooks/eval_model.ipynb. Training selects checkpoints purely on
# reconstruction MSE, which is the bottleneck across stages 0–5.


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
    ap.add_argument("--decoder", type=str, default="mlp", choices=["mlp", "conv"],
                    help="output head: 'mlp' (Stage 0 dense Linear) or 'conv' (Stage 1 conv decoder)")
    ap.add_argument("--include_goal", action="store_true", help="append goal_pos (3) to input")
    ap.add_argument("--include_task", action="store_true", help="append task one-hot to input")
    ap.add_argument("--include_rgb", action="store_true",
                    help="add pre_rgb (resized) through a small CNN encoder (Stage 3)")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        suffix = "vision" if args.include_rgb else args.decoder
        if args.include_goal:
            suffix += "+goal"
        if args.include_task:
            suffix += "+task"
        if args.include_rgb:
            suffix += "+rgb"
        args.output_dir = Path("runs") / f"heatmap_{args.scene}_{suffix}_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing artifacts to {args.output_dir}")

    # ---- Build the full dataset, split by traj_id, fit stats on train ----
    ds_kwargs = dict(include_goal=args.include_goal,
                     include_task=args.include_task,
                     include_rgb=args.include_rgb)
    full = HeatmapDataset(args.dataset, args.scene, **ds_kwargs)
    print(f"loaded {len(full)} configs, grid={full.grid_shape}, "
          f"input_dim={full.input_dim} (goal={args.include_goal}, task={args.include_task}, "
          f"|task_vocab|={len(full.task_vocab)})")
    train_keys, val_keys = split_traj_keys(full.traj_keys, val_frac=args.val_frac, seed=args.seed)
    assert set(train_keys).isdisjoint(val_keys), "traj split leakage!"
    print(f"split: {len(train_keys)} train (task,traj) keys, {len(val_keys)} val keys")

    train_ds = HeatmapDataset(args.dataset, args.scene, traj_keys=train_keys,
                              task_vocab=full.task_vocab, **ds_kwargs)
    val_ds   = HeatmapDataset(args.dataset, args.scene, traj_keys=val_keys,
                              task_vocab=full.task_vocab, **ds_kwargs)
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
    in_dim = full.input_dim
    if args.include_rgb:
        # Vision decoder always uses the conv decoder + small CNN.
        model = HeatmapVisionConvDecoder(state_dim=in_dim,
                                         grid_shape=full.grid_shape).to(args.device)
        print(f"decoder type: vision-conv, state_dim={in_dim}")
    elif args.decoder == "mlp":
        model = HeatmapMLP(in_dim=in_dim,
                           grid_shape=full.grid_shape).to(args.device)
        print(f"decoder type: mlp, in_dim={in_dim}")
    elif args.decoder == "conv":
        model = HeatmapConvDecoder(in_dim=in_dim,
                                   grid_shape=full.grid_shape).to(args.device)
        print(f"decoder type: conv, in_dim={in_dim}")
    else:
        raise ValueError(args.decoder)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params; device={args.device}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # ---- Per-pixel mean baseline (for context) ----
    y_mean_t = torch.from_numpy(stats.y_mean).to(args.device)
    y_std_t  = torch.from_numpy(stats.y_std).to(args.device)
    baseline_mse_destd = float(((stats.y_std ** 2).mean()))   # MSE if predicting train mean (in original units)

    # Grid is still loaded so the preds-panel can use scene-aligned extents.
    grid = load_grid(args.dataset / args.scene / "grid.json")

    # ---- Loop ----
    history = {"train_loss": [], "val_loss": [], "val_mse_destd": []}
    best_val = float("inf")
    best_mse_path = args.output_dir / "best_mse.pt"
    # Legacy: keep `best.pt` as an alias of best_mse.pt so existing notebooks
    # and eval scripts that load `best.pt` keep working.
    best_path = args.output_dir / "best.pt"

    def _forward(batch):
        if args.include_rgb:
            x, rgb, y, _ = batch
            x = x.to(args.device, non_blocking=True)
            rgb = rgb.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            return model(x, rgb), y
        x, y, _ = batch
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)
        return model(x), y

    for ep in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            pred, y = _forward(batch)
            loss = F.mse_loss(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        sched.step()

        # ---- Validation ----
        model.eval()
        val_losses = []
        sq_err_destd_sum = 0.0
        n_cells = 0
        with torch.no_grad():
            for batch in val_loader:
                pred, y = _forward(batch)
                val_losses.append(F.mse_loss(pred, y).item())
                pred_d = pred * y_std_t + y_mean_t
                y_d    = y    * y_std_t + y_mean_t
                sq_err_destd_sum += float(((pred_d - y_d) ** 2).sum())
                n_cells += y_d.numel()

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        val_mse_destd = sq_err_destd_sum / n_cells
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse_destd"].append(val_mse_destd)

        improved_mse = val_loss < best_val
        marker = "*" if improved_mse else " "
        print(f" ep {ep+1:>3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"val_mse_destd={val_mse_destd:.3f}  "
              f"baseline_mse={baseline_mse_destd:.3f}  {marker}")

        def _save(path: Path):
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
            }, path)

        if improved_mse:
            best_val = val_loss
            _save(best_mse_path)
            _save(best_path)  # legacy alias

    best_mse_epoch = int(np.argmin(history["val_loss"])) + 1
    print(f"\nbest val_loss = {best_val:.4f} at ep {best_mse_epoch} (saved to {best_mse_path})")
    print(f"baseline (predict train mean) MSE in original units: {baseline_mse_destd:.3f}")
    print(f"final val_mse_destd: {history['val_mse_destd'][-1]:.3f}")

    # ---- Loss curve plot ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
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
    plt.tight_layout(); plt.savefig(args.output_dir / "loss_curve.png", dpi=110)
    plt.close()

    # ---- Predicted vs actual panel for 5 held-out configs ----
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(val_ds), size=5, replace=False)
    print(f"rendering preds.png from {best_mse_path.name}...")
    ckpt = torch.load(best_mse_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    fig, axes = plt.subplots(5, 3, figsize=(11, 14))
    for k, idx in enumerate(sel):
        item = val_ds[idx]
        if args.include_rgb:
            x, rgb, y, _ = item
            with torch.no_grad():
                p = model(x.unsqueeze(0).to(args.device),
                          rgb.unsqueeze(0).to(args.device)).cpu().numpy()[0]
        else:
            x, y, _ = item
            with torch.no_grad():
                p = model(x.unsqueeze(0).to(args.device)).cpu().numpy()[0]
        target_d = y.numpy() * stats.y_std + stats.y_mean
        pred_d   = p          * stats.y_std + stats.y_mean
        err = np.abs(target_d - pred_d)
        vmax = max(target_d.max(), pred_d.max(), 1e-6)
        for c, (img, _t) in enumerate([(target_d, "target"), (pred_d, "pred"), (err, "|err|")]):
            ax = axes[k, c]
            ax.imshow(img, origin="lower", extent=grid.extent, cmap="magma",
                      vmin=0, vmax=vmax if c < 2 else err.max() + 1e-6)
            ax.set_xticks([]); ax.set_yticks([])
        row = val_ds.rows[idx]
        axes[k, 0].set_ylabel(f"{row.task_id}\ntraj{row.traj_id}", fontsize=8)
        if k == 0:
            for c, t in enumerate(["target", "pred", "|err|"]):
                axes[k, c].set_title(t)
    fig.suptitle(f"checkpoint: best_mse.pt (epoch {ckpt['epoch']})", fontsize=10)
    plt.tight_layout(); plt.savefig(args.output_dir / "preds.png", dpi=110)
    plt.close()

    # ---- Save history json for downstream use ----
    (args.output_dir / "history.json").write_text(json.dumps({
        "history": history,
        "best_val": best_val,
        "best_mse_epoch": best_mse_epoch,
        "baseline_mse_destd": baseline_mse_destd,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "n_train_keys": len(train_keys), "n_val_keys": len(val_keys),
    }, indent=2) + "\n")

    print(f"done. open {args.output_dir}/{{loss_curve,preds}}.png")


if __name__ == "__main__":
    main()
