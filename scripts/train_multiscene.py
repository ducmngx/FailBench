"""Stage 5: train one heatmap regressor across all 5 scenes jointly.

Inputs are padded to the max grid shape across involved scenes and a per-row
boolean mask flags each scene's valid cells. Loss is masked MSE so padding
contributes zero. Standardisation is per-scene per-cell.

Outputs:
    runs/heatmap_multi_<suffix>_<ts>/
      best_mse.pt, best_rho.pt, best.pt (alias)
      history.json, loss_curve.png, preds_*.png

Usage:
    python scripts/train_multiscene.py --epochs 200 --include_task
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

from planner.risk.dataset import (
    MultiSceneHeatmapDataset, fit_multiscene_stats,
    split_multiscene_traj_keys,
)
from planner.risk.model import HeatmapConvDecoder, HeatmapVisionConvDecoder
from planner.risk.spatial import (
    load_grid, entity_footprints, integrate_per_entity)


DEFAULT_SCENES = ["scene_level2", "scene_kitchen", "scene_workshop",
                  "scene_grocery", "scene_cluttered"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("datasets/v10"))
    ap.add_argument("--scenes_dir", type=Path, default=Path("scenes"))
    ap.add_argument("--scenes", type=str, default=",".join(DEFAULT_SCENES))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--include_goal", action="store_true")
    ap.add_argument("--include_task", action="store_true")
    ap.add_argument("--include_rgb", action="store_true")
    ap.add_argument("--include_depth", action="store_true",
                    help="stack pre_depth as a 4th channel on pre_rgb (Stage 4)")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        suffix = "vision" if args.include_rgb else "conv"
        if args.include_goal:
            suffix += "+goal"
        if args.include_task:
            suffix += "+task"
        if args.include_rgb:
            suffix += "+rgb"
        if args.include_depth:
            suffix += "+depth"
        args.output_dir = Path("runs") / f"heatmap_multi_{suffix}_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing artifacts to {args.output_dir}")
    print(f"scenes: {scenes}")

    # ---- Build full dataset to discover sizes / vocab / shapes ----
    full = MultiSceneHeatmapDataset(
        args.dataset, scenes,
        include_goal=args.include_goal,
        include_task=args.include_task,
        include_rgb=args.include_rgb,
        include_depth=args.include_depth,
    )
    print(f"loaded {len(full)} configs total, "
          f"max_grid={full.max_grid_shape}, input_dim={full.input_dim}, "
          f"|task_vocab|={len(full.task_vocab)}")
    for s in scenes:
        print(f"  {s:20s} {len(full.subdatasets[s]):>5d} configs   grid={full.subdatasets[s].grid_shape}")

    # ---- Per-scene traj split ----
    scenes_to_keys: dict[str, list[tuple[str, int]]] = {}
    for s in scenes:
        scenes_to_keys[s] = list({(r.task_id, r.traj_id)
                                   for r in full.subdatasets[s].rows})
    train_keys, val_keys = split_multiscene_traj_keys(scenes_to_keys,
                                                       val_frac=args.val_frac,
                                                       seed=args.seed)
    for s in scenes:
        assert set(train_keys[s]).isdisjoint(val_keys[s]), f"split leak in {s}"
        print(f"  split {s}: {len(train_keys[s])} train / {len(val_keys[s])} val keys")

    # ---- Build train and val datasets ----
    ds_kwargs = dict(include_goal=args.include_goal,
                     include_task=args.include_task,
                     include_rgb=args.include_rgb,
                     include_depth=args.include_depth,
                     task_vocab=full.task_vocab,
                     max_grid_shape=full.max_grid_shape)
    train_ds = MultiSceneHeatmapDataset(args.dataset, scenes,
                                         traj_keys_per_scene=train_keys,
                                         **ds_kwargs)
    val_ds   = MultiSceneHeatmapDataset(args.dataset, scenes,
                                         traj_keys_per_scene=val_keys,
                                         **ds_kwargs)
    print(f"  total train configs: {len(train_ds)}   val configs: {len(val_ds)}")

    print("computing per-scene standardisation stats on train split...")
    stats_per_scene = fit_multiscene_stats(train_ds)
    train_ds.stats_per_scene = stats_per_scene
    val_ds.stats_per_scene = stats_per_scene

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ---- Model: built around max_grid_shape ----
    in_dim = full.input_dim
    if args.include_rgb:
        rgb_in_ch = 4 if args.include_depth else 3
        model = HeatmapVisionConvDecoder(state_dim=in_dim,
                                          grid_shape=full.max_grid_shape,
                                          rgb_in_ch=rgb_in_ch).to(args.device)
        print(f"model: vision-conv, state_dim={in_dim}, rgb_in_ch={rgb_in_ch}, "
              f"output_grid={full.max_grid_shape}")
    else:
        model = HeatmapConvDecoder(in_dim=in_dim,
                                    grid_shape=full.max_grid_shape).to(args.device)
        print(f"model: conv, in_dim={in_dim}, output_grid={full.max_grid_shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.2f}M params; device={args.device}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # ---- Per-scene de-standardisation tensors + masks + entity footprints ----
    y_mean_t: dict[str, torch.Tensor] = {}
    y_std_t: dict[str, torch.Tensor] = {}
    mask_t: dict[str, torch.Tensor] = {}
    fps_per_scene: dict[str, dict[str, np.ndarray]] = {}
    obstacle_names_per_scene: dict[str, list[str]] = {}
    grid_per_scene = {}
    baseline_mse_per_scene: dict[str, float] = {}
    for s in scenes:
        st = stats_per_scene[s]
        ny, nx = full.subdatasets[s].grid_shape
        my, mx = full.max_grid_shape
        # pad mean/std into the full frame, with std=1 outside (no scaling)
        ymean_pad = np.zeros((my, mx), dtype=np.float32)
        ystd_pad  = np.ones((my, mx), dtype=np.float32)
        ymean_pad[:ny, :nx] = st.y_mean
        ystd_pad[:ny, :nx]  = st.y_std
        y_mean_t[s] = torch.from_numpy(ymean_pad).to(args.device)
        y_std_t[s]  = torch.from_numpy(ystd_pad).to(args.device)
        mask_pad = np.zeros((my, mx), dtype=np.float32)
        mask_pad[:ny, :nx] = 1.0
        mask_t[s]   = torch.from_numpy(mask_pad).to(args.device)
        baseline_mse_per_scene[s] = float((st.y_std ** 2).mean())
        # entity footprints / obstacle names for Spearman
        scene_grid = load_grid(args.dataset / s / "grid.json")
        grid_per_scene[s] = scene_grid
        mj_model = mujoco.MjModel.from_xml_path(str(args.scenes_dir / s / "scene.xml"))
        fps = entity_footprints(mj_model, scene_grid)
        fps_per_scene[s] = fps
        obstacle_names_per_scene[s] = [n for n in fps if "obstacle" in n.lower()]
        print(f"  {s:20s} obstacles tracked: {len(obstacle_names_per_scene[s])} "
              f"baseline_mse={baseline_mse_per_scene[s]:.3f}")

    # ---- Train / val loop ----
    history: dict = {"train_loss": [], "val_loss": [], "val_spearman": [],
                     "per_scene": {s: {"val_mse_destd": [], "val_spearman": []} for s in scenes}}
    best_val = float("inf")
    best_rho = -float("inf")
    best_mse_path = args.output_dir / "best_mse.pt"
    best_rho_path = args.output_dir / "best_rho.pt"
    best_path = args.output_dir / "best.pt"

    def _save(path):
        torch.save({
            "model_state": model.state_dict(),
            "stats_per_scene": {s: stats_per_scene[s].to_dict() for s in scenes},
            "max_grid_shape": full.max_grid_shape,
            "scenes": scenes,
            "task_vocab": full.task_vocab,
            "epoch": ep + 1,
            "history": history,
            "args": vars(args) | {"output_dir": str(args.output_dir),
                                    "dataset": str(args.dataset),
                                    "scenes_dir": str(args.scenes_dir)},
        }, path)

    def _forward(batch):
        if args.include_rgb:
            x, rgb, y, mask, scene_idx, _i = batch
            x = x.to(args.device, non_blocking=True)
            rgb = rgb.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            mask = mask.to(args.device, non_blocking=True).float()
            return model(x, rgb), y, mask, scene_idx
        x, y, mask, scene_idx, _i = batch
        x = x.to(args.device, non_blocking=True)
        y = y.to(args.device, non_blocking=True)
        mask = mask.to(args.device, non_blocking=True).float()
        return model(x), y, mask, scene_idx

    def _masked_mse(pred, y, mask):
        diff2 = (pred - y) ** 2 * mask
        return diff2.sum() / mask.sum().clamp_min(1.0)

    for ep in range(args.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            pred, y, mask, _ = _forward(batch)
            loss = _masked_mse(pred, y, mask)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        sched.step()

        # ---- Validation ----
        model.eval()
        val_losses = []
        # Per-scene accumulators
        ent_pred_acc: dict[str, dict[str, list[float]]] = {
            s: {n: [] for n in obstacle_names_per_scene[s]} for s in scenes}
        ent_targ_acc: dict[str, dict[str, list[float]]] = {
            s: {n: [] for n in obstacle_names_per_scene[s]} for s in scenes}
        sq_err_destd_sum: dict[str, float] = {s: 0.0 for s in scenes}
        n_cells: dict[str, int] = {s: 0 for s in scenes}

        with torch.no_grad():
            for batch in val_loader:
                pred, y, mask, scene_idx = _forward(batch)
                val_losses.append(_masked_mse(pred, y, mask).item())
                for k in range(pred.shape[0]):
                    si = int(scene_idx[k])
                    s = scenes[si]
                    sub_grid = full.subdatasets[s].grid_shape
                    ny, nx = sub_grid
                    pred_d = pred[k, :ny, :nx] * y_std_t[s][:ny, :nx] + y_mean_t[s][:ny, :nx]
                    y_d    = y[k, :ny, :nx]    * y_std_t[s][:ny, :nx] + y_mean_t[s][:ny, :nx]
                    sq_err_destd_sum[s] += float(((pred_d - y_d) ** 2).sum())
                    n_cells[s] += pred_d.numel()
                    pred_np = pred_d.cpu().numpy()
                    y_np    = y_d.cpu().numpy()
                    fps = fps_per_scene[s]
                    ep_scores = integrate_per_entity(pred_np, fps)
                    et_scores = integrate_per_entity(y_np, fps)
                    for n in obstacle_names_per_scene[s]:
                        ent_pred_acc[s][n].append(ep_scores.get(n, 0.0))
                        ent_targ_acc[s][n].append(et_scores.get(n, 0.0))

        per_scene_mse = {s: (sq_err_destd_sum[s] / max(n_cells[s], 1)) for s in scenes}
        per_scene_rho = {}
        for s in scenes:
            fp = np.concatenate([np.asarray(v) for v in ent_pred_acc[s].values()]) \
                 if ent_pred_acc[s] else np.zeros(0)
            ft = np.concatenate([np.asarray(v) for v in ent_targ_acc[s].values()]) \
                 if ent_targ_acc[s] else np.zeros(0)
            if fp.size > 1 and fp.std() > 0 and ft.std() > 0:
                rho_s, _ = spearmanr(fp, ft)
            else:
                rho_s = float("nan")
            per_scene_rho[s] = float(rho_s)
        # Global Spearman: pool all scenes' (pred, target) pairs.
        all_pred = []
        all_targ = []
        for s in scenes:
            for n in obstacle_names_per_scene[s]:
                all_pred.extend(ent_pred_acc[s][n])
                all_targ.extend(ent_targ_acc[s][n])
        all_pred_a = np.asarray(all_pred)
        all_targ_a = np.asarray(all_targ)
        if all_pred_a.size > 1 and all_pred_a.std() > 0 and all_targ_a.std() > 0:
            global_rho, _ = spearmanr(all_pred_a, all_targ_a)
        else:
            global_rho = float("nan")

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        # Use the average of per-scene destd MSE (relative to baseline) as the
        # selection metric — robust to per-cell-std outliers that inflate the
        # standardised val_loss.
        val_mse_destd_avg = float(np.mean([per_scene_mse[s] / max(baseline_mse_per_scene[s], 1e-6)
                                            for s in scenes]))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history.setdefault("val_mse_destd_avg", []).append(val_mse_destd_avg)
        history["val_spearman"].append(float(global_rho))
        for s in scenes:
            history["per_scene"][s]["val_mse_destd"].append(per_scene_mse[s])
            history["per_scene"][s]["val_spearman"].append(per_scene_rho[s])

        improved_mse = val_mse_destd_avg < best_val
        improved_rho = (not np.isnan(global_rho)) and (global_rho > best_rho)
        marker = ("M" if improved_mse else " ") + ("R" if improved_rho else " ")
        per_scene_summary = "  ".join(
            f"{s.replace('scene_', ''):8s}={per_scene_mse[s]:5.2f}/{per_scene_rho[s]:.2f}"
            for s in scenes)
        print(f" ep {ep+1:>3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val_destd_avg={val_mse_destd_avg:.3f}  "
              f"rho={global_rho:.3f}  {marker}  | {per_scene_summary}")
        if improved_mse:
            best_val = val_mse_destd_avg
            _save(best_mse_path)
            _save(best_path)
        if improved_rho:
            best_rho = global_rho
            _save(best_rho_path)

    best_rho_epoch = int(np.nanargmax(history["val_spearman"])) + 1
    best_mse_epoch = int(np.argmin(history["val_mse_destd_avg"])) + 1
    print(f"\nbest val_destd_avg = {best_val:.4f} at ep {best_mse_epoch} (saved to {best_mse_path})")
    print(f"best Spearman ρ = {best_rho:.4f} at ep {best_rho_epoch} (saved to {best_rho_path})")
    print(f"  ρ at best-MSE epoch: {history['val_spearman'][best_mse_epoch-1]:.4f}")
    print(f"  per-scene MSE/ρ at best-MSE epoch:")
    for s in scenes:
        m = history["per_scene"][s]["val_mse_destd"][best_mse_epoch - 1]
        r = history["per_scene"][s]["val_spearman"][best_mse_epoch - 1]
        b = baseline_mse_per_scene[s]
        print(f"    {s:20s} MSE={m:6.3f} (baseline={b:6.3f})  ρ={r:.3f}")

    # ---- Loss curve plot ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    epochs = np.arange(1, args.epochs + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set(title="masked MSE (standardised)", xlabel="epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    for s in scenes:
        axes[1].plot(epochs, history["per_scene"][s]["val_mse_destd"], label=s.replace("scene_", ""))
    axes[1].set(title="val MSE (destandardised, per scene)", xlabel="epoch", ylabel="MSE")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[2].plot(epochs, history["val_spearman"], color="black", label="all-scene")
    for s in scenes:
        axes[2].plot(epochs, history["per_scene"][s]["val_spearman"],
                     alpha=0.5, label=s.replace("scene_", ""))
    axes[2].set(title="Spearman ρ (per-obstacle, per scene)", xlabel="epoch", ylabel="ρ")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "loss_curve.png", dpi=110)
    plt.close()

    # ---- History json ----
    (args.output_dir / "history.json").write_text(json.dumps({
        "history": history,
        "best_val": best_val,
        "best_rho": best_rho,
        "best_mse_epoch": best_mse_epoch,
        "best_rho_epoch": best_rho_epoch,
        "baseline_mse_per_scene": baseline_mse_per_scene,
        "scenes": scenes,
        "n_train": len(train_ds), "n_val": len(val_ds),
    }, indent=2) + "\n")
    print(f"done. open {args.output_dir}/loss_curve.png")


if __name__ == "__main__":
    main()
