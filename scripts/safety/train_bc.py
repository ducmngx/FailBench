#!/usr/bin/env python3
"""Phase 2: BC training on predictor-augmented LIBERO demos.

Reads the augmented HDF5 produced by ``augment_demos.py`` and trains a
small MLP that predicts demo actions from (proprio, pred_per_body,
gate_prob).  RGB is NOT used here — Phase 3 will swap in a CNN encoder
when transferring to the PPO policy.  Keeping BC proprio-based makes the
sample-efficient + fast-to-train regime work on the local 3070.

Outputs a torch state_dict + a config json so the PPO trainer can find
the right architecture to copy into.

Includes optional **predictor-based curation** (OopsieVerse §V.A trick):
drop demos whose ``max_pred_risk`` exceeds a threshold so BC trains on
the safer subset.

Usage::

    PYTHONPATH=external/LIBERO:. external/LIBERO/.venv/bin/python \\
        -m scripts.safety.train_bc \\
        --augmented datasets/libero/augmented/tomato_sauce.hdf5 \\
        --out runs/bc-tomato-sauce \\
        --epochs 20 --curation_quantile 0.75
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BCDataset(Dataset):
    """Loads (proprio, pred_per_body, gate_prob, action) tuples from a list
    of demo HDF5 paths.  Optional per-demo curation filter.
    """

    def __init__(self, hdf5_paths: List[Path],
                 max_pred_risk_threshold: float = float("inf"),
                 max_demos_per_file: int = None):
        self.samples = []
        self.body_names: List[str] = []
        for path in hdf5_paths:
            with h5py.File(path, "r") as f:
                if not self.body_names:
                    self.body_names = [b.decode("utf-8") if isinstance(b, bytes)
                                        else str(b)
                                        for b in f["data"].attrs["body_names"]]
                demo_keys = sorted(f["data"].keys(),
                                    key=lambda k: int(k.split("_")[1]))
                if max_demos_per_file is not None:
                    demo_keys = demo_keys[:max_demos_per_file]
                for dk in demo_keys:
                    grp = f[f"data/{dk}"]
                    max_pred = float(grp.attrs.get("max_pred_risk", 0))
                    if max_pred > max_pred_risk_threshold:
                        continue
                    T = int(grp.attrs["T"])
                    proprio = np.asarray(grp["proprio"][:],
                                          dtype=np.float32)
                    pred = np.asarray(grp["pred_per_body"][:],
                                       dtype=np.float32)
                    gate = np.asarray(grp["gate_prob"][:],
                                       dtype=np.float32).reshape(T, 1)
                    actions = np.asarray(grp["actions"][:],
                                          dtype=np.float32)
                    for t in range(T):
                        self.samples.append(
                            (proprio[t], pred[t], gate[t], actions[t]))
        self.K = len(self.body_names)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        proprio, pred, gate, action = self.samples[idx]
        return (torch.from_numpy(proprio),
                torch.from_numpy(pred),
                torch.from_numpy(gate),
                torch.from_numpy(action))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """Small MLP that maps (proprio, pred_per_body, gate_prob) -> action.

    Architecture mirrors the shared trunk we'll use in the PPO policy so
    weight transfer is cheap.  RGB encoder will be added at PPO time as a
    parallel feature head if needed.
    """

    def __init__(self, K: int, action_dim: int = 7):
        super().__init__()
        self.proprio_enc = nn.Sequential(
            nn.Linear(18, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh())
        self.pred_enc = nn.Sequential(
            nn.Linear(K, 64), nn.Tanh())
        self.gate_enc = nn.Sequential(
            nn.Linear(1, 16), nn.Tanh())
        self.trunk = nn.Sequential(
            nn.Linear(128 + 64 + 16, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh())
        self.action_head = nn.Linear(512, action_dim)

    def forward(self, proprio, pred, gate):
        z = torch.cat([
            self.proprio_enc(proprio),
            self.pred_enc(pred),
            self.gate_enc(gate),
        ], dim=-1)
        return self.action_head(self.trunk(z))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--augmented", required=True, nargs="+",
                    help="One or more augmented HDF5 files.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output dir for the BC checkpoint + config.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--curation_quantile", type=float, default=None,
                    help="If set, keep only demos with max_pred_risk below "
                         "this quantile of all demos. Try 0.75 to drop the "
                         "top 25%.")
    ap.add_argument("--max_demos_per_file", type=int, default=None,
                    help="Cap demos per file for smoke testing.")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # First pass: collect max_pred_risk across all demos to compute the
    # curation threshold if requested.
    max_preds = []
    for p in args.augmented:
        with h5py.File(p, "r") as f:
            for dk in f["data"].keys():
                max_preds.append(
                    float(f[f"data/{dk}"].attrs.get("max_pred_risk", 0)))
    max_preds = np.array(max_preds)
    if args.curation_quantile is not None:
        threshold = float(np.quantile(max_preds, args.curation_quantile))
        print(f"Curation threshold (q={args.curation_quantile}): "
              f"max_pred_risk <= {threshold:.1f}")
    else:
        threshold = float("inf")
        print("No curation (using all demos)")
    print(f"  max_pred_risk distribution: min={max_preds.min():.0f}  "
          f"p50={np.median(max_preds):.0f}  "
          f"p90={np.quantile(max_preds, 0.9):.0f}  "
          f"max={max_preds.max():.0f}")

    # Build dataset
    ds = BCDataset(
        [Path(p) for p in args.augmented],
        max_pred_risk_threshold=threshold,
        max_demos_per_file=args.max_demos_per_file)
    print(f"Loaded {len(ds)} samples from {len(args.augmented)} file(s); "
          f"K={ds.K} bodies; body order={ds.body_names}")

    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    rng = torch.Generator().manual_seed(0)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=rng)
    print(f"  train: {len(train_ds)}, val: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    policy = BCPolicy(K=ds.K).to(args.device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print(f"\nTraining BCPolicy (K={ds.K}, action_dim=7) on {args.device}")
    print(f"  total params: {sum(p.numel() for p in policy.parameters()):,}")
    print()

    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        policy.train()
        train_loss = 0.0
        n_batches = 0
        for proprio, pred, gate, action in train_dl:
            proprio = proprio.to(args.device, non_blocking=True)
            pred = pred.to(args.device, non_blocking=True)
            gate = gate.to(args.device, non_blocking=True)
            action = action.to(args.device, non_blocking=True)
            pred_action = policy(proprio, pred, gate)
            loss = ((pred_action - action) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        policy.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for proprio, pred, gate, action in val_dl:
                proprio = proprio.to(args.device)
                pred = pred.to(args.device)
                gate = gate.to(args.device)
                action = action.to(args.device)
                pred_action = policy(proprio, pred, gate)
                val_loss += ((pred_action - action) ** 2).mean().item()
                n_val_batches += 1
        val_loss /= max(n_val_batches, 1)

        print(f"epoch {epoch+1:3d}/{args.epochs}: "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"({time.time() - t0:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": policy.state_dict(),
                "K": ds.K,
                "body_names": ds.body_names,
                "config": vars(args),
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, args.out / "bc_best.pt")

    # Save final + config
    torch.save({
        "model_state": policy.state_dict(),
        "K": ds.K,
        "body_names": ds.body_names,
        "config": vars(args),
        "epoch": args.epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }, args.out / "bc_final.pt")

    with open(args.out / "bc_config.json", "w") as f:
        cfg = vars(args).copy()
        cfg["out"] = str(cfg["out"])
        cfg["K"] = ds.K
        cfg["body_names"] = ds.body_names
        cfg["best_val_loss"] = best_val
        cfg["curation_threshold_used"] = threshold
        cfg["n_train_samples"] = len(train_ds)
        cfg["n_val_samples"] = len(val_ds)
        json.dump(cfg, f, indent=2)

    print(f"\nBest val_loss: {best_val:.4f}")
    print(f"Saved: {args.out}/bc_best.pt, bc_final.pt, bc_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
