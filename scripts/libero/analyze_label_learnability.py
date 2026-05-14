"""Learnability sanity checks for the LIBERO camera-projected heatmap target.

Runs four checks BEFORE committing to full training:

  1. Nearest-neighbour predictability — similar pre-states ⇒ similar labels?
  2. Within-group vs between-group variance — how much per-failure noise survives?
  3. Constant-baseline MSE — the floor any model must beat.
  4. Tiny-MLP — can a state-only model beat the constant baseline?

Outputs a 4-row summary table. Each check has a numeric verdict and a pass/fail
flag against the criteria in the plan.

Usage:
  python -m scripts.libero.analyze_label_learnability \\
      --labels datasets/libero/v1/label_prototypes.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

V1_ROOT = REPO_ROOT / "datasets" / "libero" / "v1"


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------


def _depth_pool_8x8(depth: np.ndarray) -> np.ndarray:
    """Mean-pool a (H, W) depth frame down to (8, 8)."""
    h, w = depth.shape
    bh, bw = h // 8, w // 8
    crop = depth[: bh * 8, : bw * 8]
    return crop.reshape(8, bh, 8, bw).mean(axis=(1, 3))


def load_group_features(labels_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build a (G, 72) feature matrix, one per group, plus aggregate labels.

    Returns (features, group_keys, group_agent_log1p, manifest_subset_df).
    """
    labels = np.load(labels_npz)
    exp_ids = labels["experiment_id"]
    group_ids = labels["group_ids"]
    group_keys = labels["group_keys"]
    G = len(group_keys)

    # Build a combined manifest over all 3 splits.
    parts = []
    for split in ["libero_spatial", "libero_goal", "libero_object"]:
        p = V1_ROOT / split / "manifest.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    manifest = pd.concat(parts, ignore_index=True).set_index("experiment_id")

    features = np.zeros((G, 7 + 1 + 64), dtype=np.float32)
    for g in range(G):
        idx = int(np.where(group_ids == g)[0][0])
        eid = str(exp_ids[idx])
        rec = manifest.loc[eid]
        npz_path = V1_ROOT / rec["split"] / rec["npz_file"]
        with np.load(npz_path) as d:
            pre_qpos = d["pre_qpos"].astype(np.float32)
            pre_qvel = d["pre_qvel"].astype(np.float32)
            pre_depth = d["pre_depth"].astype(np.float32)
        depth_pool = _depth_pool_8x8(pre_depth).astype(np.float32).flatten()
        features[g, :7] = pre_qpos
        features[g, 7] = np.linalg.norm(pre_qvel)
        features[g, 8:] = depth_pool

    # Per-group aggregated labels (mass channel only, log1p compressed).
    group_agent = labels["group_cam_heatmap_agentview"]  # (G, H, W) raw mass
    group_agent_log1p = np.log1p(group_agent).astype(np.float32)

    return features, group_keys, group_agent_log1p, manifest


# -----------------------------------------------------------------------------
# Check 1 — Nearest-neighbour predictability
# -----------------------------------------------------------------------------


def check1_nn_predictability(features: np.ndarray, labels: np.ndarray,
                             k: int = 5, rng: np.random.Generator = None):
    """NN-vs-random similarity ratio in label space."""
    if rng is None:
        rng = np.random.default_rng(0)
    G = features.shape[0]
    if G < k + 2:
        return dict(ratio=float("nan"), nn_sim=float("nan"),
                    rand_sim=float("nan"), passed=False,
                    note=f"too few groups ({G})")

    # L2-normalise features (cosine similarity = inner product).
    feat_n = features - features.mean(axis=0, keepdims=True)
    feat_n /= np.maximum(np.linalg.norm(feat_n, axis=1, keepdims=True), 1e-9)
    # Label cosine similarity matrix.
    L = labels.reshape(G, -1)
    L_n = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-9)
    label_sim = L_n @ L_n.T

    feat_sim = feat_n @ feat_n.T

    nn_label_sims = []
    rand_label_sims = []
    for g in range(G):
        # NN in feature space (exclude self).
        order = np.argsort(-feat_sim[g])
        nn = [j for j in order if j != g][:k]
        nn_label_sims.append(label_sim[g, nn].mean())
        # Random comparison: pick k different indices uniformly.
        rand = rng.choice([j for j in range(G) if j != g], size=k, replace=False)
        rand_label_sims.append(label_sim[g, rand].mean())

    nn_sim = float(np.mean(nn_label_sims))
    rand_sim = float(np.mean(rand_label_sims))
    ratio = nn_sim / max(rand_sim, 1e-9)
    return dict(ratio=ratio, nn_sim=nn_sim, rand_sim=rand_sim,
                passed=ratio >= 2.0, note="")


# -----------------------------------------------------------------------------
# Check 2 — Within-group vs between-group variance
# -----------------------------------------------------------------------------


def _within_between_ratio(per_trial, per_group, group_ids):
    """Helper: median per-pixel within/between variance ratio over pixels with
    non-trivial between-group variance."""
    n_groups = int(group_ids.max()) + 1
    within_acc = np.zeros(per_trial.shape[1:], dtype=np.float64)
    n_ok = 0
    for g in range(n_groups):
        sel = np.where(group_ids == g)[0]
        if sel.size < 2:
            continue
        within_acc += per_trial[sel].var(axis=0, ddof=0)
        n_ok += 1
    if n_ok == 0:
        return float("nan"), float("nan"), float("nan"), 0
    within_var = within_acc / n_ok
    between_var = per_group.var(axis=0, ddof=0)
    mask = between_var > 0.01 * between_var.max()
    if not mask.any():
        return float("nan"), float("nan"), float("nan"), 0
    eps = 1e-9
    ratio = within_var[mask] / np.maximum(between_var[mask], eps)
    return (float(np.median(ratio)),
            float(np.percentile(ratio, 25)),
            float(np.percentile(ratio, 75)),
            int(mask.sum()))


def check2_variance_decomposition(labels_npz: Path):
    """Within/between variance ratio on raw mass AND on log1p(mass).

    The log1p variant is closer to what the model actually optimises against,
    so we use it for pass/fail. Raw ratio is reported for diagnostic context
    (raw can FAIL just because per-failure magnitudes differ wildly while
    spatial structure is fine).
    """
    labels = np.load(labels_npz)
    cam_raw = labels["cam_heatmap_agentview"][..., 0]  # (N, H, W) per-trial mass
    cam_log = np.log1p(cam_raw)
    group_ids = labels["group_ids"]
    group_raw = labels["group_cam_heatmap_agentview"]
    group_log = np.log1p(group_raw)

    raw_med, raw_p25, raw_p75, n_pix = _within_between_ratio(cam_raw, group_raw, group_ids)
    log_med, log_p25, log_p75, _ = _within_between_ratio(cam_log, group_log, group_ids)
    return dict(median_ratio=log_med,
                p25=log_p25, p75=log_p75,
                raw_median=raw_med, raw_p25=raw_p25, raw_p75=raw_p75,
                n_meaningful_pixels=n_pix,
                passed=log_med < 0.5,
                note=f"pass/fail on log1p; raw_med={raw_med:.3f} (diagnostic)")


# -----------------------------------------------------------------------------
# Check 3 — Constant-baseline MSE
# -----------------------------------------------------------------------------


def _split_groups(G: int, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(G)
    n_val = max(1, int(round(G * val_frac)))
    val = perm[:n_val]
    train = perm[n_val:]
    return train, val


def check3_constant_baseline(labels: np.ndarray, val_frac: float = 0.2, seed: int = 0):
    """MSE of `train_mean` predictor on val set."""
    G = labels.shape[0]
    train_idx, val_idx = _split_groups(G, val_frac, seed)
    if val_idx.size < 1:
        return dict(val_mse=float("nan"), passed=False, note="no val")
    baseline = labels[train_idx].mean(axis=0)  # (H, W)
    val_mse = float(((labels[val_idx] - baseline) ** 2).mean())
    train_var = float(labels[train_idx].var())
    return dict(val_mse=val_mse, train_var=train_var,
                normalized=val_mse / max(train_var, 1e-9),
                passed=True,  # informational, not pass/fail
                note=f"floor recorded ({val_idx.size}/{G} val groups)",
                train_idx=train_idx, val_idx=val_idx, baseline=baseline)


# -----------------------------------------------------------------------------
# Check 4 — Tiny MLP
# -----------------------------------------------------------------------------


def check4_tiny_mlp(features: np.ndarray, labels: np.ndarray,
                    constant_baseline_mse: float,
                    train_idx: np.ndarray, val_idx: np.ndarray,
                    epochs: int = 200, hidden: int = 128, lr: float = 1e-3,
                    weight_decay: float = 1e-2,
                    target_hw: tuple = (30, 40),
                    seed: int = 0):
    """MLP from 72-D feature to flattened low-res downsampled label.

    Output is downsampled to (target_hw) — say 30x40 = 1200 dim — to keep the
    parameter-to-train-sample ratio sane on small prototype runs. With heavy
    weight decay this is a reasonable test of whether state-only features carry
    enough signal to beat the constant baseline.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    H, W = labels.shape[1], labels.shape[2]
    th, tw = target_hw
    bh = H // th
    bw = W // tw
    crop_h = bh * th
    crop_w = bw * tw
    labels_small = labels[:, :crop_h, :crop_w].reshape(-1, th, bh, tw, bw).mean(axis=(2, 4))
    out_dim = th * tw

    mu = features[train_idx].mean(axis=0)
    sd = features[train_idx].std(axis=0) + 1e-6
    X = (features - mu) / sd

    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    Y_t = torch.from_numpy(labels_small.reshape(labels.shape[0], -1).astype(np.float32)).to(device)
    train_t = torch.from_numpy(train_idx).long().to(device)
    val_t = torch.from_numpy(val_idx).long().to(device)

    model = nn.Sequential(
        nn.Linear(X.shape[1], hidden), nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    best_val = float("inf")
    for ep in range(epochs):
        model.train()
        pred = model(X_t[train_t])
        loss = ((pred - Y_t[train_t]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        train_losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            vp = model(X_t[val_t])
            vloss = float(((vp - Y_t[val_t]) ** 2).mean())
        val_losses.append(vloss)
        best_val = min(best_val, vloss)

    # Re-evaluate constant baseline on the SAME downsampled targets for fairness.
    baseline_small = labels_small[train_idx].mean(axis=0).flatten()
    val_const_mse = float(((labels_small[val_idx].reshape(val_idx.size, -1) - baseline_small) ** 2).mean())

    reduction = (val_const_mse - best_val) / max(val_const_mse, 1e-9)
    return dict(best_val_mse=best_val, const_mse=val_const_mse,
                reduction=reduction, passed=reduction >= 0.20,
                train_losses=train_losses, val_losses=val_losses,
                note=f"downsampled to {th}x{tw}; const_mse computed on same target")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    default=V1_ROOT / "label_prototypes.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--mlp_epochs", type=int, default=80)
    args = ap.parse_args()

    if not args.labels.exists():
        sys.exit(f"labels file not found: {args.labels}")

    t0 = time.time()
    print(f"loading features + labels from {args.labels}")
    features, group_keys, labels, manifest = load_group_features(args.labels)
    G = features.shape[0]
    print(f"  G={G} groups, feature dim={features.shape[1]}, label shape={labels.shape[1:]}")

    print("\n[check 1] NN-predictability ...")
    r1 = check1_nn_predictability(features, labels)
    print(f"  nn_sim={r1['nn_sim']:.4f}  rand_sim={r1['rand_sim']:.4f}  "
          f"ratio={r1['ratio']:.2f}  {'PASS' if r1['passed'] else 'FAIL'}  {r1.get('note','')}")

    print("\n[check 2] within/between variance ...")
    r2 = check2_variance_decomposition(args.labels)
    print(f"  median ratio={r2['median_ratio']:.3f}  "
          f"[p25={r2.get('p25', float('nan')):.3f}, p75={r2.get('p75', float('nan')):.3f}]  "
          f"{'PASS' if r2['passed'] else 'FAIL'}  {r2.get('note','')}")

    print("\n[check 3] constant-baseline MSE ...")
    r3 = check3_constant_baseline(labels, val_frac=args.val_frac, seed=args.seed)
    print(f"  val_mse={r3['val_mse']:.4f}  train_var={r3['train_var']:.4f}  "
          f"normalised={r3['normalized']:.3f}  {r3.get('note','')}")

    print("\n[check 4] tiny MLP ...")
    r4 = check4_tiny_mlp(features, labels, r3["val_mse"],
                         r3["train_idx"], r3["val_idx"],
                         epochs=args.mlp_epochs, seed=args.seed)
    print(f"  best_val_mse={r4['best_val_mse']:.4f}  const_mse={r4['const_mse']:.4f}  "
          f"reduction={r4['reduction']:.1%}  {'PASS' if r4['passed'] else 'FAIL'}")

    # Decision matrix
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    rows = [
        ("Check 1: NN predictability", r1["passed"], f"ratio={r1['ratio']:.2f}"),
        ("Check 2: within/between var", r2["passed"], f"median={r2['median_ratio']:.3f}"),
        ("Check 3: constant baseline", True, f"val_mse={r3['val_mse']:.4f}"),
        ("Check 4: tiny MLP", r4["passed"], f"reduction={r4['reduction']:.1%}"),
    ]
    for name, passed, detail in rows:
        flag = "PASS" if passed else "FAIL"
        print(f"  {name:32s}  [{flag}]  {detail}")

    print("\nDecision:")
    if not r1["passed"]:
        print("  Check 1 failed → cam-heatmap target NOT learnable from current inputs.")
        print("  Action: reconsider representation (add wrist cam / failure-mode embedding).")
    elif not r2["passed"]:
        print("  Check 2 failed → aggregation cannot remove per-failure noise.")
        print("  Action: switch to per-trial labels + failure_mode embedding input.")
    elif not r4["passed"]:
        print("  Check 4 failed (state-only inputs insufficient).")
        print("  Action: still proceed with Round 3 step 4 — image inputs add info.")
    else:
        print("  All checks pass → proceed to Round 3 step 4 (full training) as planned.")

    print(f"\nelapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
