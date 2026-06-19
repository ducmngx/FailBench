#!/usr/bin/env python3
"""Plot the 600-rollout sweep results for visual inspection.

Produces a 4-panel figure:

    1. Per-mode realized_risk distribution (boxplot, log y, baseline vs scaling)
    2. Predicted-vs-realized risk scatter (baseline only, log-log)
    3. Per-(mode, progress) mean risk heatmap, baseline minus scaling
    4. Catastrophic-rate (realized_risk > 100) per mode + policy

Saves to ``<results_dir>/viz.png``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path,
                    default=Path("out/safety_rollouts_opt1/"
                                  "pick_up_the_black_bowl_from_table_center"
                                  "_and_place_it_on_the_plate/results.csv"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG (default: <results_dir>/viz.png)")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    print(f"loaded {len(df)} rollouts from {args.results}")
    out_path = args.out or args.results.parent / "viz.png"

    MODES = ["GRIPPER_OPEN", "SLIPPERY_GRIP", "SINGLE_JOINT",
             "MULTI_JOINT", "ALL_JOINTS"]
    POLICIES = sorted(df["policy"].unique().tolist())
    PROGRESSES = sorted(df["fail_progress"].unique().tolist())

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ----- Panel 1: boxplot of realized risk per mode (log y) ---------------
    ax1 = fig.add_subplot(gs[0, 0])
    eps = 1e-3   # to plot zeros on log scale
    positions = []
    labels = []
    data = []
    colors = []
    for i, mode in enumerate(MODES):
        for j, poli in enumerate(POLICIES):
            sub = df[(df["mode"] == mode) & (df["policy"] == poli)]
            vals = sub["realized_risk"].values + eps
            data.append(vals)
            positions.append(i * (len(POLICIES) + 1) + j)
            labels.append(f"{mode[:4]}.{poli[:3]}")
            colors.append("C0" if poli == "baseline" else "C1")
    bp = ax1.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                     showfliers=True, flierprops=dict(marker=".", markersize=3))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax1.set_yscale("log")
    ax1.set_xticks([i * (len(POLICIES) + 1) + (len(POLICIES) - 1) / 2
                     for i in range(len(MODES))])
    ax1.set_xticklabels(MODES, rotation=15)
    ax1.set_ylabel("realized_risk (log)")
    ax1.set_title("(a) Realized risk distribution per failure mode")
    ax1.grid(axis="y", alpha=0.3, which="both")
    # Legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color="C0", alpha=0.6, label="baseline"),
                          Patch(color="C1", alpha=0.6, label="scaling")],
                 loc="upper left")

    # ----- Panel 2: predicted vs realized risk scatter, baseline only -------
    ax2 = fig.add_subplot(gs[0, 1])
    b = df[df["policy"] == "baseline"].copy()
    for i, mode in enumerate(MODES):
        sub = b[b["mode"] == mode]
        ax2.scatter(sub["pred_risk_pre_failure"] + eps,
                    sub["realized_risk"] + eps,
                    alpha=0.7, s=30, label=mode, color=f"C{i}")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("predicted risk at failure step (log)")
    ax2.set_ylabel("realized risk (log)")
    ax2.set_title("(b) Predictor vs realized risk (baseline only)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7, loc="lower right")

    # Add Spearman annotation
    from scipy.stats import spearmanr
    r, p = spearmanr(b["pred_risk_pre_failure"], b["realized_risk"])
    ax2.text(0.05, 0.95,
              f"Spearman ρ = {r:+.3f}\n(p = {p:.2f}, n = {len(b)})",
              transform=ax2.transAxes, va="top", fontsize=9,
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))

    # ----- Panel 3: heatmap of mean risk per (mode, progress, policy) -------
    ax3 = fig.add_subplot(gs[1, 0])
    # Build a (mode × progress × policy) array, then plot baseline and scaling
    # side by side per mode
    grid_baseline = np.zeros((len(MODES), len(PROGRESSES)))
    grid_scaling = np.zeros_like(grid_baseline)
    for i, m in enumerate(MODES):
        for j, p in enumerate(PROGRESSES):
            for poli, grid in [("baseline", grid_baseline),
                                ("scaling", grid_scaling)]:
                sub = df[(df["mode"] == m) & (df["fail_progress"] == p)
                          & (df["policy"] == poli)]
                grid[i, j] = sub["realized_risk"].mean() if len(sub) else np.nan
    # Use log scale for color
    stacked = np.concatenate([grid_baseline, np.full((len(MODES), 1), np.nan),
                                grid_scaling], axis=1)
    vmax = np.nanmax(stacked); vmin = max(1e-3, np.nanmin(stacked[stacked > 0]))
    im = ax3.imshow(np.log10(stacked + eps), aspect="auto", cmap="viridis",
                     vmin=np.log10(vmin), vmax=np.log10(vmax))
    ax3.set_yticks(range(len(MODES))); ax3.set_yticklabels(MODES)
    xticks = list(range(len(PROGRESSES))) + [len(PROGRESSES)] + \
             list(range(len(PROGRESSES) + 1, 2 * len(PROGRESSES) + 1))
    xlabels = [f"{p:.2f}" for p in PROGRESSES] + [""] + \
              [f"{p:.2f}" for p in PROGRESSES]
    ax3.set_xticks(xticks); ax3.set_xticklabels(xlabels, fontsize=8)
    ax3.text(len(PROGRESSES) / 2 - 0.5, -0.7, "baseline",
              ha="center", fontsize=10, weight="bold")
    ax3.text(1.5 * len(PROGRESSES) + 0.5, -0.7, "scaling",
              ha="center", fontsize=10, weight="bold")
    ax3.set_xlabel("fail_progress")
    ax3.set_title("(c) Mean realized risk per (mode × progress)")
    cbar = plt.colorbar(im, ax=ax3, label="log10(mean risk)")

    # Annotate values
    for i in range(len(MODES)):
        for j in range(stacked.shape[1]):
            v = stacked[i, j]
            if np.isnan(v): continue
            label = f"{v:.1f}" if v >= 1 else f"{v:.2f}"
            ax3.text(j, i, label, ha="center", va="center",
                      color="white" if np.log10(v + eps) < np.log10(vmax) - 1
                            else "black", fontsize=7)

    # ----- Panel 4: catastrophic-rate bars per mode + policy ----------------
    ax4 = fig.add_subplot(gs[1, 1])
    rows = []
    for poli in POLICIES:
        for m in MODES:
            sub = df[(df["mode"] == m) & (df["policy"] == poli)]
            rate = 100 * (sub["realized_risk"] > 100).mean()
            rows.append({"policy": poli, "mode": m, "rate": rate,
                         "max": sub["realized_risk"].max()})
    cat = pd.DataFrame(rows)
    x = np.arange(len(MODES))
    width = 0.35
    for i, poli in enumerate(POLICIES):
        sub = cat[cat["policy"] == poli].set_index("mode").reindex(MODES)
        bars = ax4.bar(x + (i - 0.5) * width, sub["rate"], width,
                        label=poli, color=f"C{i}", alpha=0.7)
        # Annotate max risk above each bar
        for b, m in zip(bars, MODES):
            mx = cat[(cat.policy == poli) & (cat["mode"] == m)]["max"].iloc[0]
            ax4.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                      f"max\n{mx:.0f}" if mx > 1 else f"max\n{mx:.2f}",
                      ha="center", fontsize=6)
    ax4.set_xticks(x); ax4.set_xticklabels(MODES, rotation=15)
    ax4.set_ylabel("rollouts with realized_risk > 100 (%)")
    ax4.set_title("(d) Catastrophic-outcome rate per failure mode")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_ylim(0, max(10, cat["rate"].max() + 5))

    fig.suptitle("Safety-rollout sweep — 600 rollouts on "
                  "pick_up_the_black_bowl_from_table_center_and_place_it"
                  "_on_the_plate", fontsize=11, y=0.995)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nsaved figure → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
