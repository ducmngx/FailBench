#!/usr/bin/env python3
"""Per-object attribution gallery: predictor's per-body risk score vs the
ground-truth damaged-objects set.

Picks N trials (highest realized_damage_total) and shows for each:
- pred_per_body_pre_failure as a horizontal bar chart, sorted descending
- ground-truth damage_per_body bar chart, same body ordering
- damaged objects highlighted in red

This complements the scalar precision/recall figure by showing whether the
predictor's per-body attention RANKS the damaged objects highly even when
the top-K binarisation may miss them.

Usage::

    python -m scripts.safety.viz_per_object_attribution \\
        --csv out/safety_rollouts_damage/.../results.csv \\
        --n 12 --topk 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


JSON_COLS = (
    "contact_mass_per_body",
    "realized_damage_per_body",
    "final_health_per_body",
    "damaged_objects",
    "damage_summary",
    "pred_per_body_pre_failure",
)


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    for c in JSON_COLS:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda s: json.loads(s) if isinstance(s, str) else s)
    return df


def _short(n: str) -> str:
    return n.replace("_main", "").replace("_1", "")[:24]


def _scaled(d: Dict[str, float]) -> Dict[str, float]:
    if not d: return {}
    vmax = max(abs(v) for v in d.values())
    if vmax < 1e-9: return d
    return {k: v / vmax for k, v in d.items()}


def plot(df: pd.DataFrame, out: Path, n: int, topk: int) -> None:
    df = df.sort_values("realized_damage_total", ascending=False).head(n)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.0 * n),
                              constrained_layout=True)
    if n == 1:
        axes = axes[None, :]

    for row, (_, r) in enumerate(df.iterrows()):
        pred = r.pred_per_body_pre_failure or {}
        dmg = r.realized_damage_per_body or {}
        damaged = set(r.damaged_objects or [])
        bodies = sorted(set(pred) | set(dmg),
                        key=lambda b: pred.get(b, 0), reverse=True)[:10]
        pred_s = _scaled({b: pred.get(b, 0) for b in bodies})
        dmg_s = _scaled({b: dmg.get(b, 0) for b in bodies})

        ax_p, ax_d = axes[row, 0], axes[row, 1]
        ypos = np.arange(len(bodies))[::-1]
        # Determine predictor top-K
        topk_pred = set(sorted(pred.items(), key=lambda kv: kv[1],
                                reverse=True)[:topk])
        topk_names = {b for b, v in topk_pred if v > 0}

        colors_p = ["#d62728" if b in damaged else
                    ("#2ca02c" if b in topk_names else "#1f77b4")
                    for b in bodies]
        ax_p.barh(ypos, [pred_s.get(b, 0) for b in bodies], color=colors_p)
        ax_p.set_yticks(ypos)
        ax_p.set_yticklabels([_short(b) for b in bodies], fontsize=8)
        ax_p.set_xlim(-0.05, 1.1)
        ax_p.set_xlabel("predicted (norm)")
        title_p = (f"trial {int(r.init_idx)} {r['mode']} p={r.fail_progress:.2f} "
                    f"{r.policy} — predicted")
        ax_p.set_title(title_p, fontsize=9)

        colors_d = ["#d62728" if b in damaged else "#999"
                    for b in bodies]
        ax_d.barh(ypos, [dmg_s.get(b, 0) for b in bodies], color=colors_d)
        ax_d.set_yticks(ypos)
        ax_d.set_yticklabels([_short(b) for b in bodies], fontsize=8)
        ax_d.set_xlim(-0.05, 1.1)
        ax_d.set_xlabel("realized damage (norm)")
        ax_d.set_title(f"realized — total dmg={r.realized_damage_total:.2f}, "
                        f"#damaged={len(damaged)}",
                        fontsize=9)

    fig.suptitle("Per-object predictor attention vs realized damage\n"
                  "red=actually damaged   green=predictor top-K   blue=other",
                  fontsize=12)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12,
                    help="Number of top-damage trials to display.")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    df = load(args.csv)
    print(f"loaded {len(df)} rollouts")
    # Filter to trials with at least some damage so the figure isn't all-zero
    with_dmg = df[df.realized_damage_total > 0]
    print(f"trials with damage > 0: {len(with_dmg)}")
    if len(with_dmg) == 0:
        print("Nothing to plot — no damaged trials.")
        return 1
    out = args.out or args.csv.parent / "per_object_attribution.png"
    plot(with_dmg, out, n=args.n, topk=args.topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
