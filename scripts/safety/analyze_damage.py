#!/usr/bin/env python3
"""Analyze the damage-augmented safety-rollout CSV.

Produces:
- Spearman rho:
  * pred_risk_t0       vs contact_mass_total      (OLD metric, reproduces -0.10)
  * pred_risk_t0       vs realized_damage_total   (NEW metric)
  * pred_risk_pre_fail vs each of the above
  Plus per-mode breakdown.

- Damaged-objects classification:
  * ground truth per trial = bodies whose damage > 5
  * predictor's flagged set per trial = top-K bodies by
    pred_per_body_pre_failure (the predictor's heatmap mass integrated
    over each body's AABB at the pre-failure step)
  * Precision / recall / F1 averaged over trials.
  * Per-object confusion matrix.

Outputs a single figure ``damage_analysis.png`` next to the input CSV plus a
text summary printed to stdout.

Usage::

    python -m scripts.safety.analyze_damage \\
        --csv out/safety_rollouts_damage/.../results.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# CSV loading + per-row enrichment
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Spearman helpers
# ---------------------------------------------------------------------------

def safe_spearman(x: Sequence[float], y: Sequence[float]
                  ) -> Tuple[float, float, int]:
    """Returns (rho, p, n_used). Drops NaNs in either series."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), int(mask.sum())


def per_mode_spearman(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    rows = []
    for mode, sub in df.groupby("mode"):
        rho, p, n = safe_spearman(sub[xcol].values, sub[ycol].values)
        rows.append(dict(mode=mode, rho=rho, p=p, n=n))
    rho_all, p_all, n_all = safe_spearman(df[xcol].values, df[ycol].values)
    rows.append(dict(mode="POOLED", rho=rho_all, p=p_all, n=n_all))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Damaged-set classification
# ---------------------------------------------------------------------------

# Bodies whose AABB dominates the agentview and that aren't manipulable
# damageable objects in our LIBERO tasks.  Their predictor mass is high by
# virtue of mask area, not failure risk, so we exclude them from the top-K
# flagged set for the precision/recall analysis.  (Their per_obj counts
# still show up in the confusion table so we can see the bias.)
BACKGROUND_BODY_PATTERNS = (
    "table", "world", "cabinet_top", "cabinet_middle", "cabinet_bottom",
    "cabinet_base", "stove_burner", "wall", "ground",
)


def is_background(body_name: str) -> bool:
    name = body_name.lower()
    return any(p in name for p in BACKGROUND_BODY_PATTERNS)


def pick_topk_bodies(per_body: Dict[str, float], k: int,
                      exclude_background: bool = True) -> Set[str]:
    if not per_body:
        return set()
    items = [(b, v) for b, v in per_body.items()
             if not (exclude_background and is_background(b))]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return {b for b, v in items[:k] if v > 0}


def classification_row(actual: Set[str], predicted: Set[str]) -> dict:
    tp = len(actual & predicted)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and not np.isnan(precision)
              and not np.isnan(recall) and (precision + recall) > 0
          else float("nan"))
    return dict(tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1)


def damaged_at_threshold(per_body_damage: Dict[str, float],
                          threshold: float) -> set:
    """Recompute the damaged set from per-body damage using a fresh threshold,
    overriding whatever the CSV recorded.  Lets us calibrate the
    'is this object damaged' cutoff at analysis time, independent of the
    rollout-time DamageAccumulator setting.
    """
    if not per_body_damage:
        return set()
    return {b for b, v in per_body_damage.items() if v > threshold}


def build_classification(df: pd.DataFrame, k: int,
                          damage_threshold: float) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        actual = damaged_at_threshold(
            r.realized_damage_per_body or {}, damage_threshold)
        predicted = pick_topk_bodies(r.pred_per_body_pre_failure or {}, k)
        row = classification_row(actual, predicted)
        row.update(mode=r["mode"], policy=r.policy,
                   fail_progress=r.fail_progress,
                   init_idx=r.init_idx, n_actual=len(actual),
                   n_pred=len(predicted))
        rows.append(row)
    return pd.DataFrame(rows)


def per_object_confusion(df: pd.DataFrame, k: int,
                          damage_threshold: float) -> pd.DataFrame:
    """For each object, count {damaged & predicted, damaged & not pred, ...}"""
    counters: Dict[str, Dict[str, int]] = {}
    for _, r in df.iterrows():
        actual = damaged_at_threshold(
            r.realized_damage_per_body or {}, damage_threshold)
        predicted = pick_topk_bodies(r.pred_per_body_pre_failure or {}, k)
        seen_objects = (set(r.realized_damage_per_body or {})
                         | set(r.pred_per_body_pre_failure or {}))
        for obj in seen_objects:
            c = counters.setdefault(obj, dict(TP=0, FP=0, FN=0, TN=0))
            in_a = obj in actual
            in_p = obj in predicted
            if in_a and in_p:     c["TP"] += 1
            elif in_p:            c["FP"] += 1
            elif in_a:            c["FN"] += 1
            else:                 c["TN"] += 1
    rows = []
    for obj, c in counters.items():
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        rows.append(dict(obj=obj, **c,
                         precision=prec, recall=rec,
                         total_damaged=tp + fn,
                         total_predicted=tp + fp))
    return (pd.DataFrame(rows)
            .sort_values("total_damaged", ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(df: pd.DataFrame, cls: pd.DataFrame, per_obj: pd.DataFrame,
         out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    # --- Row 0: scatter old vs new metric, both against pred_risk_t0 ----
    ax = fig.add_subplot(gs[0, 0])
    _scatter(ax, df, "pred_risk_t0", "contact_mass_total",
             "OLD: pred_risk_t0  vs  contact_mass_total")
    ax = fig.add_subplot(gs[0, 1])
    _scatter(ax, df, "pred_risk_t0", "realized_damage_total",
             "NEW: pred_risk_t0  vs  realized_damage_total")
    ax = fig.add_subplot(gs[0, 2])
    _scatter(ax, df, "pred_risk_pre_failure", "realized_damage_total",
             "NEW: pred_risk_pre_fail  vs  realized_damage_total")

    # --- Row 1: per-mode rho bars (OLD vs NEW) + per-mode damage ranges --
    ax = fig.add_subplot(gs[1, 0])
    _per_mode_bars(ax, df, "pred_risk_t0", "contact_mass_total",
                    "Spearman rho — pred_risk_t0 vs contact_mass (OLD)")
    ax = fig.add_subplot(gs[1, 1])
    _per_mode_bars(ax, df, "pred_risk_t0", "realized_damage_total",
                    "Spearman rho — pred_risk_t0 vs realized_damage (NEW)")
    ax = fig.add_subplot(gs[1, 2])
    _damage_distribution(ax, df)

    # --- Row 2: classification metrics, per-object precision/recall -----
    ax = fig.add_subplot(gs[2, 0])
    _classification_summary(ax, cls)
    ax = fig.add_subplot(gs[2, 1])
    _per_object_bars(ax, per_obj)
    ax = fig.add_subplot(gs[2, 2])
    _confusion_matrix(ax, cls)

    fig.suptitle(
        f"OopsieVerse-style damage augmentation — {len(df)} rollouts"
        f"  (top-K=3 predictor flagging, damaged-set threshold=5)",
        fontsize=14)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nwrote {out_path}")


def _scatter(ax, df, xcol, ycol, title):
    rho_overall, p_overall, n = safe_spearman(df[xcol].values, df[ycol].values)
    colors = {"baseline": "#1f77b4", "scaling": "#ff7f0e",
              "search": "#2ca02c"}
    for pol, sub in df.groupby("policy"):
        ax.scatter(sub[xcol], sub[ycol], s=15, alpha=0.6,
                    label=pol, c=colors.get(pol, "k"))
    ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    ax.set_title(f"{title}\nrho={rho_overall:.3f} (p={p_overall:.3g}, n={n})")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xscale("symlog", linthresh=1)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)


def _per_mode_bars(ax, df, xcol, ycol, title):
    rows = per_mode_spearman(df, xcol, ycol)
    modes = rows["mode"].tolist()
    rhos = rows["rho"].tolist()
    bars = ax.bar(range(len(modes)), rhos,
                   color=["#bbb" if m == "POOLED" else "#3a7" if r > 0
                          else "#a44" for m, r in zip(modes, rhos)])
    for i, (m, r, n) in enumerate(
            zip(modes, rhos, rows["n"].tolist())):
        ax.text(i, r + (0.02 if r > 0 else -0.06),
                f"n={n}", ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Spearman rho")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)


def _damage_distribution(ax, df):
    """Compare contact_mass and realized_damage distribution per mode."""
    ax2 = ax.twinx()
    modes = sorted(df["mode"].unique())
    for i, mode in enumerate(modes):
        sub = df[df["mode"] == mode]
        cm = sub.contact_mass_total.values
        dm = sub.realized_damage_total.values
        ax.scatter([i - 0.15] * len(cm), cm, alpha=0.5, s=10,
                    c="#1f77b4")
        ax2.scatter([i + 0.15] * len(dm), dm, alpha=0.5, s=10,
                     c="#d62728")
    ax.set_yscale("symlog", linthresh=1)
    ax2.set_yscale("symlog", linthresh=0.01)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("contact_mass (blue)", color="#1f77b4")
    ax2.set_ylabel("realized_damage (red)", color="#d62728")
    ax.set_title("Per-mode distribution comparison")
    ax.grid(True, alpha=0.3)


def _classification_summary(ax, cls):
    """Overall precision/recall/F1 + per-mode."""
    # Pooled across trials with at least one damaged object
    has_damage = cls[cls.n_actual > 0]
    if len(has_damage) == 0:
        ax.text(0.5, 0.5, "No rollouts with damaged objects",
                ha="center", va="center")
        ax.set_axis_off()
        return
    pooled_prec = has_damage.precision.mean()
    pooled_rec  = has_damage.recall.mean()
    pooled_f1   = has_damage.f1.mean()

    modes = sorted(cls["mode"].unique())
    width = 0.25
    x = np.arange(len(modes))
    precs, recs, f1s = [], [], []
    for m in modes:
        sub = cls[(cls["mode"] == m) & (cls.n_actual > 0)]
        if len(sub) == 0:
            precs.append(np.nan); recs.append(np.nan); f1s.append(np.nan)
        else:
            precs.append(sub.precision.mean())
            recs.append(sub.recall.mean())
            f1s.append(sub.f1.mean())
    ax.bar(x - width, precs, width, label="precision", color="#2ca02c")
    ax.bar(x, recs, width, label="recall", color="#1f77b4")
    ax.bar(x + width, f1s, width, label="F1", color="#9467bd")
    ax.set_xticks(x); ax.set_xticklabels(modes, rotation=30,
                                          ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Damaged-objects classification\n"
                  f"pooled: P={pooled_prec:.2f} R={pooled_rec:.2f} "
                  f"F1={pooled_f1:.2f} (n={len(has_damage)} rollouts "
                  f"with damage)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)


def _per_object_bars(ax, per_obj):
    """Per-object precision/recall stacked."""
    obj_sub = per_obj[per_obj.total_damaged > 0].copy()
    if len(obj_sub) == 0:
        ax.text(0.5, 0.5, "No damaged objects in dataset",
                ha="center", va="center")
        ax.set_axis_off()
        return
    obj_sub = obj_sub.head(8)
    x = np.arange(len(obj_sub))
    width = 0.4
    ax.bar(x - width / 2, obj_sub.precision.fillna(0).values,
            width, label="precision", color="#2ca02c")
    ax.bar(x + width / 2, obj_sub.recall.fillna(0).values,
            width, label="recall", color="#1f77b4")
    # Annotate "n damaged" above
    for i, n in enumerate(obj_sub.total_damaged):
        ax.text(i, 1.03, f"n={int(n)}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(n) for n in obj_sub.obj],
                        rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title("Per-object precision / recall")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)


def _confusion_matrix(ax, cls):
    """Aggregate TP/FP/FN/TN over all (trial, object) pairs."""
    tp = cls.tp.sum(); fp = cls.fp.sum(); fn = cls.fn.sum()
    # Total (trial × object) pairs minus the above gives TN approx — use
    # n_predicted + n_actual + per-row complement
    cm = np.array([[tp, fp], [fn, 0]], dtype=int)
    ax.imshow([[tp, fp], [fn, 0]], cmap="Greens", alpha=0.4)
    for i in range(2):
        for j in range(2):
            if (i, j) == (1, 1):
                ax.text(j, i, "—", ha="center", va="center", fontsize=14)
            else:
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["damaged", "not damaged"])
    ax.set_yticklabels(["predicted", "not predicted"])
    ax.set_title("Aggregate (trial × object) confusion\n"
                 "(TN omitted; only damaged-related counts shown)",
                 fontsize=10)


def _short_name(n: str) -> str:
    return n.replace("_main", "").replace("_1", "")[:18]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True,
                    help="results.csv produced by safety_rollout.py with "
                          "damage tracking enabled")
    ap.add_argument("--topk", type=int, default=3,
                    help="Top-K bodies (by predictor mass on AABB) flagged "
                          "as 'predictor's at-risk set' per trial.")
    ap.add_argument("--damage_threshold", type=float, default=5.0,
                    help="Body counts as damaged if accumulated damage > "
                          "this value. Set to match what was used in the "
                          "rollout runner.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG path. Defaults next to CSV.")
    args = ap.parse_args()

    df = load(args.csv)
    print(f"loaded {len(df)} rollouts from {args.csv}")
    print(f"  policies : {sorted(df.policy.unique())}")
    print(f"  modes    : {sorted(df['mode'].unique())}")
    print(f"  progresses: {sorted(df.fail_progress.unique())}")

    # Sanity stats
    print("\n=== distribution sanity ===")
    print(df[["contact_mass_total",
              "realized_damage_total",
              "pred_risk_t0",
              "pred_risk_pre_failure"]].describe(
                  percentiles=[.25, .5, .75, .95]).round(3).T)

    # Spearman
    print("\n=== Spearman rho — pred_risk_t0 vs metrics ===")
    print("OLD (contact_mass_total):")
    print(per_mode_spearman(df, "pred_risk_t0",
                             "contact_mass_total").round(3).to_string(
                                index=False))
    print("\nNEW (realized_damage_total):")
    print(per_mode_spearman(df, "pred_risk_t0",
                             "realized_damage_total").round(3).to_string(
                                index=False))

    print("\n=== Spearman rho — pred_risk_pre_failure vs metrics ===")
    print("OLD (contact_mass_total):")
    print(per_mode_spearman(df, "pred_risk_pre_failure",
                             "contact_mass_total").round(3).to_string(
                                index=False))
    print("\nNEW (realized_damage_total):")
    print(per_mode_spearman(df, "pred_risk_pre_failure",
                             "realized_damage_total").round(3).to_string(
                                index=False))

    # Damaged-objects classification (recompute set from per-body damage
    # at the analysis-time threshold, overriding the CSV's baked-in value).
    cls = build_classification(df, k=args.topk,
                                damage_threshold=args.damage_threshold)
    per_obj = per_object_confusion(df, k=args.topk,
                                    damage_threshold=args.damage_threshold)
    has_dmg = cls[cls.n_actual > 0]
    print(f"\n=== Damaged-objects classification (top-K={args.topk}) ===")
    print(f"trials with at least one damaged object: "
          f"{len(has_dmg)} / {len(cls)}")
    if len(has_dmg) > 0:
        print(f"  mean precision: {has_dmg.precision.mean():.3f}")
        print(f"  mean recall   : {has_dmg.recall.mean():.3f}")
        print(f"  mean F1       : {has_dmg.f1.mean():.3f}")
    print("\nPer-object:")
    print(per_obj.round(3).to_string(index=False))

    out_path = args.out or args.csv.parent / "damage_analysis.png"
    plot(df, cls, per_obj, out_path)

    # Save the enriched per-trial classification + per-object tables too
    cls.to_csv(out_path.parent / "classification_per_trial.csv", index=False)
    per_obj.to_csv(out_path.parent / "per_object_confusion.csv", index=False)
    print(f"wrote classification_per_trial.csv + per_object_confusion.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
