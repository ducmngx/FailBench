#!/usr/bin/env python3
"""Visualise the long-term safety evaluation (eval_trajectories.py --json).

Handles a mix of trajectory sources (e.g. human demos vs. generated routes, run
together via ``--demos``): demo trajectories are hatched / dashed so the
comparison is legible on one common safety axis.

Three panels:

  A. Safety comparison — expected / peak realized damage per trajectory (on the
     chosen channel), ordered safest -> riskiest, ☠ = bystander catastrophe rate.
  B. Damage vs. failure time — mean realized damage per sampled failure step
     along each trajectory's progress (when a failure is most costly).
  C. Damage channel split — expected environment (bystander) vs. carried-object
     self-damage per trajectory.

Usage::

    conda run -n failbench_env python -m scripts.safety.viz_traj_damage \\
        --json out/traj_eval/combined.json --out figures/traj_safety_combined.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _is_demo(t):
    return t.get("source") == "demo" or t["name"].startswith("demo")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--channel", choices=("env", "all", "carried"), default=None,
                    help="damage channel for panels A/B (default: eval's metric)")
    args = ap.parse_args()

    j = json.loads(args.json.read_text())
    ch = args.channel or j.get("metric", "env")
    n_modes = len(j.get("modes", [1] * 5))
    trajs = sorted(j["trajectories"], key=lambda s: s[ch]["expected"])
    out = args.out or Path(f"figures/traj_safety_{j['task'][:40]}.png")
    names = [t["name"] for t in trajs]
    is_demo = [_is_demo(t) for t in trajs]
    x = np.arange(len(trajs))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.9))
    cmap = plt.get_cmap("RdYlGn_r")
    col = [cmap(0.12 + 0.76 * i / max(len(trajs) - 1, 1)) for i in range(len(trajs))]
    # demos: hatched bar + navy edge; generated: plain gradient
    hatch = ["//" if d else "" for d in is_demo]
    edge = ["navy" if d else "none" for d in is_demo]
    lw = [1.6 if d else 0.0 for d in is_demo]

    # ---- A: safety comparison -------------------------------------------
    exp = [t[ch]["expected"] for t in trajs]
    peak = [t[ch]["peak"] for t in trajs]
    axA.bar(x - 0.2, exp, 0.4, color=col, hatch=hatch, edgecolor=edge,
            linewidth=lw, label="expected")
    axA.bar(x + 0.2, peak, 0.4, color=col, alpha=0.45, hatch=hatch,
            edgecolor=edge, linewidth=lw)
    for xi, t in zip(x, trajs):
        if t.get("catastrophe_bystander", 0) > 0:
            axA.text(xi, max(exp[xi], peak[xi]),
                     f"☠{t['catastrophe_bystander']:.0%}", ha="center",
                     va="bottom", fontsize=8, color="darkred")
    axA.set_xticks(x)
    tl = axA.set_xticklabels(names, rotation=25, fontsize=8, ha="right")
    for lab, d in zip(tl, is_demo):
        lab.set_color("navy" if d else "black")
    axA.set_ylabel(f"realized {ch} damage  ($d_{{mech}}$)")
    axA.set_title(f"A. Long-term safety (safest→riskiest)\n"
                  f"channel={ch}, {j['n_fail_points']}×{n_modes} inj.  "
                  f"(☠ = bystander catastrophe)", fontsize=10)
    axA.legend(handles=[
        Patch(facecolor="0.6", label="expected"),
        Patch(facecolor="0.6", alpha=0.45, label="peak"),
        Patch(facecolor="0.85", hatch="//", edgecolor="navy", label="human demo"),
        Patch(facecolor="0.85", label="generated route")],
        fontsize=7, loc="upper left")
    axA.grid(alpha=0.3, axis="y")

    # ---- B: damage vs failure time --------------------------------------
    for i, t in enumerate(trajs):
        prof = np.asarray(t["env_by_injection"], float)
        steps = t["fail_steps"]
        if len(prof) != len(steps) * n_modes:
            continue
        per_step = prof.reshape(len(steps), n_modes).mean(1)
        T = max(steps) or 1
        fr = [s / T for s in steps]
        if is_demo[i]:
            axB.plot(fr, per_step, "--", lw=2.0, color="navy", marker="s",
                     ms=4, label=t["name"], zorder=5)
        else:
            axB.plot(fr, per_step, "-o", ms=3, color=col[i], label=t["name"])
    axB.set_xlabel("failure time (trajectory progress)")
    axB.set_ylabel("mean realized env damage")
    axB.set_title("B. When is a failure most costly?", fontsize=10)
    axB.legend(fontsize=6, ncol=2); axB.grid(alpha=0.3)

    # ---- C: env vs carried channel --------------------------------------
    env_e = [t["env"]["expected"] for t in trajs]
    car_e = [t["carried"]["expected"] for t in trajs]
    axC.bar(x, env_e, 0.55, color="#c0392b", hatch=hatch, edgecolor=edge,
            linewidth=lw, label="environment (bystanders)")
    axC.bar(x, car_e, 0.55, bottom=env_e, color="#34495e", hatch=hatch,
            edgecolor=edge, linewidth=lw, label="carried-object self")
    axC.set_xticks(x)
    tl = axC.set_xticklabels(names, rotation=25, fontsize=8, ha="right")
    for lab, d in zip(tl, is_demo):
        lab.set_color("navy" if d else "black")
    axC.set_ylabel("expected realized damage")
    axC.set_title("C. Damage channel split", fontsize=10)
    axC.legend(fontsize=8); axC.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Trajectory long-term safety — demos vs generated — {j['task']}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
