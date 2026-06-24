#!/usr/bin/env python3
"""Diagnose why an injected failure produces (no) object damage.

Runs ONE failure on a trajectory/demo and, over the settle, instruments every
contact: which body pair, the world-force magnitude, and the resulting d_mech —
split into robot-vs-body vs env-vs-body contributions. Prints peak/mean forces
and the per-body damage with the held-body filter ON vs OFF, so we can see
whether impacts are (a) not happening, (b) filtered out by the held rule, or
(c) registering but crushed to ~0 by the damage-rate calibration.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="libero_spatial")
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--object", default="bowl")
    ap.add_argument("--traj", type=Path, default=None)
    ap.add_argument("--mode", default="ALL_JOINTS")
    ap.add_argument("--fail_frac", type=float, default=0.5)
    ap.add_argument("--settle_steps", type=int, default=400)
    args = ap.parse_args()

    from scripts.libero.gen_diverse_trajs import setup_scene
    from planner.experiments.libero.runner import LiberoRunner, LiberoTrialConfig
    from planner.risk.damage import (DamageAccumulator, lookup_damage_params)
    from scripts.safety.play_failure import make_failure

    ctx = setup_scene(args.task, args.split, args.demo, args.object,
                      with_predictor=False)
    runner = LiberoRunner(ctx["demo"], LiberoTrialConfig(
        resistance_mode="gravcomp_pd", post_failure_settle_steps=args.settle_steps))
    model, data, h = runner.model, runner.data, runner.handles
    robot = set(int(g) for g in h.robot_geom_ids)
    carried = ctx["carried"]["name"]
    carried_bid = ctx["carried"]["body_id"]

    def bname(gid):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[gid])) or "?"

    # seed at the failure step
    if args.traj:
        from scripts.safety.eval_trajectories import traj_from_npz, seed_step
        traj = traj_from_npz(args.traj, model)
        fq = np.asarray(ctx["demo"].finger_qpos, float); Td = len(fq)
        cf = fq[Td // 4:3 * Td // 4].mean(0) if Td >= 4 else fq.mean(0)
        T = len(traj["frames"]); t = int(args.fail_frac * (T - 1))
        arm_target = seed_step(runner, ctx, traj, t, cf, "friction")
    else:
        fs = np.asarray(ctx["demo"].full_states, float)
        T = len(fs); t = int(args.fail_frac * (T - 1))
        runner._set_full_state(fs[t])
        arm_target = np.array([data.qpos[adr] for adr in h.arm_qpos_adrs])

    fc = make_failure(args.mode, None)
    runner._inject_failure(fc)
    print(f"carried={carried} (body {carried_bid})  mode={args.mode}  "
          f"fail_step={t}/{T-1}  settle={args.settle_steps}")

    # two accumulators: with the held filter and without
    acc_on = DamageAccumulator(model, data, robot, held_body_ids={carried_bid})
    acc_off = DamageAccumulator(model, data, robot)

    pair_force = defaultdict(lambda: [0, 0.0, 0.0])  # category -> [count,sum,max]
    carried_force = defaultdict(lambda: [0, 0.0, 0.0])  # 'robot'/'env' on bowl

    def categorize(g1, g2):
        b1, b2 = bname(g1), bname(g2)
        r1, r2 = g1 in robot, g2 in robot
        def tag(b, r):
            if r:
                return "robot"
            if b == carried:
                return "BOWL"
            if "table" in b.lower():
                return "table"
            return b.replace("_main", "")
        return tuple(sorted((tag(b1, r1), tag(b2, r2))))

    for step in range(args.settle_steps):
        runner._apply_resistance(arm_target)
        mujoco.mj_step(model, data)
        acc_on.step(); acc_off.step()
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in robot and g2 in robot:
                continue
            f6 = np.zeros(6); mujoco.mj_contactForce(model, data, i, f6)
            fmag = float(np.linalg.norm(f6[:3]))
            if fmag < 1.0:
                continue
            cat = categorize(g1, g2)
            e = pair_force[cat]; e[0] += 1; e[1] += fmag; e[2] = max(e[2], fmag)
            # is the carried bowl involved, and via robot or env?
            invb = (bname(g1) == carried) or (bname(g2) == carried)
            if invb:
                other_robot = (g2 in robot) if bname(g1) == carried else (g1 in robot)
                key = "robot(grip/hit)" if other_robot else "env(table/obj)"
                e2 = carried_force[key]; e2[0] += 1; e2[1] += fmag; e2[2] = max(e2[2], fmag)

    print("\n-- contacts by body pair (count, mean|F|, max|F|) --")
    for cat, (n, s, mx) in sorted(pair_force.items(), key=lambda kv: -kv[1][2]):
        print(f"  {str(cat):34s} n={n:5d}  mean={s/max(n,1):7.1f}N  max={mx:7.1f}N")

    print("\n-- forces ON the carried bowl, by contact source --")
    for k, (n, s, mx) in carried_force.items():
        print(f"  {k:18s} n={n:5d}  mean={s/max(n,1):7.1f}N  max={mx:7.1f}N")

    p = lookup_damage_params(carried)
    print(f"\nbowl damage params: alpha={p.alpha} beta={p.beta} "
          f"threshold={p.threshold} rate={p.rate}")
    print("\n-- per-body total damage (health drop) --")
    bodies = sorted(set(acc_off.per_body_damage) | set(acc_on.per_body_damage),
                    key=lambda b: -acc_off.per_body_damage.get(b, 0))
    print(f"  {'body':34s} {'held-OFF':>10s} {'held-ON':>10s}")
    for b in bodies[:10]:
        print(f"  {b:34s} {acc_off.per_body_damage.get(b,0):10.3f} "
              f"{acc_on.per_body_damage.get(b,0):10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
