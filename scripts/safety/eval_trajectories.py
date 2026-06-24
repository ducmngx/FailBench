#!/usr/bin/env python3
"""Monte-Carlo failure-cost evaluation of a robot trajectory.

You give it a trajectory (from ANY source). For that trajectory it samples N
hardware failures — each a random *time* along the trajectory and a random
*mode* — plays to that point, injects the failure, settles the physics, and
measures the realized object-damage cost. It reports the expected cost

    E[cost] = (1/N) Σ cost_i        (± Monte-Carlo standard error)

i.e. the expected damage if a failure strikes at a random moment during
execution. No predictor and no trajectory generation are involved — this is
pure physics ground truth, so two trajectories can be compared by their E[cost].

Cost channel (``--cost``): all (total d_mech over every object, default),
env (bystander objects only, excludes the carried object's own drop), or
carried (the carried object's self-damage).

Trajectory input:
  --traj FILE [FILE ...]   one or more .npz/.npy trajectories. Each holds
                           ``arm_qpos`` (T,7) or ``qpos`` (T,nq) [+ optional
                           ``carry_flags`` (T,)]. Evaluated and compared.
  --source demo            replay the LIBERO demo's recorded full states
                           (use --demo / --demos to pick which).
The scene (MJCF + which body is the carried object) is taken from --task /
--split / --demo regardless of source.

Sampling uses common random numbers across trajectories (same sampled fractions
+ modes), so comparisons are paired and reproducible via --seed. Failure time is
uniform over the trajectory; mode is uniform over the 5 modes (``--mode_dist
v2`` weights by the v2 failure likelihood instead).

Usage::

    # one trajectory, 30 sampled failures, report E[cost]
    conda run -n failbench_env python -m scripts.safety.eval_trajectories \\
        --traj my_traj.npz --task <task> --split libero_spatial \\
        --n_samples 30 --json out/eval/mine.json

    # compare several trajectories on one common sample plan
    conda run -n failbench_env python -m scripts.safety.eval_trajectories \\
        --traj a.npz b.npz c.npz --task <task> --split libero_spatial \\
        --n_samples 30
"""
from __future__ import annotations

import argparse
import csv
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import sys
from pathlib import Path
import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CKPT = ("notebooks/model_playground/cluster_download/06202026/"
        "dualgated_state_rgb/best.pt")

# a fragile object is "destroyed" (catastrophe) below this health
CATASTROPHE_HEALTH = 50.0


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------

def _spearman(x, y):
    """Spearman rank correlation (numpy-only, matches eval_accuracy.py)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def aligned_failure_modes():
    """Five FailureConfigs, one per mode type, joints per
    ``planner.risk.inference._DEFAULT_JOINTS``."""
    from planner.experiments.config import FailureMode, FailureConfig
    from planner.risk.inference import _DEFAULT_JOINTS
    out = []
    for name in ("GRIPPER_OPEN", "SLIPPERY_GRIP", "SINGLE_JOINT",
                 "MULTI_JOINT", "ALL_JOINTS"):
        kw = dict(mode=FailureMode[name])
        joints = [f"joint{j}" for j in _DEFAULT_JOINTS[name]]
        if joints:
            kw["joint_names"] = joints
        if name == "SLIPPERY_GRIP":
            kw["grip_value"] = 180.0
        out.append((name, FailureConfig(**kw)))
    return out


def v2_mode_weights(modes):
    """Failure-mode probabilities from the v2 default distribution, aggregated
    to the five mode types and normalised over ``modes``."""
    from planner.experiments.libero.runner import _default_failures
    agg = {}
    for fc in _default_failures():
        agg[fc.mode.name] = agg.get(fc.mode.name, 0.0) + float(fc.probability)
    w = np.array([agg.get(name, 0.0) for name, _ in modes], float)
    return w / w.sum() if w.sum() > 0 else None


def carried_freejoint_dofadr(model, body_id):
    for j in range(model.njnt):
        if (model.jnt_bodyid[j] == body_id
                and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_dofadr[j])
    return -1


# --------------------------------------------------------------------------
# trajectory representation + sources
# --------------------------------------------------------------------------
# A trajectory is a dict: {name, source, kind('full'|'arm'), frames, carry_flags}

def traj_from_candidate(cand, h):
    arm = np.array([[s[adr] for adr in h.arm_qpos_adrs] for s in cand["seq"]])
    return dict(name=cand["style"], source="generated", kind="arm", frames=arm,
                carry_flags=list(cand["carry_flags"]),
                pred_risk=cand.get("score"))   # predictor's risk for validation


def traj_from_demo(demo, h, name="demo"):
    fs = np.asarray(demo.full_states, float)
    fq = np.asarray(demo.finger_qpos, float)
    gap = np.abs(fq[:, 0] - fq[:, 1]) if fq.ndim == 2 else np.abs(fq)
    closed = gap < (gap.max() * 0.6 + 1e-9)
    return dict(name=name, source="demo", kind="full", frames=fs,
                carry_flags=list(closed))


def traj_from_npz(path, model):
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
        data = {"qpos" if arr.shape[1] == model.nq else "arm_qpos": arr}
    else:
        data = dict(np.load(p))
    if "arm_qpos" in data:
        frames = np.asarray(data["arm_qpos"], float); kind = "arm"
    elif "qpos" in data:
        frames = np.asarray(data["qpos"], float)
        kind = "arm" if frames.shape[1] == 7 else "full"
    else:
        raise SystemExit(f"{path}: expected 'arm_qpos' (T,7) or 'qpos' key")
    cf = data.get("carry_flags")
    cf = list(np.asarray(cf).astype(bool)) if cf is not None \
        else [True] * len(frames)
    return dict(name=p.stem, source="npz", kind=kind, frames=frames,
                carry_flags=cf)


# --------------------------------------------------------------------------
# play + inject + settle
# --------------------------------------------------------------------------

def seed_step(runner, ctx, traj, t, closed_fingers, grip):
    """Set the sim to the trajectory's nominal state at step ``t`` and return
    the arm-joint target (for healthy-joint resistance during the settle)."""
    model, data, h = runner.model, runner.data, runner.handles
    a = ctx["carried"]["freejoint_qadr"]
    if traj["kind"] == "full":
        runner._set_full_state(np.asarray(traj["frames"][t], float))
        arm_target = np.array([data.qpos[adr] for adr in h.arm_qpos_adrs])
    else:  # arm — reset the full scene to nominal first so non-carried objects
        #        don't drift across independently-sampled failures
        data.qpos[:] = ctx["q_home"]
        data.qvel[:] = 0.0
        q = np.asarray(traj["frames"][t], float)
        for adr, val in zip(h.arm_qpos_adrs, q):
            data.qpos[adr] = val
        mujoco.mj_forward(model, data)
        if traj["carry_flags"][t] and a >= 0:
            ee = data.site_xpos[h.ee_site_id].copy()
            data.qpos[a:a + 3] = ee
            data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
            if grip == "friction" and closed_fingers is not None \
                    and len(h.finger_qpos_adrs) == len(closed_fingers):
                for adr, fv in zip(h.finger_qpos_adrs, closed_fingers):
                    data.qpos[adr] = fv
        mujoco.mj_forward(model, data)
        arm_target = q[:len(h.arm_joint_ids)]
    return arm_target


def release_carried(runner, ctx):
    """Detach the carried object from the gripper so a gripper-failure actually
    drops it. Synthetic grips pin the object's centre at the end-effector in a
    gripper narrower than the object, so it stays wedged on GRIPPER_OPEN; this
    opens the fingers and nudges the object just clear of the fingertips so it
    free-falls from (near) its carry height."""
    model, data, h = runner.model, runner.data, runner.handles
    a = ctx["carried"]["freejoint_qadr"]
    if a < 0:
        return
    for adr in h.finger_qpos_adrs:
        jid = next((j for j in range(model.njnt)
                    if model.jnt_qposadr[j] == adr), None)
        if jid is not None:
            lo, hi = model.jnt_range[jid]
            data.qpos[adr] = lo if abs(lo) > abs(hi) else hi
    mujoco.mj_forward(model, data)
    ee = data.site_xpos[h.ee_site_id].copy()
    data.qpos[a:a + 3] = [ee[0], ee[1], ee[2] - 0.07]
    dof = ctx.get("_carried_dofadr")
    if dof is None:
        dof = carried_freejoint_dofadr(model, ctx["carried"]["body_id"])
    if dof >= 0:
        data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(model, data)


def settle_damage(runner, ctx, fc, mode_name, settle_steps, arm_target,
                  carry, grip):
    """Inject ``fc``, settle physics, accumulate per-object d_mech. Returns the
    per-body damage + health. State must be seeded; the failure (not the state)
    is restored before returning."""
    from planner.risk.damage import DamageAccumulator
    from planner.risk.severity import FAILURE_MODE_COMPONENT, BODY, CARRIED_OBJECT
    model, data, h = runner.model, runner.data, runner.handles
    a = ctx["carried"]["freejoint_qadr"]
    dof = ctx["_carried_dofadr"]
    repin = (grip == "repin") and carry and a >= 0 \
        and FAILURE_MODE_COMPONENT.get(mode_name) == BODY

    runner._inject_failure(fc)
    # gripper failures drop the carried object; ensure it actually releases
    if carry and FAILURE_MODE_COMPONENT.get(mode_name) == CARRIED_OBJECT:
        release_carried(runner, ctx)
    acc = DamageAccumulator(model, data, h.robot_geom_ids,
                            held_body_ids={ctx["carried"]["body_id"]})
    for _ in range(settle_steps):
        runner._apply_resistance(arm_target)
        if repin:
            mujoco.mj_forward(model, data)
            data.qpos[a:a + 3] = data.site_xpos[h.ee_site_id]
            data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
            if dof >= 0:
                data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_step(model, data)
        acc.step()
    runner.injector.restore_all()
    return dict(per_body=dict(acc.per_body_damage),
                health=dict(acc.per_body_health),
                total=float(acc.total_damage))


# --------------------------------------------------------------------------
# evaluate one trajectory: E[cost] over N sampled failures
# --------------------------------------------------------------------------

def evaluate_trajectory(runner, ctx, traj, modes, fracs, midx, settle_steps,
                        grip, closed_fingers, fragile_objs, cost_channel,
                        sev_fn=None):
    """Sample failures (time fraction ``fracs[k]`` × mode ``midx[k]``), measure
    each one's realized-damage cost, and return E[cost] (+SE) and the records.

    ``sev_fn(name)`` weights each object's realized damage by its severity (so an
    object's HP loss counts toward the cost in proportion to how much we value
    it). Defaults to 1.0 (raw damage)."""
    if sev_fn is None:
        sev_fn = lambda _name: 1.0
    carried_name = ctx["carried"]["name"]
    T = len(traj["frames"])
    if T < 2:
        raise SystemExit(f"trajectory {traj['name']!r} too short ({T})")

    costs, env_l, car_l = [], [], []
    catC = catB = 0
    records = []
    for k in range(len(fracs)):
        t = int(round(float(fracs[k]) * (T - 1)))
        mode_name, fc = modes[int(midx[k])]
        arm_target = seed_step(runner, ctx, traj, t, closed_fingers, grip)
        carry = bool(traj["carry_flags"][t])
        d = settle_damage(runner, ctx, fc, mode_name, settle_steps,
                          arm_target, carry, grip)
        # severity-weighted per-object damage (HP loss × value)
        env = sum(sev_fn(b) * v for b, v in d["per_body"].items()
                  if b != carried_name)
        allc = sum(sev_fn(b) * v for b, v in d["per_body"].items())
        carc = sev_fn(carried_name) * d["per_body"].get(carried_name, 0.0)
        cost = {"all": allc, "env": env, "carried": carc}[cost_channel]
        costs.append(cost); env_l.append(env); car_l.append(carc)
        destroyed = [o for o in fragile_objs
                     if d["health"].get(o, 100.0) < CATASTROPHE_HEALTH]
        broke_carried = carried_name in destroyed
        bystanders = [o for o in destroyed if o != carried_name]
        catC += int(broke_carried); catB += int(bool(bystanders))
        records.append(dict(trajectory=traj["name"], sample=k, step=t,
                            frac=round(t / (T - 1), 3), mode=mode_name,
                            cost=float(cost), env_damage=float(env),
                            all_damage=float(allc), carried_damage=float(carc),
                            carried_broken=int(broke_carried),
                            bystanders_broken="|".join(bystanders)))
    c = np.asarray(costs, float)
    n = max(1, len(c))
    se = float(c.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return dict(
        name=traj["name"], source=traj.get("source", "?"),
        pred_risk=traj.get("pred_risk"),
        cost_channel=cost_channel, n_samples=len(c),
        total=float(c.sum()), E_cost=float(c.mean()), se=se, std=float(c.std()),
        median=float(np.median(c)), p90=float(np.percentile(c, 90)),
        max=float(c.max()), mean_env=float(np.mean(env_l)),
        mean_carried=float(np.mean(car_l)),
        catastrophe_carried=catC / n, catastrophe_bystander=catB / n,
        costs=[float(x) for x in c]), records


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", type=Path, nargs="+",
                    help="trajectory file(s) (.npz/.npy) — the primary input")
    ap.add_argument("--source", choices=("npz", "demo", "gen"), default="npz",
                    help="used when --traj is not given")
    ap.add_argument("--task", required=True, help="scene/MJCF + carried object")
    ap.add_argument("--split", default="libero_spatial")
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--demos", default=None,
                    help="comma-separated demo keys to ALSO evaluate")
    ap.add_argument("--object", default="bowl")
    ap.add_argument("--n_samples", type=int, default=30,
                    help="number of sampled failures per trajectory")
    ap.add_argument("--settle_steps", type=int, default=400)
    ap.add_argument("--grip", choices=("friction", "repin", "none"),
                    default="friction")
    ap.add_argument("--cost", choices=("all", "env", "carried"), default="all",
                    help="damage channel used as the cost")
    ap.add_argument("--severity_config", type=Path, default=None,
                    help="YAML/JSON {object_substring: severity}; weights each "
                         "object's HP loss in the cost and feeds the predictor")
    ap.add_argument("--mode_dist", choices=("uniform", "v2"), default="uniform")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=CKPT, help="only for --source gen")
    ap.add_argument("--n_gen", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/traj_eval/eval.csv"))
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from scripts.libero.gen_diverse_trajs import setup_scene, build_candidates
    from planner.experiments.libero.runner import LiberoRunner, LiberoTrialConfig
    from planner.risk.severity import (entity_severity, load_severity_config,
                                       resolve_severity)

    # manual severity config: weights each object's HP loss in the cost AND is
    # fed to the predictor, so both sides value objects identically.
    sev_map = load_severity_config(args.severity_config) if args.severity_config else None
    sev_fn = ((lambda name: resolve_severity(name, override=sev_map))
              if sev_map else (lambda _name: 1.0))
    if sev_map:
        print(f"severity config: {sev_map}")

    need_pred = (args.source == "gen") and not args.traj
    ctx = setup_scene(args.task, args.split, args.demo, args.object,
                      args.ckpt, args.device, with_predictor=need_pred,
                      severity_override=sev_map)

    runner = LiberoRunner(
        ctx["demo"],
        LiberoTrialConfig(resistance_mode="gravcomp_pd",
                          post_failure_settle_steps=args.settle_steps))
    h = runner.handles
    ctx["_carried_dofadr"] = carried_freejoint_dofadr(
        runner.model, ctx["carried"]["body_id"])

    fq = np.asarray(ctx["demo"].finger_qpos, float)
    Td = len(fq)
    closed_fingers = fq[Td // 4: 3 * Td // 4].mean(0) if Td >= 4 else fq.mean(0)
    fragile = [e["name"] for e in ctx["ents"]
               if entity_severity(e["name"], scale="paper") >= 10.0]

    # assemble trajectory list
    if args.traj:
        trajs = [traj_from_npz(p, runner.model) for p in args.traj]
    elif args.source == "demo":
        trajs = [traj_from_demo(ctx["demo"], h, name=args.demo)]
    else:  # gen
        trajs = [traj_from_candidate(c, h) for c in build_candidates(ctx, args.n_gen)]
    if args.demos:
        from planner.experiments.libero.adapter import load_demo
        raw = REPO / "datasets" / "libero" / "raw" / args.split / \
            f"{args.task}_demo.hdf5"
        have = {t["name"] for t in trajs}
        for dk in (d.strip() for d in args.demos.split(",")):
            if dk and dk not in have:
                trajs.append(traj_from_demo(load_demo(str(raw), dk), h, name=dk))

    # common random numbers: one sample plan (fraction × mode) for all trajectories
    modes = aligned_failure_modes()
    mode_p = v2_mode_weights(modes) if args.mode_dist == "v2" else None
    rng = np.random.default_rng(args.seed)
    fracs = rng.random(args.n_samples)
    midx = rng.choice(len(modes), size=args.n_samples, p=mode_p)

    print(f"\ncarried={ctx['carried']['name']}  fragile={fragile or '[none]'}")
    print(f"sampling {args.n_samples} failures/traj  (time~uniform, mode~"
          f"{args.mode_dist})  cost={args.cost}  settle={args.settle_steps}  "
          f"seed={args.seed}")
    print(f"{'trajectory':16s} {'E[cost]':>9s} {'±SE':>7s} {'total':>9s} "
          f"{'std':>7s} {'median':>7s} {'p90':>7s} {'max':>8s}  "
          f"catastrophe(carried/bystander)")

    summaries, rows = [], []
    for traj in trajs:
        s, recs = evaluate_trajectory(
            runner, ctx, traj, modes, fracs, midx, args.settle_steps,
            args.grip, closed_fingers, fragile, args.cost, sev_fn=sev_fn)
        summaries.append(s); rows.extend(recs)
        print(f"{s['name']:16s} {s['E_cost']:9.2f} {s['se']:7.2f} {s['total']:9.1f} "
              f"{s['std']:7.2f} {s['median']:7.2f} {s['p90']:7.2f} {s['max']:8.2f}  "
              f"{s['catastrophe_carried']:.0%} / {s['catastrophe_bystander']:.0%}")

    if len(summaries) > 1:
        ranked = sorted(summaries, key=lambda s: s["E_cost"])
        print(f"\n=== ranked by E[{args.cost} cost] (safest first) ===")
        for i, s in enumerate(ranked):
            pr = f"  pred_risk={s['pred_risk']:.0f}" if s.get("pred_risk") else ""
            print(f"  {i+1}. {s['name']:16s} E[cost]={s['E_cost']:8.2f} "
                  f"± {s['se']:.2f}  total={s['total']:8.1f}{pr}")
        if ranked[-1]["E_cost"] > 0:
            r = 100 * (1 - ranked[0]["E_cost"] / ranked[-1]["E_cost"])
            print(f"  safest '{ranked[0]['name']}' is {r:.0f}% lower E[cost] "
                  f"than riskiest '{ranked[-1]['name']}'")

        # validation: does the predictor's preferred trajectory score better?
        pr = [s["pred_risk"] for s in summaries]
        if all(p is not None for p in pr):
            ec = [s["E_cost"] for s in summaries]
            rho = _spearman(pr, ec)
            pred_best = min(summaries, key=lambda s: s["pred_risk"])
            phys_best = ranked[0]
            print(f"\n=== predictor vs physics ===")
            print(f"  Spearman(pred_risk, E[cost]) = {rho:+.3f}  (>0 ⇒ predictor "
                  f"agrees with physics)")
            print(f"  predictor's pick: {pred_best['name']:16s} "
                  f"E[cost]={pred_best['E_cost']:.2f}")
            print(f"  physics-safest  : {phys_best['name']:16s} "
                  f"E[cost]={phys_best['E_cost']:.2f}")
            print(f"  predictor's pick is physics-safest: "
                  f"{pred_best['name'] == phys_best['name']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.out}  ({len(rows)} samples)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        pr = [s["pred_risk"] for s in summaries]
        rho_pp = (_spearman(pr, [s["E_cost"] for s in summaries])
                  if len(summaries) > 1 and all(p is not None for p in pr) else None)
        args.json.write_text(json.dumps(dict(
            task=args.task, split=args.split, cost=args.cost,
            n_samples=args.n_samples, settle_steps=args.settle_steps,
            mode_dist=args.mode_dist, seed=args.seed, grip=args.grip,
            fragile=fragile, modes=[m for m, _ in modes],
            rho_pred_vs_physics=rho_pp,
            trajectories=summaries), indent=2, default=float))
        print(f"wrote {args.json}")

    try:
        ctx["rend"].close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
