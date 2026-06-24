#!/usr/bin/env python3
"""Generate diverse pick-and-carry trajectories in a LIBERO scene and rank them
by the learned contact/severity risk score.

LIBERO teleop demos for a task are near-identical, so there is nothing to choose
between. This synthesises a spread of genuinely different trajectories with the
same IK machinery the FailBench planner uses (Mink), then scores each with the
contact predictor + object severity — demonstrating trajectory selection on
LIBERO instead of only on the hand-built ``scenes/``.

Per trajectory we vary the carry height, a lateral detour, and the place goal,
IK each end-effector waypoint (arm joints only, on the combined LIBERO model),
densely interpolate, kinematically carry the grasped object, render the agentview
at sampled configs, and integrate the marginalised, severity-weighted risk.

Usage::

    conda run -n failbench_env python -m scripts.libero.gen_diverse_trajs \\
        --task pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate \\
        --n_trajs 6 --out figures/libero_gen_trajs.png
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import sys
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from planner.experiments.libero.adapter import load_demo, materialise_mjcf
from planner.experiments.libero.naming import resolve_model_handles
from planner.kinematics.inverse_kinematics import (
    IKSolver, IKConfig, EndEffectorTarget, IKResult)
from planner.experiments.data_capture import OffscreenRenderer

CKPT = ("notebooks/model_playground/cluster_download/06202026/"
        "dualgated_state_rgb/best.pt")
RAW = "datasets/libero/raw/libero_spatial"
LIBERO_ROOT = REPO / "external" / "LIBERO"
H, W = 240, 320


# --------------------------------------------------------------------------
# BDDL goal parsing — task-correct placement target
# --------------------------------------------------------------------------

def read_bddl(bddl_file_name: str) -> str:
    """Resolve the demo's bddl_file_name against the in-repo LIBERO tree."""
    import re as _re
    cands = [LIBERO_ROOT / bddl_file_name,
             LIBERO_ROOT / "libero" / bddl_file_name]
    cands += list(LIBERO_ROOT.glob(f"**/{Path(bddl_file_name).name}"))
    for c in cands:
        if Path(c).exists():
            return Path(c).read_text()
    raise FileNotFoundError(f"BDDL not found for {bddl_file_name}")


def parse_bddl(text: str) -> dict:
    """Extract obj_of_interest, the goal (pred, obj, target), and table regions."""
    import re as _re

    def _block(tag):
        m = _re.search(rf"\(:{tag}\b", text)
        if not m:
            return ""
        i = m.end()
        depth = 1
        while depth and i < len(text):
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        return text[m.start():i]

    ooi = _block("obj_of_interest").split(None, 1)
    ooi = ooi[1].rstrip(")").split() if len(ooi) > 1 else []

    goal_blk = _block("goal")
    gm = _re.search(r"\((On|In)\s+(\S+)\s+(\S+)\)", goal_blk, _re.I)
    goal = (gm.group(1).lower(), gm.group(2), gm.group(3).rstrip(")")) if gm else None

    regions = {}
    for rm in _re.finditer(
            r"\((\w+)\s+\(:target\s+(\w+)\)\s*"
            r"(?:\(:ranges\s*\(\s*\(([^)]*)\)\s*\))?", text):
        name, target, rng = rm.group(1), rm.group(2), rm.group(3)
        vals = [float(v) for v in rng.split()] if rng else []
        regions[name] = dict(target=target, ranges=vals)
    return dict(obj_of_interest=ooi, goal=goal, regions=regions)


def _table_info(model, data):
    """(table_body_xy, table_top_z) for the main table."""
    for bid in range(model.nbody):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
        if "table" in name:
            gids = np.where(model.geom_bodyid == bid)[0]
            if len(gids) == 0:
                continue
            top = max(float(data.geom_xpos[g][2] + model.geom_size[g][2])
                      for g in gids)
            return data.xpos[bid][:2].copy(), top
    return np.zeros(2), 0.9


def resolve_goal(parsed, ents, model, data, fallback):
    """World goal position from the BDDL goal target (object, fixture, or
    table region). Returns (carried_name_hint, goal_xyz)."""
    goal = parsed.get("goal")
    if not goal:
        return None, fallback
    pred, obj_tok, target_tok = goal

    def _ent(tok):
        # region targets like "basket_1_contain_region" name a sub-region of a
        # body ("basket_1_main"); strip the region suffix before matching.
        base = tok.lower()
        for suf in ("_contain_region", "_top_region", "_bottom_region",
                    "_top_side", "_region", "_side"):
            if base.endswith(suf):
                base = base[:-len(suf)]
                break
        return next((e for e in ents
                     if base.rstrip("_0123456789") in e["name"].lower()
                     or base in e["name"].lower()), None)

    # target is an object / fixture / container body
    te = _ent(target_tok)
    if te is not None:
        lo = np.asarray(te["aabb_min"]); hi = np.asarray(te["aabb_max"])
        c = 0.5 * (lo + hi)
        # "in" → lower into the container interior; "on" → rest on top-centre
        z = hi[2] - 0.03 if pred == "in" else hi[2] + 0.05
        return obj_tok, np.array([c[0], c[1], z])

    # target is a table region → fixture origin + region centre
    reg = parsed["regions"].get(target_tok)
    if reg and len(reg["ranges"]) >= 4:
        x0, y0, x1, y1 = reg["ranges"][:4]
        txy, tz = _table_info(model, data)
        return obj_tok, np.array([txy[0] + (x0 + x1) / 2,
                                  txy[1] + (y0 + y1) / 2, tz + 0.05])
    return obj_tok, fallback


# --------------------------------------------------------------------------
# scene helpers (live model → entities, masks, state)
# --------------------------------------------------------------------------

def scene_entities(model, data):
    """Non-robot, non-table bodies with geoms → [{name, aabb_min, aabb_max,
    body_id, freejoint_qadr}]. AABB from current geom world positions."""
    from planner.experiments.data_capture import _ROBOT_BODY_NAMES
    ents = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not name or name == "world" or name in _ROBOT_BODY_NAMES:
            continue
        if name.startswith(("robot0_", "gripper0_", "mount0_", "base0_")):
            continue
        if any(t in name.lower() for t in ("table", "wall", "floor")):
            continue
        gids = np.where(model.geom_bodyid == bid)[0]
        if len(gids) == 0:
            continue
        lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
        for g in gids:
            c = data.geom_xpos[g]; s = model.geom_size[g]
            lo = np.minimum(lo, c - s); hi = np.maximum(hi, c + s)
        # freejoint qpos address, if the body has one
        fq = -1
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == bid and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                fq = model.jnt_qposadr[j]; break
        ents.append(dict(name=name, aabb_min=lo.tolist(), aabb_max=hi.tolist(),
                         body_id=bid, freejoint_qadr=fq))
    return ents


def build_masks(entities, model, data, cam_id):
    from planner.risk.inference import aabb_to_image_mask
    cam_pos = data.cam_xpos[cam_id].astype(np.float64)
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).astype(np.float64)
    fovy = float(model.cam_fovy[cam_id])
    masks = {}
    for e in entities:
        m = aabb_to_image_mask(np.asarray(e["aabb_min"]), np.asarray(e["aabb_max"]),
                               cam_pos, cam_mat, fovy, (H, W))
        if m.any():
            masks[e["name"]] = m
    return masks


def state_vec(model, data, h):
    qpos = np.array([data.qpos[model.jnt_qposadr[j]] for j in h.arm_joint_ids],
                    np.float32)
    qvel = np.zeros(7, np.float32)
    ee = data.site_xpos[h.ee_site_id].astype(np.float32)
    grip = np.float32(data.qpos[model.jnt_qposadr[h.finger_joint_ids[0]]]
                      if getattr(h, "finger_joint_ids", None) else 0.0)
    return np.concatenate([qpos, qvel, ee, [grip]])[None]   # (1, 18)


# --------------------------------------------------------------------------
# trajectory synthesis
# --------------------------------------------------------------------------

def ik_arm(ik, model, h, seed_q, pos, quat):
    tgt = EndEffectorTarget(position=pos, orientation=quat,
                            frame_name=mujoco.mj_id2name(
                                model, mujoco.mjtObj.mjOBJ_SITE, h.ee_site_id),
                            frame_type="site")
    sol, res = ik.solve(tgt, seed_q)
    q = seed_q.copy()
    for j in h.arm_joint_ids:
        adr = model.jnt_qposadr[j]
        q[adr] = sol[adr]
    return q, res == IKResult.SUCCESS


def obstacle_clearance(ents, model, data, carried_name, obj_pos, goal, pad=0.12):
    """Tallest obstacle top-z near the straight pick→place segment (excluding the
    carried object, table, robot). Used to keep the transport arc above clutter
    so the arm/payload don't sweep through tall fixtures (e.g. the wine rack)."""
    a, b = np.asarray(obj_pos)[:2], np.asarray(goal)[:2]
    seg = b - a
    seg_len2 = float(seg @ seg) or 1.0
    top = max(obj_pos[2], goal[2])
    for e in ents:
        if e["name"] == carried_name:
            continue
        c = 0.5 * (np.asarray(e["aabb_min"]) + np.asarray(e["aabb_max"]))
        # distance from entity centre to the pick→place segment in xy
        t = np.clip(float((c[:2] - a) @ seg) / seg_len2, 0.0, 1.0)
        proj = a + t * seg
        if np.linalg.norm(c[:2] - proj) <= pad + 0.10:
            top = max(top, float(e["aabb_max"][2]))
    return top


def cluster_centroid(ents, model, data, carried_name, target_name):
    """XY centroid of the manipulable object clutter (excludes the carried
    object, the place target, and table/robot). Used to route detours away
    from the cluster."""
    pts = []
    for e in ents:
        nm = e["name"].lower()
        if e["name"] in (carried_name, target_name):
            continue
        if target_name and target_name.lower().rstrip("_0123456789") in nm:
            continue
        c = 0.5 * (np.asarray(e["aabb_min"]) + np.asarray(e["aabb_max"]))
        pts.append(c[:2])
    return np.mean(pts, axis=0) if pts else None


def perp_away(obj_pos, goal, cluster_xy):
    """Unit XY vector perpendicular to the pick→place line, pointing away from
    the cluster centroid (so +detour bows the arc around the clutter)."""
    d = (np.asarray(goal) - np.asarray(obj_pos))[:2]
    if np.linalg.norm(d) < 1e-6:
        return np.array([1.0, 0.0])
    dn = d / np.linalg.norm(d)
    perp = np.array([-dn[1], dn[0]])
    if cluster_xy is not None:
        m0 = 0.5 * (np.asarray(obj_pos) + np.asarray(goal))[:2]
        if perp @ (np.asarray(cluster_xy) - m0) > 0:   # perp points toward cluster
            perp = -perp
    return perp


def waypoints(obj_pos, ee_quat, carry_h, detour_vec, goal, clear_z=None):
    """EE waypoints for a pick-and-carry: above→grasp→lift→transport→place.

    Transport runs at ``transport_z`` = max(object-relative carry height,
    obstacle clearance + margin) and the goal is approached from directly above
    (over_goal → straight-down place), so the arm clears tall obstacles instead
    of cutting laterally through them. ``detour_vec`` is an XY offset applied to
    the transport leg as a half-sine bow, so a large value arcs the path wide
    around the object cluster rather than carrying straight over it. No full
    collision planner, but this is the same clearance heuristic scenes/ uses."""
    transport_z = obj_pos[2] + carry_h
    if clear_z is not None:
        transport_z = max(transport_z, clear_z + 0.08)
    obj_xy = np.asarray(obj_pos)[:2]
    goal_xy = np.asarray(goal)[:2]
    dv = np.asarray(detour_vec)

    above = obj_pos + [0, 0, 0.08]
    lift = np.array([obj_pos[0], obj_pos[1], transport_z])
    # bowed transport: two intermediate points offset perpendicular (half-sine)
    pts = [above, obj_pos, lift]
    for t in (1.0 / 3.0, 2.0 / 3.0):
        xy = obj_xy + t * (goal_xy - obj_xy) + dv * np.sin(np.pi * t)
        pts.append(np.array([xy[0], xy[1], transport_z]))
    over_goal = np.array([goal[0], goal[1], transport_z])
    place = goal + [0, 0, 0.04]
    pts += [over_goal, place]
    grasp_at = 1   # index after which the object is carried
    return pts, grasp_at


def interp(qa, qb, n):
    return [qa + (qb - qa) * t for t in np.linspace(0, 1, n)[1:]]


# --------------------------------------------------------------------------
# reusable scene setup + candidate generation (shared with the physics eval)
# --------------------------------------------------------------------------

def setup_scene(task, split="libero_spatial", demo="demo_0", object_hint="bowl",
                ckpt=CKPT, device=None, predictor=None, with_predictor=True,
                severity_override=None):
    """Load the LIBERO scene + predictor and resolve the carried object, place
    goal, routing geometry, and routing styles. Returns a context dict reused by
    :func:`build_candidates` and the physics evaluation (scripts/safety/
    eval_trajectories.py). Pass ``with_predictor=False`` to skip loading the
    predictor when only the scene geometry is needed."""
    if predictor is None and with_predictor:
        from scripts.safety.filter_demos import load_predictor
        predictor, _ = load_predictor(ckpt, device)

    raw_dir = REPO / "datasets" / "libero" / "raw" / split
    demo_obj = load_demo(str(raw_dir / f"{task}_demo.hdf5"), demo)
    model = mujoco.MjModel.from_xml_path(materialise_mjcf(demo_obj.model_xml))
    data = mujoco.MjData(model)
    h = resolve_model_handles(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, h.agentview_cam)
    ik = IKSolver(model, IKConfig(check_joint_limits=False))
    rend = OffscreenRenderer(model, height=H, width=W, camera_name=h.agentview_cam)

    # init state (places objects)
    fs = np.asarray(demo_obj.init_state, float)
    off = 1 if fs.shape[0] == 1 + model.nq + model.nv else 0
    data.qpos[:model.nq] = fs[off:off + model.nq]
    mujoco.mj_forward(model, data)
    q_home = data.qpos.copy()
    ee_mat = data.site_xmat[h.ee_site_id].reshape(3, 3)
    ee_quat = np.zeros(4); mujoco.mju_mat2Quat(ee_quat, ee_mat.ravel())

    ents = scene_entities(model, data)

    # task-correct carried object + goal from the BDDL
    parsed = parse_bddl(read_bddl(demo_obj.bddl_file_name))
    carried_hint = (parsed["goal"][1] if parsed.get("goal")
                    else (parsed["obj_of_interest"][0] if parsed["obj_of_interest"]
                          else object_hint))

    def _match(tok):
        return next((e for e in ents if e["freejoint_qadr"] >= 0 and
                     (tok.lower().rstrip("_0123456789") in e["name"].lower()
                      or tok.lower() in e["name"].lower())), None)

    carried = _match(carried_hint) or _match(object_hint) \
        or next(e for e in ents if e["freejoint_qadr"] >= 0)
    obj_pos = 0.5 * (np.asarray(carried["aabb_min"]) + np.asarray(carried["aabb_max"]))

    _, goal = resolve_goal(parsed, ents, model, data,
                           fallback=obj_pos + np.array([0.18, -0.12, 0.0]))
    tgt_tok = parsed["goal"][2] if parsed.get("goal") else "?"
    print(f"task={task}\n  goal predicate: {parsed.get('goal')}")
    print(f"  carried={carried['name']}  obj_pos={np.round(obj_pos,3)}")
    print(f"  place target={tgt_tok}  goal_pos={np.round(goal,3)}")

    # routing geometry: where is the clutter, which way is "around" it
    clust = cluster_centroid(ents, model, data, carried["name"], tgt_tok)
    pa = perp_away(obj_pos, goal, clust)   # unit XY, away from the cluster
    print(f"  cluster centroid xy={np.round(clust,3) if clust is not None else None}"
          f"  away-dir={np.round(pa,2)}")

    # diverse routing STYLES: (name, carry_height, signed detour magnitude along
    # pa). +detour bows the transport arc away from the cluster; -detour cuts
    # across the cluster side (deliberately riskier, for contrast).
    styles = [
        ("direct-low",   0.06,  0.00),
        ("direct-high",  0.20,  0.00),
        ("over-cluster", 0.10, -0.16),
        ("around",       0.12,  0.18),
        ("around-wide",  0.14,  0.28),
        ("around-high",  0.20,  0.22),
    ]

    clear_z = obstacle_clearance(ents, model, data, carried["name"], obj_pos, goal)
    print(f"  obstacle clearance top-z along path={clear_z:.3f} "
          f"(transport stays >= {clear_z + 0.08:.3f})")

    return dict(task=task, split=split, demo=demo_obj, predictor=predictor,
                model=model, data=data, h=h, cam_id=cam_id, ik=ik, rend=rend,
                q_home=q_home, ee_quat=ee_quat, ents=ents, carried=carried,
                obj_pos=obj_pos, goal=goal, tgt_tok=tgt_tok, pa=pa, clust=clust,
                styles=styles, clear_z=clear_z,
                severity_override=severity_override)


def build_candidates(ctx, n_score=10):
    """Run the routing-styles loop, returning per-trajectory result dicts with
    predicted severity-weighted risk plus the dense ``seq``/``carry_flags`` and
    the exact ``sample_idxs`` scored (so the physics eval can pair realized
    damage to predicted risk per config). Sorted ascending by risk."""
    from planner.risk.inference import marginal_heatmap, risk_score
    from planner.risk.severity import severity_for_entities

    model, data, h = ctx["model"], ctx["data"], ctx["h"]
    ik, rend, cam_id, predictor = ctx["ik"], ctx["rend"], ctx["cam_id"], ctx["predictor"]
    carried, obj_pos, goal = ctx["carried"], ctx["obj_pos"], ctx["goal"]
    ee_quat, q_home, pa, clear_z = (ctx["ee_quat"], ctx["q_home"], ctx["pa"],
                                    ctx["clear_z"])

    results = []
    for ti, (style, carry_h, det) in enumerate(ctx["styles"]):
        detour_vec = pa * det
        pts, grasp_at = waypoints(obj_pos, ee_quat, carry_h, detour_vec, goal,
                                  clear_z=clear_z)
        # IK the waypoints
        q = q_home.copy(); arm_qs = []; ok = True
        for p in pts:
            q, good = ik_arm(ik, model, h, q, np.asarray(p), ee_quat)
            arm_qs.append(q.copy()); ok = ok and good
        # dense config sequence + carry flags
        seq = [arm_qs[0]]; carry_flags = [False]
        for s in range(len(arm_qs) - 1):
            for qq in interp(arm_qs[s], arm_qs[s + 1], 8):
                seq.append(qq); carry_flags.append(s >= grasp_at)
        # score n_score evenly-spaced configs
        idxs = np.linspace(0, len(seq) - 1, n_score).round().astype(int)
        prof = []; risk_by_idx = {}
        for rank, ci in enumerate(idxs):
            data.qpos[:] = seq[ci]
            if carry_flags[ci] and carried["freejoint_qadr"] >= 0:
                mujoco.mj_forward(model, data)
                ee = data.site_xpos[h.ee_site_id]
                a = carried["freejoint_qadr"]
                data.qpos[a:a + 3] = ee
                data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            rgb = rend.render(data)                              # (H,W,3) u8
            rgb_w = (np.asarray(rgb, np.float32) / 255.0).transpose(2, 0, 1)[None]
            sv = state_vec(model, data, h)
            ents_now = scene_entities(model, data)
            masks = build_masks(ents_now, model, data, cam_id)
            if not masks:
                continue
            heat, _ = marginal_heatmap(predictor, rgb_w, sv)
            ovals = severity_for_entities(masks.keys(),
                                          override=ctx.get("severity_override"))
            total, _ = risk_score(heat, masks, object_values=ovals)
            prof.append((rank / (n_score - 1), float(total)))
            risk_by_idx[int(ci)] = float(total)
        score = float(np.mean([r for _, r in prof])) if prof else float("nan")
        results.append(dict(idx=ti, style=style, carry_h=carry_h, detour=det,
                            score=score, ik_ok=ok, profile=prof,
                            seq=[s.copy() for s in seq],
                            carry_flags=list(carry_flags),
                            sample_idxs=[int(x) for x in idxs],
                            pred_risk_by_idx=risk_by_idx))
        print(f"  traj {ti}: {style:13s} carry_h={carry_h:.2f} detour={det:+.2f} "
              f"ik_ok={ok}  risk={score:8.1f}")

    results.sort(key=lambda r: r["score"])
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="libero_spatial")
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--object", default="bowl",
                    help="substring of the carried object's body name")
    ap.add_argument("--n_trajs", type=int, default=6)
    ap.add_argument("--n_score", type=int, default=10,
                    help="configs scored per trajectory")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", type=Path, default=Path("figures/libero_gen_trajs.png"))
    ap.add_argument("--play_out", type=Path,
                    default=Path("figures/libero_trajs_play.mp4"))
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ctx = setup_scene(args.task, args.split, args.demo, args.object,
                      args.ckpt, args.device)
    model, data, h = ctx["model"], ctx["data"], ctx["h"]
    rend, predictor, carried = ctx["rend"], ctx["predictor"], ctx["carried"]

    results = build_candidates(ctx, args.n_score)

    print(f"\n  safest : {results[0]['style']} "
          f"(carry_h={results[0]['carry_h']:.2f} detour={results[0]['detour']:+.2f}) "
          f"risk={results[0]['score']:.1f}")
    print(f"  riskiest: {results[-1]['style']} "
          f"(carry_h={results[-1]['carry_h']:.2f} detour={results[-1]['detour']:+.2f}) "
          f"risk={results[-1]['score']:.1f}")

    # comparison figure: risk vs progress per generated trajectory
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for r in results:
        if not r["profile"]:
            continue
        xs = [p for p, _ in r["profile"]]; ys = [v for _, v in r["profile"]]
        frac = (r["score"] - results[0]["score"]) / max(
            results[-1]["score"] - results[0]["score"], 1e-6)
        ax.plot(xs, ys, "-o", ms=3, color=cmap(0.15 + 0.7 * frac),
                label=f"{r['style']} (risk {r['score']:.0f})")
    ax.set_xlabel("trajectory progress"); ax.set_ylabel("severity-weighted risk")
    ax.set_title(f"Generated LIBERO trajectories, scored by contact risk\n{args.task}",
                 fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")

    # play safest vs riskiest side by side, with predicted-contact overlay
    playable = [r for r in results if r["seq"]]
    if playable:
        sel = [(playable[0], "SAFEST", (60, 200, 60)),
               (playable[-1], "RISKIEST", (60, 60, 230))]
        play_trajs(sel, model, data, h, carried, rend, args.play_out, args.fps,
                   predictor=predictor)
    try:
        rend.close()
    except Exception:
        pass   # EGL context teardown is cosmetic on this host
    return 0


def play_trajs(sel, model, data, h, carried, rend, out, fps, predictor=None):
    """Render selected trajectories' full sequences side by side to an mp4.

    When ``predictor`` is given, the per-frame marginal contact heatmap (the
    "if a failure fired now, where would impact land" prediction) is blended
    onto each frame as a TURBO overlay, so the video shows *why* one route is
    riskier than the other."""
    import cv2
    from planner.risk.inference import marginal_heatmap
    PW, PH, HEAD, DIV = 384, 288, 0, 8
    a = carried["freejoint_qadr"]

    panels = []      # (rgb_frames, heat_frames, tag, color, result)
    for r, tag, color in sel:
        rgb_frames, heat_frames = [], []
        for ci, q in enumerate(r["seq"]):
            data.qpos[:] = q
            if r["carry_flags"][ci] and a >= 0:
                mujoco.mj_forward(model, data)
                data.qpos[a:a + 3] = data.site_xpos[h.ee_site_id]
                data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            rgb = np.asarray(rend.render(data))            # (H,W,3) RGB u8
            rgb_frames.append(rgb)
            if predictor is not None:
                rgb_w = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
                heat, _ = marginal_heatmap(predictor, rgb_w, state_vec(model, data, h))
                heat_frames.append(heat.astype(np.float32))
        panels.append((rgb_frames, heat_frames, tag, color, r))

    # shared overlay scale so the two panels are directly comparable
    vmax = 1.0
    if predictor is not None:
        allh = np.concatenate([np.stack(hf).ravel() for _, hf, *_ in panels if hf])
        vmax = float(np.percentile(allh, 99.5)) or 1.0

    def _overlay(rgb, heat):
        bgr = cv2.cvtColor(cv2.resize(rgb, (PW, PH)), cv2.COLOR_RGB2BGR)
        if heat is None:
            return bgr
        hn = np.clip(cv2.resize(heat, (PW, PH)) / vmax, 0, 1)
        cmap = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        al = (hn ** 0.7)[..., None] * 0.65          # hotter -> more opaque
        return (bgr * (1 - al) + cmap * al).astype(np.uint8)

    n = max(len(f) for f, *_ in panels) + fps        # hold last second
    W = PW * len(panels) + DIV * (len(panels) - 1)
    Ht = HEAD + PH
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Ht))
    for i in range(n):
        canvas = np.full((Ht, W, 3), 24, np.uint8)
        for j, (rgb_frames, heat_frames, tag, color, r) in enumerate(panels):
            k = min(i, len(rgb_frames) - 1)
            heat = heat_frames[k] if heat_frames else None
            img = _overlay(rgb_frames[k], heat)
            x0 = j * (PW + DIV)
            canvas[HEAD:HEAD + PH, x0:x0 + PW] = img
            cv2.rectangle(canvas, (x0, HEAD), (x0 + PW - 1, HEAD + PH - 1), color, 2)
        vw.write(canvas)
    vw.release()
    tail = "  +contact overlay" if predictor is not None else ""
    print(f"wrote {out}  ({n} frames @ {fps}fps, {len(panels)} trajectories{tail})")


if __name__ == "__main__":
    raise SystemExit(main())
