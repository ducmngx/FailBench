#!/usr/bin/env python3
"""Per-step OopsieVerse-style damage video for one or more LIBERO trials.

Two modes:

1. From a results CSV — top-N highest-damage trials of a chosen policy::

       python -m scripts.safety.viz_damage_video \\
           --csv out/.../results.csv --ckpt .../best.pt \\
           --policy baseline --n 2

2. Direct task list — one trial per task with a default failure config::

       python -m scripts.safety.viz_damage_video --ckpt .../best.pt \\
           --tasks pick_up_the_black_bowl_on_the_stove_..., \\
                    pick_up_the_alphabet_soup_..., \\
                    put_the_wine_bottle_on_the_rack \\
           --mode MULTI_JOINT --joints 2,4 --fail_progress 0.55

Both modes render the same 2-panel video:

- Left: live agentview RGB + per-body health bars + red tint over damaged
  bodies, updating every step.
- Right: live predictor heatmap (always-on — queried at every step) + green
  circles around its current top-K flagged objects.

Banner color flips PRE-FAILURE → FAILURE INJECTED → POST-FAILURE.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

JSON_COLS = (
    "contact_mass_per_body", "realized_damage_per_body",
    "final_health_per_body", "damaged_objects", "damage_summary",
    "pred_per_body_pre_failure",
)


def _load(csv: Path):
    import pandas as pd
    df = pd.read_csv(csv)
    for c in JSON_COLS:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda s: json.loads(s) if isinstance(s, str) else s)
    return df


def _flip(rgb: np.ndarray) -> np.ndarray:
    return np.flipud(np.asarray(rgb)).copy()


def main() -> int:
    # Import overlay helpers from the still-image viz to avoid duplication.
    from scripts.safety.viz_damage_overlay import (
        _project_world_to_pixel, _get_camera, _short_name, _draw_health_bar,
        _tint_damaged, overlay_heatmap, overlay_topk_boxes,
        overlay_health_bars, BACKGROUND_PATTERNS,
    )
    import cv2
    import h5py
    import mujoco

    from libero.libero.envs import OffScreenRenderEnv
    from planner.experiments.config import FailureConfig, FailureMode
    from planner.experiments.libero.naming import resolve_model_handles
    from planner.policy.libero_env_failure import (
        EnvFailureScheduler, unwrap_sim)
    from planner.policy.safe_action import ObsWindow, query_risk
    from planner.risk.damage import DamageAccumulator
    from planner.risk.inference import ContactPredictor
    from scripts.safety.safety_rollout import build_entity_masks

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=None,
                    help="Path to results.csv. If given, picks the top-N "
                          "highest-damage trials of --policy.")
    ap.add_argument("--tasks", type=str, default=None,
                    help="Comma-separated task names (no _demo suffix). "
                          "Renders one trial per task with the default "
                          "failure config. Overrides --csv when set.")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--libero_root", type=Path,
                    default=Path("external/LIBERO"))
    ap.add_argument("--demo_root", type=Path,
                    default=Path("datasets/libero/raw"))
    ap.add_argument("--policy", default="baseline")
    ap.add_argument("--n", type=int, default=2,
                    help="Number of top-damage trials to render (CSV mode).")
    ap.add_argument("--init_idx", type=int, default=0,
                    help="Demo index per task in direct-tasks mode.")
    ap.add_argument("--mode", default="MULTI_JOINT",
                    help="Failure mode in direct-tasks mode.")
    ap.add_argument("--joints", default="2,4",
                    help="Comma-separated joint indices for joint-failure "
                          "modes (direct-tasks mode).")
    ap.add_argument("--fail_progress", type=float, default=0.55,
                    help="Failure progress in [0, 1] (direct-tasks mode).")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--image_h", type=int, default=240)
    ap.add_argument("--image_w", type=int, default=320)
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--damage_full_scale", type=float, default=0.30)
    ap.add_argument("--damage_thr_for_tint", type=float, default=0.05)
    ap.add_argument("--gate_threshold", type=float, default=0.5,
                    help="Gatekeeper probability cutoff. Below this, the "
                          "right-panel heatmap and top-K flags are blanked: "
                          "the model is telling us 'no contact expected, "
                          "so the localization head is not meaningful'.")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    # Resolve trial list — either from CSV (top-N) or direct tasks.
    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
        joint_list = ([int(j) for j in args.joints.split(",") if j.strip()]
                       if args.joints else [])
        trials = [dict(task=t, init_idx=args.init_idx, mode=args.mode,
                       joints=joint_list, fail_progress=args.fail_progress,
                       policy="baseline",
                       realized_damage_total=float("nan"))
                  for t in task_list]
        default_out = Path("out/safety_rollouts_damage/multi_task_videos")
    else:
        if args.csv is None:
            print("either --csv or --tasks is required"); return 1
        df = _load(args.csv)
        df = df[df.policy == args.policy].copy()
        df = df.sort_values("realized_damage_total",
                             ascending=False).head(args.n)
        if len(df) == 0:
            print("no matching trials"); return 1
        trials = df.to_dict(orient="records")
        default_out = args.csv.parent / "damage_videos"

    out_dir = args.out_dir or default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = ContactPredictor.from_checkpoint(str(args.ckpt))
    print(f"loaded predictor: {predictor.meta.arch}")

    # Group trials by task — we rebuild env + masks + clean snapshot per task
    # since each task has its own MJCF.
    from collections import defaultdict
    by_task: dict = defaultdict(list)
    for r in trials:
        by_task[r["task"]].append(r)

    trial_counter = 0
    for task, trials_for_task in by_task.items():
        bddls = list(args.libero_root.glob(f"**/bddl_files/**/{task}.bddl"))
        if not bddls:
            print(f"  ! no BDDL found for {task} — skipping"); continue
        bddl = bddls[0]
        demo = None
        for pat in (f"**/{task}.hdf5", f"**/{task}_demo.hdf5"):
            cands = list(args.demo_root.glob(pat))
            if cands: demo = cands[0]; break
        if demo is None:
            print(f"  ! no HDF5 found for {task} — skipping"); continue
        env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                  camera_heights=args.image_h,
                                  camera_widths=args.image_w)
        masks = build_entity_masks(env, (args.image_h, args.image_w))
        print(f"\n=== {task} === built {len(masks)} entity masks")
        model0, _ = unwrap_sim(env.sim)
        clean = dict(
            actuator_gainprm=model0.actuator_gainprm.copy(),
            actuator_biastype=model0.actuator_biastype.copy(),
            actuator_gaintype=model0.actuator_gaintype.copy(),
            jnt_stiffness=model0.jnt_stiffness.copy(),
            dof_damping=model0.dof_damping.copy(),
            jnt_range=model0.jnt_range.copy(),
            dof_frictionloss=model0.dof_frictionloss.copy(),
        )

        for r in trials_for_task:
            trial_counter += 1
            init_idx = int(r["init_idx"])
            with h5py.File(demo, "r") as f:
                key = f"demo_{init_idx}"
                if key not in f["data"]:
                    print(f"  skip {task}/init{init_idx}: no {key}"); continue
                actions = np.asarray(f[f"data/{key}/actions"],
                                       dtype=np.float32)
                init_state = np.asarray(f[f"data/{key}/states"][0],
                                          dtype=np.float32)
            fail_step = max(1, int(r["fail_progress"] * len(actions)))
            joints = r.get("joints", []) or []
            if isinstance(joints, str):
                joints = json.loads(joints)
            failure = FailureConfig(
                mode=FailureMode[r["mode"]], probability=1.0,
                joint_names=([f"joint{j}" for j in joints]
                              if joints else None))
            rdmg = r.get("realized_damage_total", float("nan"))
            rdmg_s = (f"dmg={rdmg:.2f}"
                       if (isinstance(rdmg, (int, float))
                           and not np.isnan(rdmg))
                       else "live")
            print(f"  trial {trial_counter}: init{init_idx} {r['mode']} "
                  f"p={r['fail_progress']:.2f}  {rdmg_s}")

            # Restore clean model state
            m_chk, _ = unwrap_sim(env.sim)
            for k, v in clean.items():
                getattr(m_chk, k)[:] = v

            sched = EnvFailureScheduler(env, failure, fail_step)
            obs = sched.reset()

            # Seed sim from demo init_state
            model, data = unwrap_sim(env.sim)
            nq, nv = model.nq, model.nv
            flat = np.asarray(init_state, dtype=np.float64)
            off = 1 if flat.shape[0] == 1 + nq + nv else 0
            data.qpos[:nq] = flat[off:off + nq]
            data.qvel[:nv] = flat[off + nq:off + nq + nv]
            mujoco.mj_forward(model, data)
            handles = resolve_model_handles(model)
            accum = DamageAccumulator(model, data, handles.robot_geom_ids)
            accum.reset()

            cam_pos, cam_mat, fovy = _get_camera(env)
            last_obs = obs
            last_pred_risk = float("nan")

            # Frame buffer for the video
            H_disp = args.image_h * args.upscale
            W_disp = args.image_w * args.upscale
            out_path = (out_dir
                        / f"trial{trial_counter:02d}_{task}_init{init_idx}_"
                           f"{r['mode']}_p{r['fail_progress']:.2f}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Banner adds 80 px on top
            banner_h = 80 * args.upscale // 2
            full_W = 2 * W_disp
            full_H = H_disp + banner_h
            writer = cv2.VideoWriter(str(out_path), fourcc, args.fps,
                                      (full_W, full_H))

            # Reusable obs window so we get the right "window context" each step
            ow = ObsWindow()

            for i in range(len(actions)):
                # Step the env first; predictor is queried POST-step so the
                # right-panel heatmap always corresponds to the same agentview
                # frame shown on the left.
                last_obs, _, done, _ = sched.step(actions[i])
                accum.step()

                # Always-on predictor query at every step.
                rgb_now = _flip(last_obs["agentview_image"])
                rgb_chw = np.transpose(rgb_now, (2, 0, 1)).astype(np.uint8)
                qpos = np.asarray(
                    last_obs.get("robot0_joint_pos", np.zeros(7)))[:7]
                qvel = np.asarray(
                    last_obs.get("robot0_joint_vel", np.zeros(7)))[:7]
                ee = np.asarray(
                    last_obs.get("robot0_eef_pos", np.zeros(3)))[:3]
                grip = np.asarray(
                    last_obs.get("robot0_gripper_qpos",
                                  np.zeros(1))).ravel()[:1]
                state_vec = np.concatenate(
                    [qpos, qvel, ee, grip]).astype(np.float32)
                ow.push(rgb_chw, state_vec)
                rq = query_risk(predictor, ow, masks=masks)
                heat = np.asarray(rq.heatmap, dtype=np.float32)
                pred_per_body = dict(rq.per_entity)
                last_pred_risk = float(rq.total_risk)
                gate_prob = float(rq.gate_prob) if rq.gate_prob is not None \
                    else float("nan")
                # Gatekeeper-gated rendering: if the classifier head says
                # "no contact expected", blank the heatmap and top-K so the
                # right panel honestly reflects the model's "no risk" claim.
                gate_fires = (not np.isnan(gate_prob)
                               and gate_prob >= args.gate_threshold)

                # Current body positions for both panels
                now_positions = {}
                for bid in range(model.nbody):
                    name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                bid) or "")
                    if name in masks:
                        now_positions[name] = np.asarray(
                            data.xpos[bid]).copy()

                if gate_fires:
                    with_heat = overlay_heatmap(rgb_now, heat, alpha=0.55)
                    right_panel = overlay_topk_boxes(
                        with_heat, now_positions, pred_per_body,
                        cam_pos, cam_mat, fovy, k=3,
                        background_patterns=BACKGROUND_PATTERNS)
                else:
                    # Gate did not fire — present a plain RGB with a small
                    # banner so the viewer knows the localization is being
                    # honestly suppressed (not a rendering bug).
                    right_panel = rgb_now.copy()
                    cv2.putText(
                        right_panel,
                        f"gate p={gate_prob:.2f} < {args.gate_threshold:.2f}",
                        (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (200, 200, 200), thickness=1, lineType=cv2.LINE_AA)
                    cv2.putText(right_panel, "no contact predicted",
                                 (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                 (140, 220, 140), thickness=1,
                                 lineType=cv2.LINE_AA)
                per_body_damage = dict(accum.per_body_damage)
                damaged = {b for b, d in per_body_damage.items()
                            if d > args.damage_thr_for_tint}
                if damaged:
                    tint_mask = np.zeros(rgb_now.shape[:2], dtype=bool)
                    for nm in damaged:
                        if nm in masks:
                            tint_mask |= masks[nm]
                    tinted = _tint_damaged(rgb_now, tint_mask)
                else:
                    tinted = rgb_now
                left_panel = overlay_health_bars(
                    tinted, now_positions, accum.per_body_health,
                    per_body_damage, cam_pos, cam_mat, fovy,
                    damage_full_scale=args.damage_full_scale,
                    skip_background=BACKGROUND_PATTERNS)
                rp = right_panel

                # Upscale
                if args.upscale > 1:
                    left_panel = cv2.resize(
                        left_panel, (W_disp, H_disp),
                        interpolation=cv2.INTER_LANCZOS4)
                    rp = cv2.resize(rp, (W_disp, H_disp),
                                     interpolation=cv2.INTER_LANCZOS4)

                stitched = np.concatenate([left_panel, rp], axis=1)

                # Banner with caption + step counter + failure marker
                banner = np.full((banner_h, full_W, 3), 30, dtype=np.uint8)
                step_label = f"step {i+1}/{len(actions)}"
                stage = ("PRE-FAILURE" if i + 1 < fail_step
                         else ("FAILURE INJECTED" if i + 1 == fail_step
                                else "POST-FAILURE"))
                stage_color = ((200, 200, 200) if "PRE" in stage
                                else ((255, 80, 80) if "INJECT" in stage
                                      else (255, 200, 80)))
                cv2.putText(banner,
                            f"init{init_idx}  {r['mode']}  fail@p="
                            f"{r['fail_progress']:.2f}  {step_label}",
                            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (235, 235, 235), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(banner, stage, (12, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, stage_color,
                            thickness=2, lineType=cv2.LINE_AA)
                gate_txt = (f"gate ON p={gate_prob:.2f}" if gate_fires
                             else (f"gate OFF p={gate_prob:.2f}"
                                    if not np.isnan(gate_prob)
                                    else "gate n/a"))
                gate_color = ((130, 240, 130) if gate_fires
                               else (240, 130, 130))
                cv2.putText(
                    banner,
                    f"damaged: {len(damaged)} body(s)   "
                    f"total damage: {sum(per_body_damage.values()):.2f}   "
                    f"pred risk: {last_pred_risk:.1f}",
                    (full_W // 2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (200, 220, 255), thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(banner, gate_txt,
                            (full_W // 2, banner_h // 2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, gate_color,
                            thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(banner, "LEFT: live state + health bars     "
                                      "RIGHT: heatmap (only when gate fires)",
                            (12, banner_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                            (170, 170, 170), thickness=1, lineType=cv2.LINE_AA)

                composite = np.concatenate([banner, stitched], axis=0)
                # cv2 wants BGR
                writer.write(composite[..., ::-1])

                if done:
                    # Hold the final frame for ~1 second so the viewer sees the
                    # outcome before the clip ends.
                    for _ in range(args.fps):
                        writer.write(composite[..., ::-1])
                    break

            writer.release()
            print(f"    wrote {out_path.name}")

        env.close()
    print(f"\nDone. Output: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
