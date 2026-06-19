#!/usr/bin/env python3
"""OopsieVerse-style damage overlay on LIBERO agentview frames.

Re-runs a small number of high-damage rollouts (deterministic from
init_state + demo replay).  For each, captures:

- the pre-failure RGB frame (last step before the failure injects)
- the post-rollout RGB frame (end of episode)

and overlays:

- Per-object health bars at each body's projected 2D centroid
- Red tint on bodies whose health dropped below a threshold ("damaged")
- Predictor's heatmap (semi-transparent) on the pre-failure frame
- Predictor's top-K AABB flags as bounding boxes

Output: one mp4 per trial, or one PNG with both frames side-by-side.

Usage::

    python -m scripts.safety.viz_damage_overlay \\
        --csv out/safety_rollouts_damage/.../results.csv \\
        --ckpt notebooks/.../best_ep08_val0.0648.pt \\
        --n 4 --policy baseline
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

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


def _project_world_to_pixel(point_world: np.ndarray, cam_pos: np.ndarray,
                              cam_mat: np.ndarray, fovy_deg: float,
                              image_hw: Tuple[int, int]
                              ) -> Optional[Tuple[int, int]]:
    """Project a world-space 3D point to pixel coordinates of the agentview.

    Uses the same convention as planner/risk/inference.py::aabb_to_image_mask.
    Returns ``(row, col)`` in image-space (Y-down).  None if behind camera.
    """
    H, W = image_hw
    # Camera frame: cam_mat columns are camera basis vectors in world.
    # MuJoCo convention: camera looks down -Z in its own frame; +Y is up.
    rel = point_world - cam_pos
    cam_x = np.array([cam_mat[0, 0], cam_mat[1, 0], cam_mat[2, 0]])
    cam_y = np.array([cam_mat[0, 1], cam_mat[1, 1], cam_mat[2, 1]])
    cam_z = np.array([cam_mat[0, 2], cam_mat[1, 2], cam_mat[2, 2]])
    px = rel @ cam_x
    py = rel @ cam_y
    pz = rel @ cam_z
    if pz >= 0:
        return None  # behind camera (MuJoCo: -Z is forward)
    f = 0.5 * H / np.tan(np.deg2rad(fovy_deg) / 2)
    # u/v in image-plane coords; (0,0) at image center, x right, y up
    u = -f * px / pz
    v = -f * py / pz
    # To pixel coords (col, row) in image-down convention.  np.flipud is
    # applied to the rendered RGB to align with the v2 dataset convention,
    # so the row index is computed from the flipped image.
    col = int(round(W / 2 + u))
    row_y_up = int(round(H / 2 + v))
    row = H - 1 - row_y_up  # flipud convention
    if not (0 <= row < H and 0 <= col < W):
        return None
    return row, col


def _get_camera(env):
    """Pull agentview camera pose + intrinsics from the env's MjModel."""
    import mujoco
    from planner.policy.libero_env_failure import unwrap_sim
    from planner.experiments.libero.naming import resolve_model_handles
    model, data = unwrap_sim(env.sim)
    handles = resolve_model_handles(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                handles.agentview_cam)
    cam_pos = np.asarray(data.cam_xpos[cam_id]).copy()
    cam_mat = np.asarray(data.cam_xmat[cam_id]).reshape(3, 3).copy()
    fovy = float(model.cam_fovy[cam_id])
    return cam_pos, cam_mat, fovy


def _short_name(s: str) -> str:
    return (s.replace("_main", "")
              .replace("akita_black_", "")
              .replace("glazed_rim_porcelain_", "")
              .replace("wooden_", "")
              .replace("flat_stove_1_", "stove_")
              .replace("_1", ""))[:16]


def _draw_health_bar(img: np.ndarray, row: int, col: int,
                      health: float, h_max: float,
                      label: str, *,
                      bar_w: int = 60, bar_h: int = 7,
                      offset_row: int = -22) -> None:
    """Render an OopsieVerse-style horizontal health bar on ``img`` (RGB)."""
    import cv2
    H, W = img.shape[:2]
    r = max(0, min(H - bar_h - 4, row + offset_row))
    c = max(2, min(W - bar_w - 2, col - bar_w // 2))
    pct = float(np.clip(health / h_max, 0, 1))
    # Color ramp: green -> yellow -> red as health drops
    if pct > 0.7:
        color = (40, 200, 40)
    elif pct > 0.3:
        color = (240, 200, 40)
    else:
        color = (220, 50, 50)
    # Background bar
    cv2.rectangle(img, (c, r), (c + bar_w, r + bar_h),
                  (40, 40, 40), thickness=-1)
    # Filled portion
    fill = max(0, int(pct * bar_w))
    cv2.rectangle(img, (c, r), (c + fill, r + bar_h),
                  color, thickness=-1)
    # Border
    cv2.rectangle(img, (c, r), (c + bar_w, r + bar_h),
                  (10, 10, 10), thickness=1)
    # Label above
    cv2.putText(img, label, (c, r - 3), cv2.FONT_HERSHEY_SIMPLEX,
                0.32, (10, 10, 10), thickness=1, lineType=cv2.LINE_AA)


def _tint_damaged(img: np.ndarray, mask: np.ndarray,
                   strength: float = 0.45) -> np.ndarray:
    """Apply a red tint where ``mask`` is True."""
    out = img.copy()
    red = np.array([220, 30, 30], dtype=np.float32)
    out[mask] = (out[mask].astype(np.float32) * (1 - strength)
                  + red * strength).astype(np.uint8)
    return out


def _render_agentview_from_obs(obs: dict) -> np.ndarray:
    """Read agentview RGB from the latest step's obs dict, in v2-convention.

    Robosuite's OffScreenRenderEnv returns the OpenGL framebuffer (Y-up);
    the v2 dataset and our predictor expect Y-down.  Flip on read.
    """
    rgb = obs.get("agentview_image")
    if rgb is None:
        for k in obs:
            if "agentview" in k and "image" in k:
                rgb = obs[k]; break
    if rgb is None:
        raise RuntimeError(f"no agentview image in obs: keys={sorted(obs)}")
    return np.flipud(np.asarray(rgb)).copy()


def _replay_until(env, demo_actions, target_step: int):
    """Replay env from current state until step `target_step`.  Returns
    the observation dict after stepping `target_step` times (1-based count).
    """
    obs = None
    for i in range(target_step):
        obs, _, done, _ = env.step(demo_actions[i])
        if done:
            break
    return obs


def replay_trial(env, deps, demo_actions, init_state, failure_cfg: dict,
                  fail_step: int, masks: dict,
                  clean_model_state: dict) -> Dict:
    """Replay one trial, capturing pre-failure and post-rollout state.

    Returns a dict with rendered RGBs, per-body positions, per-body health,
    predictor heatmap + per-body attention at pre-failure.
    """
    import mujoco
    from planner.experiments.config import FailureConfig, FailureMode
    from planner.policy.libero_env_failure import (
        EnvFailureScheduler, unwrap_sim)
    from planner.policy.safe_action import (
        ObsWindow, PassthroughPolicy, query_risk)
    from planner.risk.damage import DamageAccumulator
    from planner.experiments.libero.naming import resolve_model_handles

    # Restore clean state
    _model_check, _ = unwrap_sim(env.sim)
    for k, v in clean_model_state.items():
        getattr(_model_check, k)[:] = v

    failure = FailureConfig(
        mode=FailureMode[failure_cfg["mode"]],
        probability=1.0,
        joint_names=[f"joint{j}" for j in failure_cfg.get("joints", [])]
                     if failure_cfg.get("joints") else None,
    )
    scheduler = EnvFailureScheduler(env, failure, fail_step)
    obs = scheduler.reset()

    # Seed from init_state to align with the demo
    model, data = unwrap_sim(env.sim)
    nq, nv = model.nq, model.nv
    flat = np.asarray(init_state, dtype=np.float64)
    off = 1 if flat.shape[0] == 1 + nq + nv else 0
    data.qpos[:nq] = flat[off:off + nq]
    data.qvel[:nv] = flat[off + nq:off + nq + nv]
    mujoco.mj_forward(model, data)

    handles = resolve_model_handles(model)
    damage_accum = DamageAccumulator(model, data, handles.robot_geom_ids)
    damage_accum.reset()

    # Replay up to (fail_step - 1) — that's the snapshot just before failure
    pre_step = max(1, fail_step - 1)
    last_obs = obs
    for i in range(pre_step):
        last_obs, _, done, _ = scheduler.step(demo_actions[i])
        damage_accum.step()
        if done:
            break

    # Capture pre-failure frame + body positions
    pre_rgb = _render_agentview_from_obs(last_obs)
    pre_positions = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name and name in masks:
            pre_positions[name] = np.asarray(data.xpos[bid]).copy()

    # Predictor query at pre-failure step
    pred_heatmap = None
    pred_per_body = {}
    pred_total = float("nan")
    if predictor_global is not None:
        ow = ObsWindow()
        rgb_chw = np.transpose(pre_rgb, (2, 0, 1)).astype(np.uint8)
        qpos = np.asarray(obs.get("robot0_joint_pos", np.zeros(7)))[:7]
        qvel = np.asarray(obs.get("robot0_joint_vel", np.zeros(7)))[:7]
        ee = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)))[:3]
        grip = np.asarray(obs.get("robot0_gripper_qpos",
                                     np.zeros(1))).ravel()[:1]
        state_vec = np.concatenate([qpos, qvel, ee, grip]).astype(np.float32)
        ow.push(rgb_chw, state_vec)
        rq = query_risk(predictor_global, ow, masks=masks)
        pred_heatmap = np.asarray(rq.heatmap, dtype=np.float32)
        pred_per_body = dict(rq.per_entity)
        pred_total = rq.total_risk

    # Step through the rest of the demo
    for i in range(pre_step, len(demo_actions)):
        last_obs, _, done, _ = scheduler.step(demo_actions[i])
        damage_accum.step()
        if done:
            break

    # Capture post-rollout frame + final body positions + health
    post_rgb = _render_agentview_from_obs(last_obs)
    post_positions = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name and name in masks:
            post_positions[name] = np.asarray(data.xpos[bid]).copy()

    health = damage_accum.per_body_health
    damage_total = damage_accum.total_damage

    return dict(
        pre_rgb=pre_rgb,
        post_rgb=post_rgb,
        pre_positions=pre_positions,
        post_positions=post_positions,
        health=health,
        per_body_damage=dict(damage_accum.per_body_damage),
        damage_total=damage_total,
        pred_heatmap=pred_heatmap,
        pred_per_body=pred_per_body,
        pred_total=pred_total,
        damaged_set=damage_accum.damaged_set,
    )


predictor_global = None  # set in main(); used by replay_trial for risk query


def overlay_health_bars(rgb: np.ndarray, positions: Dict[str, np.ndarray],
                         health: Dict[str, float],
                         damage: Dict[str, float],
                         cam_pos, cam_mat, fovy,
                         damage_full_scale: float = 0.5,
                         skip_background: Iterable[str] = ()) -> np.ndarray:
    """Draw OopsieVerse-style health bars over each named body.

    Maps raw damage (typically 0..0.5 with our LIBERO rate constants) to a
    visible 0..100 percentage by dividing by ``damage_full_scale``.  Bodies
    not in ``damage`` are assumed full-health.
    """
    out = rgb.copy()
    H, W = out.shape[:2]
    skips = tuple(skip_background)
    for name, p in positions.items():
        if any(s in name.lower() for s in skips):
            continue
        proj = _project_world_to_pixel(p, cam_pos, cam_mat, fovy, (H, W))
        if proj is None:
            continue
        row, col = proj
        d = float(damage.get(name, 0.0))
        h_pct = max(0.0, 100.0 * (1.0 - d / max(damage_full_scale, 1e-9)))
        _draw_health_bar(out, row, col, h_pct, 100.0, _short_name(name))
    return out


def overlay_heatmap(rgb: np.ndarray, heat: np.ndarray,
                     alpha: float = 0.55) -> np.ndarray:
    """Overlay a normalized heatmap on the RGB.  Uses MAGMA colormap."""
    import cv2
    if heat is None:
        return rgb.copy()
    H, W = rgb.shape[:2]
    if heat.shape != (H, W):
        heat = cv2.resize(heat.astype(np.float32), (W, H),
                          interpolation=cv2.INTER_LINEAR)
    h = heat - heat.min()
    if h.max() > 1e-9:
        h = h / h.max()
    h_u8 = (255 * h).astype(np.uint8)
    cmap = cv2.applyColorMap(h_u8, cv2.COLORMAP_MAGMA)[..., ::-1]  # BGR->RGB
    # Only blend where heatmap > small threshold to keep RGB visible elsewhere
    mask = (h > 0.05)[..., None]
    out = rgb.astype(np.float32).copy()
    out = np.where(mask, out * (1 - alpha) + cmap.astype(np.float32) * alpha,
                    out)
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_topk_boxes(rgb: np.ndarray, positions: Dict[str, np.ndarray],
                        pred_per_body: Dict[str, float],
                        cam_pos, cam_mat, fovy, k: int,
                        background_patterns: Iterable[str]) -> np.ndarray:
    """Draw green circles around the top-K predictor-flagged objects."""
    import cv2
    H, W = rgb.shape[:2]
    bg = tuple(background_patterns)
    items = [(n, v) for n, v in pred_per_body.items()
             if not any(b in n.lower() for b in bg)]
    items.sort(key=lambda kv: kv[1], reverse=True)
    out = rgb.copy()
    for name, val in items[:k]:
        if val <= 0:
            continue
        p = positions.get(name)
        if p is None:
            continue
        proj = _project_world_to_pixel(p, cam_pos, cam_mat, fovy, (H, W))
        if proj is None:
            continue
        row, col = proj
        cv2.circle(out, (col, row), 18, (40, 220, 40), thickness=2)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BACKGROUND_PATTERNS = ("table", "cabinet", "world", "wall", "ground",
                         "stove_burner")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--libero_root", type=Path,
                    default=Path("external/LIBERO"))
    ap.add_argument("--demo_root", type=Path,
                    default=Path("datasets/libero/raw"))
    ap.add_argument("--policy", default="baseline",
                    help="Filter trials by policy name (baseline / scaling).")
    ap.add_argument("--n", type=int, default=4,
                    help="Number of top-damage trials to render.")
    ap.add_argument("--image_h", type=int, default=240)
    ap.add_argument("--image_w", type=int, default=320)
    ap.add_argument("--upscale", type=int, default=2,
                    help="Upscale output for legibility.")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    import cv2
    import pandas as pd
    df = _load(args.csv)
    df = df[df.policy == args.policy].copy()
    df = df.sort_values("realized_damage_total", ascending=False).head(args.n)
    if len(df) == 0:
        print("No trials match the filter.")
        return 1
    out_dir = args.out_dir or (args.csv.parent / "damage_overlay")
    out_dir.mkdir(parents=True, exist_ok=True)
    task = df.task.iloc[0]
    print(f"task: {task}")
    print(f"top-{args.n} trials by damage:")
    print(df[["init_idx", "mode", "fail_progress", "policy",
              "realized_damage_total"]].to_string(index=False))

    # Lazy imports — heavy
    from libero.libero.envs import OffScreenRenderEnv
    from planner.policy.libero_env_failure import unwrap_sim
    from planner.risk.inference import ContactPredictor

    # BDDL + demo
    bddls = list(args.libero_root.glob(f"**/bddl_files/**/{task}.bddl"))
    bddl = bddls[0]
    demos = []
    for pat in (f"**/{task}.hdf5", f"**/{task}_demo.hdf5"):
        demos.extend(args.demo_root.glob(pat))
    demo = demos[0]

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.image_h, camera_widths=args.image_w)

    # Build masks and clean snapshot once
    from scripts.safety.safety_rollout import build_entity_masks
    masks = build_entity_masks(env, (args.image_h, args.image_w))
    print(f"  built {len(masks)} entity masks")
    model0, _ = unwrap_sim(env.sim)
    clean_state = dict(
        actuator_gainprm=model0.actuator_gainprm.copy(),
        actuator_biastype=model0.actuator_biastype.copy(),
        actuator_gaintype=model0.actuator_gaintype.copy(),
        jnt_stiffness=model0.jnt_stiffness.copy(),
        dof_damping=model0.dof_damping.copy(),
        jnt_range=model0.jnt_range.copy(),
        dof_frictionloss=model0.dof_frictionloss.copy(),
    )

    # Load predictor
    global predictor_global
    predictor_global = ContactPredictor.from_checkpoint(str(args.ckpt))
    print(f"  loaded predictor: {predictor_global.meta.arch}")

    # Load demo actions per init_idx
    import h5py
    deps = None  # unused

    for trial_i, (_, r) in enumerate(df.iterrows()):
        init_idx = int(r.init_idx)
        with h5py.File(demo, "r") as f:
            demo_actions = np.asarray(
                f[f"data/demo_{init_idx}/actions"], dtype=np.float32)
            init_state = np.asarray(
                f[f"data/demo_{init_idx}/states"][0], dtype=np.float32)

        fail_step = max(1, int(r.fail_progress * len(demo_actions)))
        failure_cfg = dict(mode=r["mode"],
                            joints=json.loads(r.joints)
                                if isinstance(r.joints, str) else r.joints)
        print(f"  trial {trial_i+1}/{args.n}: init{init_idx} "
              f"{r['mode']} p={r.fail_progress:.2f} "
              f"dmg={r.realized_damage_total:.2f}")

        result = replay_trial(env, deps, demo_actions, init_state,
                              failure_cfg, fail_step, masks, clean_state)

        cam_pos, cam_mat, fovy = _get_camera(env)

        # Build the four panels:
        #  1) pre-failure RGB
        #  2) pre-failure + predictor heatmap + top-K flags
        #  3) post-rollout RGB
        #  4) post-rollout + per-object health bars + damaged tint
        h_max = 100.0

        # Damaged set computed with the analysis-time threshold so the
        # overlay matches the precision/recall tables.
        # Raw d_mech damage per body (NOT 100-health, since health is
        # clamped to [0, h_max] when total_damage exceeds 100).
        per_body_damage = result["per_body_damage"]
        DAMAGE_THR_FOR_TINT = 0.05  # raw d_mech units; matches analyzer thr
        DAMAGE_FULL_SCALE = 0.30    # half-saturation point for bar color
        damaged = {b for b, d in per_body_damage.items()
                    if d > DAMAGE_THR_FOR_TINT}

        if damaged:
            tint_mask = np.zeros(result["post_rgb"].shape[:2], dtype=bool)
            for name in damaged:
                if name in masks:
                    tint_mask |= masks[name]
            post_tinted = _tint_damaged(result["post_rgb"], tint_mask)
        else:
            post_tinted = result["post_rgb"]
        post_with_bars = overlay_health_bars(
            post_tinted, result["post_positions"], result["health"],
            per_body_damage, cam_pos, cam_mat, fovy,
            damage_full_scale=DAMAGE_FULL_SCALE,
            skip_background=BACKGROUND_PATTERNS)

        # Panel 2: predictor heat + top-K flags on pre-failure
        pre_with_heat = overlay_heatmap(result["pre_rgb"],
                                          result["pred_heatmap"], alpha=0.55)
        pre_with_flags = overlay_topk_boxes(
            pre_with_heat, result["pre_positions"], result["pred_per_body"],
            cam_pos, cam_mat, fovy, k=3,
            background_patterns=BACKGROUND_PATTERNS)

        # Stitch 2x2
        H, W = result["post_rgb"].shape[:2]
        row1 = np.concatenate([result["pre_rgb"], pre_with_flags], axis=1)
        row2 = np.concatenate([result["post_rgb"], post_with_bars], axis=1)
        grid = np.concatenate([row1, row2], axis=0)
        if args.upscale > 1:
            grid = cv2.resize(grid, (grid.shape[1] * args.upscale,
                                       grid.shape[0] * args.upscale),
                              interpolation=cv2.INTER_LANCZOS4)

        # Caption banner
        caption_lines = [
            f"init{init_idx}  {r['mode']}  fail@p={r.fail_progress:.2f}",
            f"realized damage total: {r.realized_damage_total:.2f}   "
            f"damaged: {len(damaged)} body(s)",
            f"predictor risk pre-failure: {result['pred_total']:.1f}",
        ]
        banner_h = 26 * len(caption_lines) + 12
        banner = np.full((banner_h, grid.shape[1], 3), 30, dtype=np.uint8)
        for li, line in enumerate(caption_lines):
            cv2.putText(banner, line, (12, 22 + 24 * li),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235),
                        thickness=1, lineType=cv2.LINE_AA)
        labels = [("pre-failure RGB", 0, 0),
                  ("predictor heatmap + top-3 flags", 0, W * args.upscale),
                  ("post-rollout RGB", H * args.upscale, 0),
                  ("post-rollout + health bars + damaged tint",
                   H * args.upscale, W * args.upscale)]
        composite = np.concatenate([banner, grid], axis=0)
        # Quadrant labels
        for txt, rr, cc in labels:
            cv2.putText(composite, txt,
                        (cc + 8, rr + banner_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0),
                        thickness=1, lineType=cv2.LINE_AA)

        # Save
        out_path = (out_dir
                    / f"trial{trial_i+1:02d}_init{init_idx}_"
                       f"{r['mode']}_p{r.fail_progress:.2f}.png")
        cv2.imwrite(str(out_path), composite[..., ::-1])
        print(f"    wrote {out_path}")

    env.close()
    print(f"\nDone. Output: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
