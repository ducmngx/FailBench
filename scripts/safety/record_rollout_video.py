#!/usr/bin/env python3
"""Record a side-by-side rollout video: agentview frame + same frame with
predicted heatmap overlay.

At each control step we run the env, render the agentview, query the
predictor (marginal heatmap over the 5 failure modes), and write a paired
frame to MP4 — left panel is the bare RGB, right panel is the RGB with the
predicted heatmap masked in. A small caption strip on top shows the step
index, traj_progress, predictor gate prob, and whether the failure has
fired yet.

Run::

    external/LIBERO/.venv/bin/python -u -m scripts.safety.record_rollout_video \\
        --ckpt notebooks/model_playground/cluster_download/06172026/dualgated_state_rgb/best_ep08_val0.0648.pt \\
        --task pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate \\
        --init 0 --mode SINGLE_JOINT --joints 4 --progress 0.4 \\
        --out /tmp/rollout.mp4

The mp4 plays at the LIBERO control rate (~20 Hz). Use ``--fps`` to speed
up or slow down playback.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_MODE_JOINTS = {
    "GRIPPER_OPEN":  [],
    "SLIPPERY_GRIP": [],
    "SINGLE_JOINT":  [4],
    "MULTI_JOINT":   [2, 4],
    "ALL_JOINTS":    [1, 2, 3, 4, 5, 6, 7],
}

# Per-mode RGB color (0-255) for the combined-modes overlay panel.
# Picked from matplotlib's tab10 categorical palette so the 5 modes are
# distinguishable both individually and when blended additively.
_MODE_COLORS_RGB = {
    "GRIPPER_OPEN":  (214,  39,  40),   # red
    "SLIPPERY_GRIP": (255, 127,  14),   # orange
    "SINGLE_JOINT":  (255, 215,   0),   # yellow
    "MULTI_JOINT":   ( 44, 160,  44),   # green
    "ALL_JOINTS":    ( 31, 119, 180),   # blue
}


def _heatmap_to_rgb(heat: np.ndarray, vmax: float) -> np.ndarray:
    """Convert a (H, W) heatmap to a (H, W, 3) uint8 BGR overlay using the
    'hot' colormap. Pixels below 15 % of vmax become transparent (alpha=0).
    Returns (rgb_uint8, alpha_uint8). We don't return RGBA so the caller can
    do an alpha-blended composite."""
    import matplotlib.cm as cm
    if vmax <= 0:
        vmax = 1.0
    norm = np.clip(heat / vmax, 0.0, 1.0)
    rgba = (cm.get_cmap("hot")(norm) * 255).astype(np.uint8)   # (H, W, 4)
    alpha = (norm > 0.15).astype(np.uint8) * int(255 * 0.55)   # 55 % blend
    return rgba[..., :3], alpha


def _composite(rgb: np.ndarray, heat: np.ndarray, vmax: float) -> np.ndarray:
    """Overlay a heat map onto an RGB image with per-pixel alpha."""
    overlay, alpha = _heatmap_to_rgb(heat, vmax)
    a = alpha.astype(np.float32) / 255.0
    out = rgb.astype(np.float32) * (1.0 - a[..., None]) \
            + overlay.astype(np.float32) * a[..., None]
    return out.astype(np.uint8)


def _composite_per_mode(rgb: np.ndarray,
                        heats: dict[str, np.ndarray],
                        thresh_frac: float = 0.30,
                        alpha_max: float = 0.85) -> np.ndarray:
    """Overlay multiple per-mode heatmaps onto an RGB image, each in its
    own color from ``_MODE_COLORS_RGB``, using winner-take-all blending.

    Each mode's heatmap is normalised by ITS OWN max so different modes show
    up regardless of relative magnitudes. Pixels below ``thresh_frac * max``
    contribute no alpha for that mode. At each pixel the mode with the
    highest normalised intensity owns the pixel — produces crisp colored
    regions instead of the muddy wash you get from naïve alpha-summing N
    modes (each transparent, all of them all but invisible).
    """
    H, W = rgb.shape[:2]
    mode_names = list(heats.keys())
    n = len(mode_names)
    if n == 0:
        return rgb

    # Build (n, H, W) of soft-thresholded normalised intensities.
    soft_stack = np.zeros((n, H, W), dtype=np.float32)
    for mi, m in enumerate(mode_names):
        h = heats[m]
        vmax = max(float(h.max()), 1e-6)
        norm = np.clip(h / vmax, 0.0, 1.0)
        soft_stack[mi] = np.clip((norm - thresh_frac) /
                                  (1.0 - thresh_frac), 0.0, 1.0)

    # Per-pixel: which mode has the highest soft intensity?
    winner = soft_stack.argmax(axis=0)              # (H, W) int
    winner_intensity = soft_stack.max(axis=0)        # (H, W) float

    # Per-mode RGB color lookup table → (H, W, 3)
    palette = np.array(
        [list(_MODE_COLORS_RGB[m]) for m in mode_names], dtype=np.float32)
    color_field = palette[winner]                    # (H, W, 3)

    alpha = (winner_intensity * alpha_max)[..., None]
    out = rgb.astype(np.float32) * (1.0 - alpha) + color_field * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _annotate(frame: np.ndarray, lines: list[str],
              colored_lines: list = None) -> np.ndarray:
    """Burn a text strip onto the top of the frame.

    ``colored_lines`` is an optional list of ``[(text, BGR_color), ...]``
    sequences; one such list per extra line. Each segment in a sequence is
    drawn left-to-right with its own color so a "legend line" can show
    several colored labels in sequence.
    """
    import cv2
    H, W = frame.shape[:2]
    n_lines = len(lines) + (len(colored_lines) if colored_lines else 0)
    strip_h = 16 * n_lines + 6
    out = np.full((H + strip_h, W, 3), 0, dtype=np.uint8)
    out[strip_h:] = frame
    for i, line in enumerate(lines):
        cv2.putText(out, line, (6, 14 + i * 16),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255),
                     1, cv2.LINE_AA)
    if colored_lines:
        for k, segments in enumerate(colored_lines):
            x = 6
            y = 14 + (len(lines) + k) * 16
            for text, bgr in segments:
                cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                             0.42, bgr, 1, cv2.LINE_AA)
                # advance roughly by text width
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.42, 1)
                x += tw + 6
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--libero_root", type=Path, default=Path("external/LIBERO"))
    ap.add_argument("--demo_root", type=Path, default=Path("datasets/libero/raw"))
    ap.add_argument("--task", required=True)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--mode", default="SINGLE_JOINT",
                    choices=list(_MODE_JOINTS))
    ap.add_argument("--joints", default=None,
                    help="Comma-separated 1-based joint indices. Default "
                          "uses _MODE_JOINTS for the chosen mode.")
    ap.add_argument("--progress", type=float, default=0.4)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=20,
                    help="Output mp4 frame rate (default 20 = control rate)")
    ap.add_argument("--image_h", type=int, default=240)
    ap.add_argument("--image_w", type=int, default=320)
    ap.add_argument("--mode_prior", default=None,
                    help="JSON dict of mode→weight for marginal_heatmap "
                          "(default uniform over the 5 modes).")
    ap.add_argument("--combined_modes", action="store_true",
                    help="Render the right panel as a per-mode color overlay "
                          "showing all 5 failure-mode predictions simultaneously, "
                          "each in its own color (red=GRIPPER_OPEN, "
                          "orange=SLIPPERY_GRIP, yellow=SINGLE_JOINT, "
                          "green=MULTI_JOINT, blue=ALL_JOINTS). Default is a "
                          "single marginal heatmap.")
    args = ap.parse_args()

    import cv2

    # Resolve paths
    from scripts.safety.safety_rollout import (
        _resolve_bddl, _resolve_demo_hdf5)
    bddl = _resolve_bddl(args.libero_root, args.task)
    demo_hdf5 = _resolve_demo_hdf5(args.demo_root, args.task)
    print(f"task : {args.task}\nbddl : {bddl}\ndemo : {demo_hdf5}")

    # Load demo actions
    import h5py
    with h5py.File(demo_hdf5, "r") as f:
        actions = np.asarray(f["data"][f"demo_{args.init}"]["actions"],
                             dtype=np.float32)
    n_actions = len(actions)
    fail_step = max(1, int(args.progress * n_actions))
    joints = ([int(j) for j in args.joints.split(",") if j.strip()]
               if args.joints else _MODE_JOINTS[args.mode])
    print(f"demo length={n_actions}  fail_step={fail_step}  "
          f"mode={args.mode}  joints={joints}")

    # Env + predictor
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.image_h, camera_widths=args.image_w)
    print("env loaded")

    from planner.risk.inference import (
        ContactPredictor, marginal_heatmap)
    from planner.policy.safe_action import ObsWindow
    cp = ContactPredictor.from_checkpoint(args.ckpt)
    print(f"predictor: {cp.meta.arch}  ep={cp.meta.epoch}  "
          f"val_heat={cp.meta.val_heat}  device={cp.device}")

    # Failure scheduler
    from planner.experiments.config import FailureConfig, FailureMode
    from planner.policy.libero_env_failure import EnvFailureScheduler
    failure = FailureConfig(
        mode=FailureMode[args.mode], probability=1.0,
        joint_names=[f"joint{j}" for j in joints] if joints else None,
    )
    sched = EnvFailureScheduler(env, failure, fail_step)
    obs = sched.reset()

    # Optional mode prior
    mode_prior = None
    if args.mode_prior:
        import json
        mode_prior = json.loads(args.mode_prior)

    # Video writer: panel is 2 * W wide, with caption strip on top.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel_w = 2 * args.image_w
    # Caption: 3 lines (4 if --combined_modes adds a legend) × 16 + 6 padding.
    n_caption_lines = 4 if args.combined_modes else 3
    panel_h = args.image_h + n_caption_lines * 16 + 6
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps,
                              (panel_w, panel_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {args.out}")

    obs_win = ObsWindow(H=args.image_h, W=args.image_w)
    contact_dt = 0.0
    t_start = time.time()
    for i in range(n_actions):
        # Pull RGB + state from current obs and flip to v2 orientation
        env_rgb = np.flipud(np.asarray(obs["agentview_image"])).copy()
        qpos = np.asarray(obs.get("robot0_joint_pos", np.zeros(7)),
                          dtype=np.float32)[:7]
        qvel = np.asarray(obs.get("robot0_joint_vel", np.zeros(7)),
                          dtype=np.float32)[:7]
        ee = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)),
                         dtype=np.float32)[:3]
        grip = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(1)),
                          dtype=np.float32).ravel()[:1]
        state = np.concatenate([qpos, qvel, ee, grip]).astype(np.float32)

        rgb_chw = env_rgb.transpose(2, 0, 1).astype(np.uint8)
        obs_win.push(rgb_chw, state)

        gate_prob = float("nan")
        per_mode_heats: dict[str, np.ndarray] = {}
        right_caption_extra: str = ""
        colored_legend = None

        if args.combined_modes:
            # Query the predictor once per failure mode. We use this mode's
            # default joints (as set in _MODE_JOINTS), since at the moment of
            # action selection we wouldn't yet know exactly which joints will
            # fail — we'd just know each mode's joint policy.
            for mode_name in ("GRIPPER_OPEN", "SLIPPERY_GRIP", "SINGLE_JOINT",
                               "MULTI_JOINT", "ALL_JOINTS"):
                m_joints = _MODE_JOINTS[mode_name]
                h, g = cp.predict(obs_win.rgb_window, obs_win.state_window,
                                   mode_name, m_joints)
                per_mode_heats[mode_name] = h
                if mode_name == args.mode:
                    gate_prob = float(g)
            right = _composite_per_mode(env_rgb, per_mode_heats)
            # Legend: one colored chip per mode
            colored_legend = [[("legend:", (255, 255, 255))] +
                              [(m, _MODE_COLORS_RGB[m][::-1])
                                for m in per_mode_heats]]
            # In combined mode the caption summarises per-mode heat max
            max_summary = "  ".join(
                f"{m[:2]}={per_mode_heats[m].max():.2f}"
                for m in per_mode_heats)
            right_caption_extra = max_summary
        else:
            # Single marginal predicted heatmap (default behavior)
            heat, gate_prob = marginal_heatmap(
                cp, obs_win.rgb_window, obs_win.state_window,
                mode_prior=mode_prior)
            right = _composite(env_rgb, heat,
                                vmax=max(float(heat.max()), 1e-6))
            right_caption_extra = f"heat_max: {heat.max():.3f}"

        # Compose frame: bare RGB | annotated overlay
        side_by_side = np.concatenate([env_rgb, right], axis=1)
        # Caption strip
        prog = i / max(1, n_actions - 1)
        # ASCII-only labels — OpenCV's putText with HERSHEY_SIMPLEX doesn't
        # support unicode and renders ✓/✗ as "???".
        triggered = "PRE" if i < fail_step else "POST"
        lines = [
            f"step {i+1:3d}/{n_actions}   prog {prog:.3f}   "
            f"fail_step {fail_step}   [{triggered}-failure]",
            f"task : {args.task[:70]}",
            f"injected: {args.mode}  joints: {joints}   "
            f"gate_prob: {gate_prob:.3f}   {right_caption_extra}",
        ]
        frame = _annotate(side_by_side, lines, colored_lines=colored_legend)

        # cv2 expects BGR; matplotlib gives RGB.
        frame_bgr = frame[..., ::-1]
        writer.write(frame_bgr)

        # Step the env (failure scheduler may inject this step)
        obs, reward, done, info = sched.step(actions[i])
        if done:
            print(f"  env reported done at step {i+1}")
            break

    writer.release()
    env.close()
    print(f"\nwrote {args.out}  ({time.time()-t_start:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
