#!/usr/bin/env python3
"""Qualitative figure: predicted heatmap vs realized post-failure contacts.

For a handful of (init, mode, progress) trials, run the LIBERO env up to
``fail_step - 1``, render the pre-failure RGB + predicted heatmap, inject the
failure, replay the demo through the rest, accumulate real contacts and
project them into the agentview image, then render a 4-panel row per trial:

    [pre RGB] [pre RGB + predicted heatmap] [post RGB] [post RGB + actual
                                                        contact dots]

Saves the figure to ``<out>/viz_pred_vs_real.png``.

Run::

    external/LIBERO/.venv/bin/python -u -m scripts.safety.viz_prediction_vs_reality \\
        --ckpt notebooks/model_playground/cluster_download/06172026/dualgated_state_rgb/best_ep08_val0.0648.pt \\
        --task pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate \\
        --out out/safety_rollouts_opt1/pick_..._on_the_plate/viz_pred_vs_real.png
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


# Per-mode joint set, mirrors dataset/safety_rollout convention.
_MODE_JOINTS = {
    "GRIPPER_OPEN":  [],
    "SLIPPERY_GRIP": [],
    "SINGLE_JOINT":  [4],
    "MULTI_JOINT":   [2, 4],
    "ALL_JOINTS":    [1, 2, 3, 4, 5, 6, 7],
}


def _project_world_to_agentview(world_pts, env):
    """Project world-frame contact positions to agentview pixel coords.

    Same convention as :mod:`planner.risk.v2_targets`:
    image-down Y axis (mat0.T with Y row negated).
    """
    import mujoco
    from planner.policy.libero_env_failure import unwrap_sim
    model, data = unwrap_sim(env.sim)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
    mat0 = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    fovy_deg = float(model.cam_fovy[cam_id])
    H, W = 240, 320

    # v2 convention: mat0.T with Y-row negated (image Y points down).
    # Display setup: the env RGB is np.flipud'd before display so the result
    # is in v2 (image) convention. Projecting world → v2-pixel and plotting
    # on a v2-oriented image is a direct match — no extra flip needed.
    R = mat0.T.copy(); R[1] = -R[1]
    fy = (H / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0); fx = fy
    cx, cy = W / 2.0, H / 2.0

    pts = np.asarray(world_pts, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cam_coords = (pts - cam_pos) @ R.T
    depth = -cam_coords[:, 2]
    in_front = depth > 1e-6
    u = fx * cam_coords[in_front, 0] / depth[in_front] + cx
    v = fy * cam_coords[in_front, 1] / depth[in_front] + cy
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.stack([u[in_img], v[in_img]], axis=1).astype(np.float32)


def _load_demo_actions(demo_hdf5, init_idx):
    import h5py
    with h5py.File(demo_hdf5, "r") as f:
        return np.asarray(f["data"][f"demo_{init_idx}"]["actions"],
                          dtype=np.float32)


def run_one_trial(env, predictor, demo_actions, fail_step, mode, joints,
                  obs_window, image_hw=(240, 320)):
    """Replay through fail_step-1, predict, inject, continue. Returns dict."""
    from planner.experiments.config import FailureConfig, FailureMode
    from planner.policy.libero_env_failure import (
        EnvFailureScheduler, unwrap_sim)
    from planner.risk.inference import marginal_heatmap
    import mujoco

    failure = FailureConfig(
        mode=FailureMode[mode], probability=1.0,
        joint_names=[f"joint{j}" for j in joints] if joints else None,
    )
    sched = EnvFailureScheduler(env, failure, fail_step)
    obs = sched.reset()
    obs_window.reset()

    pre_rgb_chw = None
    pre_heatmap = None
    pre_gate = float("nan")

    # Accumulator for world-frame contact positions during the post-failure run.
    contact_positions: list = []
    contact_force_world: list = []

    model_unwrapped, data_unwrapped = unwrap_sim(env.sim)
    from planner.experiments.libero.naming import resolve_model_handles
    handles = resolve_model_handles(model_unwrapped)
    robot_geoms = set(int(g) for g in handles.robot_geom_ids)

    n_actions = len(demo_actions)
    for i in range(n_actions):
        # Robosuite's OffScreenRenderEnv returns the agentview image with the
        # OpenGL framebuffer convention (Y points up), but the v2 dataset
        # stored RGB in image convention (Y points down) — same camera, same
        # intrinsics, just flipped. The predictor was trained on the v2
        # orientation, so we np.flipud here before pushing into the window.
        env_rgb = np.flipud(np.asarray(obs["agentview_image"])).copy()
        rgb = np.transpose(env_rgb, (2, 0, 1)).astype(np.uint8)
        qpos = np.asarray(obs.get("robot0_joint_pos", np.zeros(7)),
                           dtype=np.float32)[:7]
        qvel = np.asarray(obs.get("robot0_joint_vel", np.zeros(7)),
                           dtype=np.float32)[:7]
        ee   = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)),
                           dtype=np.float32)[:3]
        grip = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(1)),
                           dtype=np.float32).ravel()[:1]
        state = np.concatenate([qpos, qvel, ee, grip]).astype(np.float32)
        obs_window.push(rgb, state)

        # Snapshot at fail_step - 1: capture pre-failure RGB + predictor output
        if i == fail_step - 1:
            pre_rgb_chw = rgb.copy()
            heat, gate = marginal_heatmap(predictor,
                                          obs_window.rgb_window,
                                          obs_window.state_window)
            pre_heatmap = heat
            pre_gate = gate

        obs, reward, done, info = sched.step(demo_actions[i])

        # Accumulate post-failure contacts in world frame
        if i >= fail_step - 1:
            ncon = int(data_unwrapped.ncon)
            for k in range(ncon):
                con = data_unwrapped.contact[k]
                g1, g2 = int(con.geom1), int(con.geom2)
                r1, r2 = g1 in robot_geoms, g2 in robot_geoms
                if r1 and r2:
                    continue
                contact_positions.append(con.pos.copy())
                cf = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model_unwrapped, data_unwrapped, k, cf)
                contact_force_world.append(cf[:3].copy())
        if done:
            break

    # Post-failure RGB = last rendered frame (same flip as above for display
    # consistency with the v2-trained heatmap).
    post_env_rgb = np.flipud(np.asarray(obs["agentview_image"])).copy()
    post_rgb_chw = np.transpose(post_env_rgb, (2, 0, 1)).astype(np.uint8)
    # Project the accumulated contacts into pixel space.
    contact_pixels = _project_world_to_agentview(contact_positions, env)

    # Compute the failure-induced contact mass for the title
    forces = np.asarray(contact_force_world) if contact_force_world else np.zeros((0, 3))
    realized_mass = float(np.linalg.norm(forces, axis=1).sum()) if forces.size else 0.0

    return {
        "pre_rgb": pre_rgb_chw,
        "pre_heatmap": pre_heatmap,
        "pre_gate": pre_gate,
        "post_rgb": post_rgb_chw,
        "contact_pixels": contact_pixels,
        "n_contacts": len(contact_positions),
        "realized_mass": realized_mass,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--libero_root", type=Path, default=Path("external/LIBERO"))
    ap.add_argument("--demo_root", type=Path, default=Path("datasets/libero/raw"))
    ap.add_argument("--task", required=True)
    # Hand-picked diverse-mode trials. Each entry: (init_idx, mode, fail_progress).
    ap.add_argument("--trials",
                    default="0,SINGLE_JOINT,0.40;0,GRIPPER_OPEN,0.70;"
                            "0,ALL_JOINTS,0.55;0,MULTI_JOINT,0.55;"
                            "2,SLIPPERY_GRIP,0.70",
                    help="Semicolon-separated list of init,mode,progress tuples.")
    ap.add_argument("--out", type=Path,
                    default=Path("out/safety_rollouts_opt1/viz_pred_vs_real.png"))
    ap.add_argument("--image_h", type=int, default=240)
    ap.add_argument("--image_w", type=int, default=320)
    args = ap.parse_args()

    # Resolve BDDL + demo HDF5
    from scripts.safety.safety_rollout import (
        _resolve_bddl, _resolve_demo_hdf5)
    bddl = _resolve_bddl(args.libero_root, args.task)
    demo_hdf5 = _resolve_demo_hdf5(args.demo_root, args.task)
    print(f"task : {args.task}\nbddl : {bddl}\ndemo : {demo_hdf5}\n")

    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                              camera_heights=args.image_h,
                              camera_widths=args.image_w)
    print("env loaded")

    from planner.risk.inference import ContactPredictor
    from planner.policy.safe_action import ObsWindow
    cp = ContactPredictor.from_checkpoint(args.ckpt)
    print(f"predictor: {cp.meta.arch}  ep={cp.meta.epoch}  "
          f"val_heat={cp.meta.val_heat}")

    obs_window = ObsWindow(H=args.image_h, W=args.image_w)

    # Parse trials
    spec = []
    for tok in args.trials.split(";"):
        a, b, c = tok.split(",")
        spec.append((int(a), b.strip(), float(c)))

    # Resolve fail_step per trial from demo length
    rows = []
    for (init_idx, mode, prog) in spec:
        actions = _load_demo_actions(str(demo_hdf5), init_idx)
        joints = _MODE_JOINTS.get(mode, [])
        fail_step = max(1, int(prog * len(actions)))
        print(f"\n=== init {init_idx}  {mode}  progress={prog:.2f}  "
              f"fail_step={fail_step}/{len(actions)} ===")
        t0 = time.time()
        res = run_one_trial(env, cp, actions, fail_step, mode, joints,
                             obs_window, image_hw=(args.image_h, args.image_w))
        res.update(dict(init_idx=init_idx, mode=mode, progress=prog,
                         fail_step=fail_step, n_steps=len(actions)))
        rows.append(res)
        print(f"  {len(rows)}/{len(spec)} done in {time.time()-t0:.1f}s   "
              f"contacts={res['n_contacts']}  mass={res['realized_mass']:.2f}  "
              f"gate={res['pre_gate']:.3f}")

    env.close()

    # ----- Plot -----
    import matplotlib.pyplot as plt
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for ri, r in enumerate(rows):
        # Column 1: pre-failure RGB
        ax = axes[ri, 0]
        rgb = r["pre_rgb"].transpose(1, 2, 0)
        ax.imshow(rgb)
        ax.set_title(f"init{r['init_idx']}  {r['mode']}  "
                      f"prog={r['progress']:.2f}\n"
                      f"step {r['fail_step']}/{r['n_steps']}",
                      fontsize=9)
        ax.set_xlabel("pre-failure RGB")
        ax.set_xticks([]); ax.set_yticks([])

        # Column 2: pre RGB + predicted heatmap overlay
        ax = axes[ri, 1]
        heat = r["pre_heatmap"]
        ax.imshow(rgb)
        vmax = max(float(heat.max()), 1e-6)
        masked = np.ma.masked_where(heat < 0.15 * vmax, heat)
        ax.imshow(masked, cmap="hot", alpha=0.6, vmin=0, vmax=vmax)
        ax.set_xlabel(f"PREDICTED heatmap (max={heat.max():.2f}, "
                       f"gate={r['pre_gate']:.2f})")
        ax.set_xticks([]); ax.set_yticks([])

        # Column 3: post-failure RGB
        ax = axes[ri, 2]
        post = r["post_rgb"].transpose(1, 2, 0)
        ax.imshow(post)
        ax.set_xlabel(f"post-failure RGB  ({r['n_contacts']} contacts)")
        ax.set_xticks([]); ax.set_yticks([])

        # Column 4: post RGB + actual contact dots (projected from world)
        ax = axes[ri, 3]
        ax.imshow(post)
        cp_pix = r["contact_pixels"]
        if cp_pix.shape[0] > 0:
            ax.scatter(cp_pix[:, 0], cp_pix[:, 1], s=4,
                        c="cyan", alpha=0.4, edgecolors="none")
        ax.set_xlabel(f"ACTUAL contacts (n={cp_pix.shape[0]} in-frame, "
                       f"mass={r['realized_mass']:.1f})")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Predicted heatmap (col 2, hot) vs realized contacts "
                  f"(col 4, cyan) — {args.task[:70]}",
                  fontsize=11, y=1.01)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nsaved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
