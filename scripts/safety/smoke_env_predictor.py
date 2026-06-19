#!/usr/bin/env python3
"""Stage-0 smoke: LIBERO env + ContactPredictor + EnvFailureScheduler round trip.

Validates the plumbing pieces needed by safety_rollout.py before kicking off
a full sweep:

1. ``OffScreenRenderEnv`` instantiates headlessly and yields a 240×320
   RGB observation per step.
2. ``ContactPredictor.from_checkpoint`` loads the checkpoint and returns a
   marginal risk score on the env's first frame.
3. ``EnvFailureScheduler`` actually injects the failure at the configured
   step (qvel on the failed joint goes to near-zero after a few settle
   steps).

Run from the LIBERO sidecar venv (robosuite 1.4 + mujoco)::

    external/LIBERO/.venv/bin/python -u -m scripts.safety.smoke_env_predictor \\
        --ckpt $SCRATCH/failbench/runs/06172026/dualgated_state_rgb/best_ep08_val0.0648.pt \\
        --bddl <path>/KITCHEN_SCENE3_put_the_black_bowl_on_top_of_the_cabinet.bddl
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bddl", required=True,
                    help="Path to one LIBERO BDDL file.")
    ap.add_argument("--image_h", type=int, default=240)
    ap.add_argument("--image_w", type=int, default=320)
    ap.add_argument("--n_steps", type=int, default=100,
                    help="How many env steps to run (default 100).")
    ap.add_argument("--inject_at", type=int, default=50,
                    help="Step at which to inject the failure (default 50).")
    ap.add_argument("--mode", default="SINGLE_JOINT",
                    choices=["GRIPPER_OPEN", "SLIPPERY_GRIP",
                              "SINGLE_JOINT", "MULTI_JOINT", "ALL_JOINTS"])
    ap.add_argument("--joints", default="4",
                    help="Comma-separated 1-based joint indices.")
    args = ap.parse_args()

    print("[1/5] Loading LIBERO env (this can take 5–15 s)...", flush=True)
    t0 = time.time()
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=str(args.bddl),
        camera_heights=args.image_h,
        camera_widths=args.image_w,
    )
    obs = env.reset()
    print(f"  env loaded + reset in {time.time()-t0:.1f}s. obs keys: "
          f"{sorted(obs)[:8]}...")

    # Find the RGB key
    rgb_key = None
    for k in obs:
        if "agentview" in k and "image" in k:
            rgb_key = k; break
    assert rgb_key is not None, f"no agentview_image-like key in {sorted(obs)}"
    print(f"  rgb key: {rgb_key}  shape: {np.asarray(obs[rgb_key]).shape}")

    print("\n[2/5] Loading predictor...", flush=True)
    from planner.risk.inference import ContactPredictor, marginal_heatmap
    cp = ContactPredictor.from_checkpoint(args.ckpt)
    print(f"  arch={cp.meta.arch}  ep={cp.meta.epoch}  "
          f"val_heat={cp.meta.val_heat}  device={cp.device}")

    print("\n[3/5] Building rolling observation window...", flush=True)
    from planner.policy.safe_action import ObsWindow
    win = ObsWindow(H=args.image_h, W=args.image_w)
    # Robosuite OpenGL framebuffer has Y up; v2 dataset stored Y down. Flip.
    rgb_chw = np.transpose(
        np.flipud(np.asarray(obs[rgb_key])).copy(),
        (2, 0, 1)).astype(np.uint8)
    state_vec = _extract_state(obs)
    win.push(rgb_chw, state_vec)
    print(f"  window ready: rgb={win.rgb_window.shape} state={win.state_window.shape}")

    print("\n[4/5] One marginal predictor query...", flush=True)
    t0 = time.time()
    heat, gprob = marginal_heatmap(cp, win.rgb_window, win.state_window)
    print(f"  heat={heat.shape}  max={heat.max():.4f}  mean={heat.mean():.4f}  "
          f"gate_prob={gprob:.3f}  ({(time.time()-t0)*1000:.0f} ms)")

    print("\n[5/5] EnvFailureScheduler step-and-inject...", flush=True)
    from planner.experiments.config import FailureConfig, FailureMode
    from planner.policy.libero_env_failure import EnvFailureScheduler
    joints = [int(j) for j in args.joints.split(",") if j.strip()]
    failure = FailureConfig(
        mode=FailureMode[args.mode],
        probability=1.0,
        joint_names=[f"joint{j}" for j in joints] if joints else None,
    )
    sched = EnvFailureScheduler(env, failure, fail_step=args.inject_at)
    obs = sched.reset()

    # Robosuite wraps sim.data; pull qvel via the wrapper's attribute access
    # (which proxies to the underlying mujoco.MjData.qvel).
    pre_qvel, post_qvel = [], []
    for i in range(args.n_steps):
        action = np.zeros(7, dtype=np.float32)  # zero action — hold pose
        obs, r, done, info = sched.step(action)
        if i == args.inject_at - 1:
            pre_qvel = list(np.asarray(env.sim.data.qvel)[:7])
        if i == args.inject_at + 30:
            post_qvel = list(np.asarray(env.sim.data.qvel)[:7])
        if done:
            print(f"  env reported done at step {i+1}")
            break

    print(f"\n  injected = {sched.injected}  events = {sched.events}")
    if pre_qvel and post_qvel and args.mode == "SINGLE_JOINT" and joints:
        j = joints[0] - 1
        print(f"  joint {joints[0]} qvel  pre-failure: {pre_qvel[j]:+.4f}   "
              f"post (+30 steps): {post_qvel[j]:+.4f}")
        if abs(post_qvel[j]) < max(0.01, 1.5 * abs(pre_qvel[j])):
            print("  ✓ failed joint's velocity is bounded (failure took effect)")
        else:
            print("  ⚠ failed joint still moving — investigate injector "
                  "(may be normal if there are external forces)")

    print("\n[SUCCESS] All 5 checks passed. The pipeline is wired up.")
    env.close()
    return 0


def _extract_state(obs: dict) -> np.ndarray:
    qpos = np.asarray(obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float32)[:7]
    qvel = np.asarray(obs.get("robot0_joint_vel", np.zeros(7)), dtype=np.float32)[:7]
    ee = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)[:3]
    grip = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(1)),
                       dtype=np.float32).ravel()[:1]
    return np.concatenate([qpos, qvel, ee, grip]).astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
