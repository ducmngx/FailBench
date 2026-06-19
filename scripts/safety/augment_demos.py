#!/usr/bin/env python3
"""Phase 1: augment LIBERO demos with predictor features.

For every (obs, action) pair in every demo of a task, run the predictor
once and save:

- ``proprio`` (T, 18)      qpos + qvel + ee_pos + gripper_qpos
- ``actions`` (T, 7)       teleop actions, copied from source
- ``agentview_rgb`` (T, H, W, 3)  Y-down (predictor convention), copied + flipped
- ``pred_per_body`` (T, K)  per-body risk score over the canonical body order
- ``gate_prob`` (T,)        classifier head probability
- ``init_state``            full sim init state for env re-seeding (copied)

Plus per-demo aggregate attributes for predictor-based curation:

- ``max_pred_risk``         max(sum(pred_per_body)) across the trajectory
- ``mean_gate_prob``        mean(gate_prob)
- ``n_high_pred_steps``     count of steps where sum(pred_per_body) > 1000

Output: one augmented HDF5 next to the input.  Reuses the canonical body
order from ``SafeLiberoEnv.build_entity_masks`` so the augmented dataset
is policy-input-compatible.

Usage::

    PYTHONPATH=external/LIBERO:. external/LIBERO/.venv/bin/python \\
        -m scripts.safety.augment_demos \\
        --bddl <bddl_file> --demo <demo.hdf5> \\
        --ckpt <predictor.pt> \\
        --out datasets/libero/augmented/<task>.hdf5
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bddl", required=True)
    ap.add_argument("--demo", required=True,
                    help="Source LIBERO demo HDF5 (with obs/agentview_rgb).")
    ap.add_argument("--ckpt", required=True,
                    help="Predictor checkpoint .pt")
    ap.add_argument("--out", required=True,
                    help="Output augmented HDF5 path.")
    ap.add_argument("--image_h", type=int, default=240)
    ap.add_argument("--image_w", type=int, default=320)
    ap.add_argument("--max_demos", type=int, default=None,
                    help="Cap number of demos for smoke testing.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Build env once just to get masks + body order.  We do NOT step the env;
    # the demo HDF5 has the RGB+state we need.
    from libero.libero.envs import OffScreenRenderEnv
    from planner.policy.libero_env_failure import unwrap_sim
    from planner.policy.safe_action import ObsWindow, query_risk
    from planner.risk.inference import ContactPredictor
    from scripts.safety.safety_rollout import build_entity_masks

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl,
        camera_heights=args.image_h, camera_widths=args.image_w)
    masks = build_entity_masks(env, (args.image_h, args.image_w))
    body_names = sorted(masks.keys())
    K = len(body_names)
    print(f"built {K} entity masks for {Path(args.bddl).stem}")
    print(f"  body order: {body_names}")
    env.close()

    predictor = ContactPredictor.from_checkpoint(args.ckpt, device=args.device)
    print(f"loaded predictor: {predictor.meta.arch} "
          f"val_heat={predictor.meta.val_heat:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_steps = 0

    with h5py.File(args.demo, "r") as fin, \
         h5py.File(args.out, "w") as fout:
        # Mirror the source file's top-level "data" group
        gdata = fout.create_group("data")
        # Save the body order so downstream readers don't need to rebuild
        # the env to know what each pred_per_body column means.
        gdata.attrs["body_names"] = np.array(body_names, dtype="S")
        gdata.attrs["K"] = K
        gdata.attrs["source_demo"] = str(args.demo)
        gdata.attrs["predictor_ckpt"] = str(args.ckpt)

        demo_keys = sorted(fin["data"].keys(),
                           key=lambda k: int(k.split("_")[1]))
        if args.max_demos is not None:
            demo_keys = demo_keys[:args.max_demos]

        for dk in demo_keys:
            grp_in = fin[f"data/{dk}"]
            T = grp_in["actions"].shape[0]
            rgbs = np.asarray(grp_in["obs/agentview_rgb"])  # (T, H, W, 3) Y-up
            ee_pos = np.asarray(grp_in["obs/ee_pos"])       # (T, 3)
            joints = np.asarray(grp_in["obs/joint_states"]) # (T, 7) qpos
            grip = np.asarray(grp_in["obs/gripper_states"]) # (T, 2)
            actions = np.asarray(grp_in["actions"], dtype=np.float32)
            init_state = np.asarray(grp_in["states"][0], dtype=np.float32)

            # Build proprio in the same order SafeLiberoEnv uses:
            # qpos(7) + qvel(7) + ee_pos(3) + grip(1).
            # The HDF5 only has joint_states (qpos) — qvel is not recorded,
            # zero it.  At PPO time qvel comes from the env directly.
            qpos = joints.astype(np.float32)
            qvel = np.zeros_like(qpos, dtype=np.float32)
            grip_single = grip[:, :1].astype(np.float32)
            proprio = np.concatenate(
                [qpos, qvel, ee_pos.astype(np.float32), grip_single],
                axis=1)  # (T, 18)

            # Flip RGB to Y-down (v2 / predictor convention).  Demos store
            # RGB at LIBERO's recording resolution (typically 128x128);
            # the predictor was trained on (image_h, image_w) — usually
            # 240x320.  Resize before flip.
            import cv2
            src_h, src_w = rgbs.shape[1], rgbs.shape[2]
            if (src_h, src_w) != (args.image_h, args.image_w):
                rgbs_resized = np.stack([
                    cv2.resize(rgbs[t], (args.image_w, args.image_h),
                               interpolation=cv2.INTER_LINEAR)
                    for t in range(T)], axis=0)
            else:
                rgbs_resized = rgbs
            rgbs_v2 = rgbs_resized[:, ::-1, :, :].copy()  # (T, H, W, 3)

            pred_per_body = np.zeros((T, K), dtype=np.float32)
            gate_prob = np.zeros((T,), dtype=np.float32)
            ow = ObsWindow()
            for t in range(T):
                rgb_chw = np.transpose(rgbs_v2[t], (2, 0, 1)).astype(np.uint8)
                state = proprio[t]
                ow.push(rgb_chw, state)
                rq = query_risk(predictor, ow, masks=masks)
                pred_per_body[t] = np.array(
                    [rq.per_entity.get(b, 0.0) for b in body_names],
                    dtype=np.float32)
                gate_prob[t] = float(rq.gate_prob) if rq.gate_prob is not None \
                    else 0.0

            # Save augmented demo
            grp_out = gdata.create_group(dk)
            grp_out.create_dataset("actions", data=actions)
            grp_out.create_dataset("proprio", data=proprio)
            grp_out.create_dataset(
                "agentview_rgb", data=rgbs_v2,
                compression="gzip", compression_opts=4)
            grp_out.create_dataset("pred_per_body", data=pred_per_body)
            grp_out.create_dataset("gate_prob", data=gate_prob)
            grp_out.create_dataset("init_state", data=init_state)

            # Per-demo aggregate stats for curation
            total_pred = pred_per_body.sum(axis=1)  # (T,)
            grp_out.attrs["T"] = T
            grp_out.attrs["max_pred_risk"] = float(total_pred.max())
            grp_out.attrs["mean_pred_risk"] = float(total_pred.mean())
            grp_out.attrs["mean_gate_prob"] = float(gate_prob.mean())
            grp_out.attrs["n_high_pred_steps"] = int(
                (total_pred > 1000).sum())

            total_steps += T
            elapsed = time.time() - t0
            rate = total_steps / max(elapsed, 1e-6)
            print(f"  {dk}: T={T:3d}  max_pred={total_pred.max():7.1f}  "
                  f"mean_pred={total_pred.mean():6.1f}  "
                  f"mean_gate={gate_prob.mean():.2f}  "
                  f"({total_steps} steps, {rate:.1f} steps/s, "
                  f"{elapsed:.1f}s elapsed)")

    print(f"\nWrote {args.out}")
    print(f"Total: {total_steps} steps in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
