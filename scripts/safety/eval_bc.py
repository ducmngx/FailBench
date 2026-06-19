#!/usr/bin/env python3
"""Phase 2.5: Eval a BC checkpoint in SafeLiberoEnv.

Loads bc_best.pt, rolls out N episodes (no failure injection), reports
success rate, mean cumulative damage, and per-episode max pred_risk.
This is the "did BC reproduce demo behavior?" gate before PPO.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bc_ckpt", required=True, type=Path)
    ap.add_argument("--bddl", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--predictor_ckpt", required=True)
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--deterministic", action="store_true",
                    help="Use BC action directly (no exploration noise).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    from planner.policy.safe_rl_env import SafeLiberoEnv
    from scripts.safety.train_bc import BCPolicy

    ckpt = torch.load(args.bc_ckpt, map_location=args.device,
                       weights_only=False)
    K = ckpt["K"]
    body_names = ckpt["body_names"]
    policy = BCPolicy(K=K).to(args.device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()
    print(f"loaded BC checkpoint: epoch={ckpt['epoch']}  "
          f"val_loss={ckpt['val_loss']:.4f}")
    print(f"  body_names: {body_names}")

    env = SafeLiberoEnv(
        bddl_file=args.bddl, demo_hdf5=args.demo,
        ckpt=args.predictor_ckpt,
        lambda_pred=0.0, lambda_dmg=0.0,
        failure_prob=0.0,
        max_episode_steps=args.max_steps,
        pred_features_in_obs=True, rgb_in_obs=False,
    )
    # Verify body order alignment
    if env._body_names != body_names:
        print("WARNING: body order mismatch between BC and env!")
        print(f"  BC:  {body_names}")
        print(f"  env: {env._body_names}")
    else:
        print("body orders aligned")

    n_success = 0
    damages = []
    max_pred_risks = []

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=ep)
        ep_max_pred = 0.0
        ep_success = False
        for t in range(args.max_steps):
            proprio = torch.from_numpy(obs["proprio"]).unsqueeze(0).to(args.device)
            pred = torch.from_numpy(obs["pred_per_body"]).unsqueeze(0).to(args.device)
            gate = torch.from_numpy(obs["gate_prob"]).unsqueeze(0).to(args.device)
            with torch.no_grad():
                action = policy(proprio, pred, gate).cpu().numpy()[0]
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            obs, r, term, trunc, info = env.step(action)
            ep_max_pred = max(ep_max_pred, info["pred_risk_total"])
            if info.get("success"):
                ep_success = True
                break
            if term or trunc:
                break
        n_success += int(ep_success)
        damages.append(info["damage_total"])
        max_pred_risks.append(ep_max_pred)
        print(f"  ep {ep+1:2d}: success={int(ep_success)}  "
              f"steps={t+1:3d}  damage={info['damage_total']:.3f}  "
              f"max_pred={ep_max_pred:.0f}")

    env.close()
    print(f"\n=== BC evaluation summary ===")
    print(f"  success rate    : {n_success}/{args.n_episodes} "
          f"= {100*n_success/args.n_episodes:.1f}%")
    print(f"  mean damage     : {np.mean(damages):.3f}")
    print(f"  mean max_pred   : {np.mean(max_pred_risks):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
