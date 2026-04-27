#!/usr/bin/env python
"""Preview failure-point sampling strategies over a clean trajectory replay.

Simulates each trajectory fully through MuJoCo (no failure injection),
records (qpos, qvel, ee_pos) at every sim step, then applies each sampling
strategy and reports the resulting distribution across mission segments,
joint-velocity magnitude, and end-effector z.

Usage:
    python scripts/visualize_failure_sampling.py \
        --trajs scenes/scene_level2/trajs/scene_level2_clean_far_00.pkl \
                scenes/scene_workshop/trajs/scene_workshop_stack_nominal_00.pkl \
        --n_samples 100 \
        --strategies uniform stratified_segments stratified_bands \
        --save_fig /tmp/failure_sampling_preview.png
"""

import argparse
import os
import pickle
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.experiments.failure_sampling import (
    sample_fail_sim_step,
    compute_segment_boundaries_sim,
)
from planner.utils.trajectory_interpolation import interpolate_trajectory
from planner.grasp_lock import GraspLock


DEFAULT_TRAJS = [
    "scenes/scene_level2/trajs/scene_level2_clean_far_00.pkl",
    "scenes/scene_kitchen/trajs/scene_kitchen_clean_far_00.pkl",
    "scenes/scene_workshop/trajs/scene_workshop_stack_nominal_00.pkl",
    "scenes/scene_grocery/trajs/scene_grocery_clean_far_00.pkl",
    "scenes/scene_cluttered/trajs/scene_cluttered_clean_far_00.pkl",
]


def _detect_scene_xml(pkl_path: str) -> str:
    # scenes/<scene>/trajs/<file>.pkl -> scenes/<scene>/scene.xml
    parts = os.path.normpath(pkl_path).split(os.sep)
    idx = parts.index("trajs")
    scene_dir = os.sep.join(parts[:idx])
    return os.path.join(scene_dir, "scene.xml")


def simulate_trajectory(pkl_path: str, interp_points_per_segment: int = 100,
                        steps_per_interp_point: int = 8):
    """Replay the trajectory end-to-end, recording per-sim-step state.

    Returns
    -------
    states : dict
        {"qpos": (T, 7), "qvel": (T, 7), "ee_pos": (T, 3),
         "segment_boundaries_sim": [0, ..., T],
         "segment_names": [...], "total_sim_steps": T}
    """
    scene_xml = _detect_scene_xml(pkl_path)
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    sd = d[list(d)[0]]
    segments = sd["segments"]
    grasped_object = sd.get("grasped_object", "object3")

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Pre-position arm at trajectory start (matches runner/verifier behavior).
    start_qpos = segments[0]["trajectory"][0][:7]
    data.qpos[:7] = start_qpos
    data.ctrl[:7] = start_qpos
    data.ctrl[7] = 255.0  # gripper open
    for _ in range(200):
        mujoco.mj_step(model, data)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    # Densify each segment.
    dense_segments = []
    for seg in segments:
        dense = interpolate_trajectory(
            seg["trajectory"],
            num_points_per_segment=interp_points_per_segment,
            method="cubic",
            joint_limits=None,
        )
        dense_segments.append({"name": seg["name"], "dense": dense,
                               "action_after": seg.get("action_after")})

    segment_boundaries_sim = compute_segment_boundaries_sim(
        [len(seg["dense"]) for seg in dense_segments], steps_per_interp_point
    )
    total_sim_steps = segment_boundaries_sim[-1]

    qpos_hist = np.zeros((total_sim_steps, 7), dtype=np.float64)
    qvel_hist = np.zeros((total_sim_steps, 7), dtype=np.float64)
    ee_hist = np.zeros((total_sim_steps, 3), dtype=np.float64)

    grip_ctrl = data.ctrl[7]
    grasp_lock = GraspLock(model)
    sim_step = 0
    for seg in dense_segments:
        for pt in seg["dense"]:
            data.ctrl[:7] = pt
            data.ctrl[7] = grip_ctrl
            for _ in range(steps_per_interp_point):
                mujoco.mj_step(model, data)
                grasp_lock.update(data)
                qpos_hist[sim_step] = data.qpos[:7]
                qvel_hist[sim_step] = data.qvel[:7]
                ee_hist[sim_step] = data.site_xpos[ee_site_id]
                sim_step += 1

        action = seg["action_after"]
        if action == "grasp":
            data.ctrl[7] = 0.0
            grip_ctrl = 0.0
            for _ in range(200):
                mujoco.mj_step(model, data)
                grasp_lock.update(data)
            grasp_lock.attach(model, data, grasped_object)
        elif action == "release":
            grasp_lock.release(data)
            data.ctrl[7] = 255.0
            grip_ctrl = 255.0
            for _ in range(200):
                mujoco.mj_step(model, data)

    return {
        "qpos": qpos_hist,
        "qvel": qvel_hist,
        "ee_pos": ee_hist,
        "segment_boundaries_sim": segment_boundaries_sim,
        "segment_names": [s["name"] for s in dense_segments],
        "total_sim_steps": total_sim_steps,
    }


def segment_for(sim_step: int, boundaries: list, names: list) -> str:
    for i in range(len(names)):
        if boundaries[i] <= sim_step < boundaries[i + 1]:
            return names[i]
    return names[-1]


def summarize(samples, states):
    """Return per-strategy summary stats over the sampled sim-steps."""
    qvel_norms = np.linalg.norm(states["qvel"][samples], axis=1)
    ee_z = states["ee_pos"][samples, 2]
    seg_counts = {n: 0 for n in states["segment_names"]}
    for s in samples:
        seg_counts[segment_for(s, states["segment_boundaries_sim"],
                               states["segment_names"])] += 1
    return {
        "n": len(samples),
        "seg_counts": seg_counts,
        "qvel_p10": float(np.percentile(qvel_norms, 10)),
        "qvel_p50": float(np.percentile(qvel_norms, 50)),
        "qvel_p90": float(np.percentile(qvel_norms, 90)),
        "ee_z_p10": float(np.percentile(ee_z, 10)),
        "ee_z_p50": float(np.percentile(ee_z, 50)),
        "ee_z_p90": float(np.percentile(ee_z, 90)),
        "near_static_frac": float((qvel_norms < 0.05).mean()),
    }


def text_histogram(samples, total, bins=40, width=60):
    """Compact ASCII histogram of sample positions along the timeline."""
    hist, _ = np.histogram(samples, bins=bins, range=(0, total))
    if hist.max() == 0:
        return "(no samples)"
    scale = width / hist.max()
    lines = []
    for i, v in enumerate(hist):
        bar = "#" * max(1, int(v * scale)) if v > 0 else ""
        lines.append(f"  {i * total // bins:>6}..{((i + 1) * total // bins):>6}  {v:>4}  {bar}")
    return "\n".join(lines)


def save_fig(path, per_traj, strategies):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  (matplotlib not available, skipping {path})")
        return

    n = len(per_traj)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n), squeeze=False)
    for row, (pkl, entry) in enumerate(per_traj):
        states = entry["states"]
        qvel_norms = np.linalg.norm(states["qvel"], axis=1)
        ax_t = axes[row, 0]
        ax_v = axes[row, 1]

        # Timeline with segment shading
        colors = ["#d7e5ff", "#c9f4d4", "#fff0c0", "#ffd2c0", "#d0c0ff"]
        for i, name in enumerate(states["segment_names"]):
            a = states["segment_boundaries_sim"][i]
            b = states["segment_boundaries_sim"][i + 1]
            ax_t.axvspan(a, b, color=colors[i % len(colors)], alpha=0.5)
            ax_v.axvspan(a, b, color=colors[i % len(colors)], alpha=0.5)

        ax_v.plot(qvel_norms, color="black", linewidth=0.7)
        ax_v.set_ylabel("|qvel| (rad/s)")
        ax_v.set_title(f"{os.path.basename(pkl)} — joint velocity")
        ax_v.set_xlabel("sim step")

        for j, strat in enumerate(strategies):
            samples = entry["samples_by_strategy"][strat]
            y = np.full_like(samples, -0.3 - 0.1 * j, dtype=float)
            ax_t.scatter(samples, y, label=strat, s=6, alpha=0.7)
            ax_v.scatter(samples, qvel_norms[samples], s=8, alpha=0.6, label=strat)

        ax_t.set_yticks([])
        ax_t.set_ylim(-1, 0)
        ax_t.set_title(f"{os.path.basename(pkl)} — sample positions")
        ax_t.set_xlabel("sim step")
        ax_t.legend(loc="upper right", fontsize=7)
        ax_v.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nFigure saved to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajs", nargs="+", default=DEFAULT_TRAJS)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--strategies", nargs="+",
                   default=["uniform", "stratified_segments", "stratified_bands"])
    p.add_argument("--n_bands", type=int, default=6)
    p.add_argument("--boundary_margin", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_fig", type=str, default="")
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    per_traj = []
    for pkl in args.trajs:
        if not os.path.exists(pkl):
            print(f"  skipping missing {pkl}")
            continue
        print(f"\n=== {pkl} ===")
        states = simulate_trajectory(pkl)
        total = states["total_sim_steps"]
        boundaries = states["segment_boundaries_sim"]
        print(f"  total sim steps: {total}")
        print(f"  segments (sim-step range): " +
              ", ".join(f"{n}=[{a},{b})" for n, a, b in zip(
                  states["segment_names"], boundaries[:-1], boundaries[1:])))

        entry = {"states": states, "samples_by_strategy": {}}
        for strat in args.strategies:
            if strat == "stratified_segments":
                samples = sample_fail_sim_step(
                    total, strat, rng,
                    segment_boundaries_sim=boundaries,
                    boundary_margin=args.boundary_margin,
                )
                # Repeat over n_samples for distribution stats, with fresh seeds.
                big = []
                for _ in range(args.n_samples):
                    big.extend(sample_fail_sim_step(
                        total, strat, rng,
                        segment_boundaries_sim=boundaries,
                        boundary_margin=args.boundary_margin))
                samples_for_stats = np.array(big)
            elif strat == "stratified_bands":
                big = []
                for _ in range(args.n_samples):
                    big.extend(sample_fail_sim_step(
                        total, strat, rng, n_bands=args.n_bands))
                samples_for_stats = np.array(big)
            else:  # uniform
                samples_for_stats = np.array(sample_fail_sim_step(
                    total, strat, rng, n=args.n_samples))

            entry["samples_by_strategy"][strat] = samples_for_stats

            stats = summarize(samples_for_stats, states)
            total_s = sum(stats["seg_counts"].values())
            pct = {k: 100 * v / total_s for k, v in stats["seg_counts"].items()}

            print(f"\n  [{strat}] n={stats['n']}")
            print(f"    segment %:  " +
                  "  ".join(f"{k}={pct[k]:5.1f}%" for k in states["segment_names"]))
            print(f"    qvel_norm:  p10={stats['qvel_p10']:.3f}"
                  f"  p50={stats['qvel_p50']:.3f}"
                  f"  p90={stats['qvel_p90']:.3f}"
                  f"  near_static={100 * stats['near_static_frac']:.1f}%")
            print(f"    ee_z:       p10={stats['ee_z_p10']:.3f}"
                  f"  p50={stats['ee_z_p50']:.3f}"
                  f"  p90={stats['ee_z_p90']:.3f}")
        per_traj.append((pkl, entry))

    if args.save_fig and per_traj:
        save_fig(args.save_fig, per_traj, args.strategies)


if __name__ == "__main__":
    main()
