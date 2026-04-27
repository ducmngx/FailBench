#!/usr/bin/env python
"""Full-scale PoC dataset generator.

Iterates enabled tasks x trajectories across all requested scenes, drawing
K_per_segment continuous fail_fraction samples stratified per mission segment
(approach / descend / lift / transport / place). Each trial runs all 6 default
failure modes (failure_sample_mode="all"); one npz per trial aggregates
contacts across modes.

Output layout:

    datasets/<output_version>/<scene>/<task>/{manifest.csv, exp_*.npz}

Per-task subfolders keep train/val/split logic easy downstream and make it
trivial to inspect a single task's samples.

Defaults produce ~16,775 samples across 5 scenes at K=25 per trajectory.
"""
import argparse
import glob
import os
import pickle
import random
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.experiments.config import ExperimentConfig
from planner.experiments.manager import BatchExperimentManager
from planner.tasks import load_tasks
from scripts.generate_task_trajs import _detect_robot_xml

DEFAULT_SCENES = [
    "scene_level2",
    "scene_kitchen",
    "scene_cluttered",
    "scene_workshop",
    "scene_grocery",
]


def segment_bounds(segments: List[dict], points_per_segment: int = 100
                   ) -> List[Tuple[float, float]]:
    """Per-segment (lo, hi) fractional bounds over the concatenated dense timeline.

    `interpolate_trajectory` emits (n_waypoints-1)*points_per_segment+1 dense
    points per segment, so segment lengths in fraction-space depend on each
    segment's sparse-waypoint count.
    """
    lens = [(len(s["trajectory"]) - 1) * points_per_segment + 1 for s in segments]
    total = sum(lens)
    bounds = []
    cum = 0
    for L in lens:
        lo = cum / total
        cum += L
        hi = cum / total
        bounds.append((lo, hi))
    return bounds


def build_task_configs(
    scene: str,
    task_id: str,
    task_def: dict,
    scene_xml: str,
    robot_xml: str,
    k_per_segment: int,
    base_seed: int,
    start_idx: int,
    grasped_default: str,
) -> Tuple[List[ExperimentConfig], int, int]:
    """Return (configs, next_idx, n_trajs) for one (scene, task)."""
    grasped = task_def.get("grasped_object", grasped_default)
    traj_dir = f"scenes/{scene}/trajs"
    pkls = sorted(glob.glob(os.path.join(traj_dir, f"{scene}_{task_id}_*.pkl")))
    if not pkls:
        return [], start_idx, 0

    rng = random.Random(base_seed + abs(hash((scene, task_id))) % 10_000_000)

    configs: List[ExperimentConfig] = []
    idx = start_idx
    for traj_id, pkl_path in enumerate(pkls):
        with open(pkl_path, "rb") as f:
            entry = pickle.load(f)[scene]
        bounds = segment_bounds(entry["segments"])

        for (lo, hi) in bounds:
            for _ in range(k_per_segment):
                frac = rng.uniform(lo, hi)
                configs.append(ExperimentConfig(
                    scene_xml_path=scene_xml,
                    robot_xml_path=robot_xml,
                    trajectory_file=pkl_path,
                    task_id=task_id,
                    traj_id=traj_id,
                    seed=base_seed + idx,
                    experiment_id=f"exp_{idx:06d}",
                    fail_fraction=frac,
                    extra_cameras=["ee_cam"],
                    failure_sample_mode="all",
                    grasped_object_name=grasped,
                ))
                idx += 1

    return configs, idx, len(pkls)


def run_task(scene: str, task_id: str, task_def: dict, grasped_default: str,
             scene_xml: str, robot_xml: str, out_root: str,
             k_per_segment: int, workers: int, base_seed: int,
             start_idx: int) -> Tuple[int, int]:
    """Generate + run one task's configs. Returns (next_idx, n_trials_run)."""
    out_dir = os.path.join(out_root, scene, task_id)

    configs, next_idx, n_trajs = build_task_configs(
        scene, task_id, task_def, scene_xml, robot_xml,
        k_per_segment, base_seed, start_idx, grasped_default,
    )

    if not configs:
        print(f"  [{scene}/{task_id}] no pkls — skip")
        return next_idx, 0

    print(f"  [{scene}/{task_id}] {n_trajs} trajs -> {len(configs)} trials -> {out_dir}")
    t0 = time.time()
    mgr = BatchExperimentManager(output_dir=out_dir, num_workers=workers)
    mgr.run_batch_chunked(configs, chunk_size=max(1, len(configs) // 5))
    dt = time.time() - t0
    rate = len(configs) / dt if dt > 0 else 0.0
    print(f"  [{scene}/{task_id}] done in {dt/60:.1f} min ({rate:.2f} trials/s)")
    return next_idx, len(configs)


def run_scene(scene: str, out_root: str, k_per_segment: int,
              workers: int, base_seed: int, start_idx: int,
              task_filter: List[str] = None) -> int:
    scene_xml = f"scenes/{scene}/scene.xml"
    robot_xml = _detect_robot_xml(scene_xml)

    print(f"\n=== {scene} (robot_xml={os.path.basename(robot_xml)}) ===")

    tasks_yaml = load_tasks(scene)
    task_defs = tasks_yaml.get("tasks", {}) or {}
    grasped_default = tasks_yaml.get("grasped_object", "object3")
    enabled = [name for name, d in task_defs.items() if d.get("enabled", True)]
    if task_filter:
        enabled = [t for t in enabled if t in task_filter]

    idx = start_idx
    total_trials = 0
    for task_id in sorted(enabled):
        idx, n = run_task(
            scene=scene, task_id=task_id, task_def=task_defs[task_id],
            grasped_default=grasped_default,
            scene_xml=scene_xml, robot_xml=robot_xml, out_root=out_root,
            k_per_segment=k_per_segment, workers=workers, base_seed=base_seed,
            start_idx=idx,
        )
        total_trials += n

    print(f"=== {scene}: {total_trials} trials across {len(enabled)} tasks ===")
    return idx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Optional filter: only generate these task ids")
    parser.add_argument("--k_per_segment", type=int, default=5,
                        help="Continuous fail_fraction draws per segment (K_total = 5*k_per_segment)")
    parser.add_argument("--output_version", default="v10",
                        help="Output subdir under datasets/ (default v10)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base_seed", type=int, default=20260424)
    args = parser.parse_args()

    out_root = os.path.join("datasets", args.output_version)
    os.makedirs(out_root, exist_ok=True)

    print(f"Scenes:          {args.scenes}")
    print(f"Task filter:     {args.tasks or 'all enabled'}")
    print(f"K per segment:   {args.k_per_segment} (K per traj = {args.k_per_segment*5})")
    print(f"Output root:     {out_root}")
    print(f"Workers:         {args.workers}")
    print(f"Base seed:       {args.base_seed}")

    t_start = time.time()
    idx = 0
    for scene in args.scenes:
        idx = run_scene(
            scene=scene, out_root=out_root,
            k_per_segment=args.k_per_segment,
            workers=args.workers, base_seed=args.base_seed,
            start_idx=idx, task_filter=args.tasks,
        )
    dt = time.time() - t_start
    print(f"\nTotal: {idx} trials in {dt/60:.1f} min -> {out_root}")


if __name__ == "__main__":
    main()
