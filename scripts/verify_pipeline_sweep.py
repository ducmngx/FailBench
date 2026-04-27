#!/usr/bin/env python
"""Small-scale pipeline sweep to verify continuous fail_fraction sampling
and ee_cam rendering before full-scale dataset generation.

Shape:
    scene_level2 x 2 tasks x 2 trajs x 4 trials = 16 experiments
    Each trial draws fail_fraction ~ U(0, 1) from a seeded RNG.
    All 5 failure modes run per sample.
    Output: scenes/scene_level2/datasets/verify_sweep/{manifest.csv, exp_*.npz}
"""
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.experiments.config import ExperimentConfig
from planner.experiments.manager import BatchExperimentManager
from scripts.generate_task_trajs import _detect_robot_xml

SCENE = "scene_level2"
SCENE_XML = f"scenes/{SCENE}/scene.xml"
TASKS = ["clean_nominal", "stack_nominal"]
TRAJS_PER_TASK = 2
TRIALS_PER_TRAJ = 4
BASE_SEED = 20260424
OUTPUT_DIR = f"scenes/{SCENE}/datasets/verify_sweep"
NUM_WORKERS = 4


def main():
    robot_xml = _detect_robot_xml(SCENE_XML)
    rng = random.Random(BASE_SEED)

    traj_dir = f"scenes/{SCENE}/trajs"
    configs = []
    idx = 0
    for task_id in TASKS:
        pattern = os.path.join(traj_dir, f"{SCENE}_{task_id}_*.pkl")
        pkls = sorted(glob.glob(pattern))[:TRAJS_PER_TASK]
        if len(pkls) < TRAJS_PER_TASK:
            raise RuntimeError(
                f"Not enough pkls for {task_id}: found {len(pkls)}, need {TRAJS_PER_TASK}"
            )
        for traj_id, pkl in enumerate(pkls):
            for trial in range(TRIALS_PER_TRAJ):
                frac = rng.uniform(0.05, 0.98)
                configs.append(ExperimentConfig(
                    scene_xml_path=SCENE_XML,
                    robot_xml_path=robot_xml,
                    trajectory_file=pkl,
                    task_id=task_id,
                    traj_id=traj_id,
                    seed=BASE_SEED + idx,
                    experiment_id=f"exp_{idx:05d}",
                    fail_fraction=frac,
                    extra_cameras=["ee_cam"],
                    failure_sample_mode="all",
                    post_failure_settle_steps=200,
                ))
                idx += 1

    print(f"Generated {len(configs)} configs")
    for c in configs[:4]:
        print(f"  {c.experiment_id}: task={c.task_id} traj={c.traj_id} "
              f"frac={c.fail_fraction:.3f}")
    print("  ...")

    mgr = BatchExperimentManager(output_dir=OUTPUT_DIR, num_workers=NUM_WORKERS)
    manifest = mgr.run_batch_chunked(configs, chunk_size=len(configs))
    print(f"\nManifest: {manifest}")


if __name__ == "__main__":
    main()
