#!/usr/bin/env python3
"""Generate FailBench v9 dataset — interpolated trajectories + mid-motion failure.

Changes from v8:
  - Cubic spline interpolation between sparse RRT waypoints (50 pts/segment)
  - Failure injected mid-motion at canonical trajectory fractions
  - Single failure mode sampled per trial (probability-weighted)
  - Arm has nonzero velocity at failure time
"""

import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planner.experiments.manager import BatchExperimentManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

SCENE_XML = os.path.join(PROJECT_ROOT, "scenes", "scene_level2", "scene.xml")
ROBOT_XML = os.path.join(PROJECT_ROOT, "franka_emika_panda", "panda.xml")
TRAJ_DIR  = os.path.join(PROJECT_ROOT, "scenes", "scene_level2", "trajs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scenes", "scene_level2", "datasets", "v9")

TRAJECTORY_FILES = [
    os.path.join(TRAJ_DIR, "scene_level2_RRTConnect_new_baseline.pkl"),
    os.path.join(TRAJ_DIR, "scene_level2_RRTConnect_sample_0.pkl"),
    os.path.join(TRAJ_DIR, "scene_level2_RRTConnect_sample_1.pkl"),
    os.path.join(TRAJ_DIR, "scene_level2_RRTConnect_sample_7.pkl"),
]

NUM_TRIALS_PER_TRAJECTORY = 10
BASE_SEED = 0


def main():
    for tf in TRAJECTORY_FILES:
        if not os.path.exists(tf):
            print(f"ERROR: Trajectory file not found: {tf}")
            sys.exit(1)

    manager = BatchExperimentManager(output_dir=OUTPUT_DIR, num_workers=1)

    configs = manager.generate_configs(
        scene_xml_path=SCENE_XML,
        robot_xml_path=ROBOT_XML,
        trajectory_files=TRAJECTORY_FILES,
        num_trials_per_trajectory=NUM_TRIALS_PER_TRAJECTORY,
        base_seed=BASE_SEED,
        camera_name="front_cam",
        extra_cameras=["ee_cam"],
        # v9: interpolated trajectory + mid-motion failure
        use_interpolation=True,
        interp_points_per_segment=50,
        interp_method="cubic",
        steps_per_interp_point=4,
        # v9: sample one failure mode per trial
        failure_sample_mode="sample",
        num_failure_samples=1,
    )

    print(f"Generated {len(configs)} experiment configs")
    print(f"Output directory: {OUTPUT_DIR}")

    manifest_path = manager.run_batch(configs)
    print(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
