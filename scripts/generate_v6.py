#!/usr/bin/env python3
"""Generate FailBench v6 dataset with depth camera + EE camera enabled.

Same experiments as v5 but with extra_cameras=["ee_cam"] and depth capture.
Uses free camera params matching v5: lookat=[0.0, -0.35, 0.4], distance=1.1,
azimuth=150, elevation=-65.
"""

import logging
import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planner.experiments.manager import BatchExperimentManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

SCENE_XML = os.path.join(PROJECT_ROOT, "franka_emika_panda", "scene_level2.xml")
ROBOT_XML = os.path.join(PROJECT_ROOT, "franka_emika_panda", "panda.xml")
TRAJ_DIR = os.path.join(PROJECT_ROOT, "collected_trajs", "scene2_trajs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "v6")

# Same trajectory files as v5 (in order)
TRAJECTORY_FILES = [
    os.path.join(TRAJ_DIR, "scene_level2_RRTConnect_new_baseline.pkl"),
    os.path.join(TRAJ_DIR, "scene_level2_RRTConnect_sample_0.pkl"),
    os.path.join(TRAJ_DIR, "scene_level2_RRTConnect_sample_1.pkl"),
    os.path.join(TRAJ_DIR, "scene_level2_RRTConnect_sample_7.pkl"),
]

NUM_TRIALS_PER_TRAJECTORY = 5
BASE_SEED = 0


def main():
    # Verify all trajectory files exist
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
        # Free camera params (same as v5)
        camera_name=None,
        camera_lookat=[0.0, -0.35, 0.4],
        camera_distance=1.1,
        camera_azimuth=150,
        camera_elevation=-65,
        # NEW: enable EE camera
        extra_cameras=["ee_cam"],
    )

    print(f"Generated {len(configs)} experiment configs")
    print(f"Output directory: {OUTPUT_DIR}")

    manifest_path = manager.run_batch(configs)
    print(f"Dataset generation complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
