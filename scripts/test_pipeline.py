#!/usr/bin/env python
"""End-to-end pipeline smoke test for segmented full-mission trajectories.

1. Generates 1 trajectory via generate_task_trajs (approach→descend→lift→transport→place)
2. Runs ExperimentRunner with segmented replay
3. Asserts DataSample and npz fields are correct

Usage:
    python scripts/test_pipeline.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCENE = "scene_level2"
SCENE_XML = "scenes/scene_level2/scene.xml"
TASK = "pick_place_nominal"

# ---------------------------------------------------------------------------
# 1. Generate one trajectory
# ---------------------------------------------------------------------------

print("=== Step 1: Generate one segmented trajectory ===")
out_dir = tempfile.mkdtemp(prefix="test_trajs_")
from scripts.generate_task_trajs import generate, _detect_robot_xml
robot_xml = _detect_robot_xml(SCENE_XML)

generate(scene=SCENE, task=TASK, n_trajs=1, seed=99,
         scene_xml=SCENE_XML, robot_xml=robot_xml, out_dir=out_dir)

import glob
pkl_files = glob.glob(os.path.join(out_dir, "*.pkl"))
assert len(pkl_files) == 1, f"Expected 1 pkl, got {len(pkl_files)}"
traj_path = pkl_files[0]

# Verify pkl structure
import pickle
with open(traj_path, "rb") as f:
    data = pickle.load(f)
scene_data = data[SCENE]
assert "segments" in scene_data, "Missing 'segments' key in pkl"
segments = scene_data["segments"]
assert len(segments) == 5, f"Expected 5 segments, got {len(segments)}"
expected_names = ["approach", "descend", "lift", "transport", "place"]
actual_names = [s["name"] for s in segments]
assert actual_names == expected_names, f"Segment names: {actual_names}"
assert segments[1]["action_after"] == "grasp"
assert segments[4]["action_after"] == "release"
total_wps = sum(len(s["trajectory"]) for s in segments)
print(f"  [PASS] pkl structure correct: {total_wps} total waypoints across 5 segments")

# ---------------------------------------------------------------------------
# 2. Run ExperimentRunner with segmented trajectory
# ---------------------------------------------------------------------------

print("\n=== Step 2: Run ExperimentRunner ===")
from planner.experiments.config import ExperimentConfig
from planner.experiments.runner import ExperimentRunner

for frac_label, frac in [("early (approach)", 0.1), ("mid (transport)", 0.6)]:
    print(f"\n  Testing fail_fraction={frac} ({frac_label}) ...")
    config = ExperimentConfig(
        scene_xml_path=SCENE_XML,
        robot_xml_path=robot_xml,
        trajectory_file=traj_path,
        task_id=TASK,
        traj_id=0,
        seed=42,
        fail_fraction=frac,
        experiment_id=f"test_{frac}",
        extra_cameras=["ee_cam"],
        post_failure_settle_steps=50,
    )

    runner = ExperimentRunner(config)
    sample = runner.run()
    runner.close()

    assert sample is not None, "Runner returned None"
    assert sample.task_id == TASK
    assert sample.traj_id == 0
    assert abs(sample.traj_progress - frac) < 1e-5, f"traj_progress={sample.traj_progress}"
    assert not hasattr(sample, "fail_phase"), "Still has old fail_phase field"
    print(f"    [PASS] traj_progress={sample.traj_progress:.3f}, "
          f"contacts={len(sample.aggregate_contacts)}")

# ---------------------------------------------------------------------------
# 3. Verify npz serialization
# ---------------------------------------------------------------------------

print("\n=== Step 3: Verify npz serialization ===")
from planner.experiments.manager import save_sample_npz

npz_tmp = tempfile.mktemp(suffix=".npz")
save_sample_npz(sample, npz_tmp)
npz = dict(np.load(npz_tmp, allow_pickle=True))
os.unlink(npz_tmp)

required = ["pre_rgb", "pre_qpos", "pre_ee_pos", "task_id", "traj_id",
            "traj_progress", "contact_positions", "contact_forces",
            "contact_failure_id", "ee_cam_rgb", "ee_cam_depth"]
missing = [k for k in required if k not in npz]
assert not missing, f"npz missing: {missing}"

forbidden = ["fail_phase", "fail_step"]
present = [k for k in forbidden if k in npz]
assert not present, f"npz has old keys: {present}"

print(f"  [PASS] npz keys: {sorted(npz.keys())}")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

import shutil
shutil.rmtree(out_dir)
print("\nAll checks passed.")
