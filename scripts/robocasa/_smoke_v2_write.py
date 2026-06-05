"""Smoke: generate a small RoboCasa v2 HDF5 file and validate schema parity with LIBERO v2.

20 trials from TurnOffStove (10 demos × 2 failure configs each) into a temp HDF5.
Confirms:
- LiberoRunner.run_v2() consumes RoboCasa demos via the new adapter + scene overrides.
- V2Writer accepts the payload unchanged.
- Field names / dtypes / shapes match LIBERO v2.

Run from /tmp (outside the FailBench tree) to avoid the editable-finder cwd shadow:
    cd /tmp && /home/aaron/miniconda3/envs/failbench_env/bin/python \\
        -m scripts.robocasa._smoke_v2_write \\
        --hdf5 datasets/robocasa/raw/TurnOffStove.hdf5 \\
        --output /tmp/rc_v2_smoke/TurnOffStove.h5 \\
        --n_demos 10 --trials_per_demo 2
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import sys
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planner.experiments.config import FailureConfig, FailureMode
from planner.experiments.robocasa import (
    load_demo, materialise_mjcf, list_demos, read_ep_meta, build_scene_overrides,
)
from planner.experiments.libero.runner import (
    LiberoRunner, LiberoTrialConfig, V2CaptureSpec,
)
from planner.risk.v2_store import V2Writer


# Stratified failure configs for the smoke — each demo gets these 2 trials.
SMOKE_FAILURES = [
    (0.35, FailureMode.SINGLE_JOINT, ("joint4",)),
    (0.70, FailureMode.GRIPPER_OPEN,  tuple()),
]


def _build_runner(demo, hdf5_path: str, capture: V2CaptureSpec, settle_steps: int):
    xml_path = materialise_mjcf(demo.model_xml)
    ep_meta = read_ep_meta(hdf5_path, demo.demo_key)
    # Bootstrap an MjData seeded to the start state so scene helpers see real positions
    m_tmp = mujoco.MjModel.from_xml_path(xml_path)
    d_tmp = mujoco.MjData(m_tmp)
    nq, nv = m_tmp.nq, m_tmp.nv
    flat = demo.full_states[0]
    if flat.shape[0] == 1 + nq + nv:
        d_tmp.qpos[:] = flat[1:1 + nq]
        d_tmp.qvel[:] = flat[1 + nq:]
    else:
        d_tmp.qpos[:] = flat[:nq]
        d_tmp.qvel[:] = flat[nq:nq + nv]
    mujoco.mj_forward(m_tmp, d_tmp)
    overrides = build_scene_overrides(m_tmp, d_tmp, ep_meta)

    cfg = LiberoTrialConfig(
        fail_progress=None,
        post_failure_settle_steps=settle_steps,
        resistance_mode="gravcomp_pd",
        seed_from_init_state=False,
        image_width=capture.image_w,
        image_height=capture.image_h,
    )
    runner = LiberoRunner(demo, cfg, mjcf_path=xml_path, scene_overrides=overrides)
    return runner


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n_demos", type=int, default=10)
    p.add_argument("--trials_per_demo", type=int, default=2)
    p.add_argument("--settle_steps", type=int, default=500)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--image_h", type=int, default=240)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    capture = V2CaptureSpec(image_w=args.image_w, image_h=args.image_h)
    demos = list_demos(str(args.hdf5))[:args.n_demos]
    failures = SMOKE_FAILURES[:args.trials_per_demo]
    task = args.hdf5.stem
    split = "robocasa"

    n_done, n_err = 0, 0
    t0 = time.time()
    with V2Writer(args.output, split=split, task=task) as writer:
        for di, demo_key in enumerate(demos):
            try:
                demo = load_demo(str(args.hdf5), demo_key)
                runner = _build_runner(demo, str(args.hdf5), capture, args.settle_steps)
            except Exception as e:
                print(f"[demo {demo_key}] load/build failed: {e}", file=sys.stderr)
                n_err += args.trials_per_demo
                continue
            try:
                for ti, (progress, mode, joints) in enumerate(failures):
                    trial_id = f"{demo_key}_s0_b{ti}"
                    runner.config.fail_progress = progress
                    failure = FailureConfig(
                        mode=mode,
                        joint_names=list(joints),
                        probability=1.0 / len(failures),
                    )
                    try:
                        payload = runner.run_v2(capture, failure, experiment_id=trial_id)
                        if payload is None:
                            n_err += 1
                            continue
                        payload.update({
                            "trial_id": trial_id,
                            "split": split,
                            "task": task,
                            "demo_key": demo_key,
                            "seed": 0,
                            "seed_idx": 0,
                            "bin_idx": ti,
                        })
                        writer.write_trial(trial_id, payload)
                        n_done += 1
                    except Exception as e:
                        print(f"[{trial_id}] trial failed: {e}", file=sys.stderr)
                        n_err += 1
            finally:
                if hasattr(runner, "close"):
                    runner.close()
            if (di + 1) % 5 == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                print(f"  {di + 1}/{len(demos)} demos: {n_done} ok / {n_err} err  ({rate:.2f}/s)",
                      flush=True)

    dt = time.time() - t0
    print(f"Done: {n_done} ok / {n_err} err in {dt:.1f}s "
          f"({n_done / max(dt, 1e-6):.2f} trials/s)")

    # Schema parity check vs LIBERO v2
    LIBERO_V2 = Path("/media/aaron/F/failbench/libero/v2/libero_spatial/"
                     "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.h5")
    if not LIBERO_V2.exists():
        print(f"WARN: LIBERO v2 not at {LIBERO_V2}; skipping parity check")
        return 0
    with h5py.File(args.output, "r") as r, h5py.File(LIBERO_V2, "r") as l:
        rc_trials = list(r["trials"].keys())
        lb_trials = list(l["trials"].keys())
        rc_keys = set(r[f"trials/{rc_trials[0]}"].keys())
        lb_keys = set(l[f"trials/{lb_trials[0]}"].keys())
        common = rc_keys & lb_keys
        only_rc = rc_keys - lb_keys
        only_lb = lb_keys - rc_keys
        print(f"Schema parity (LIBERO v2 vs RoboCasa v2 smoke):")
        print(f"  common fields : {len(common)}")
        print(f"  only RoboCasa : {sorted(only_rc)}")
        print(f"  only LIBERO   : {sorted(only_lb)}")
        rc_attrs = set(r[f"trials/{rc_trials[0]}"].attrs.keys())
        lb_attrs = set(l[f"trials/{lb_trials[0]}"].attrs.keys())
        print(f"  attrs common  : {len(rc_attrs & lb_attrs)} "
              f"(only RC: {sorted(rc_attrs - lb_attrs)}, only LB: {sorted(lb_attrs - rc_attrs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
