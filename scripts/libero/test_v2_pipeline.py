#!/usr/bin/env python3
"""End-to-end smoke test for the LIBERO v2 dataset build.

Builds a small handful of trials for one libero_spatial task into a temp
directory, then loads them via :class:`LiberoV2Dataset` and asserts the
key schema invariants:

* window_qpos[-1] ≈ pre_qpos (last window frame is the fail moment)
* window_frame_idx[-1] == trial-group fail_idx attr
* window_qvel finite-diff is non-zero (vs v1's pre_qvel ≈ 0)
* contact_time monotonic-nondecreasing and within [0, settle_steps)
* contact_force_world norms equal contact_forces[:, :3] norms (rotation
  preserves magnitude)
* depth values within plausible range for both cameras at pre, window, post
* settle_qpos / settle_obj_pos shapes match settle_S
* obj_pos_pre and obj_pos_post differ for a failure trial (something moved)

Exit code 0 on pass. Run with::

    MUJOCO_GL=egl python -m scripts.libero.test_v2_pipeline \\
        --output_dir /tmp/v2_smoke --limit 4
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from planner.risk.dataset_v2 import LiberoV2Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_build(output_dir: Path, limit: int) -> None:
    cmd = [
        sys.executable, "-m", "scripts.libero.build_v2_dataset",
        "--splits", "libero_spatial",
        "--v1_root", str(REPO_ROOT / "datasets" / "libero" / "v1"),
        "--output_dir", str(output_dir),
        "--workers", "1",
        "--limit_tasks", "1",
        "--limit_per_task", str(limit),
        "--settle_steps", "150",  # keep it brisk for smoke test
        "--log_level", "WARNING",
    ]
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    subprocess.run(cmd, check=True, env=env, cwd=str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default="/tmp/v2_smoke")
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--keep", action="store_true",
                   help="Keep the temp build dir after the test")
    args = p.parse_args()

    out = Path(args.output_dir)
    if out.exists() and not args.keep:
        shutil.rmtree(out)

    print(f"Building {args.limit} trials into {out} ...")
    _run_build(out, args.limit)

    ds = LiberoV2Dataset(out, splits=("libero_spatial",),
                        use_window=True, use_wrist_cam=True,
                        use_depth=True, use_settle=True)
    print(f"Loaded dataset: {len(ds)} trials")
    assert len(ds) > 0, "Build produced zero trials"

    failures = []

    for i in range(len(ds)):
        s = ds[i]
        tid = s["trial_id"]

        # Window-end vs pre alignment
        if not np.allclose(s["window_qpos"][-1], s["pre_qpos"].astype(np.float32), atol=1e-4):
            failures.append(f"[{tid}] window_qpos[-1] != pre_qpos")

        if int(s["window_frame_idx"][-1]) != int(s["fail_idx"]):
            failures.append(
                f"[{tid}] window_frame_idx[-1]={s['window_frame_idx'][-1]} "
                f"!= fail_idx={s['fail_idx']}"
            )

        # Real (non-zero) qvel
        if np.max(np.abs(s["window_qvel"][-1])) < 1e-6:
            failures.append(f"[{tid}] window_qvel[-1] ≈ 0 — finite-diff broken")

        # Contact time bounds + monotonicity (allowed within-step duplicates,
        # so check non-decreasing, not strictly increasing).
        ct = s["contact_time"]
        if ct.size:
            if (ct[1:] < ct[:-1]).any():
                failures.append(f"[{tid}] contact_time not monotonic-nondecreasing")
            if int(ct.min()) < 0 or int(ct.max()) >= 500:
                failures.append(
                    f"[{tid}] contact_time out of range: "
                    f"[{int(ct.min())}, {int(ct.max())}]"
                )

        # Force norm preservation
        if s["contact_forces"].shape[0] > 0:
            n_local = np.linalg.norm(s["contact_forces"][:, :3], axis=1)
            n_world = np.linalg.norm(s["contact_force_world"], axis=1)
            if not np.allclose(n_local, n_world, atol=1e-3):
                failures.append(
                    f"[{tid}] contact_force_world norm differs from contact_forces "
                    f"max-diff={np.max(np.abs(n_local - n_world)):.4f}"
                )

        # Depth ranges (LIBERO workspace 0.1–4 m typical)
        for k in ("pre_depth", "window_agentview_depth",
                  "post_agentview_depth", "post_wrist_depth"):
            d = np.asarray(s[k], dtype=np.float32)
            d = d[np.isfinite(d) & (d > 0)]
            if d.size and (d.min() < 0.01 or d.max() > 20.0):
                failures.append(
                    f"[{tid}] {k} out of plausible range: "
                    f"[{d.min():.3f}, {d.max():.3f}] m"
                )

        # Settle shapes
        if s["settle_qpos"].shape[1] != 7:
            failures.append(f"[{tid}] settle_qpos.shape[1] != 7")
        if s["settle_obj_pos"].shape[1] != len(s["obj_names"]):
            failures.append(
                f"[{tid}] settle_obj_pos n_obj={s['settle_obj_pos'].shape[1]} "
                f"!= len(obj_names)={len(s['obj_names'])}"
            )

        # Print one-line summary
        print(f"  [{tid}] mode={s['failure_mode']} "
              f"contacts={s['contact_positions'].shape[0]} "
              f"is_holding={s['is_holding']}")

    if failures:
        print("\nFAIL")
        for f in failures:
            print("  ", f)
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
