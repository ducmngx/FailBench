#!/usr/bin/env python3
"""Add baseline-equilibrium contact arrays to existing RoboCasa v2 HDF5s.

For each trial:
1. Load the source demo + materialise the cached MJCF.
2. Seed the sim to the trial's recorded pre_qpos / pre_qvel (the failure boundary
   state), with PD on every arm joint targeting pre_target_qpos so the arm is
   held still.
3. Step 50 physics frames with the gravity-comp + PD active resistance on ALL
   seven arm joints (i.e. as if no joint had failed) so the kitchen settles
   into static equilibrium.
4. Extract contacts at the final baseline frame and write four new arrays to
   the trial group::

       baseline_contact_positions   (N_b, 3)
       baseline_contact_forces      (N_b, 6)
       baseline_contact_force_world (N_b, 3)
       baseline_contact_geom_pairs  (N_b, 2)

Training-time loaders subtract this baseline from ``contact_*`` by matching
``(geom1, geom2, ~pos)`` to isolate failure-induced contacts.

Run::

    /home/aaron/miniconda3/envs/failbench_env/bin/python -u \\
        -m scripts.robocasa.add_baseline_contacts \\
        --v2_root /media/aaron/F/failbench/robocasa/v2

Idempotent: skips trials that already carry the baseline fields.
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
from typing import Optional

import h5py
import hdf5plugin  # noqa: F401 — register filters
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planner.experiments.libero.naming import resolve_model_handles
from planner.experiments.robocasa.adapter import load_demo, materialise_mjcf


_DEFAULT_KP = np.array([600.0, 600.0, 600.0, 600.0, 300.0, 120.0, 120.0])
_DEFAULT_KD = 2.0 * np.sqrt(_DEFAULT_KP)
BASELINE_FIELDS = (
    "baseline_contact_positions",
    "baseline_contact_forces",
    "baseline_contact_force_world",
    "baseline_contact_geom_pairs",
)


def hold_all_joints(model: mujoco.MjModel, data: mujoco.MjData, h,
                    target: np.ndarray) -> None:
    """Apply gravcomp + PD on every arm joint (NO failed joints)."""
    mujoco.mj_forward(model, data)
    for i, aid in enumerate(h.arm_actuator_ids):
        if aid < 0:
            continue
        q = float(data.qpos[h.arm_qpos_adrs[i]])
        qd = float(data.qvel[h.arm_dof_adrs[i]])
        grav = float(data.qfrc_bias[h.arm_dof_adrs[i]])
        tau = grav + _DEFAULT_KP[i] * (target[i] - q) - _DEFAULT_KD[i] * qd
        lo, hi = model.actuator_ctrlrange[aid]
        data.ctrl[aid] = float(np.clip(tau, lo, hi))


def extract_contacts(model, data, robot_geom_ids: set, min_force: float = 1.0):
    pos, fl, fw, pairs = [], [], [], []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in robot_geom_ids and g2 in robot_geom_ids:
            continue
        f6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f6)
        if np.linalg.norm(f6[:3]) < min_force:
            continue
        frame = c.frame.reshape(3, 3)
        fwv = frame.T @ f6[:3]
        pos.append(c.pos.copy())
        fl.append(f6.copy())
        fw.append(fwv)
        pairs.append([g1, g2])
    if not pos:
        return (np.zeros((0, 3), np.float32),
                np.zeros((0, 6), np.float32),
                np.zeros((0, 3), np.float32),
                np.zeros((0, 2), np.int32))
    return (np.asarray(pos, np.float32),
            np.asarray(fl, np.float32),
            np.asarray(fw, np.float32),
            np.asarray(pairs, np.int32))


def trial_has_baseline(grp: h5py.Group) -> bool:
    return all(k in grp for k in BASELINE_FIELDS)


def write_baseline(grp: h5py.Group, pos, fl, fw, pairs) -> None:
    # Compression matches the rest of v2's contact arrays.
    try:
        from hdf5plugin import Blosc
        comp = Blosc(cname="lz4", clevel=5)
    except ImportError:  # pragma: no cover
        comp = "lzf"
    chunks_n = max(1, min(pos.shape[0], 1024))
    chunk_arr = lambda shape: (chunks_n,) + tuple(shape[1:])
    if isinstance(comp, str):
        kw = {"compression": comp}
    else:
        kw = dict(comp)
    grp.create_dataset("baseline_contact_positions", data=pos,
                       chunks=chunk_arr(pos.shape), **kw) if pos.shape[0] > 0 else grp.create_dataset("baseline_contact_positions", data=pos)
    grp.create_dataset("baseline_contact_forces", data=fl,
                       chunks=chunk_arr(fl.shape), **kw) if fl.shape[0] > 0 else grp.create_dataset("baseline_contact_forces", data=fl)
    grp.create_dataset("baseline_contact_force_world", data=fw,
                       chunks=chunk_arr(fw.shape), **kw) if fw.shape[0] > 0 else grp.create_dataset("baseline_contact_force_world", data=fw)
    grp.create_dataset("baseline_contact_geom_pairs", data=pairs,
                       chunks=chunk_arr(pairs.shape), **kw) if pairs.shape[0] > 0 else grp.create_dataset("baseline_contact_geom_pairs", data=pairs)


def process_task(h5_path: Path, raw_root: Path,
                 baseline_steps: int = 50, max_trials: Optional[int] = None) -> dict:
    """Pass through one task's HDF5; cache RoboCasa demos / MjModels per (hdf5, demo)."""
    t0 = time.time()
    n_done = n_skip = n_err = 0
    task = h5_path.stem
    src_hdf5 = raw_root / f"{task}.hdf5"
    if not src_hdf5.exists():
        return {"task": task, "n_done": 0, "n_err": 0, "n_skip": 0,
                "error": f"raw hdf5 missing: {src_hdf5}"}

    # Cache by demo_key — many trials share the same demo.
    demo_cache: dict = {}

    with h5py.File(h5_path, "r+") as h:
        trial_ids = list(h["trials"].keys())
        if max_trials is not None:
            trial_ids = trial_ids[:max_trials]
        for ti, tid in enumerate(trial_ids):
            g = h[f"trials/{tid}"]
            if trial_has_baseline(g):
                n_skip += 1
                continue
            try:
                demo_key = g.attrs["demo_key"] if "demo_key" in g.attrs else tid.split("_s", 1)[0]
                if isinstance(demo_key, bytes):
                    demo_key = demo_key.decode("utf-8")
                if demo_key not in demo_cache:
                    demo = load_demo(str(src_hdf5), demo_key)
                    xml_path = materialise_mjcf(demo.model_xml)
                    model = mujoco.MjModel.from_xml_path(xml_path)
                    data = mujoco.MjData(model)
                    handles = resolve_model_handles(model)
                    demo_cache[demo_key] = (model, data, handles)
                    # Limit cache size — drop oldest if too big.
                    if len(demo_cache) > 4:
                        demo_cache.pop(next(iter(demo_cache)))
                else:
                    model, data, handles = demo_cache[demo_key]

                # Seed pre-failure qpos/qvel
                pre_qpos = np.asarray(g["pre_qpos"][...]).astype(np.float64)
                pre_qvel = np.asarray(g["pre_qvel"][...]).astype(np.float64)
                pre_target = np.asarray(g["pre_target_qpos"][...]).astype(np.float64)

                # Reset full state by re-seeding from full_states[fail_idx]
                # (so object positions are right too — pre_qpos is arm-only).
                fail_idx = int(g.attrs["fail_idx"])
                from planner.experiments.robocasa.adapter import load_demo as _ld  # local import
                # Reuse the per-demo full_states via the demo loader cache.
                # Cheaper: reload only states (small array) on the fly.
                with h5py.File(str(src_hdf5), "r") as srcf:
                    flat = np.asarray(srcf[f"data/{demo_key}/states"][fail_idx]).astype(np.float64)
                nq, nv = model.nq, model.nv
                offset = 1 if flat.shape[0] == 1 + nq + nv else 0
                data.qpos[:] = flat[offset:offset + nq]
                data.qvel[:] = flat[offset + nq:offset + nq + nv]
                mujoco.mj_forward(model, data)

                # Step physics with PD on all arm joints (no failure) so the
                # kitchen reaches static equilibrium.
                for _ in range(baseline_steps):
                    hold_all_joints(model, data, handles, pre_target)
                    mujoco.mj_step(model, data)

                pos, fl, fw, pairs = extract_contacts(
                    model, data, handles.robot_geom_ids, min_force=1.0,
                )
                write_baseline(g, pos, fl, fw, pairs)
                n_done += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [{task}/{tid}] failed: {e}", flush=True)
                n_err += 1
            if (ti + 1) % 200 == 0:
                dt = time.time() - t0
                print(f"  {task}: {ti + 1}/{len(trial_ids)}  "
                      f"({n_done} new / {n_skip} skip / {n_err} err)  "
                      f"{(ti + 1) / max(dt, 1e-6):.2f}/s", flush=True)
        h.flush()

    return {
        "task": task,
        "n_done": n_done,
        "n_skip": n_skip,
        "n_err": n_err,
        "elapsed_s": time.time() - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--v2_root", required=True, type=Path,
                   help="e.g. /media/aaron/F/failbench/robocasa/v2")
    p.add_argument("--raw_root", default=str(REPO_ROOT / "datasets" / "robocasa" / "raw"),
                   type=Path)
    p.add_argument("--baseline_steps", type=int, default=50)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--max_trials", type=int, default=None)
    args = p.parse_args()

    h5_files = sorted(args.v2_root.glob("*.h5"))
    if args.tasks:
        h5_files = [p for p in h5_files if p.stem in args.tasks]
    if not h5_files:
        print(f"No HDF5s found in {args.v2_root}", file=sys.stderr)
        return 2

    t0 = time.time()
    for h5_path in h5_files:
        print(f"=== {h5_path.name} ===", flush=True)
        res = process_task(h5_path, args.raw_root,
                           baseline_steps=args.baseline_steps,
                           max_trials=args.max_trials)
        if "error" in res:
            print(f"  ERROR: {res['error']}", flush=True)
            continue
        print(f"  done: {res['n_done']} new / {res['n_skip']} skip / "
              f"{res['n_err']} err  in {res['elapsed_s']:.1f}s "
              f"({res['n_done'] / max(res['elapsed_s'], 1e-6):.2f}/s)", flush=True)

    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
