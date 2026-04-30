#!/usr/bin/env python3
"""Visual sanity check: replay one LIBERO demo in the MuJoCo viewer.

No failure injection — just kinematic playback of arm + finger qpos through
our adapter, to confirm the cached MJCF + name resolution + qpos addressing
all line up before we trust them in :class:`LiberoRunner`.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from planner.experiments.libero.adapter import list_demos, load_demo, materialise_mjcf
from planner.experiments.libero.naming import resolve_model_handles


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True, help="Path to LIBERO demo HDF5 file")
    p.add_argument("--demo", default="demo_0",
                   help="Demo key inside HDF5 (e.g. demo_3)")
    p.add_argument("--list", action="store_true",
                   help="List demo keys in the HDF5 file and exit")
    p.add_argument("--rate", type=float, default=20.0,
                   help="Playback rate (steps/sec)")
    args = p.parse_args()

    if args.list:
        for k in list_demos(args.hdf5):
            print(k)
        return 0

    demo = load_demo(args.hdf5, args.demo)
    xml_path = materialise_mjcf(demo.model_xml)
    print(f"Cached MJCF → {xml_path}")
    print(f"Demo {demo.demo_key}: T={demo.arm_qpos.shape[0]}, "
          f"task_id={demo.task_id}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    h = resolve_model_handles(model)
    print(f"Resolved arm joints: {h.arm_joint_names}")
    print(f"Cameras: agentview={h.agentview_cam}, eye_in_hand={h.ee_cam}")

    if demo.init_state is not None:
        nq, nv = model.nq, model.nv
        data.qpos[:nq] = demo.init_state[:nq]
        data.qvel[:nv] = demo.init_state[nq:nq + nv]
        mujoco.mj_forward(model, data)

    dt = 1.0 / args.rate
    with mujoco.viewer.launch_passive(model, data) as viewer:
        T = demo.arm_qpos.shape[0]
        for t in range(T):
            for adr, q in zip(h.arm_qpos_adrs, demo.arm_qpos[t]):
                data.qpos[adr] = q
            if len(h.finger_qpos_adrs) == 2:
                for adr, q in zip(h.finger_qpos_adrs, demo.finger_qpos[t]):
                    data.qpos[adr] = q
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(dt)
            if not viewer.is_running():
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
