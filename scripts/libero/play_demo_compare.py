#!/usr/bin/env python3
"""Side-by-side video: LIBERO's recorded agentview vs our reproduction.

For each step ``t`` in a demo:
  - left panel  = ``data/<demo>/obs/agentview_rgb[t]``  (LIBERO truth)
  - right panel = our render of ``agentview`` after seeding the sim state
                  from ``data/<demo>/states[t]`` via the post-fix
                  :meth:`LiberoRunner._set_full_state`.

If the two panels track each other through the demo (object pickup, lift,
place all at the same step) then the runner's state-seeding + renderer are
faithful to LIBERO's source data.

Usage::

    conda run -n failbench_env python -m scripts.libero.play_demo_compare \\
        --hdf5 datasets/libero/raw/libero_spatial/<task>.hdf5 \\
        --demo demo_0
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import mujoco
import numpy as np

from planner.experiments.data_capture import OffscreenRenderer
from planner.experiments.libero.adapter import list_demos, load_demo, materialise_mjcf
from planner.experiments.libero.naming import resolve_model_handles
from planner.experiments.libero.runner import LiberoRunner, LiberoTrialConfig

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "out", "libero_videos", "play")


def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
    """Stamp a text label in the top-left of ``frame`` (BGR, in-place modifies a copy)."""
    out = frame.copy()
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _split_path_to_label(hdf5_path: str) -> str:
    parts = os.path.normpath(hdf5_path).split(os.sep)
    # .../datasets/libero/raw/<split>/<task>_demo.hdf5
    if "raw" in parts:
        i = parts.index("raw")
        if i + 1 < len(parts):
            split = parts[i + 1]
            task = os.path.splitext(parts[-1])[0]
            return f"{split}__{task}"
    return os.path.splitext(os.path.basename(hdf5_path))[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", required=True)
    p.add_argument("--demo", default="demo_0")
    p.add_argument("--list", action="store_true",
                   help="List demo keys in the HDF5 and exit")
    p.add_argument("--output", default=None,
                   help="Output mp4 path. Default: out/libero_videos/play/<split>__<task>__<demo>.mp4")
    p.add_argument("--rate", type=float, default=20.0,
                   help="Output mp4 frame rate (LIBERO demos run at ~20 Hz)")
    p.add_argument("--height", type=int, default=384,
                   help="Output panel height (LIBERO truth is upsampled, our render is rendered at this size)")
    args = p.parse_args()

    if args.list:
        for k in list_demos(args.hdf5):
            print(k)
        return 0

    demo = load_demo(args.hdf5, args.demo)
    if demo.full_states is None:
        print(f"ERROR: demo {args.demo} has no full_states; can't seed sim.",
              file=sys.stderr)
        return 1

    # Read LIBERO's recorded agentview frames directly from HDF5 — these are the
    # truth panel. Already in RGB uint8 from robosuite's renderer. Note: LIBERO
    # records frames upside-down relative to OpenGL; flip to match our renderer.
    import h5py
    with h5py.File(args.hdf5, "r") as f:
        truth_rgb_all = f[f"data/{args.demo}/obs/agentview_rgb"][...]  # (T, 128, 128, 3)
    # Flip vertically to match our agentview orientation.
    truth_rgb_all = truth_rgb_all[:, ::-1, :, :]
    T_truth = truth_rgb_all.shape[0]

    T = demo.arm_qpos.shape[0]
    T_play = min(T, T_truth)
    if T_truth != T:
        print(f"NOTE: T_truth({T_truth}) != T_states({T}); playing first {T_play} steps",
              file=sys.stderr)

    # Build our reproduction renderer at the requested panel size.
    H = args.height
    W = H  # square — matches LIBERO's 1:1 aspect
    xml_path = materialise_mjcf(demo.model_xml)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    handles = resolve_model_handles(model)
    if handles.agentview_cam is None:
        print("ERROR: model has no agentview camera.", file=sys.stderr)
        return 1
    renderer = OffscreenRenderer(model, height=H, width=W,
                                  camera_name=handles.agentview_cam)

    # Use a runner only for its (post-fix) _set_full_state. We don't call run().
    cfg = LiberoTrialConfig(fail_progress=0.5, seed=0,
                             image_height=H, image_width=W)
    runner = LiberoRunner(demo, cfg)

    # Track bowl_z if the scene has a bowl, for sanity output.
    bowl_qadr = -1
    for name in ("akita_black_bowl_1_main", "akita_black_bowl_1",
                  "akita_black_bowl_2_main"):
        bid = mujoco.mj_name2id(runner.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0 and runner.model.body_jntadr[bid] >= 0:
            bowl_qadr = int(runner.model.jnt_qposadr[int(runner.model.body_jntadr[bid])])
            break

    out_path = args.output or os.path.join(
        DEFAULT_OUT_DIR, f"{_split_path_to_label(args.hdf5)}__{args.demo}.mp4")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    panel_w = W
    pad = 8
    canvas_w = 2 * panel_w + 3 * pad
    canvas_h = H + 2 * pad + 24  # room for caption strip
    writer = cv2.VideoWriter(out_path, fourcc, args.rate, (canvas_w, canvas_h))
    if not writer.isOpened():
        print(f"ERROR: cv2.VideoWriter failed to open {out_path}", file=sys.stderr)
        return 1

    print(f"Rendering {T_play} frames to {out_path} ...")
    bowl_zs = []
    for t in range(T_play):
        runner._set_full_state(demo.full_states[t])
        ours_rgb = renderer.render(runner.data)  # (H, W, 3) uint8 RGB

        truth_rgb = cv2.resize(truth_rgb_all[t], (W, H), interpolation=cv2.INTER_CUBIC)

        # Convert to BGR for cv2 writer.
        truth_bgr = cv2.cvtColor(truth_rgb, cv2.COLOR_RGB2BGR)
        ours_bgr = cv2.cvtColor(ours_rgb, cv2.COLOR_RGB2BGR)

        truth_bgr = _annotate(truth_bgr, f"LIBERO truth   t={t}/{T_play - 1}")
        bowl_z = (float(runner.data.qpos[bowl_qadr + 2])
                  if bowl_qadr >= 0 else float("nan"))
        bowl_zs.append(bowl_z)
        ours_bgr = _annotate(ours_bgr,
                              f"our render     bowl_z={bowl_z:.3f}"
                              if bowl_qadr >= 0 else "our render")

        canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
        canvas[pad:pad + H, pad:pad + W] = truth_bgr
        canvas[pad:pad + H, pad + W + pad:pad + 2 * W + pad] = ours_bgr
        cv2.putText(canvas, f"{_split_path_to_label(args.hdf5)} / {args.demo}",
                    (pad, canvas_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    renderer.close()
    runner.close()

    if bowl_zs:
        zs = np.array(bowl_zs)
        print(f"bowl_z range over demo: min={zs.min():.3f}  max={zs.max():.3f}  "
              f"final={zs[-1]:.3f}")
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB, "
          f"{T_play} frames @ {args.rate} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
