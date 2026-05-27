"""Precompute DINOv2 ViT-S/14 CLS features for v2 agentview windows.

Writes ``cache/dinov2_v2/<task>/<trial_id>.npy`` of shape ``(T, 384)`` for
each v2 trial. Sibling of :mod:`scripts.precompute_dinov2` adapted for the v2
HDF5 layout (one file per task, T-frame window) instead of v10's per-trial
``exp_*.npz``.

Each frame is letterboxed to 224×224 (DINOv2 ViT-S/14 input), ImageNet-
normalised, and the CLS token is saved. With T=8 across 15k trials per split,
the cache is ~45 MB/split (3 splits ≈ 135 MB total).

Usage:
    PYTHONPATH=. python -m scripts.benchmark.precompute_dinov2_v2 \\
        --v2_root /home/aaron/scratch/v2_ssd \\
        --splits libero_spatial
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import hdf5plugin  # noqa: F401
import h5py        # noqa: E402


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_HW = 224


def letterbox_224(rgb: np.ndarray) -> np.ndarray:
    """uint8 (H, W, 3) → float32 (3, 224, 224), letterboxed + ImageNet-normed."""
    h, w = rgb.shape[:2]
    scale = TARGET_HW / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((TARGET_HW, TARGET_HW, 3), dtype=np.uint8)
    y0 = (TARGET_HW - new_h) // 2
    x0 = (TARGET_HW - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    out = canvas.astype(np.float32) / 255.0
    out = (out - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(out, (2, 0, 1))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_root", type=Path,
                    default=Path(os.environ.get("FAILBENCH_V2_ROOT",
                                                "/home/aaron/scratch/v2_ssd")))
    ap.add_argument("--cache_dir", type=Path,
                    default=Path("cache/dinov2_v2"))
    ap.add_argument("--splits", nargs="+",
                    default=["libero_spatial"])
    ap.add_argument("--batch_size", type=int, default=64,
                    help="trials per batch (each trial contributes T=8 frames)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"v2_root={args.v2_root}  cache={args.cache_dir}  splits={args.splits}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("loading DINOv2 ViT-S/14 (frozen)...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    total_new = 0
    total_skip = 0
    t0 = time.perf_counter()
    for split in args.splits:
        split_root = args.v2_root / split
        h5_files = sorted(split_root.glob("*.h5"))
        if not h5_files:
            print(f"  skip {split}: no .h5 in {split_root}")
            continue
        for h5_path in h5_files:
            task = h5_path.stem
            cache_task = args.cache_dir / split / task
            cache_task.mkdir(parents=True, exist_ok=True)

            with h5py.File(h5_path, "r") as f:
                tids = sorted(f["trials"].keys())
                pending = (tids if args.force
                           else [t for t in tids
                                 if not (cache_task / f"{t}.npy").exists()])
                if not pending:
                    total_skip += len(tids)
                    continue

                # Read T per trial from file attr (default 8 if absent).
                T = int(f.attrs.get("window_T", 8))

                for i in range(0, len(pending), args.batch_size):
                    batch = pending[i:i + args.batch_size]
                    n = len(batch)
                    imgs = np.zeros((n * T, 3, TARGET_HW, TARGET_HW),
                                     dtype=np.float32)
                    for j, tid in enumerate(batch):
                        rgb = f[f"trials/{tid}/window_agentview_rgb"][()]
                        # rgb: (T, H, W, 3) u8
                        for t in range(T):
                            imgs[j * T + t] = letterbox_224(rgb[t])

                    with torch.no_grad():
                        x = torch.from_numpy(imgs).to(args.device, non_blocking=True)
                        feats = model(x).cpu().numpy().astype(np.float32)  # (n*T, 384)
                    feats = feats.reshape(n, T, 384)

                    for j, tid in enumerate(batch):
                        np.save(cache_task / f"{tid}.npy", feats[j])
                    total_new += n

                print(f"  {split}/{task}: {len(tids)} trials "
                      f"({len(pending)} new, {len(tids) - len(pending)} cached)")
                total_skip += len(tids) - len(pending)

    dt = time.perf_counter() - t0
    print(f"\ndone. {total_new} new, {total_skip} already cached, {dt:.1f}s "
          f"({total_new/(dt+1e-9):.1f} trials/s)")


if __name__ == "__main__":
    main()
