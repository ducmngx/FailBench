"""Precompute DINOv2 ViT-S/14 features for every config in datasets/v10.

Two modes:
  --mode cls   → save (384,) CLS token. Cache dir default: cache/dinov2.
  --mode patch → take 16×16 patch grid, average-pool to --patch_pool×--patch_pool
                 (default 4×4), save (P, P, 384). Cache dir default:
                 cache/dinov2_patch{P}x{P}.

Per-image preprocessing in both modes:
  Load pre_rgb → letterbox 224×224 → ImageNet-normalise → DINOv2 forward
  → extract CLS or pooled patches → save .npy.

Storage at v10 (16,532 configs):
  cls:    ~25 MB
  patch4: ~400 MB

Usage:
    python scripts/precompute_dinov2.py --mode cls
    python scripts/precompute_dinov2.py --mode patch --patch_pool 4
    python scripts/precompute_dinov2.py --mode patch --force --scenes scene_level2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SCENES = ["scene_level2", "scene_kitchen", "scene_workshop",
                  "scene_grocery", "scene_cluttered"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_HW = 224


def letterbox_224(rgb: np.ndarray) -> np.ndarray:
    """Resize keeping aspect ratio, pad with zeros to 224×224. uint8 → float32 [0,1]."""
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
    return np.transpose(out, (2, 0, 1))   # (3, 224, 224)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("datasets/v10"))
    ap.add_argument("--cache_dir", type=Path, default=None,
                    help="default: cache/dinov2 (cls) or cache/dinov2_patch{P}x{P} (patch)")
    ap.add_argument("--mode", type=str, default="cls", choices=["cls", "patch"])
    ap.add_argument("--patch_pool", type=int, default=4,
                    help="patch grid is 16×16 from DINOv2 ViT-S/14 at 224×224 input; "
                         "average-pool to (patch_pool, patch_pool) before saving")
    ap.add_argument("--scenes", type=str, default=",".join(DEFAULT_SCENES))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing cached features")
    return ap.parse_args()


def main():
    args = parse_args()
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    if args.cache_dir is None:
        args.cache_dir = (Path("cache/dinov2") if args.mode == "cls"
                          else Path(f"cache/dinov2_patch{args.patch_pool}x{args.patch_pool}"))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"mode={args.mode}, cache_dir={args.cache_dir}")

    print(f"loading DINOv2 ViT-S/14 (frozen)...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M params; device={args.device}")

    total_done = 0
    total_skipped = 0
    t0 = time.time()
    for scene in scenes:
        scene_dir = args.dataset / scene
        if not scene_dir.is_dir():
            print(f"  skip (not found): {scene}")
            continue
        for task_dir in sorted(scene_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            cache_task_dir = args.cache_dir / scene / task_dir.name
            cache_task_dir.mkdir(parents=True, exist_ok=True)

            npz_files = sorted(task_dir.glob("exp_*.npz"))
            if not npz_files:
                continue

            # Filter out already-cached entries unless --force
            if not args.force:
                pending = [f for f in npz_files
                           if not (cache_task_dir / (f.stem + ".npy")).exists()]
            else:
                pending = npz_files

            if not pending:
                total_skipped += len(npz_files)
                continue

            # Process in batches
            for i in range(0, len(pending), args.batch_size):
                batch_files = pending[i:i + args.batch_size]
                batch_imgs = np.zeros((len(batch_files), 3, TARGET_HW, TARGET_HW),
                                       dtype=np.float32)
                for j, npz_path in enumerate(batch_files):
                    d = np.load(npz_path)
                    try:
                        rgb = np.asarray(d["pre_rgb"])
                    finally:
                        d.close()
                    batch_imgs[j] = letterbox_224(rgb)

                with torch.no_grad():
                    x = torch.from_numpy(batch_imgs).to(args.device, non_blocking=True)
                    if args.mode == "cls":
                        feats = model(x)                          # (B, 384)
                    else:
                        out = model.forward_features(x)
                        patches = out["x_norm_patchtokens"]       # (B, 256, 384)
                        # Reshape to (B, 16, 16, 384) → (B, 384, 16, 16) for AvgPool
                        B = patches.shape[0]
                        patches = patches.view(B, 16, 16, 384).permute(0, 3, 1, 2)
                        if args.patch_pool != 16:
                            patches = torch.nn.functional.adaptive_avg_pool2d(
                                patches, (args.patch_pool, args.patch_pool))
                        # Store channels-last for friendlier loading: (P, P, 384)
                        feats = patches.permute(0, 2, 3, 1).contiguous()
                feats = feats.cpu().numpy().astype(np.float32)

                for j, npz_path in enumerate(batch_files):
                    np.save(cache_task_dir / (npz_path.stem + ".npy"), feats[j])
                    total_done += 1

            print(f"  {scene}/{task_dir.name}: {len(npz_files)} configs "
                  f"({len(pending)} new, {len(npz_files) - len(pending)} skipped)")
            total_skipped += len(npz_files) - len(pending)

    dt = time.time() - t0
    print(f"\ndone. {total_done} new, {total_skipped} already cached, "
          f"{dt:.1f}s elapsed.")


if __name__ == "__main__":
    main()
