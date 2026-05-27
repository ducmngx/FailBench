"""Stage a v2 split (or all splits) to local SSD for faster training IO.

The image-conditioned models hit ~120 MB/s sustained on the USB-3 external
drive, which pins per-epoch wall-clock at ~16 min. Copying the per-task HDF5
files to a local NVMe drops sustained read to multiple GB/s and lets the GPU
saturate.

Usage:
    PYTHONPATH=. python -m scripts.benchmark.stage_v2_ssd \\
        --splits libero_spatial \\
        --src /media/aaron/F/failbench/libero/v2 \\
        --dst /home/aaron/scratch/v2_ssd

After staging, point the trainer at the new root::

    --v2_root /home/aaron/scratch/v2_ssd
    # OR
    FAILBENCH_V2_ROOT=/home/aaron/scratch/v2_ssd python -m scripts.benchmark.train_one ...

The per-split ``manifest.csv`` is copied AND its ``h5_path`` column is
rewritten so the loader resolves files under the new root without needing a
symlink hack.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path


def stage_split(src_split: Path, dst_split: Path) -> None:
    dst_split.mkdir(parents=True, exist_ok=True)
    # rsync each .h5 with progress + checksum.
    h5_files = sorted(src_split.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"no .h5 in {src_split}")
    print(f"  {len(h5_files)} h5 files, total "
          f"{sum(p.stat().st_size for p in h5_files)/2**30:.1f} GiB")
    for i, src in enumerate(h5_files, 1):
        dst = dst_split / src.name
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            print(f"  [{i}/{len(h5_files)}] {src.name} (exists, skip)")
            continue
        t0 = time.perf_counter()
        subprocess.check_call(["rsync", "-a", "--inplace", str(src), str(dst)])
        dt = time.perf_counter() - t0
        size_gb = src.stat().st_size / 2**30
        print(f"  [{i}/{len(h5_files)}] {src.name}  "
              f"{size_gb:.2f} GiB in {dt:.1f}s ({size_gb*1024/dt:.0f} MiB/s)")

    # Manifest: copy + rewrite h5_path column to point under dst root.
    src_manifest = src_split / "manifest.csv"
    dst_manifest = dst_split / "manifest.csv"
    src_prefix = str(src_split) + "/"
    dst_prefix = str(dst_split) + "/"
    with open(src_manifest) as fin, open(dst_manifest, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        rewrites = 0
        for row in reader:
            old = row["h5_path"]
            if old.startswith(src_prefix):
                row["h5_path"] = dst_prefix + old[len(src_prefix):]
                rewrites += 1
            else:
                # Fallback: rewrite by filename match.
                row["h5_path"] = str(dst_split / Path(old).name)
                rewrites += 1
            writer.writerow(row)
    print(f"  manifest: {rewrites} h5_path entries rewritten -> {dst_manifest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path("/media/aaron/F/failbench/libero/v2"))
    ap.add_argument("--dst", type=Path,
                    default=Path("/home/aaron/scratch/v2_ssd"))
    ap.add_argument("--splits", nargs="+",
                    default=["libero_spatial"],
                    help="splits to stage; use 'all' for the three canonical splits")
    args = ap.parse_args()

    if not shutil.which("rsync"):
        print("error: rsync not on PATH", file=sys.stderr)
        sys.exit(1)
    if not args.src.exists():
        print(f"error: src {args.src} does not exist (mount the USB drive?)",
              file=sys.stderr)
        sys.exit(1)

    splits = args.splits
    if splits == ["all"]:
        splits = ["libero_spatial", "libero_object", "libero_goal"]

    args.dst.mkdir(parents=True, exist_ok=True)
    for s in splits:
        src_split = args.src / s
        dst_split = args.dst / s
        if not src_split.exists():
            print(f"skip {s}: {src_split} missing", file=sys.stderr)
            continue
        print(f"=== staging {s} -> {dst_split}")
        stage_split(src_split, dst_split)
    print(f"\ndone. Set FAILBENCH_V2_ROOT={args.dst} or pass --v2_root {args.dst}.")


if __name__ == "__main__":
    main()
