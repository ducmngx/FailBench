"""Pull the v2 dataset from HuggingFace into a local directory.

Used on the cluster (or anywhere) to materialise a local copy of the v2
dataset uploaded by ``upload_v2_to_hf.py``. Rewrites each manifest's
``h5_path`` column to point at the local download directory so the
``planner.risk.v2_store.LiberoV2Dataset`` loader works directly.

One-time setup:

    pip install -U "huggingface_hub[hf_xet]"
    huggingface-cli login    # only needed if repo is private

Usage:

    # Pull all three splits (~177 GB)
    PYTHONPATH=. python -m scripts.data.pull_v2_from_hf \\
        --dst /scratch/$USER/failbench/v2

    # Pull just one split
    PYTHONPATH=. python -m scripts.data.pull_v2_from_hf \\
        --dst /scratch/$USER/failbench/v2 --splits libero_spatial
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


DEFAULT_REPO = "aaronngx/failbench-libero-v2"
DEFAULT_SPLITS = ("libero_spatial", "libero_object", "libero_goal")


def rewrite_manifest(manifest_path: Path, dst_split_dir: Path) -> int:
    """Rewrite manifest.csv's h5_path column to point under dst_split_dir.

    Returns the number of rows rewritten.
    """
    tmp = manifest_path.with_suffix(".csv.tmp")
    with open(manifest_path) as fin, open(tmp, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        n = 0
        for row in reader:
            old = row["h5_path"]
            row["h5_path"] = str(dst_split_dir / Path(old).name)
            writer.writerow(row)
            n += 1
    tmp.replace(manifest_path)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default=DEFAULT_REPO)
    ap.add_argument("--dst", type=Path, required=True,
                    help="local destination root (will mirror libero_*/...)")
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    ap.add_argument("--max_tasks", type=int, default=None,
                    help="(debug) only pull this many tasks per split")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel download workers (default: 8)")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download, list_repo_files

    # Build allow_patterns based on splits + optional max_tasks.
    allow_patterns: list = []
    if args.max_tasks is not None:
        # We need to enumerate to pick the first N tasks per split.
        all_files = list_repo_files(args.repo_id, repo_type="dataset")
        for split in args.splits:
            h5s = sorted(f for f in all_files
                         if f.startswith(f"{split}/") and f.endswith(".h5"))
            for h5 in h5s[:args.max_tasks]:
                allow_patterns.append(h5)
            allow_patterns.append(f"{split}/manifest.csv")
    else:
        for split in args.splits:
            allow_patterns.append(f"{split}/*.h5")
            allow_patterns.append(f"{split}/manifest.csv")
    allow_patterns.append("README.md")

    print(f"repo:    {args.repo_id}")
    print(f"dst:     {args.dst}")
    print(f"splits:  {args.splits}")
    print(f"patterns: {allow_patterns}")
    print(f"\ndownloading (workers={args.workers})...")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.dst),
        allow_patterns=allow_patterns,
        max_workers=args.workers,
    )

    # Rewrite manifests so the loader can resolve h5 paths locally.
    for split in args.splits:
        split_dir = args.dst / split
        manifest = split_dir / "manifest.csv"
        if manifest.exists():
            n = rewrite_manifest(manifest, split_dir)
            print(f"  {split}/manifest.csv: {n} rows rewritten -> {manifest}")
        else:
            print(f"  warn: {manifest} not found (split not pulled?)", file=sys.stderr)

    print("\ndone.")


if __name__ == "__main__":
    main()
