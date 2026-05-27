"""Upload the v2 dataset to a private HuggingFace dataset repo.

Backup + cluster-prep workflow. Repo stays private; raw HDF5 + manifest.csv
files uploaded as-is. The pull-back script (``pull_v2_from_hf.py``) rewrites
manifest h5_paths to the local download dir.

One-time setup before running:

    pip install -U "huggingface_hub[hf_xet]"      # (already installed in env)
    huggingface-cli login                          # paste token from hf.co/settings/tokens
    huggingface-cli repo create --type dataset --private \
        aaronngx/failbench-libero-v2

Then:

    PYTHONPATH=. python -m scripts.data.upload_v2_to_hf --dry-run    # sanity
    PYTHONPATH=. python -m scripts.data.upload_v2_to_hf              # real upload

Re-runs are safe — ``upload_large_folder`` resumes via xet/multipart and
skips files that already match remotely.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_REPO = "aaronngx/failbench-libero-v2"
DEFAULT_SRC = Path("/media/aaron/FAILBENCH/failbench/libero/v2")
DEFAULT_SPLITS = ("libero_spatial", "libero_object", "libero_goal")


def enumerate_uploads(src: Path, splits: Iterable[str]) -> list:
    """Return the list of files we expect to upload + their sizes (bytes)."""
    out = []
    for split in splits:
        split_dir = src / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing split dir: {split_dir}")
        h5s = sorted(split_dir.glob("*.h5"))
        if not h5s:
            raise FileNotFoundError(f"no .h5 files in {split_dir}")
        manifest = split_dir / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"missing manifest: {manifest}")
        for p in h5s + [manifest]:
            out.append(p)
    return out


def _format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def write_repo_readme(repo_id: str, dst: Path) -> None:
    """Write the README that ships at the HF repo root."""
    content = f"""\
# FailBench LIBERO v2 dataset

This is the v2 contact-prediction dataset built by the [FailBench project](https://github.com/ducmngx/FailBench).
Schema version: 2 (frozen 2026-05-15).

Each per-task HDF5 file contains 1,500 trials with the following per-trial groups:
- pre-failure window (T=8 frames of state + agentview/wrist RGB-D)
- goal lookahead (K=3 future commanded states)
- contact arrays during the 500-step settle after failure
- camera calibration, object poses, failure descriptor

See `docs/libero_v2_dataset.md` in the FailBench repo for the full schema.

## Layout

```
libero_spatial/
  manifest.csv
  pick_up_the_black_bowl_*.h5   (10 files, ~6 GB each)
libero_object/
  manifest.csv
  *.h5   (10 files)
libero_goal/
  manifest.csv
  *.h5   (10 files)
```

Total size: ~178 GB.

## Usage

To pull a copy back to local disk:

```bash
# Install
pip install -U huggingface_hub[hf_xet]

# Authenticate (one-time)
huggingface-cli login

# Download
python -m scripts.data.pull_v2_from_hf \\
    --dst /path/to/local/v2 \\
    --splits libero_spatial libero_object libero_goal
```

The pull script rewrites the manifest CSV's `h5_path` column to point under
your local download directory, so the FailBench loaders work directly.

## Notes

- This is a *private snapshot* for backup + cluster-pulling. License clearance
  for public release pending.
- Repo: [{repo_id}](https://huggingface.co/datasets/{repo_id})
- Derivative of LIBERO (MIT) + robosuite (MIT).
"""
    dst.write_text(content)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help=f"v2 root (default: {DEFAULT_SRC})")
    ap.add_argument("--repo_id", default=DEFAULT_REPO,
                    help=f"target HF dataset repo (default: {DEFAULT_REPO})")
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS),
                    help="splits to upload (default: all three)")
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate files + total size, do not upload")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel upload workers (default: 4)")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"error: src {args.src} does not exist (USB mounted?)", file=sys.stderr)
        sys.exit(1)

    files = enumerate_uploads(args.src, args.splits)
    sizes = [p.stat().st_size for p in files]
    total = sum(sizes)
    print(f"src:    {args.src}")
    print(f"repo:   {args.repo_id} (private)")
    print(f"splits: {', '.join(args.splits)}")
    print(f"files:  {len(files)}  total: {_format_size(total)}")
    for p, sz in zip(files, sizes):
        rel = p.relative_to(args.src)
        print(f"  {_format_size(sz):>10s}  {rel}")

    if args.dry_run:
        print("\n[dry-run] no upload performed.")
        return

    # Lazy import so --dry-run works without huggingface_hub installed.
    from huggingface_hub import HfApi, upload_large_folder

    api = HfApi()
    try:
        api.repo_info(args.repo_id, repo_type="dataset")
        print(f"\nrepo {args.repo_id} exists; proceeding to upload.")
    except Exception as e:
        print(f"\nrepo {args.repo_id} not accessible: {type(e).__name__}: {e}")
        print("create it with:")
        print(f"  huggingface-cli repo create --type dataset --private {args.repo_id}")
        sys.exit(1)

    # Write a README into the source dir's parent (tempfile) — we want it at
    # the repo root, not inside a split dir. Use a scratch dir.
    import tempfile
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        # Symlink-mirror only the splits we're uploading (to avoid sweeping
        # any other content in args.src) + drop README.md at the root.
        for split in args.splits:
            (scratch_path / split).symlink_to(args.src / split)
        write_repo_readme(args.repo_id, scratch_path / "README.md")

        print(f"\nstaging dir: {scratch_path}")
        print(f"uploading via upload_large_folder (workers={args.workers})...")
        upload_large_folder(
            folder_path=str(scratch_path),
            repo_id=args.repo_id,
            repo_type="dataset",
            num_workers=args.workers,
            print_report=True,
        )
    print("\nupload complete.")


if __name__ == "__main__":
    main()
