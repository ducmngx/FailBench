#!/usr/bin/env python
"""Scan a dataset directory and remove any corrupted/truncated npz files.

Workers that crash mid-write leave partial npz files (often a few hundred KB
to ~2.5 MB, vs healthy ones at ~2.5 MB). They fail to open as zipfiles. This
script identifies them by attempting to load each npz and removes the failures.

Also prunes the matching manifest.csv row(s) so the manifest stays consistent
with on-disk files.

Usage
-----
    # dry run (default — only report)
    python scripts/cleanup_corrupted.py --root datasets/v10

    # actually delete and rewrite manifests
    python scripts/cleanup_corrupted.py --root datasets/v10 --apply
"""
import argparse
import csv
import glob
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np


def find_corrupted(root: str):
    """Return list of (path, error_name) for npz files that can't be opened."""
    bad = []
    files = sorted(glob.glob(os.path.join(root, "*", "*", "*.npz")))
    for i, f in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"  scanned {i+1}/{len(files)}", file=sys.stderr)
        try:
            with np.load(f, allow_pickle=True) as npz:
                # Force a real read of one key — np.load is lazy
                _ = npz["pre_qpos"]
        except (zipfile.BadZipFile, OSError, KeyError, EOFError, ValueError) as e:
            bad.append((f, type(e).__name__))
    return bad, len(files)


def prune_manifest(manifest_path: str, removed_basenames: set):
    """Rewrite a manifest.csv excluding rows whose npz_file is in `removed_basenames`."""
    if not os.path.exists(manifest_path):
        return 0
    rows = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            if r.get("npz_file") in removed_basenames:
                continue
            rows.append(r)
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="dataset root, e.g. datasets/v10")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete corrupted files and rewrite manifests "
                         "(default is dry-run)")
    args = ap.parse_args()

    print(f"Scanning {args.root} ...", file=sys.stderr)
    bad, total = find_corrupted(args.root)
    print(f"\nFound {len(bad)} corrupted / {total} total npz "
          f"({100 * len(bad) / max(1, total):.2f}%)\n")

    if not bad:
        return

    # Group by manifest dir for efficient pruning
    by_taskdir: defaultdict = defaultdict(list)
    for path, err in bad:
        by_taskdir[os.path.dirname(path)].append((path, err))

    for taskdir, items in sorted(by_taskdir.items()):
        rel = os.path.relpath(taskdir, args.root)
        print(f"[{rel}]  {len(items)} corrupted")
        for path, err in items[:5]:
            sz = os.path.getsize(path)
            print(f"    {err:14s} size={sz:>9d}B  {os.path.basename(path)}")
        if len(items) > 5:
            print(f"    ...and {len(items) - 5} more")

    if not args.apply:
        print("\n(dry run — pass --apply to delete and rewrite manifests)")
        return

    # Delete + prune manifests
    print("\nApplying changes...")
    removed_total = 0
    for taskdir, items in sorted(by_taskdir.items()):
        basenames = set()
        for path, _ in items:
            try:
                os.remove(path)
                basenames.add(os.path.basename(path))
                removed_total += 1
            except OSError as e:
                print(f"  WARN failed to remove {path}: {e}", file=sys.stderr)
        manifest_path = os.path.join(taskdir, "manifest.csv")
        kept = prune_manifest(manifest_path, basenames)
        rel = os.path.relpath(taskdir, args.root)
        print(f"  [{rel}] removed {len(basenames)}, manifest now {kept} rows")
    print(f"\nDone — removed {removed_total} corrupted files.")


if __name__ == "__main__":
    main()
