#!/usr/bin/env python3
"""Download a LIBERO dataset split into FailBench's local cache.

Wraps ``external/LIBERO/benchmark_scripts/download_libero_datasets.py`` so the
files land in ``datasets/libero/raw/<split>/`` instead of the default LIBERO
location. Must be run with the LIBERO sidecar venv active::

    source external/LIBERO/.venv/bin/activate
    python scripts/libero/download_libero.py --datasets libero_spatial
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIBERO_DOWNLOADER = os.path.join(
    REPO_ROOT, "external", "LIBERO", "benchmark_scripts", "download_libero_datasets.py")
DEFAULT_DOWNLOAD_DIR = os.path.join(REPO_ROOT, "datasets", "libero", "raw")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="libero_spatial",
                   choices=["all", "libero_goal", "libero_spatial",
                            "libero_object", "libero_100"])
    p.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR)
    args = p.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)

    cmd = [sys.executable, LIBERO_DOWNLOADER,
           "--datasets", args.datasets,
           "--download-dir", args.download_dir,
           "--use-huggingface"]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
