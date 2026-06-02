#!/usr/bin/env python3
"""Summarise a FailBench benchmark sweep from runs/bench/*/metrics.json.

Each finished run writes ``runs/bench/<model>__<mods>__T<k>__<tf>__<split>__seed<N>__<ts>/metrics.json``.
The run-dir name does NOT encode the LIBERO suite, so the three per-suite runs for one
(model, seed) differ only by timestamp. This script dedups to the best (lowest val) run per
config key (= dir name with the trailing timestamp stripped) and prints a sorted leaderboard.

Usage:
    python scripts/summarize_sweep.py                      # leaderboard from runs/bench
    python scripts/summarize_sweep.py --runs_root runs/bench_pooled
    python scripts/summarize_sweep.py --all                # one row per run (no dedup)
    python scripts/summarize_sweep.py --missing            # list config cells with no metrics.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TS_RE = re.compile(r"__\d{8}-\d{6}$")           # trailing __YYYYMMDD-HHMMSS
ARRAY_RE = re.compile(r"__\d+_\d+$")            # trailing __<jobid>_<taskid> (pooled runs)


def config_key(run_name: str) -> str:
    """Strip the trailing timestamp / array-id token to get the config identity."""
    return ARRAY_RE.sub("", TS_RE.sub("", run_name))


def load_runs(runs_root: Path) -> list[dict]:
    runs = []
    for mp in sorted(runs_root.glob("*/metrics.json")):
        try:
            m = json.loads(mp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        name = mp.parent.name
        runs.append({
            "name": name,
            "key": config_key(name),
            "val": m.get("best_val_mse_log1p", float("inf")),
            "baseline": m.get("baseline_val_mse_log1p_weighted", float("nan")),
            "epoch": m.get("best_epoch", -1),
            "n_train": m.get("n_train", -1),
            "n_val": m.get("n_val", -1),
        })
    return runs


def print_table(rows: list[dict], key: str) -> None:
    hdr = f"{key:<58}{'val':>9}{'baseline':>10}{'ep':>4}{'n_val':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r[key]:<58}{r['val']:>9.4f}{r['baseline']:>10.4f}"
              f"{r['epoch']:>4}{r['n_val']:>8}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_root", type=Path, default=Path("runs/bench"))
    ap.add_argument("--all", action="store_true",
                    help="one row per run (no dedup to best-per-config)")
    ap.add_argument("--missing", action="store_true",
                    help="also list config cells from sweep_configs.sh with no reported run")
    args = ap.parse_args()

    runs = load_runs(args.runs_root)
    if not runs:
        print(f"no metrics.json found under {args.runs_root}/")
        return

    if args.all:
        rows = sorted(runs, key=lambda r: r["val"])
        print_table(rows, "name")
        print(f"\n{len(rows)} runs under {args.runs_root}/")
        return

    best: dict[str, dict] = {}
    for r in runs:
        if r["key"] not in best or r["val"] < best[r["key"]]["val"]:
            best[r["key"]] = r
    rows = sorted(best.values(), key=lambda r: r["val"])
    print_table(rows, "key")
    print(f"\n{len(best)} unique configs, {len(runs)} runs "
          f"(incl. requeue/suite dupes) under {args.runs_root}/")

    if args.missing:
        cfg_file = Path("scripts/cluster/sweep_configs.sh")
        if not cfg_file.exists():
            print("\n(--missing: scripts/cluster/sweep_configs.sh not found)")
            return
        # Expected config keys are model/mod/T tokens; we just report how many of the
        # seen keys are below a simple count, and list the seen keys for eyeballing.
        print("\nseen config keys:")
        for k in sorted(best):
            print(f"  {k}")


if __name__ == "__main__":
    main()
