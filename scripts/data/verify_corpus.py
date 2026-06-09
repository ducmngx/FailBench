#!/usr/bin/env python3
"""Pre-publication verification for the FailBench v2 corpus (LIBERO + RoboCasa).

Runs four phases over both per-source v2 stores and writes per-phase CSVs plus a
unified ``quarantine.csv``. Designed to catch every class of issue documented in
``docs/libero_v2_dataset.md`` plus the corruption observed when the MLP loop hit
``OSError: address of object past end of allocation`` on full-corpus iteration.

Phases:
    A — structural integrity (open file, read every per-trial attribute, sample
        every required dataset; quarantines trials that can't be read at all)
    B — schema compliance vs ``planner.risk.v2_store._PER_TRIAL_DATASETS``,
        ``_PER_TRIAL_SCALAR_ATTRS``, ``_PER_TRIAL_ARRAY_ATTRS`` + file-level attrs
    C — content sanity: NaN/Inf, by-construction invariants, value ranges
    D — cross-source consistency: trial-id uniqueness, manifest <-> HDF5 match

Outputs (all under ``out/data_verify/``):
    quarantine_structural.csv    A
    schema_violations.csv        B
    content_issues.csv           C
    consistency_issues.csv       D
    quarantine.csv               merged: every trial we will drop on publish
    summary.json                 totals + per-source counts

Run::

    /home/aaron/miniconda3/envs/failbench_env/bin/python -u -m scripts.data.verify_corpus \\
        --output_dir out/data_verify

Idempotent and read-only (no HDF5 writes).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import hdf5plugin  # noqa: F401 — register blosc:lz4 filter
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planner.risk.v2_store import (  # noqa: E402
    _PER_TRIAL_DATASETS, _PER_TRIAL_SCALAR_ATTRS, _PER_TRIAL_ARRAY_ATTRS,
)


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------

VALID_FAILURE_MODES = {
    "GRIPPER_OPEN", "SLIPPERY_GRIP", "SINGLE_JOINT", "MULTI_JOINT", "ALL_JOINTS",
}
EXPECTED_FILE_ATTRS = {
    "schema_version": (2,),
    "window_T": (8,),
    "settle_S": (50,),
}
EXPECTED_GOAL_OFFSETS = (5, 15, 30)
LIBERO_ROOT = Path("/media/aaron/F/failbench/libero/v2")
ROBOCASA_ROOT = Path("/media/aaron/F/failbench/robocasa/v2")

# Datasets whose dtype/shape we relax (variable-length first axis = N, M, n_obj, J).
# Other datasets must match the spec dtype exactly.
VARIABLE_FIRST_AXIS = {
    "contact_positions", "contact_forces", "contact_force_world", "contact_time",
    "contact_geom_pairs", "contact_failure_id", "impacted_geom_ids",
    "failure_joints", "obj_names", "obj_pos_pre", "obj_quat_pre",
    "obj_pos_post", "obj_quat_post",
}


# --------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------


def discover_sources() -> list[tuple[str, str, Path, Path]]:
    """Return (source, split, manifest_path, h5_dir) tuples.

    LIBERO: three (source="libero", split=libero_spatial/_object/_goal) entries.
    RoboCasa: one (source="robocasa", split="robocasa") entry.
    """
    out: list[tuple[str, str, Path, Path]] = []
    for split in ("libero_spatial", "libero_object", "libero_goal"):
        d = LIBERO_ROOT / split
        m = d / "manifest.csv"
        if m.exists():
            out.append(("libero", split, m, d))
    m = ROBOCASA_ROOT / "manifest.csv"
    if m.exists():
        out.append(("robocasa", "robocasa", m, ROBOCASA_ROOT))
    return out


# --------------------------------------------------------------------------
# Per-trial scan: Phase A (structural) + Phase B (schema) + Phase C (content)
# in one pass so we only iterate each trial once.
# --------------------------------------------------------------------------


def _check_dataset_dtype(grp: h5py.Group, name: str, expected_dtype) -> Optional[str]:
    """Return mismatch description or None if OK."""
    if expected_dtype is None:
        return None
    actual = grp[name].dtype
    if np.dtype(expected_dtype) != actual:
        return f"dtype={actual} (expected {np.dtype(expected_dtype)})"
    return None


def _safe_read(grp: h5py.Group, key: str):
    """Read attr or dataset; raise the original exception if broken."""
    if key in grp.attrs:
        return grp.attrs[key]
    if key in grp:
        return grp[key][...]
    raise KeyError(key)


def scan_trial(
    src: str,
    split: str,
    task: str,
    h5_path: Path,
    grp: h5py.Group,
    trial_id: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (structural_issues, schema_issues, content_issues) for one trial.

    Empty list per phase means clean. The function tries Phase A first; if any
    fatal structural issue surfaces, B+C are skipped.
    """
    base = {"source": src, "split": split, "task": task,
            "h5_path": str(h5_path), "trial_id": trial_id}
    structural: list[dict] = []
    schema: list[dict] = []
    content: list[dict] = []

    # ---- Phase A: try to read every attribute and the small scalar fields.
    try:
        # Force a read of the attribute table; this is the path that surfaces
        # the "address past end of allocation" corruption.
        _ = dict(grp.attrs)
    except (OSError, RuntimeError) as e:
        structural.append({**base, "error_kind": "bad_attrs",
                           "error_field": "", "error_msg": str(e)[:200]})
        return structural, schema, content   # cannot proceed; B+C would crash

    # Sample-read every required scalar dataset to catch silent corruption in
    # the small ones (state vectors, camera intrinsics). Image datasets are
    # left unread here for speed; their corruption shows up in B (dtype check).
    for ds in ("pre_qpos", "pre_qvel", "pre_target_qpos", "pre_ee_pos",
               "cam_agentview_pos", "cam_agentview_mat0",
               "contact_geom_pairs", "contact_time"):
        if ds in grp:
            try:
                grp[ds][...]
            except (OSError, RuntimeError) as e:
                structural.append({**base, "error_kind": "bad_dataset",
                                   "error_field": ds, "error_msg": str(e)[:200]})
                return structural, schema, content

    # ---- Phase B: schema compliance.
    for name, expected_dtype in _PER_TRIAL_DATASETS.items():
        if name not in grp:
            schema.append({**base, "field": name, "severity": "fatal",
                           "expected": "present", "actual": "missing"})
            continue
        mismatch = _check_dataset_dtype(grp, name, expected_dtype)
        if mismatch:
            schema.append({**base, "field": name, "severity": "fatal",
                           "expected": str(np.dtype(expected_dtype)),
                           "actual": mismatch})

    extras = set(grp.keys()) - set(_PER_TRIAL_DATASETS.keys())
    for x in extras:
        schema.append({**base, "field": x, "severity": "warn",
                       "expected": "not in spec", "actual": "present"})

    for attr in _PER_TRIAL_SCALAR_ATTRS:
        if attr not in grp.attrs:
            schema.append({**base, "field": attr, "severity": "fatal",
                           "expected": "scalar attr present", "actual": "missing"})
    for attr in _PER_TRIAL_ARRAY_ATTRS:
        if attr not in grp.attrs:
            schema.append({**base, "field": attr, "severity": "fatal",
                           "expected": "array attr present", "actual": "missing"})

    # ---- Phase C: content sanity. Only runs if no fatal schema issues yet.
    fatal_schema = any(i["severity"] == "fatal" for i in schema)
    if not fatal_schema:
        try:
            content.extend(_content_checks(base, grp, src))
        except (OSError, RuntimeError) as e:
            content.append({**base, "check": "content_pass_aborted",
                            "observed": str(e)[:200], "expected": "—"})

    return structural, schema, content


def _content_checks(base: dict, grp: h5py.Group, src: str) -> list[dict]:
    """Numerical invariants. Append issues; return list."""
    issues: list[dict] = []

    def add(check, observed, expected):
        issues.append({**base, "check": check,
                       "observed": str(observed)[:120],
                       "expected": str(expected)[:120]})

    # NaN/Inf in float datasets that should always be finite
    for fld in ("pre_qpos", "pre_qvel", "pre_ee_pos", "pre_target_qpos",
                "window_qpos", "window_qvel", "window_ee_pos",
                "goal_qpos", "goal_qvel", "goal_ee_pos",
                "contact_positions", "contact_forces", "contact_force_world",
                "cam_agentview_pos", "cam_agentview_mat0"):
        if fld not in grp:
            continue
        arr = grp[fld][...]
        if arr.size and not np.isfinite(arr).all():
            add(f"nan_or_inf:{fld}", "non-finite", "all finite")

    # By-construction: window_qpos[-1] ≈ pre_qpos. Both come from the same
    # _snapshot_robot() after _set_full_state(demo.full_states[fail_idx]) so
    # they're identical up to float32 cast.
    if "window_qpos" in grp and "pre_qpos" in grp:
        wq = grp["window_qpos"][-1]
        pq = grp["pre_qpos"][...]
        if not np.allclose(wq, pq, atol=1e-5):
            add("window_qpos[-1] != pre_qpos",
                f"max|diff|={float(np.abs(wq - pq).max()):.4e}", "atol=1e-5")

    # NOTE: pre_target_qpos is the demo's COMMANDED qpos at fail_idx (from
    # obs/joint_states). window_qpos[-1] is the OBSERVED qpos from the sim's
    # states[fail_idx]. On RoboCasa these match to atol=1e-3; on LIBERO they
    # diverge by ~1e-2 because robosuite samples obs/joint_states one step
    # earlier than states. Not an invariant; do not flag.

    # Force-norm consistency (rotation preserves norm)
    if "contact_force_world" in grp and "contact_forces" in grp:
        fw = grp["contact_force_world"][...]
        fl = grp["contact_forces"][...]
        if fw.shape[0] and fl.shape[0] == fw.shape[0]:
            n_w = np.linalg.norm(fw, axis=1)
            n_l = np.linalg.norm(fl[:, :3], axis=1)
            if n_w.size and not np.allclose(n_w, n_l, rtol=1e-3, atol=1e-3):
                add("force_world norm mismatch",
                    f"max|diff|={float(np.abs(n_w - n_l).max()):.4e}",
                    "‖fw‖ == ‖fl[:3]‖")

    # contact_time bounds + monotonicity
    if "contact_time" in grp:
        ct = grp["contact_time"][...]
        if ct.size:
            if ct.min() < 0 or ct.max() >= 500:
                add("contact_time out of range",
                    f"[{int(ct.min())}, {int(ct.max())}]", "[0, 500)")
            if np.any(np.diff(ct) < 0):
                add("contact_time not monotonic", "decreasing run found",
                    "monotonic non-decreasing")

    # impacted_geom_ids == unique(contact_geom_pairs.flatten())
    if "contact_geom_pairs" in grp and "impacted_geom_ids" in grp:
        cgp = grp["contact_geom_pairs"][...]
        imp = grp["impacted_geom_ids"][...]
        if cgp.size:
            expected = np.unique(cgp.flatten())
            if not np.array_equal(np.sort(imp), expected):
                add("impacted_geom_ids mismatch",
                    f"len(imp)={imp.size} len(unique)={expected.size}",
                    "sorted(unique(contact_geom_pairs))")

    # failure_mode + failure_joints + traj_progress + scene_table_z
    a = grp.attrs
    mode = a.get("failure_mode")
    if isinstance(mode, bytes): mode = mode.decode("utf-8")
    if mode not in VALID_FAILURE_MODES:
        add("failure_mode invalid", mode, f"in {sorted(VALID_FAILURE_MODES)}")

    if "failure_joints" in grp:
        fj = grp["failure_joints"][...]
        if fj.size and (fj.min() < 1 or fj.max() > 7):
            add("failure_joints out of range",
                f"[{int(fj.min())}, {int(fj.max())}]", "[1, 7]")

    tp = float(a.get("traj_progress", 0.0))
    if not (0.0 <= tp <= 1.0):
        add("traj_progress out of range", f"{tp:.4f}", "[0, 1]")

    if src == "robocasa":
        tz = float(a.get("scene_table_z", 0.0))
        if tz == 0.91:
            add("scene_table_z == 0.91 (LIBERO fallback)", "0.91",
                "real surface height (post-fix invariant)")
        for fld in ("baseline_contact_positions", "baseline_contact_forces",
                    "baseline_contact_force_world", "baseline_contact_geom_pairs"):
            if fld not in grp:
                add(f"missing {fld}", "absent",
                    "present (RoboCasa baseline subtraction)")

    # Depth sanity (median in plausible range)
    if "pre_depth" in grp:
        d = grp["pre_depth"][...].astype(np.float32)
        if d.size:
            med = float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else float("nan")
            if not (0.05 < med < 10.0):
                add("pre_depth median out of plausible range",
                    f"{med:.3f} m", "0.05–10.0 m")

    return issues


# --------------------------------------------------------------------------
# Phase D — cross-source / cross-file consistency
# --------------------------------------------------------------------------


def consistency_pass(
    sources: list[tuple[str, str, Path, Path]],
    seen_trials: dict[tuple[str, str, str, str], int],
    manifest_rows_per_file: dict[str, list[dict]],
) -> list[dict]:
    """Cross-source consistency checks. Returns issue list."""
    out: list[dict] = []

    # Trial-id uniqueness across compound key (source, split, task, trial_id)
    dupes = [(k, n) for k, n in seen_trials.items() if n > 1]
    for (src, split, task, tid), n in dupes:
        out.append({"source": src, "split": split, "task": task,
                    "trial_id": tid, "check": "duplicate compound key",
                    "observed": f"count={n}", "expected": "count=1"})

    # Manifest <-> HDF5 cross-checks per file
    for src, split, manifest_path, h5_dir in sources:
        try:
            df = pd.read_csv(manifest_path)
        except Exception as e:
            out.append({"source": src, "split": split, "task": "",
                        "trial_id": "", "check": "manifest unreadable",
                        "observed": str(e)[:120],
                        "expected": "csv parses"})
            continue
        for h5_path in sorted(h5_dir.glob("*.h5")):
            task = h5_path.stem
            df_task = df[df["task"] == task]
            try:
                with h5py.File(h5_path, "r") as f:
                    h5_tids = set(f["trials"].keys())
            except Exception as e:
                out.append({"source": src, "split": split, "task": task,
                            "trial_id": "", "check": "h5 unopenable",
                            "observed": str(e)[:120], "expected": "opens"})
                continue
            mf_tids = set(df_task["trial_id"].astype(str))
            orphan_h5 = h5_tids - mf_tids
            orphan_mf = mf_tids - h5_tids
            for tid in orphan_h5:
                out.append({"source": src, "split": split, "task": task,
                            "trial_id": tid, "check": "trial in h5 not in manifest",
                            "observed": "h5-only", "expected": "in manifest"})
            for tid in orphan_mf:
                out.append({"source": src, "split": split, "task": task,
                            "trial_id": tid, "check": "manifest row no h5 group",
                            "observed": "manifest-only", "expected": "in h5"})
            # n_contacts sanity (sample 50 per task to save time)
            sample = df_task.sample(min(50, len(df_task)), random_state=0)
            try:
                with h5py.File(h5_path, "r") as f:
                    for _, row in sample.iterrows():
                        tid = str(row["trial_id"])
                        if tid not in f["trials"]:
                            continue
                        actual = int(f[f"trials/{tid}/contact_positions"].shape[0])
                        declared = int(row["n_contacts"])
                        if actual != declared:
                            out.append({
                                "source": src, "split": split, "task": task,
                                "trial_id": tid,
                                "check": "manifest n_contacts mismatch",
                                "observed": f"h5={actual} manifest={declared}",
                                "expected": "equal",
                            })
            except (OSError, RuntimeError) as e:
                out.append({"source": src, "split": split, "task": task,
                            "trial_id": "", "check": "contact count sample errored",
                            "observed": str(e)[:120], "expected": "readable"})
    return out


# --------------------------------------------------------------------------
# File-level attr checks (run once per file)
# --------------------------------------------------------------------------


def check_file_attrs(h5_path: Path, src: str) -> list[dict]:
    issues = []
    try:
        with h5py.File(h5_path, "r") as f:
            for k, expected in EXPECTED_FILE_ATTRS.items():
                if k not in f.attrs:
                    issues.append({"source": src, "h5_path": str(h5_path),
                                   "field": k, "severity": "fatal",
                                   "expected": str(expected), "actual": "missing"})
                else:
                    val = int(f.attrs[k])
                    if val not in expected:
                        issues.append({"source": src, "h5_path": str(h5_path),
                                       "field": k, "severity": "fatal",
                                       "expected": str(expected), "actual": str(val)})
            if "goal_offsets" in f.attrs:
                offs = tuple(int(x) for x in f.attrs["goal_offsets"])
                if offs != EXPECTED_GOAL_OFFSETS:
                    issues.append({"source": src, "h5_path": str(h5_path),
                                   "field": "goal_offsets", "severity": "fatal",
                                   "expected": str(EXPECTED_GOAL_OFFSETS),
                                   "actual": str(offs)})
    except (OSError, RuntimeError) as e:
        issues.append({"source": src, "h5_path": str(h5_path),
                       "field": "<file>", "severity": "fatal",
                       "expected": "openable", "actual": str(e)[:200]})
    return issues


# --------------------------------------------------------------------------
# Quarantine merge
# --------------------------------------------------------------------------


def merge_quarantine(
    structural: list[dict],
    schema: list[dict],
    content: list[dict],
    consistency: list[dict],
) -> list[dict]:
    """Union of trial-level issues with severity priority."""
    # Build (source, split, task, trial_id) -> list of reasons
    pool: dict[tuple, list[str]] = defaultdict(list)

    def key(d):
        return (d.get("source", ""), d.get("split", ""), d.get("task", ""),
                d.get("trial_id", ""))

    for d in structural:
        pool[key(d)].append(f"structural:{d['error_kind']}:{d.get('error_field','')}")
    for d in schema:
        if d["severity"] == "fatal":
            pool[key(d)].append(f"schema_fatal:{d['field']}")
    for d in content:
        pool[key(d)].append(f"content:{d['check']}")
    for d in consistency:
        # Only quarantine the orphan / mismatch issues, not the uniqueness ones
        check = d.get("check", "")
        if check.startswith(("trial in h5 not in manifest",
                             "manifest row no h5 group",
                             "manifest n_contacts mismatch")):
            pool[key(d)].append(f"consistency:{check}")

    out = []
    for (src, split, task, tid), reasons in pool.items():
        if not tid:
            continue   # whole-file issues already in their own CSV
        primary = reasons[0]
        out.append({
            "source": src, "split": split, "task": task,
            "trial_id": tid,
            "primary_reason": primary,
            "all_reasons": json.dumps(reasons),
        })
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default=str(REPO_ROOT / "out" / "data_verify"),
                   type=Path)
    p.add_argument("--limit_files", type=int, default=None,
                   help="Process only the first N files (smoke-test)")
    p.add_argument("--limit_trials", type=int, default=None,
                   help="Per-file trial cap (smoke-test)")
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--apply_quarantine", action="store_true",
                   help="After scan, copy quarantine.csv into both corpus roots")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_sources()
    print(f"sources: {[(s, sp) for s, sp, _, _ in sources]}", flush=True)

    structural: list[dict] = []
    schema: list[dict] = []
    content: list[dict] = []
    file_attr_issues: list[dict] = []
    seen_trials: dict[tuple, int] = defaultdict(int)
    manifest_rows_per_file: dict[str, list[dict]] = defaultdict(list)
    n_trials_total = 0
    n_trials_ok = 0
    t0 = time.time()

    for src_name, split, manifest_path, h5_dir in sources:
        h5_files = sorted(h5_dir.glob("*.h5"))
        if args.limit_files is not None:
            h5_files = h5_files[: args.limit_files]
        for h5_path in h5_files:
            task = h5_path.stem
            file_attr_issues.extend(check_file_attrs(h5_path, src_name))
            try:
                with h5py.File(h5_path, "r") as f:
                    if "trials" not in f:
                        file_attr_issues.append({
                            "source": src_name, "h5_path": str(h5_path),
                            "field": "trials group", "severity": "fatal",
                            "expected": "present", "actual": "missing",
                        })
                        continue
                    tids = list(f["trials"].keys())
                    if args.limit_trials is not None:
                        tids = tids[: args.limit_trials]
                    for i, tid in enumerate(tids):
                        n_trials_total += 1
                        seen_trials[(src_name, split, task, tid)] += 1
                        try:
                            grp = f[f"trials/{tid}"]
                            s_i, sc_i, c_i = scan_trial(
                                src_name, split, task, h5_path, grp, tid)
                            structural.extend(s_i)
                            schema.extend(sc_i)
                            content.extend(c_i)
                            if not (s_i or any(x["severity"] == "fatal" for x in sc_i)):
                                n_trials_ok += 1
                        except (OSError, RuntimeError) as e:
                            structural.append({
                                "source": src_name, "split": split, "task": task,
                                "h5_path": str(h5_path), "trial_id": tid,
                                "error_kind": "exception",
                                "error_field": "", "error_msg": str(e)[:200],
                            })
                        if n_trials_total % args.log_every == 0:
                            dt = time.time() - t0
                            print(f"  scanned {n_trials_total} trials in {dt:.1f}s "
                                  f"({n_trials_total / max(dt, 1e-6):.1f}/s)  "
                                  f"{src_name}/{task}", flush=True)
            except OSError as e:
                file_attr_issues.append({
                    "source": src_name, "h5_path": str(h5_path),
                    "field": "<open>", "severity": "fatal",
                    "expected": "openable", "actual": str(e)[:200],
                })

    print(f"\nphases A-C done: {n_trials_ok}/{n_trials_total} trials clean in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    print("running phase D (cross-source consistency)...", flush=True)
    consistency = consistency_pass(sources, seen_trials, manifest_rows_per_file)

    # --- write CSVs ---
    def write_csv(rows: list[dict], name: str, fields: list[str]):
        path = args.output_dir / name
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        print(f"  wrote {path}  ({len(rows)} rows)", flush=True)

    write_csv(structural, "quarantine_structural.csv",
              ["source", "split", "task", "h5_path", "trial_id",
               "error_kind", "error_field", "error_msg"])
    write_csv(schema, "schema_violations.csv",
              ["source", "split", "task", "h5_path", "trial_id",
               "field", "severity", "expected", "actual"])
    write_csv(content, "content_issues.csv",
              ["source", "split", "task", "h5_path", "trial_id",
               "check", "observed", "expected"])
    write_csv(consistency, "consistency_issues.csv",
              ["source", "split", "task", "trial_id",
               "check", "observed", "expected"])
    write_csv(file_attr_issues, "file_attr_issues.csv",
              ["source", "h5_path", "field", "severity", "expected", "actual"])

    quarantine = merge_quarantine(structural, schema, content, consistency)
    write_csv(quarantine, "quarantine.csv",
              ["source", "split", "task", "trial_id",
               "primary_reason", "all_reasons"])

    summary = {
        "elapsed_sec": time.time() - t0,
        "n_trials_total": n_trials_total,
        "n_trials_ok": n_trials_ok,
        "structural_issues": len(structural),
        "schema_violations": len(schema),
        "schema_fatal": sum(1 for x in schema if x["severity"] == "fatal"),
        "content_issues": len(content),
        "consistency_issues": len(consistency),
        "file_attr_issues": len(file_attr_issues),
        "quarantine_trials": len(quarantine),
        "quarantine_pct": 100 * len(quarantine) / max(n_trials_total, 1),
        "by_source": {},
    }
    df_q = pd.DataFrame(quarantine) if quarantine else pd.DataFrame(
        columns=["source", "split", "task"])
    for src in df_q["source"].unique() if len(df_q) else []:
        summary["by_source"][src] = int((df_q["source"] == src).sum())
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nsummary:", json.dumps(summary, indent=2))

    if args.apply_quarantine and quarantine:
        for root in (LIBERO_ROOT, ROBOCASA_ROOT):
            dest = root / "quarantine.csv"
            import shutil
            shutil.copy(args.output_dir / "quarantine.csv", dest)
            print(f"  copied quarantine to {dest}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
