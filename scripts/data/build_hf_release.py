#!/usr/bin/env python3
"""Assemble a HuggingFace Datasets release for the FailBench contact-prediction v2 corpus.

Produces a self-contained ``hf_release/`` tree ready for ``huggingface-cli upload``:

    hf_release/
      README.md                         dataset card (YAML frontmatter + body)
      LICENSE                           CC BY 4.0
      SCHEMA.md                         copy of docs/libero_v2_dataset.md
      CITATION.cff                      structured citation
      citation.bib                      bibtex form
      .gitattributes                    LFS filter for *.h5
      quarantine.csv                    transparency: dropped trials
      sample/sample_trial.h5            one trial, standalone (~3 MB)
      load_example.py                   minimal h5py + datasets snippet
      libero/
        manifest.csv                    quarantine-applied, h5_path repo-relative
        libero_spatial/*.h5             symlinks (or copies) of the per-task HDF5s
        libero_object/*.h5
        libero_goal/*.h5
      robocasa/
        manifest.csv
        *.h5

By default, ``*.h5`` are symlinked from their source locations (saves ~200 GB
of duplicate disk). Use ``--copy`` to copy them physically (needed if the
upload host is different from the build host).

Run::

    /home/aaron/miniconda3/envs/failbench_env/bin/python -u -m scripts.data.build_hf_release \\
        --output_dir hf_release --quarantine out/data_verify/quarantine.csv

The output directory is not pushed automatically; run::

    huggingface-cli upload <user>/failbench-contact-v2 hf_release/ . --repo-type dataset

separately (or use the wrapper in ``scripts/data/upload_v2_to_hf.py``).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import hdf5plugin  # noqa: F401
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

LIBERO_SRC = Path("/media/aaron/F/failbench/libero/v2")
ROBOCASA_SRC = Path("/media/aaron/F/failbench/robocasa/v2")
SCHEMA_DOC = REPO_ROOT / "docs" / "libero_v2_dataset.md"


LICENSE_TEXT = """\
Creative Commons Attribution 4.0 International (CC BY 4.0)

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the
  license, and indicate if changes were made.

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode

This dataset is derived from:
- LIBERO (MIT licensed) — https://github.com/Lifelong-Robot-Learning/LIBERO
- mimicdroid-robocasa — https://github.com/UT-Austin-RPL/mimicdroid-robocasa
- robosuite (MIT licensed) — https://github.com/ARISE-Initiative/robosuite

Citation requirements for derived works: see CITATION.cff and citation.bib.
"""


CITATION_CFF = """\
cff-version: 1.2.0
title: "FailBench Contact Prediction Dataset v2 (LIBERO + RoboCasa)"
message: "If you use this dataset, please cite both this work and the upstream LIBERO and RoboCasa datasets."
type: dataset
authors:
  - family-names: Nguyen
    given-names: Aaron
abstract: >
  A pooled simulated dataset for failure-induced contact prediction. Each
  trial records a robot manipulation demo with a single failure mode injected
  mid-trajectory; the post-failure settle is captured as projected contact
  points, world-frame forces, and per-frame robot + object state. Built on
  LIBERO and RoboCasa kitchen scenes.
keywords:
  - robotics
  - contact prediction
  - mujoco
  - failure injection
  - libero
  - robocasa
license: CC-BY-4.0
references:
  - type: software
    title: "LIBERO"
    url: "https://github.com/Lifelong-Robot-Learning/LIBERO"
  - type: software
    title: "RoboCasa (mimicdroid-robocasa fork)"
    url: "https://github.com/UT-Austin-RPL/mimicdroid-robocasa"
  - type: software
    title: "robosuite"
    url: "https://github.com/ARISE-Initiative/robosuite"
"""


CITATION_BIB = """\
@dataset{failbench_contact_v2,
  title  = {FailBench Contact Prediction Dataset v2 (LIBERO + RoboCasa)},
  author = {Nguyen, Aaron},
  year   = {2026},
  url    = {https://huggingface.co/datasets/<user>/failbench-contact-v2},
  note   = {Built on LIBERO and mimicdroid-robocasa kitchen scenes.}
}

@inproceedings{liu2024libero,
  title     = {LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author    = {Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year      = {2023}
}

@inproceedings{nasiriany2024robocasa,
  title  = {{RoboCasa}: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author = {Nasiriany, Soroush and Maddukuri, Abhiram and Zhang, Lance and Parikh, Adeet and Lo, Aaron and Joshi, Abhishek and Mandlekar, Ajay and Zhu, Yuke},
  booktitle = {Robotics: Science and Systems},
  year   = {2024}
}

@article{zhu2020robosuite,
  title   = {{robosuite}: A Modular Simulation Framework and Benchmark for Robot Learning},
  author  = {Zhu, Yuke and Wong, Josiah and Mandlekar, Ajay and Mart{\\'i}n-Mart{\\'i}n, Roberto and Joshi, Abhishek and Nasiriany, Soroush and Zhu, Yifeng},
  journal = {arXiv preprint arXiv:2009.12293},
  year    = {2020}
}
"""


GITATTRIBUTES = """\
*.h5 filter=lfs diff=lfs merge=lfs -text
sample/*.h5 filter=lfs diff=lfs merge=lfs -text
"""


LOAD_EXAMPLE = '''\
"""Minimal loader for FailBench Contact Prediction v2.

This dataset ships as per-task HDF5 files + per-corpus manifests. Use h5py
directly; the HuggingFace ``datasets`` library is supported via streaming for
the per-row interface.

Quick start with h5py
---------------------

    import h5py, hdf5plugin   # hdf5plugin must be imported BEFORE h5py.File
    import pandas as pd

    mf = pd.read_csv("libero/manifest.csv")
    row = mf.iloc[0]
    with h5py.File(row.h5_path) as f:
        trial = f[f"trials/{row.trial_id}"]
        rgb = trial["pre_rgb"][...]               # (240, 320, 3) uint8
        contacts = trial["contact_positions"][...] # (N, 3) float32
        forces   = trial["contact_force_world"][...] # (N, 3) float32

See SCHEMA.md for the full field list.

Quick start with datasets
-------------------------

    from datasets import load_dataset
    ds = load_dataset("<user>/failbench-contact-v2", streaming=True)
    sample = next(iter(ds["train"]))
'''


# --------------------------------------------------------------------------
# Quarantine load
# --------------------------------------------------------------------------


def load_quarantine(path: Path | None) -> dict:
    """Return {source: set((split, task, trial_id))} or {} if no path."""
    if path is None or not path.exists():
        print(f"WARN: no quarantine at {path}; manifests will include all trials",
              flush=True)
        return {}
    out: dict[str, set] = defaultdict(set)
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            out[r["source"]].add((r["split"], r["task"], r["trial_id"]))
    return out


# --------------------------------------------------------------------------
# Manifest rewrite: drop quarantined rows, repath h5_path to repo-relative
# --------------------------------------------------------------------------


def rewrite_manifest(src_path: Path, dst_path: Path,
                     quarantine_set: set,
                     repo_relative_dir: str) -> tuple[int, int]:
    """Copy src_path → dst_path, applying quarantine and rewriting h5_path."""
    df = pd.read_csv(src_path)
    before = len(df)
    if quarantine_set:
        keep_mask = ~df.apply(
            lambda r: (str(r["split"]), str(r["task"]), str(r["trial_id"])) in quarantine_set,
            axis=1,
        )
        df = df[keep_mask].copy()
    after = len(df)

    # Repath: replace absolute /media/... with repo-relative path
    if "h5_path" in df.columns:
        def _repath(p):
            stem = Path(p).stem
            split = Path(p).parent.name if Path(p).parent != Path("/") else ""
            # LIBERO: <repo_relative_dir>/<split>/<task>.h5
            # RoboCasa: <repo_relative_dir>/<task>.h5
            if split.startswith("libero_"):
                return f"{repo_relative_dir}/{split}/{stem}.h5"
            return f"{repo_relative_dir}/{stem}.h5"
        df["h5_path"] = df["h5_path"].map(_repath)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_path, index=False)
    return before, after


# --------------------------------------------------------------------------
# Dataset card
# --------------------------------------------------------------------------


def make_dataset_card(stats: dict, quarantine_count: int) -> str:
    yaml = f"""---
license: cc-by-4.0
pretty_name: FailBench Contact Prediction v2 (LIBERO + RoboCasa)
task_categories:
  - robotics
  - reinforcement-learning
tags:
  - robotics
  - contact-prediction
  - mujoco
  - failure-injection
  - libero
  - robocasa
size_categories:
  - 10K<n<100K
configs:
  - config_name: libero
    data_files:
      - split: train
        path: libero/manifest.csv
  - config_name: robocasa
    data_files:
      - split: train
        path: robocasa/manifest.csv
---
"""

    libero_n = stats["libero"]["trials"]
    libero_size_gb = stats["libero"]["size_gb"]
    rc_n = stats["robocasa"]["trials"]
    rc_size_gb = stats["robocasa"]["size_gb"]
    total_n = libero_n + rc_n
    total_size_gb = libero_size_gb + rc_size_gb

    body = f"""\
# FailBench Contact Prediction Dataset v2

Simulated dataset for **failure-induced contact prediction** in robotic manipulation.
Each trial captures a Franka Panda demo with a single failure mode (gripper
release, joint freeze, etc.) injected mid-trajectory. The post-failure settle is
recorded as projected contact points, world-frame forces, per-step robot state,
and object pose trajectories — designed to train models that predict *where on
the scene the failure will produce contacts* from pre-failure context.

## Quick stats

| Source | Trials | Size | Tasks |
|---|---|---|---|
| LIBERO | {libero_n:,} | {libero_size_gb:.1f} GB | 30 |
| RoboCasa | {rc_n:,} | {rc_size_gb:.1f} GB | 5 |
| **Total** | **{total_n:,}** | **{total_size_gb:.1f} GB** | **35** |

{quarantine_count:,} trials were quarantined during pre-publication verification
and are excluded from the shipped manifests (see `quarantine.csv` for the list
and per-trial reasons; primarily metadata-attribute corruption from an interim
build artifact, not content corruption).

## What's in each trial

54 datasets + 22 attrs per trial. See `SCHEMA.md` for the full spec. Highlights:

- **Pre-failure window (T=8 frames)** — agentview RGB + depth, wrist-cam RGB + depth, per-frame arm state.
- **Goal (K=3 future demo states)** — the intended trajectory the demo was driving toward.
- **Contact arrays** — `contact_positions`, `contact_forces`, `contact_force_world`, `contact_time`, `contact_geom_pairs`.
- **Post-failure pose + RGB** — the settled scene.
- **Camera calibration** — agentview + wrist intrinsics/extrinsics so contacts can be reprojected.
- **Settle trajectory (S=50 snapshots)** — per-step robot + object pose for world-model training.

RoboCasa trials additionally carry `baseline_contact_*` arrays — the static
contacts present at the pre-failure equilibrium. Subtract from `contact_*` to
isolate failure-induced contacts.

## Schema

See `SCHEMA.md` (a copy of the authoritative `docs/libero_v2_dataset.md`).

## How to use

See `load_example.py` for minimal h5py + HuggingFace `datasets` snippets.

## Known limitations

- ~5 % of LIBERO and ~20 % of RoboCasa trials have **zero failure-induced contacts** — failures at low `traj_progress` or with benign modes that don't propagate. Documented in the manifests via `n_contacts == 0` rows.
- **TurnOnMicrowave** is a sparsity outlier (median 15 contacts vs ~1,000 elsewhere). Filter or upweight at training time.
- The `scene_entities_json` attr is **stale on ~20 % of RoboCasa trials**. Numerical scene attrs (`scene_table_z`, `scene_aabb_min/max`) are correct everywhere. Per-entity risk users should rebuild the entity list from these numerical attrs at load time.
- `trial_id` is unique within `(source, split, task)` but **not globally unique** — the manipulation manifest's compound key `(source, split, task, trial_id)` is the actual unique identifier.

## License

Released under **CC BY 4.0**. See `LICENSE`. The upstream LIBERO and robosuite
projects are MIT-licensed; mimicdroid-robocasa data is derived from RoboCasa
(CC BY 4.0).

## Citation

If you use this dataset, please cite this dataset (`CITATION.cff` and
`citation.bib`) AND the upstream LIBERO and RoboCasa papers. Full bibtex in
`citation.bib`.

## Reproducibility

The build pipeline is in the FailBench repo:

- Generation: `scripts/libero/build_v2_dataset.py`, `scripts/robocasa/build_v2_dataset.py`
- Baseline subtraction: `scripts/robocasa/add_baseline_contacts.py`
- Scene metadata fix: `scripts/robocasa/fix_scene_metadata.py`
- Verification: `scripts/data/verify_corpus.py`
- This release assembly: `scripts/data/build_hf_release.py`

Sidecar venv setup for source HDF5 dependencies is documented in the FailBench
README. The corpus was generated against the commit referenced in the GitHub
release.
"""
    return yaml + body


# --------------------------------------------------------------------------
# Sample trial extraction
# --------------------------------------------------------------------------


def extract_sample_trial(libero_src: Path, dst: Path) -> None:
    """Copy one full LIBERO trial group into a standalone HDF5 for inspection."""
    sample_task = libero_src / "libero_spatial"
    h5s = list(sample_task.glob("*.h5"))
    if not h5s:
        print(f"WARN: no LIBERO HDF5 at {sample_task}; skipping sample trial",
              flush=True)
        return
    src_h5 = h5s[0]
    with h5py.File(src_h5, "r") as fsrc:
        tids = list(fsrc["trials"].keys())
        if not tids:
            return
        sample_tid = tids[0]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst, "w") as fdst:
            for k, v in fsrc.attrs.items():
                fdst.attrs[k] = v
            grp_src = fsrc[f"trials/{sample_tid}"]
            grp_dst = fdst.create_group(f"trials/{sample_tid}")
            for k, v in grp_src.attrs.items():
                grp_dst.attrs[k] = v
            for name in grp_src:
                fdst.copy(grp_src[name], grp_dst, name=name)
    print(f"  extracted sample trial {sample_tid} from {src_h5.name}", flush=True)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def link_or_copy(src: Path, dst: Path, do_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if do_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default=str(REPO_ROOT / "hf_release"), type=Path)
    p.add_argument("--quarantine", default=str(REPO_ROOT / "out" / "data_verify" / "quarantine.csv"),
                   type=Path)
    p.add_argument("--copy", action="store_true",
                   help="Copy HDF5s instead of symlinking (adds ~200 GB of duplicate disk)")
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"building HF release at {out}", flush=True)

    # ---- 1. license + citations + .gitattributes + load example ----
    (out / "LICENSE").write_text(LICENSE_TEXT)
    (out / "CITATION.cff").write_text(CITATION_CFF)
    (out / "citation.bib").write_text(CITATION_BIB)
    (out / ".gitattributes").write_text(GITATTRIBUTES)
    (out / "load_example.py").write_text(LOAD_EXAMPLE)

    # ---- 2. SCHEMA.md = copy of docs/libero_v2_dataset.md ----
    if SCHEMA_DOC.exists():
        shutil.copy2(SCHEMA_DOC, out / "SCHEMA.md")
    else:
        print(f"WARN: missing {SCHEMA_DOC}; SCHEMA.md will be a stub", flush=True)
        (out / "SCHEMA.md").write_text("# Schema\n\nSee FailBench repo docs/libero_v2_dataset.md\n")

    # ---- 3. quarantine.csv ----
    if args.quarantine.exists():
        shutil.copy2(args.quarantine, out / "quarantine.csv")
    quarantine = load_quarantine(args.quarantine)

    # ---- 4. LIBERO manifests + HDF5s ----
    libero_n = 0
    libero_bytes = 0
    libero_q = quarantine.get("libero", set())
    for split in ("libero_spatial", "libero_object", "libero_goal"):
        src_dir = LIBERO_SRC / split
        if not src_dir.exists():
            continue
        dst_dir = out / "libero" / split
        for h5 in sorted(src_dir.glob("*.h5")):
            dst = dst_dir / h5.name
            link_or_copy(h5, dst, args.copy)
            libero_bytes += h5.stat().st_size
    # Manifest rewriting per split (then concat into libero/manifest.csv)
    libero_frames = []
    for split in ("libero_spatial", "libero_object", "libero_goal"):
        src_mf = LIBERO_SRC / split / "manifest.csv"
        if not src_mf.exists():
            continue
        tmp = out / "libero" / split / "manifest.csv"
        before, after = rewrite_manifest(src_mf, tmp, libero_q, "libero")
        libero_n += after
        libero_frames.append(pd.read_csv(tmp))
        print(f"  libero/{split}: {before} → {after} (after quarantine)", flush=True)
    if libero_frames:
        pd.concat(libero_frames, ignore_index=True).to_csv(out / "libero" / "manifest.csv", index=False)

    # ---- 5. RoboCasa manifests + HDF5s ----
    robocasa_n = 0
    robocasa_bytes = 0
    rc_q = quarantine.get("robocasa", set())
    src_mf = ROBOCASA_SRC / "manifest.csv"
    if src_mf.exists():
        for h5 in sorted(ROBOCASA_SRC.glob("*.h5")):
            link_or_copy(h5, out / "robocasa" / h5.name, args.copy)
            robocasa_bytes += h5.stat().st_size
        before, after = rewrite_manifest(src_mf, out / "robocasa" / "manifest.csv",
                                         rc_q, "robocasa")
        robocasa_n = after
        print(f"  robocasa: {before} → {after} (after quarantine)", flush=True)

    # ---- 6. sample trial ----
    extract_sample_trial(LIBERO_SRC, out / "sample" / "sample_trial.h5")

    # ---- 7. README.md (dataset card) ----
    stats = {
        "libero":  {"trials": libero_n,  "size_gb": libero_bytes / 2**30},
        "robocasa": {"trials": robocasa_n, "size_gb": robocasa_bytes / 2**30},
    }
    quarantine_total = sum(len(v) for v in quarantine.values())
    (out / "README.md").write_text(make_dataset_card(stats, quarantine_total))

    print(f"\ndone. tree at {out}", flush=True)
    print(f"  libero  : {libero_n:,} trials, {libero_bytes / 2**30:.1f} GB", flush=True)
    print(f"  robocasa: {robocasa_n:,} trials, {robocasa_bytes / 2**30:.1f} GB", flush=True)
    print(f"  quarantined: {quarantine_total:,} trials", flush=True)
    print(f"\nnext: `huggingface-cli upload <user>/failbench-contact-v2 {out}/ . --repo-type dataset`",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
