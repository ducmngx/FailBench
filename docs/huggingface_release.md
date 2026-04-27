# HuggingFace Dataset Release — Migration Plan

This document describes how to publish the v10 FailBench dataset (~40 GB, 16,538 npz across 5 scenes / 54 tasks) to a public HuggingFace dataset repository tied to the `ducmngx` account.

The plan is split into preparation (one-time), packaging (per-version), upload (per-version), and post-release verification. Each step is independently runnable so the release can be staged, paused, and resumed.

---

## 0. Decisions

| Choice | Picked | Why |
|---|---|---|
| Visibility | **Public** | Reviewers and downstream users can `datasets.load_dataset("ducmngx/FailBench")`; standard PoC release. |
| Repo name | `ducmngx/FailBench` (mirrors GitHub) | Discoverability — same name as the code repo. |
| Storage format | **Per-scene tarballs of native npz** + **per-task parquet manifest** (see §2) | Bulk download stays as native npz (no schema lossy conversion of contact arrays); parquet gives fast metadata indexing without unzipping. |
| Versioning | **HF tags** (`v10`, `v10.1`, …) | One HF tag per dataset version; matches the on-disk `datasets/v10/` convention. |
| License | **CC BY 4.0** | Permissive academic-friendly; allows redistribution with attribution. Lock in before first upload. |
| Citation | Optional **Zenodo DOI** for the paper-bound release | HF link is unstable in principle; Zenodo gives a citeable DOI tied to a tagged snapshot. |

Toggle these in §3 / §4 if you change your mind before upload.

---

## 1. One-time account / repo prep

Items the human has to do once. Listed as a checklist; estimated 15 min total.

- [ ] **HF account** — `ducmngx`. If not yet, sign up at huggingface.co.
- [ ] **HF token** with `write` scope:
  ```bash
  huggingface-cli login    # paste a token from huggingface.co/settings/tokens
  ```
  The token persists in `~/.cache/huggingface/token`.
- [ ] **Install client tools** in the failbench env (one-time):
  ```bash
  conda activate failbench_env
  pip install --upgrade huggingface_hub datasets pyarrow
  ```
- [ ] **Create the dataset repo** (empty) on the hub. Either via the website (huggingface.co/new-dataset) or:
  ```bash
  huggingface-cli repo create FailBench --type dataset
  ```
- [ ] **Decide license** (CC BY 4.0 recommended) and add it to the YAML frontmatter of the dataset card (§3).
- [ ] **Initialise git-lfs locally** (HF datasets are git repos with LFS for binaries):
  ```bash
  git lfs install
  ```

---

## 2. Packaging strategy

### Native shape vs HF best practice

The natural HF "load_dataset" experience is parquet shards with column-oriented arrays. But:

- Each FailBench npz is heterogeneously shaped (480×640×3 RGB + 480×640 depth + variable-length contact arrays). Serializing all of that as parquet would either flatten contacts (lossy) or produce nested-list columns that are awkward to query.
- The dataset's natural unit *is* the npz file. Converting to parquet would add round-trip cost and obscure provenance.

So we ship two artefacts side-by-side:

1. **`<scene>.tar.zst`** — one zstd-compressed tarball per scene containing all the raw `<task>/exp_*.npz` files unchanged. Native, lossless, exactly what the local pipeline produces. Total ~40 GB → ~25–30 GB after zstd-19.
2. **`metadata.parquet`** — one row per npz (16,538 rows, ~2 MB), columns from each task's `manifest.csv` plus a `scene` and `task_id` column and a derived `tar_path` column pointing into the per-scene tarball. Lets users filter / split / load metadata without ever touching the binary blobs.

### Why per-scene tarballs (not one giant tar, not per-task)

- One giant 40 GB tar = one upload-failure restart from scratch. Bad.
- Per-task = 54 tarballs ~700 MB each — too many small files for HF's LFS pointer system to be efficient (each tar tracked independently).
- Per-scene = 5 tarballs in the 6–10 GB range — lines up with HF's recommended chunk size, parallelizable, individually re-uploadable on failure.

### Preserve the directory structure inside each tarball

```
scene_level2.tar.zst
├── scene_level2/
│   ├── clean_far/
│   │   ├── manifest.csv
│   │   ├── exp_000000.npz
│   │   └── ...
│   ├── clean_nominal/
│   │   └── ...
│   └── ...
```

Users untar with `tar xf scene_level2.tar.zst` and get the same `<scene>/<task>/` layout the local pipeline produces — `notebooks/inspect_dataset.ipynb` then works against the unpacked dir without modification.

---

## 3. Dataset card (`README.md` on HF)

The single most important file for discoverability. Sketch:

```markdown
---
license: cc-by-4.0
tags:
  - robotics
  - manipulation
  - failure-recovery
  - mujoco
  - franka-panda
  - simulation
size_categories:
  - 10K<n<100K
task_categories:
  - other
pretty_name: FailBench v10
---

# FailBench v10 — Failure-injection dataset for Franka Panda manipulation

16,538 simulated pick-and-place trials across 5 scenes, with each trial
forking the same pre-failure state into 6 failure modes (gripper open /
slippery grip / two single-joint locks / multi-joint lock / all-joint
lock). Used to study how pre-failure robot configuration determines
post-failure contact outcomes — the contact-prediction problem set up
in https://github.com/ducmngx/FailBench .

## Quick start

[curl one tarball, untar, open notebook]

## Scenes
[per-scene table — sample counts, OOD candidate, layout image]

## Per-sample contents
[npz schema reproduced from data_generation.md §3]

## Splits
[recommended traj-level 80/10/10 + scene_grocery as OOD test]

## How it was generated
[link to GitHub repo + commit hash that produced v10]

## License & citation
[CC BY 4.0; bibtex; Zenodo DOI if minted]
```

Source the schema and pipeline description from the existing `docs/data_generation.md` so it stays in one place.

---

## 4. Upload script (new — to be written)

Add `scripts/release_to_hf.py`. Skeleton:

```python
"""Package datasets/v10 and push it to huggingface.co/datasets/ducmngx/FailBench."""

import argparse, glob, os, subprocess
from huggingface_hub import HfApi, login, upload_file

REPO_ID = "ducmngx/FailBench"
SRC = "datasets/v10"

def pack_scene_tar(scene: str, src_root: str, out_dir: str) -> str:
    """tar -I 'zstd -19 -T0' for the scene; returns output path."""
    out = os.path.join(out_dir, f"{scene}.tar.zst")
    subprocess.check_call([
        "tar", "-I", "zstd -19 -T0",
        "-cf", out,
        "-C", src_root, scene,
    ])
    return out

def build_metadata_parquet(src_root: str, out_path: str) -> None:
    """Read every per-task manifest.csv, add scene/task/tar_path cols, write parquet."""
    import pandas as pd
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(src_root, "*", "*", "manifest.csv"))):
        scene = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        task = os.path.basename(os.path.dirname(csv_path))
        df = pd.read_csv(csv_path)
        df["scene"] = scene
        df["_task"] = task
        df["tar_path"] = f"{scene}.tar.zst"
        df["intra_path"] = df["npz_file"].map(lambda f: f"{scene}/{task}/{f}")
        rows.append(df)
    pd.concat(rows).to_parquet(out_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v10")
    parser.add_argument("--release_dir", default="release")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.release_dir, exist_ok=True)

    # 1. Build tarballs (one per scene)
    scenes = sorted(os.listdir(SRC))
    for s in scenes:
        path = pack_scene_tar(s, SRC, args.release_dir)
        print(f"  {s}: {os.path.getsize(path) / 1024**3:.2f} GB")

    # 2. Build metadata parquet
    build_metadata_parquet(SRC, os.path.join(args.release_dir, "metadata.parquet"))

    if args.dry_run:
        return

    # 3. Push to HF as a new tag/revision
    api = HfApi()
    api.upload_folder(
        folder_path=args.release_dir,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"release {args.version}",
        ignore_patterns=["*.tmp"],
    )
    api.create_tag(REPO_ID, repo_type="dataset", tag=args.version,
                   tag_message=f"FailBench {args.version} ({sum(...)} samples)")

if __name__ == "__main__":
    main()
```

The actual implementation (~80 lines) goes in `scripts/release_to_hf.py` when we execute the plan.

### Disk and time budget

- Tarball build: 5 scenes × ~7 GB each at zstd-19 → CPU-bound, ~10–15 min total on 6 cores.
- Upload: HF allows resumable LFS; 30 GB at 100 Mbit/s upload ≈ 40 min. Plan for an hour with overhead.
- Local disk needs **+30 GB free** during packaging (tarballs sit alongside the source dataset).

---

## 5. Verification (post-upload)

Run from a clean clone or a different machine to confirm a downstream user can actually load it:

1. **List the repo contents**:
   ```python
   from huggingface_hub import HfApi
   HfApi().list_repo_files("ducmngx/FailBench", repo_type="dataset")
   ```
   Expect 5 `.tar.zst` + `metadata.parquet` + `README.md` + `.gitattributes`.

2. **Download metadata only** (cheap sanity check):
   ```python
   from huggingface_hub import hf_hub_download
   import pandas as pd
   path = hf_hub_download("ducmngx/FailBench", "metadata.parquet", repo_type="dataset")
   df = pd.read_parquet(path)
   assert len(df) == 16538
   assert set(df["scene"].unique()) == {"scene_level2", "scene_kitchen",
                                         "scene_cluttered", "scene_workshop",
                                         "scene_grocery"}
   ```

3. **Pull one scene tarball, untar, open notebook**:
   ```bash
   python -c "from huggingface_hub import hf_hub_download; \
              hf_hub_download('ducmngx/FailBench', 'scene_level2.tar.zst', \
                              repo_type='dataset', local_dir='hf_check')"
   tar -xf hf_check/scene_level2.tar.zst -C hf_check/
   # point notebooks/inspect_dataset.ipynb's DATASET_ROOT at hf_check/ and run
   ```

4. **Inspect dataset card** rendering on huggingface.co — preview the README, check the schema table renders, confirm size_categories shows correctly.

---

## 6. Optional: Zenodo DOI

If the paper needs a citeable DOI (most reviewers will appreciate this):

- Zenodo has a 1-click GitHub integration: a release tag on `ducmngx/FailBench` (the code repo) auto-mints a DOI for that snapshot of the code.
- For the *dataset* itself, upload the same 5 tarballs + parquet to a new Zenodo deposit (50 GB free quota). Get a separate DOI.
- Cite both in the paper:
  - Code DOI: `zenodo.X` (auto-minted from GitHub release)
  - Dataset DOI: `zenodo.Y` (manual upload of the v10 tarballs)
- Cross-reference in the HF dataset card README.

This step is independent of the HF upload — do it any time after the HF release stabilises.

---

## 7. Critical files / artefacts

- `datasets/v10/` — source data (gitignored; 40 GB local).
- `scripts/release_to_hf.py` — **new**, to write when executing.
- `release/` — staging directory the script creates (5 tarballs + parquet + README).
- HF repo: `huggingface.co/datasets/ducmngx/FailBench`.
- License header in dataset card.

## 8. Non-goals for this release

- Don't convert to a HF `Dataset.from_dict(...)` parquet-only schema. The npz shape is non-trivial and the conversion would either be lossy or unwieldy.
- Don't upload the 671 source `.pkl` trajectories alongside — those are large, not the point of this dataset, and re-derivable from the GitHub repo via `scripts/generate_task_trajs.py`.
- Don't ship `scripts/`, `notebooks/`, or any code in the HF repo. Code lives in GitHub; the HF repo is data + a card linking to GitHub.

---

## Execution order

1. Section 1 checklist (account + token + LFS).
2. Implement `scripts/release_to_hf.py` per §4.
3. Run with `--dry_run` to build tarballs locally; spot-check sizes and untar one to confirm structure.
4. Author the HF dataset card (`release/README.md`).
5. Run without `--dry_run` to push.
6. Verify per §5.
7. Optionally mint Zenodo DOIs per §6.
