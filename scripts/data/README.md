# scripts/data — dataset transport

Helpers for moving the v2 dataset between the local USB drive, HuggingFace,
and any cluster. Private-repo backup workflow; not for public release.

## upload_v2_to_hf.py

Uploads `/media/aaron/FAILBENCH/failbench/libero/v2/` (~177 GB) to a private
HF dataset repo. Mirrors layout exactly; no format conversion.

```bash
# One-time setup
huggingface-cli login
huggingface-cli repo create --type dataset --private aaronngx/failbench-libero-v2

# Sanity check
PYTHONPATH=. python -m scripts.data.upload_v2_to_hf --dry-run

# Real upload (run in background; resumable on failure)
nohup PYTHONPATH=. python -m scripts.data.upload_v2_to_hf \
    > logs/hf_upload.log 2>&1 &
```

Realistic upload time: 4-10 h on home upload speeds, ~25 min on a 1 Gbps
institutional uplink.

## pull_v2_from_hf.py

Materialises a local copy on the cluster (or anywhere). Rewrites manifest
h5_paths so the FailBench loaders work directly.

```bash
PYTHONPATH=. python -m scripts.data.pull_v2_from_hf \
    --dst /scratch/$USER/failbench/v2

# Or just one split for a focused experiment:
PYTHONPATH=. python -m scripts.data.pull_v2_from_hf \
    --dst /scratch/$USER/failbench/v2 --splits libero_spatial
```

After pulling, set `FAILBENCH_V2_ROOT=/scratch/$USER/failbench/v2` and
benchmark scripts pick it up.

## What's NOT here

- Per-trial DINOv2 cache (~700 MB at full scale): regenerable in ~7 min
  via `scripts/benchmark/precompute_dinov2_v2.py`. Re-run on cluster.
- Marginal-target cache (~280 MB at full scale): regenerable in ~5 min
  via `scripts/benchmark/build_marginal_targets.py`. Re-run on cluster.
- Trained checkpoints (`runs/bench/`): not in this repo, separate concern.

## Prereqs

- `huggingface_hub >= 0.34.0` (xet-core support for large-folder upload).
- HF account with token; repo `aaronngx/failbench-libero-v2` created.
- For upload: USB mounted at `/media/aaron/FAILBENCH`.
- For pull: ~180 GB free on `--dst` filesystem.
