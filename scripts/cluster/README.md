# Running FailBench sweeps on the GMU Hopper cluster

This directory contains SLURM scaffolding for scaling the contact-prediction
benchmark beyond what fits on the local 3070. Use it once the local
architecture-design work is done (see `docs/benchmark_experiments_log.md`)
and you need multi-seed, multi-split paper-quality numbers.

## Cluster facts (Hopper, as of 2026-06)

- **Scheduler:** SLURM
- **GPU nodes:**
  - 31 × A100 80GB nodes (4 GPUs/node, 64-core AMD, 512 GB RAM) — primary target
  - 2 × DGX A100 40GB nodes (8 GPUs/node, 128-core AMD, 1 TB RAM) — useful for big-batch jobs
  - 1 × H100 80GB node (4 GPUs, 112-core Intel, 2 TB RAM) — for the largest sweeps
- **Storage:**
  - `$HOME` (= `/home/$USER`) — 60 GB, backed up. Code lives here.
  - `$SCRATCH` (= `/scratch/$USER`) — unlimited, **90-day purge**. Data lives here.
  - `/projects/<advisor>/` — persistent shared group space, ask advisor for access.

## Partitions (confirmed via `sinfo` 2026-06-01)

| Partition | Time limit | GPUs | Notes |
|---|---|---|---|
| **`gpuq`** | 5 days | A100 80GB + A100 40GB (dgx001-002) | Default for regular users. SBATCH script targets this |
| `contrib-gpuq` | 5 days | More A100s, gated by contributor access | Ask advisor if your group has nodes here |
| `contrib-H100` | 5 days | H100 80GB (gpu032) | Contributor-gated. Use for the biggest sweeps |
| `contrib-B200` | 5 days | B200 (dgx003) | Newest. Likely contributor-gated |

To **force A100 80GB only** (instead of getting an A100 40GB DGX node),
check the feature name via `sinfo -o "%P %f" -p gpuq` and add a
`#SBATCH --constraint=...` line. For our sweep, 40GB is fine — even the
biggest model (UNet+failure_oracle) trains in <12 GB at batch 64.

## TODO before first submission

1. Confirm the conda module name. Run `module avail anaconda` after login
   (or `module avail miniconda`). Edit the `module load` line in
   `sweep_headline.sbatch` if it isn't `anaconda3`.
2. If your group has an allocation code, uncomment `#SBATCH --account=...`
   in the SBATCH script.

## One-time setup

**Training on Hopper is pure PyTorch + h5py + numpy** — no MuJoCo at runtime
(the dataset is already pre-rendered). So we use a pip `venv`, not conda.

```bash
# 1. SSH in
ssh $USER@hopper.orc.gmu.edu

# 2. Clone the repo into $HOME
cd $HOME
git clone -b refactor/cleanup https://github.com/ducmngx/FailBench.git
cd FailBench

# 3. Build the venv (uses Hopper's Python module, ~5 GB into $HOME)
module load gnu10
module load python/3.9.9-jh
python -m venv $HOME/failbench_env
source $HOME/failbench_env/bin/activate
pip install --upgrade pip

# 4. PyTorch with CUDA (cu121 wheels are forward-compatible with the
#    cluster's drivers; check `nvidia-smi` on a GPU node if unsure)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121

# 5. Rest of the training deps
pip install h5py==3.11.0 hdf5plugin numpy==2.0.1 scipy==1.15.3 \
            pandas matplotlib pillow pyyaml \
            transformers==4.55.4   # transformers only needed if you run DINOv2 modality

# 6. Quick import check (no GPU needed; head node is fine)
python -c "
import torch, h5py, hdf5plugin, numpy, pandas
print('torch', torch.__version__)
print('h5py', h5py.__version__)
print('all imports OK')
"
```

## Smoke-test the trainer on a GPU node

Never run training on the head node — grab a GPU via `salloc` first.
Hopper requires `-q gpu` (QOS) and an explicit GPU type in the gres:

```bash
salloc -p gpuq -q gpu --gres=gpu:A100.80gb:1 --cpus-per-task=8 --mem=32G --time=00:15:00
# (you get dropped onto a GPU node)

module load gnu10 && module load python/3.9.9-jh
source $HOME/failbench_env/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# expect: NVIDIA A100-SXM4-80GB, 81920 MiB (or similar)

cd $HOME/FailBench
PYTHONPATH=. python -m scripts.benchmark.train_one \
  --model convdec --modalities state --T 1 --epochs 1 --max_trials 100 \
  --v2_root /scratch/$USER/data/failbench_data/libero/v2 \
  --splits libero_spatial
# should print one "ep 1/1 train=... val=..." line in ~30 s
exit
```

If `salloc` queues for a long time, try `--gres=gpu:A100.40gb:1` instead —
DGX A100 40GB nodes are typically less contended.

## Stage the data into $SCRATCH (one-time, ~5-10 h overnight)

Either rsync direct from your local machine:

```bash
# From your local machine
rsync -avP --bwlimit=20000 \
  /home/aaron/scratch/v2_ssd/ \
  $USER@hopper.orc.gmu.edu:/scratch/$USER/failbench/v2/
```

…or pull from HuggingFace once we publish it (faster on cluster network):

```bash
# On Hopper, after huggingface-cli login
mkdir -p $SCRATCH/failbench/v2
huggingface-cli download aaronngx/failbench-libero-v2 \
  --repo-type dataset --local-dir $SCRATCH/failbench/v2
```

Verify the layout:

```bash
ls $SCRATCH/failbench/v2/
# should show: libero_spatial/ libero_object/ libero_goal/
ls $SCRATCH/failbench/v2/libero_spatial/ | head
# should show: manifest.csv plus 10 .h5 files
```

## Optional caches

DINOv2 (`cache/dinov2_v2`) and marginal targets (`cache/marginal_targets_v2`)
are only needed for runs that include `--modalities dino` or
`--target_form marginal`. The headline sweep does not use either. If you
end up running those:

```bash
# DINOv2 cache: precompute on the cluster (uses A100, ~1 h/split)
PYTHONPATH=. conda run -n failbench_env python -m scripts.benchmark.precompute_dinov2_v2 \
  --v2_root $SCRATCH/failbench/v2 --splits libero_spatial libero_object libero_goal \
  --out_root $SCRATCH/failbench/cache/dinov2_v2

# Marginal targets: precompute
PYTHONPATH=. conda run -n failbench_env python -m scripts.benchmark.build_marginal_targets \
  --v2_root $SCRATCH/failbench/v2 --splits libero_spatial libero_object libero_goal \
  --out_root $SCRATCH/failbench/cache/marginal_targets_v2
```

## Submitting the headline sweep

The headline sweep is **3 seeds × 5 model configs × 3 splits = 45 runs**.
Edit `sweep_configs.sh` if you want to add or drop configs; the SBATCH
script picks one config per array task.

```bash
cd $HOME/FailBench
# Dry-run: print what the array would launch
bash scripts/cluster/sweep_configs.sh

# Actually submit
sbatch scripts/cluster/sweep_headline.sbatch
# returns a job ID; track with:
squeue -u $USER
# or live with:
watch -n 5 'squeue -u $USER | head -50'
```

Each run writes to `$HOME/FailBench/runs/bench/<run_dir>/` with `best.pt`,
`metrics.json`, `args.json`. `runs/` lives in `$HOME` (not `$SCRATCH`)
because checkpoints + metrics are small and you want them backed up.

## Collecting results

After the array completes, aggregate locally on the cluster head node:

```bash
PYTHONPATH=. conda run -n failbench_env python -m scripts.benchmark.eval_all \
  --runs_root runs/bench \
  --v2_root $SCRATCH/failbench/v2
# writes runs/bench/_eval/bench_table.{csv,md}
```

Then rsync `runs/bench/_eval/` back to your local machine for figure-making
notebooks.

## Cost estimate

- Headline sweep: 45 runs × ~70 min avg on A100 = ~52 GPU-hours total.
  With 8 GPUs concurrent → ~7 hours wall-clock.
  With 16 GPUs concurrent → ~3.5 hours wall-clock.
- A100 80GB nodes have 4 GPUs each; the cluster typically lets you have
  multiple nodes concurrent for a job array, so 8-16 concurrent is realistic.

## Failure modes to watch for

- **EGL libraries missing**: if MuJoCo import fails, do `conda install -c conda-forge libegl libgl libglvnd` in the env. Usually conda's `environment.yml` already includes them.
- **HDF5 file locking**: trainer already sets `HDF5_USE_FILE_LOCKING=FALSE`. If a worker hangs on file open, double-check.
- **`$HOME` quota**: 60 GB. Checkpoints are ~22 MB each × 45 runs = 1 GB. Plenty of room. But don't accidentally put data in `$HOME`.
- **Job array task failures**: SLURM by default does not requeue. The script sets `--requeue` so transient failures (node reboot) auto-retry. For deterministic failures (bug), inspect `logs/sweep-<jobid>_<taskid>.out`.

## Future sweeps (queued, not built yet)

- `sweep_ood.sbatch` — 3 seeds × 4 configs × 3 task-held-out folds = 36 runs.
- `sweep_oracle.sbatch` — failure-descriptor oracle variants (§12-15 of the contact-prediction doc).
- `sweep_marginal.sbatch` — marginal target form; needs `cache/marginal_targets_v2` staged first.

Each follows the same pattern as `sweep_headline.sbatch`: a sweep_configs file enumerates rows, the SBATCH script picks one per array task.
