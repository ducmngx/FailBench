#!/bin/bash
#
# Headline FailBench sweep for the contact-prediction paper.
# 3 seeds × 5 model configs × 3 splits = 45 runs.
#
# Submit with:   sbatch scripts/cluster/sweep_headline.sbatch
# Track with:    squeue -u $USER
#
# Each array task picks one row from scripts/cluster/sweep_configs.sh and
# runs scripts.benchmark.train_one with the corresponding flags.

#SBATCH --job-name=fb-headline
#SBATCH --partition=gpuq                # Hopper main GPU partition (5-day timelimit)
#SBATCH --qos=gpu                       # required for any GPU job on Hopper
#SBATCH --array=0-44%8                  # 45 jobs, max 8 concurrent — adjust if cluster lets you run more
#SBATCH --gres=gpu:A100.80gb:1          # 1 full A100 80GB per task. See "GPU type alternatives" below
#SBATCH --cpus-per-task=8               # match --num_workers below
#SBATCH --mem=64G                       # per task
#SBATCH --time=02:00:00                 # most runs are <90 min; raise if your sweep gets bigger models
#SBATCH --output=logs/sweep-%x-%A_%a.out
#SBATCH --error=logs/sweep-%x-%A_%a.err
#SBATCH --requeue                       # auto-retry on transient node failure

# ---- GPU type alternatives ------------------------------------------------
# Switch the --gres line above if you want a different GPU type:
#    --gres=gpu:A100.40gb:1     (DGX A100s — typically less queued)
#    --gres=gpu:H100.80gb:1     (single H100 node, contrib-H100 partition)
# MIG slices (1g.10gb, 2g.20gb, 3g.40gb) are too small for our models — skip.
#
# If your group has contributor access:
#    #SBATCH --partition=contrib-gpuq
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- Cluster environment --------------------------------------------------

module purge
module load gnu10
module load python/3.9.9-jh             # cluster default Python; works for training-only

# Activate the venv (no conda needed — training never imports MuJoCo).
source "$HOME/failbench_env/bin/activate"

# Repo + data paths.
export PYTHONPATH="$HOME/FailBench:$PYTHONPATH"
export FAILBENCH_V2_ROOT="$SCRATCH/data/failbench_data/libero/v2"

cd "$HOME/FailBench"
mkdir -p logs runs/bench

# ---- Resolve this array task's config -------------------------------------

# Loads SWEEP_CONFIGS bash array.
# shellcheck disable=SC1091
source scripts/cluster/sweep_configs.sh

N=${#SWEEP_CONFIGS[@]}
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N )); then
  echo "Array task $SLURM_ARRAY_TASK_ID out of range [0, $((N-1))]"
  exit 1
fi

ROW="${SWEEP_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086
set -- $ROW
MODEL=$1; MOD=$2; T=$3
shift 3

# Remaining tokens: optional EXTRA flags, then SPLIT, SEED (last two).
ALL_REST=("$@")
N_REST=${#ALL_REST[@]}
SEED=${ALL_REST[$((N_REST-1))]}
SPLIT=${ALL_REST[$((N_REST-2))]}
EXTRA=("${ALL_REST[@]:0:$((N_REST-2))}")

echo "=========================================================="
echo "FailBench sweep task $SLURM_ARRAY_TASK_ID / $((N-1))"
echo "  model       = $MODEL"
echo "  modalities  = $MOD"
echo "  T           = $T"
echo "  split       = $SPLIT"
echo "  seed        = $SEED"
echo "  extra flags = ${EXTRA[*]}"
echo "  host        = $(hostname)"
echo "  gpu         = $CUDA_VISIBLE_DEVICES"
echo "  v2 root     = $FAILBENCH_V2_ROOT"
echo "=========================================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- Run the trainer ------------------------------------------------------

python -u -m scripts.benchmark.train_one \
  --model "$MODEL" \
  --modalities "$MOD" \
  --T "$T" \
  --splits "$SPLIT" \
  --seed "$SEED" \
  --target_form per_trial \
  --split_by demo \
  --epochs 30 \
  --batch_size 64 \
  --num_workers 8 \
  --warmup_epochs 2 \
  --patience 5 \
  --v2_root "$FAILBENCH_V2_ROOT" \
  "${EXTRA[@]}"

echo "task $SLURM_ARRAY_TASK_ID done"
