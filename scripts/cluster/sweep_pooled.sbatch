#!/bin/bash
#
# Pooled FailBench headline sweep for the contact-prediction paper.
# 5 model configs × 3 seeds = 15 runs. Each run trains ONE model on ALL three
# LIBERO suites pooled (libero_spatial + libero_object + libero_goal), instead of
# the per-suite sweep_headline.sbatch (which trains one model per suite = 45 runs).
#
# Why pooled: a single model gives a clean headline number + one coherent set of
# qualitative panels, and pure seed-variance error bars (vs the per-suite sweep,
# whose headline conflates suite-difficulty with seed variance). Per-suite numbers
# are recovered at eval time via eval_all.py's "Per-suite breakdown".
#
# Submit with:   sbatch scripts/cluster/sweep_pooled.sbatch
# Track with:    squeue -u $USER
# Eval with:     python -m scripts.benchmark.eval_all --runs_root runs/bench_pooled
#
# Each array task picks one row from POOLED_CONFIGS in scripts/cluster/sweep_configs.sh.

#SBATCH --job-name=fb-pooled
#SBATCH --partition=gpuq                # Hopper main GPU partition (5-day timelimit)
#SBATCH --qos=gpu                       # required for any GPU job on Hopper
#SBATCH --array=0-14%8                  # 15 runs, max 8 concurrent
#SBATCH --gres=gpu:A100.80gb:1          # 1 full A100 80GB per task
#SBATCH --cpus-per-task=8               # match --num_workers below
#SBATCH --mem=96G                       # per task — pooled set is ~3x trials vs per-suite
#SBATCH --time=04:00:00                 # ~3x trials/epoch over 30 epochs ≈ 3x wall vs per-suite
#SBATCH --output=logs/sweep-%x-%A_%a.out
#SBATCH --error=logs/sweep-%x-%A_%a.err
#SBATCH --requeue                       # auto-retry on transient node failure

# ---- GPU type alternatives ------------------------------------------------
# Switch the --gres line above if you want a different GPU type:
#    --gres=gpu:A100.40gb:1     (DGX A100s — typically less queued)
#    --gres=gpu:H100.80gb:1     (single H100 node, contrib-H100 partition)
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- Cluster environment --------------------------------------------------

module purge
module load gnu10
module load python/3.9.9-jh             # cluster default Python; works for training-only

# Activate the venv (no conda needed — training never imports MuJoCo).
source "$HOME/failbench_env/bin/activate"

# Repo + data paths.
export PYTHONPATH="$HOME/FailBench:${PYTHONPATH:-}"
export FAILBENCH_V2_ROOT="$SCRATCH/data/failbench_data/libero/v2"

cd "$HOME/FailBench"
mkdir -p logs runs/bench_pooled

# ---- Resolve this array task's config -------------------------------------

# Loads POOLED_CONFIGS bash array.
# shellcheck disable=SC1091
source scripts/cluster/sweep_configs.sh

N=${#POOLED_CONFIGS[@]}
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N )); then
  echo "Array task $SLURM_ARRAY_TASK_ID out of range [0, $((N-1))]"
  exit 1
fi

ROW="${POOLED_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086
set -- $ROW
MODEL=$1; MOD=$2; T=$3
shift 3

# Remaining tokens: optional EXTRA flags, then SEED (last token — NO split token
# in pooled rows, unlike the per-suite sweep where SPLIT and SEED are the last two).
ALL_REST=("$@")
N_REST=${#ALL_REST[@]}
SEED=${ALL_REST[$((N_REST-1))]}
EXTRA=("${ALL_REST[@]:0:$((N_REST-1))}")

# Immutable, pooled-specific output dir (array id, not timestamp → no overwrite on
# requeue, and never confused with the 45 per-suite dirs under runs/bench/).
MOD_TAG=${MOD//,/+}
OUT="runs/bench_pooled/${MODEL}__${MOD_TAG}__T${T}__pertrial__splitD__pooled__seed${SEED}__${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "=========================================================="
echo "FailBench POOLED sweep task $SLURM_ARRAY_TASK_ID / $((N-1))"
echo "  model       = $MODEL"
echo "  modalities  = $MOD"
echo "  T           = $T"
echo "  splits      = libero_spatial libero_object libero_goal (POOLED)"
echo "  seed        = $SEED"
echo "  extra flags = ${EXTRA[*]}"
echo "  output_dir  = $OUT"
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
  --splits libero_spatial libero_object libero_goal \
  --seed "$SEED" \
  --target_form per_trial \
  --split_by demo \
  --epochs 30 \
  --batch_size 64 \
  --num_workers 8 \
  --warmup_epochs 2 \
  --patience 5 \
  --v2_root "$FAILBENCH_V2_ROOT" \
  --output_dir "$OUT" \
  "${EXTRA[@]}"

echo "task $SLURM_ARRAY_TASK_ID done"
