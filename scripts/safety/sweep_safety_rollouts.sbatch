#!/bin/bash
#SBATCH --job-name=safety_rollouts
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/safety_rollouts_%j.out
#SBATCH --error=logs/safety_rollouts_%j.err

# Safety-rollout cluster sweep: 3 LIBERO tasks × 20 inits × 5 failure modes
# × 6 fail_progresses × 3 policies ≈ 5,400 rollouts.
#
# Wall-clock estimate: baseline+scaling at ~6 s/rollout, search at ~26 s.
# 3 × 20 × 5 × 6 × (6 + 6 + 26) = 68,400 s ≈ 19 h with all three policies.
# Cutting to baseline+scaling only: ~2,160 rollouts × 6 s = 3.6 h.
#
# Required on the cluster:
#  - external/LIBERO/.venv with libero, robosuite==1.4.0, bddl==1.0.1,
#    future, hydra-core, easydict, einops, gym, cloudpickle, matplotlib,
#    opencv-python, h5py, hdf5plugin, torch (cu121 build).
#  - The cluster_download checkpoint at
#    $SCRATCH/failbench/runs/06172026/dualgated_state_rgb/best_ep08_val0.0648.pt
#    (or update CKPT below).
#  - Raw LIBERO demos at $SCRATCH/failbench/libero/raw/<split>/<task>_demo.hdf5.
#
# Submit::
#
#   sbatch scripts/safety/sweep_safety_rollouts.sbatch
#
# Then watch::
#
#   tail -F logs/safety_rollouts_*.out

set -euo pipefail
mkdir -p logs

# --- env ---
module load gnu10 || true
module load python || true

cd "${SLURM_SUBMIT_DIR:-$HOME/FailBench}"

# LIBERO venv (built per scripts/libero/setup notes — robosuite + libero
# editable). The cluster install must include all of LIBERO's
# requirements.txt plus h5py/hdf5plugin/torch on top.
source external/LIBERO/.venv/bin/activate

# Make the in-repo libero package importable.
export PYTHONPATH="$PWD/external/LIBERO:$PWD:${PYTHONPATH:-}"

# Headless rendering. EGL on Hopper compute nodes works per CLAUDE.md note.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# HDF5 multiprocess
export HDF5_USE_FILE_LOCKING=FALSE

# --- paths ---
CKPT="${CKPT:-$SCRATCH/failbench/runs/06172026/dualgated_state_rgb/best_ep08_val0.0648.pt}"
DEMO_ROOT="${DEMO_ROOT:-$SCRATCH/failbench/libero/raw}"
LIBERO_ROOT="${LIBERO_ROOT:-external/LIBERO}"
OUT_ROOT="${OUT_ROOT:-$SCRATCH/failbench/safety_rollouts}"

# Pick three LIBERO tasks that exercise different geometric regimes.
# Replace these names if the cluster demo set differs — `find $DEMO_ROOT -name '*_demo.hdf5'`
# to list available tasks.
TASKS="pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate,pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate,pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"

# Drop --policies search if you want to skip the slowest mechanism (recommended
# until σ is tuned; current σ=0.02 is degenerate per the smoke).
POLICIES="${POLICIES:-baseline,scaling}"

python -u -m scripts.safety.safety_rollout \
    --ckpt "$CKPT" \
    --libero_root "$LIBERO_ROOT" \
    --demo_root  "$DEMO_ROOT" \
    --tasks "$TASKS" \
    --n_inits 20 \
    --modes GRIPPER_OPEN,SLIPPERY_GRIP,SINGLE_JOINT,MULTI_JOINT,ALL_JOINTS \
    --progresses 0.1,0.25,0.4,0.55,0.7,0.85 \
    --policies "$POLICIES" \
    --out_root "$OUT_ROOT" \
    --scaling_tau 0.0 \
    --scaling_alpha 0.3 \
    --search_sigma 0.05 \
    --search_n 6

echo "All tasks complete."
echo "Results under: $OUT_ROOT/<task>/results.csv"
