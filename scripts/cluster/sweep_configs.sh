#!/usr/bin/env bash
# Defines the headline sweep: 3 seeds × 5 model configs × 3 splits = 45 runs.
#
# Source this file (don't execute) to get the SWEEP_CONFIGS bash array:
#   source scripts/cluster/sweep_configs.sh
#   echo "${#SWEEP_CONFIGS[@]}"            # number of runs
#   echo "${SWEEP_CONFIGS[0]}"             # first row
#
# Each row is: MODEL MODALITIES T SPLIT SEED EXTRA_FLAGS
# EXTRA_FLAGS may be empty or contain trainer-specific flags (e.g. unet_temporal).
# The SBATCH script (sweep_headline.sbatch) reads one row per array task.
#
# Run this script directly to print the sweep as a table for inspection:
#   bash scripts/cluster/sweep_configs.sh

SWEEP_CONFIGS=()

# 5 model configs to compare across the paper headline:
#   1. ConvDec state-only T=1            — kinematic-only leader (run #9 locally)
#   2. UNet late_fusion state+rgb+depth  — in-dist leader (run #6 locally)
#   3. UNet state+rgb+depth+failure_mode+failure_joints (oracle)
#                                         — full-info upper bound (run #15)
#   4. Transformer state+rgb+depth T=8   — sequence-model baseline (run #22)
#   5. Transformer rgb+depth T=8 (true vision-only) — vision-only (run #24)

MODEL_CONFIGS=(
  "convdec state                                             1 "
  "unet    state,rgb,depth                                   8 --unet_temporal late_fusion"
  "unet    state,rgb,depth,failure_mode,failure_joints       8 --unet_temporal late_fusion"
  "transformer state,rgb,depth                               8 "
  "transformer rgb,depth                                     8 "
)

SPLITS=(libero_spatial libero_object libero_goal)
SEEDS=(0 1 2)

for CFG in "${MODEL_CONFIGS[@]}"; do
  for SPLIT in "${SPLITS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      SWEEP_CONFIGS+=("${CFG} ${SPLIT} ${SEED}")
    done
  done
done

# Pretty-print when invoked directly.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  printf "%-4s  %-12s  %-50s  %-3s  %-15s  %-4s  %s\n" \
    "idx" "model" "modalities" "T" "split" "seed" "extra"
  printf "%-4s  %-12s  %-50s  %-3s  %-15s  %-4s  %s\n" \
    "----" "-----" "----------" "-" "-----" "----" "-----"
  i=0
  for ROW in "${SWEEP_CONFIGS[@]}"; do
    # shellcheck disable=SC2086
    set -- ${ROW}
    MODEL=$1; MOD=$2; T=$3
    shift 3
    # The remaining args contain optional --unet_temporal X, then SPLIT, SEED.
    # We parse from the right: last two tokens are SPLIT and SEED.
    ALLREST=("$@")
    SEED=${ALLREST[-1]}
    SPLIT=${ALLREST[-2]}
    EXTRA="${ALLREST[*]:0:${#ALLREST[@]}-2}"
    printf "%-4d  %-12s  %-50s  %-3d  %-15s  %-4d  %s\n" \
      "$i" "$MODEL" "$MOD" "$T" "$SPLIT" "$SEED" "$EXTRA"
    i=$((i+1))
  done
  echo
  echo "Total runs: ${#SWEEP_CONFIGS[@]}"
fi
