#!/usr/bin/env bash
# Downloads GraspGen model checkpoints from HuggingFace into
# external/GraspGen/checkpoints (gitignored).
#
# Usage:  bash scripts/download_graspgen_models.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="$REPO_ROOT/external/GraspGen/checkpoints"
VENV_PY="$REPO_ROOT/external/GraspGen/.venv/bin/python"

mkdir -p "$CKPT_DIR"

# Prefer the isolated venv's huggingface-cli so we don't force a global install.
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -m pip show huggingface_hub >/dev/null 2>&1; then
  HF_CMD=("$VENV_PY" -m huggingface_hub.commands.huggingface_cli)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CMD=(huggingface-cli)
else
  if [[ -x "$VENV_PY" ]]; then
    echo "Installing huggingface_hub into GraspGen venv..."
    "$VENV_PY" -m pip install -q "huggingface_hub[cli]"
    HF_CMD=("$VENV_PY" -m huggingface_hub.commands.huggingface_cli)
  else
    echo "ERROR: neither huggingface-cli nor GraspGen venv available." >&2
    echo "Run  bash scripts/install_graspgen.sh  first." >&2
    exit 1
  fi
fi

"${HF_CMD[@]}" download adithyamurali/GraspGenModels --local-dir "$CKPT_DIR"

echo ""
echo "Checkpoints downloaded to: $CKPT_DIR"
echo "Franka config: $CKPT_DIR/checkpoints/graspgen_franka_panda.yml"
