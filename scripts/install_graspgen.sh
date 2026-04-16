#!/usr/bin/env bash
# Installs GraspGen into an isolated UV venv at external/GraspGen/.venv.
# Does NOT touch the failbench_env conda environment.
#
# Usage:  conda deactivate && bash scripts/install_graspgen.sh
set -euo pipefail

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "ERROR: conda env '$CONDA_DEFAULT_ENV' is active. Run 'conda deactivate' first." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found on PATH. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRASPGEN_DIR="$REPO_ROOT/external/GraspGen"

if [[ ! -f "$GRASPGEN_DIR/pyproject.toml" && ! -f "$GRASPGEN_DIR/setup.py" ]]; then
  echo "ERROR: $GRASPGEN_DIR does not look like a GraspGen checkout." >&2
  echo "Run:  git submodule update --init external/GraspGen" >&2
  exit 1
fi

cd "$GRASPGEN_DIR"

uv python install 3.10
uv venv --python 3.10 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install torch-cluster torch-scatter \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
uv pip install -e .

# pointnet2_ops needs nvcc with a CUDA major version matching torch (12.x).
# If system nvcc is older (e.g. Ubuntu ships 10.1 at /usr/bin/nvcc), look for a
# newer CUDA toolkit under /usr/local and point CUDA_HOME at it.
if command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -qE "release 1[2-9]\."; then
  :  # system nvcc is fine
else
  NEW_CUDA=$(ls -d /usr/local/cuda-1[2-9]* 2>/dev/null | sort -V | tail -1 || true)
  if [[ -n "$NEW_CUDA" && -x "$NEW_CUDA/bin/nvcc" ]]; then
    echo "System nvcc is too old — using $NEW_CUDA for pointnet2_ops build."
    export CUDA_HOME="$NEW_CUDA"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  else
    echo "WARNING: no CUDA 12.x toolkit found. pointnet2_ops build will likely fail." >&2
    echo "Install one with:  sudo apt install cuda-toolkit-12-1" >&2
  fi
fi

if [[ -x ./install_uv_pointnet.sh ]]; then
  ./install_uv_pointnet.sh
elif [[ -x ./install_pointnet.sh ]]; then
  ./install_pointnet.sh
else
  echo "WARNING: no pointnet install script found — point cloud backbone may be unavailable." >&2
fi

echo ""
echo "GraspGen installed at: $GRASPGEN_DIR/.venv"
echo "Smoke test:"
echo "  $GRASPGEN_DIR/.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo ""
echo "Next: download model checkpoints with  bash scripts/download_graspgen_models.sh"
