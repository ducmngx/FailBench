# GraspGen setup

GraspGen (NVLabs) is installed as a git submodule at `external/GraspGen` with its own isolated UV venv. It does **not** share packages with the `failbench_env` conda environment — the two coexist safely.

## Requirements

- NVIDIA GPU with driver ≥ 525 (CUDA 12.1 compatible). Tested on RTX 3070 (8 GB), driver 535.
- Python 3.10 (installed automatically by `uv`).
- `uv` on PATH: install with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- Internet access to fetch PyTorch wheels and HuggingFace checkpoints.

## Install

```bash
# 1. Fetch the submodule (first time only)
git submodule update --init external/GraspGen

# 2. Deactivate any active conda env, then install
conda deactivate          # repeat until $CONDA_DEFAULT_ENV is empty
bash scripts/install_graspgen.sh

# 3. Download model checkpoints (~hundreds of MB, gitignored)
bash scripts/download_graspgen_models.sh
```

The installer refuses to run while a conda env is active — this is the primary guard against polluting `failbench_env`.

## Verify isolation

```bash
# GraspGen venv — torch 2.1.0, CUDA available
external/GraspGen/.venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"

# failbench_env — whatever torch version it had before, unchanged
conda activate failbench_env
python -c "import torch; print(torch.__version__)"
conda deactivate
```

## Run inference (sanity check)

```bash
external/GraspGen/.venv/bin/python \
  external/GraspGen/scripts/demo_object_mesh.py \
  --mesh_file scenes/scene_kitchen/assets/cubesmall.stl \
  --mesh_scale 1.0 \
  --gripper_config external/GraspGen/checkpoints/checkpoints/graspgen_franka_panda.yml \
  --num_grasps 10 \
  --output_file /tmp/cubesmall_grasps.yml \
  --no-visualization
```

Expected: `/tmp/cubesmall_grasps.yml` with 10 ranked 6-DoF grasp poses.

## Troubleshooting

- **`install_uv_pointnet.sh` fails with nvcc errors**: install CUDA 12.1 toolkit system-wide (`sudo apt install cuda-toolkit-12-1`) or re-run after verifying `nvcc --version` matches PyTorch's bundled CUDA.
- **OOM during inference on 8 GB GPU**: drop `--num_grasps` from 50 to 25.
- **`uv` not on PATH**: add `source $HOME/.local/bin/env` to `~/.bashrc`.
- **HuggingFace download is slow / hangs**: `huggingface-cli login` if your network requires auth, or set `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads.

## Rollback

```bash
git submodule deinit -f external/GraspGen
git rm -f external/GraspGen
rm -rf .git/modules/external/GraspGen external/GraspGen
```

`failbench_env` is untouched throughout.

## License

GraspGen is distributed under the NVIDIA Research License. The submodule contains a pointer only (not a redistribution). Confirm acceptable use with your institution before shipping publicly.
