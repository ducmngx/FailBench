# GenAISim

# MuJoCo Installation and Running Guide (Ubuntu, Conda)

This guide helps you install **MuJoCo** and **mujoco-py** on Ubuntu using a Conda environment.

---

## 1. Prerequisites

- Ubuntu 18.04/20.04/22.04
- Anaconda/Miniconda installed

---

## 2. Create and Activate Conda Environment

conda create -n mujoco_env python=3.8
conda activate mujoco_env

---

## 3. Install System Dependencies

sudo apt update
sudo apt install git patchelf libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev python3-pip

---

## 4. Download and Extract MuJoCo

1. Download MuJoCo 2.1.0 (or latest; tested with 2.1.0) from [mujoco.org/download](https://mujoco.org/download).
2. Extract to your home directory:

mkdir -p ~/.mujoco
tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/

---

## 5. Set Environment Variables

Add the following lines to your `~/.bashrc` (replace `<username>` with your username):

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/.mujoco/mujoco210/bin
export MUJOCO_PY_MUJOCO_PATH=/home/<username>/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so


Then reload your bash settings:

source ~/.bashrc

---

## 6. Install Python Dependencies

conda install -c conda-forge glfw glew
pip install mujoco-py

---

## 7. (Optional) Install Additional Packages

sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

---

## 8. Test Your Installation

Try importing mujoco-py in Python:

python -c "import mujoco_py"

Or run an example (if available):

python test_mujoco.py

---

## 9. Troubleshooting

- Ensure all environment variables are set correctly and `~/.mujoco/mujoco210` exists.
- If you see library errors, check that `LD_LIBRARY_PATH` includes the MuJoCo bin directory.
- For GPU rendering, additional NVIDIA libraries may be required.

---

**References:**  
- [MuJoCo official docs](https://mujoco.org/docs/)
- [Community installation guides](https://github.com/openai/mujoco-py)

---
