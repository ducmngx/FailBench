# **FailBench**: Simulating Robot Failures in MuJoCo

**FailBench** is a MuJoCo-based simulation framework for studying Franka Panda robot behavior during sudden hardware failures. It generates labeled datasets of contact patterns, RGB/depth images, and robot state across multiple failure modes and tasks.

**Research goal**: Train a planner that chooses safer robot configurations by learning which pre-failure joint configs lead to worse outcomes. The dataset covers a wide range of pre-failure configurations across different arm poses, carry directions, and mission phases.

---

## 🔧 Installation

```bash
conda env create -f environment.yml
conda activate failbench_env
```

Key dependencies: MuJoCo 3.3.4, Python 3.10, Mink 0.0.11 (IK), PyTorch, OpenCV.

---

## 🧭 Pipeline Overview

### 1. Generate trajectories

```bash
# One task, 10 trajectories
python scripts/generate_task_trajs.py --scene scene_level2 --task clean_nominal --n_trajs 10

# All 12 tasks at once
python scripts/generate_task_trajs.py --scene scene_level2 --task all --n_trajs 10
```

Output: `scenes/<scene>/trajs/<scene>_<task>_NN.pkl`

### 2. Visually verify

```bash
python scripts/play_task_trajs.py --scene scene_level2 --task clean_nominal
```

### 3. Run smoke test

```bash
python scripts/test_pipeline.py
```

### 4. Generate failure dataset (TODO)

`scripts/generate_dataset_tasks.py` — iterates over all tasks × trajectories × failure fractions, runs ExperimentRunner, writes `.npz` + `manifest.csv`.

---

## 🗂️ Task Taxonomy

Tasks are defined in per-scene YAML files (`scenes/<scene>/tasks.yaml`). Adding new tasks requires only editing YAML — no Python changes needed.

**Semantic categories** describe the real-world action:

| Category | Description |
|---|---|
| `stack` | Pick object, place on top of another object |
| `clean` | Pick object, carry to a designated drop zone |
| `sort` | Pick alternate object, deliver to a specific zone |
| `handover` | Pick object, carry to table edge for handover |

**Geometric subtasks** describe the configuration challenge:

| Suffix | What it tests |
|---|---|
| `_nominal` | Short carry, clear space |
| `_far` | Long carry, arm extended at delivery |
| `_over` | Goal on far side of obstacle field, requires cross-table traverse |

**scene_level2** has 12 tasks: `stack_nominal`, `stack_far`, `stack_over`, `clean_nominal`, `clean_far`, `clean_over`, `sort_nominal`, `sort_far`, `sort_over`, `handover_nominal`, `handover_far`, `handover_over`.

---

## 🤖 Motion Planner

Each trajectory is planned using:

- **RRT** (Rapidly-Exploring Random Tree) for path planning
- **IK** via [Mink](https://github.com/kevinzakka/mink)
- **MuJoCo collision checking**

Trajectories have 5 segments: `approach → descend/grasp → lift → transport → place/release`. Heights (carry height, lift height, place height) are derived at runtime from the scene model — no hardcoded per-scene values. Approach direction is sampled on a hemisphere around the object to maximise pre-failure joint configuration diversity.

![Demo](docs/media/mujoco_arm_planner_demo.gif)

---

## 💥 Failure Modes

Injected at a configurable point along the trajectory (`traj_progress ∈ [0,1]`):

| Mode | Description |
|---|---|
| `GRIPPER_OPEN` | Gripper fully opens, drops object |
| `SLIPPERY_GRIP` | Partial gripper closure, reduced grip |
| `SINGLE_JOINT` | One arm joint frozen |
| `MULTI_JOINT` | Multiple joints frozen |
| `ALL_JOINTS` | All arm joints frozen |

Canonical failure fractions: `[0.1, 0.25, 0.4, 0.55, 0.7, 0.85]`

---

## 📦 Dependencies

- [MuJoCo 3.3.4](https://mujoco.org/)
- Python 3.10

All other packages are installed via `environment.yml`.


---

## 🙏 Acknowledgments

This project is inspired by and builds upon the excellent work of existing simulation platforms such as:

- [RoboCasa](https://robocasa.ai/)
- [Robosuite](https://robosuite.ai/)

We extend their design philosophies and modular stacks to focus specifically on simulating and understanding robotic failures.


---

## 📚 Citation

If you use this project or find it helpful, please consider citing the foundational work we build upon:

```bibtex
@inproceedings{[FAILBENCH2025],
  title={TBA},
  author={Duc M. Nguyen, Saad Ghani, Andrew Marshall, Allison Andreyev, Gregory J. Stein and Xuesu Xiao},
  booktitle={TBA},
  year={2025}
}
```
---
## 📄 License

This project is licensed under the [MIT License](LICENSE).

