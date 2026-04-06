# Scenes

Each subdirectory is a self-contained scene environment for data generation and training.

## Directory structure

```
scenes/
  <scene_name>/
    scene.xml          # MuJoCo scene XML (includes panda.xml via relative path)
    trajs/             # Precomputed trajectories (.pkl files)
    datasets/          # Generated experiment datasets (v1/, v2/, etc.)
      <version>/
        manifest.csv
        exp_*.npz
```

## Adding a new scene

1. Create `scenes/<scene_name>/`
2. Write `scene.xml` with the desired object layout (must include `../../franka_emika_panda/panda.xml`)
3. Generate trajectories (RRT/STOMP) and save to `trajs/`
4. Run data generation targeting `datasets/<version>/`

## Existing scenes

- **scene_level2** — 3x5 grid of soft/hard obstacles + 2 target objects on a table. 4 trajectories, dataset v6 (20 experiments, RGBD + EE camera).
