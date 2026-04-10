# External Assets Registry

Third-party MuJoCo MJCF assets used in FailBench scenes. Each repository is cloned into this directory via `setup.sh`.

## Repositories

### vikashplus/object_sim
- **URL**: https://github.com/vikashplus/object_sim
- **License**: Apache-2.0
- **Commit**: *(pinned by setup.sh)*
- **Contents**: 60+ household objects (mugs, bottles, bowls, bananas, cubes, tools, etc.)
- **Used in**: scene_kitchen_counter, scene_cluttered_desk
- **Objects used**:
  - `mug_beer` — grasped object in kitchen scene
  - `bottle_beer`, `bottle_wine` — tall obstacles
  - `banana` — irregular-shape obstacle

### vikashplus/furniture_sim
- **URL**: https://github.com/vikashplus/furniture_sim
- **License**: Apache-2.0
- **Commit**: *(pinned by setup.sh)*
- **Contents**: Tables, bins, counters, cabinets, appliances
- **Used in**: scene_kitchen_counter, scene_sparse_bin
- **Objects used**:
  - `counter` — kitchen counter surface
  - `bin` — constrained workspace container

### vikashplus/YCB_sim
- **URL**: https://github.com/vikashplus/YCB_sim
- **License**: Apache-2.0
- **Commit**: *(pinned by setup.sh)*
- **Contents**: YCB benchmark manipulation objects
- **Used in**: scene_kitchen_counter, scene_cluttered_desk, scene_sparse_bin
- **Objects used**:
  - `003_cracker_box` — large flat obstacle
  - `005_tomato_soup_can` — cylindrical grasped object
  - `006_mustard_bottle` — tall obstacle
  - `010_potted_meat_can` — short obstacle
  - `024_bowl` — wide low obstacle

### kevinzakka/mujoco_scanned_objects
- **URL**: https://github.com/kevinzakka/mujoco_scanned_objects
- **License**: MIT (code), CC-BY-4.0 (meshes)
- **Commit**: *(pinned by setup.sh)*
- **Contents**: 1030 photogrammetry-scanned household objects with textures
- **Used in**: scene_cluttered_desk (selective download, 4 objects only)
- **Note**: Large repo. Only specific objects are downloaded, not the full dataset.

## Setup

```bash
cd external_assets
bash setup.sh
```

## Adding new assets

1. Add the repository to `setup.sh` with a pinned commit
2. Document it in this README with: URL, license, commit, objects used, scenes
3. Reference assets from scene XMLs via relative paths: `../../external_assets/<repo>/<object>/`
