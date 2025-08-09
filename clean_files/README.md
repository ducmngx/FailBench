# FailBench [Merge Files] Documentation

Builds a complete MuJoCo XML by composing robots and objects from per-entity XMLs, driven by a YAML config. Also includes utilities to load environments, generate collision maps, and programmatically build models.

## Repo structure

```
.
├── config.yaml                  # Scene recipe used by merge_xml.py
├── merge_xml.py                 # XML composer and pretty-printer
├── load_env.py                  # Loader and runner for generated envs
├── model_builder.py             # Programmatic MJCF construction helpers
├── collision_map_generator.py   # 3D scene to 2D obstacle grid mapper
└── assets/                      # Meshes, textures, materials
```

## What each file does

### `merge_xml.py`

Composes a full MuJoCo model from modular XML parts described in `config.yaml`, then writes `output.xml`.

Key capabilities:

* Parses an environment include via `<include file=...>` and inserts it under the root.
* Locates a target `<body name="...">` in each entity XML, deep-copies it, and applies optional `pos` and `quat` from YAML.
* Recursively attaches child bodies for hierarchical assemblies.
* Collects and de-duplicates global sections: `<asset>`, `<default>`, `<sensor>`, `<tendon>`, `<equality>`, `<actuator>`.
* Orders sections and pretty-prints the final XML with consistent spacing and blank lines between sibling bodies.

Important functions:

* `find_body_by_name(xml_root, target_name)` returns a deep copy of a matching `<body>`. If no name is given, falls back to the first `<worldbody>/<body>`.
* `collect_assets|collect_defaults|collect_tendons|collect_equalities(...)` gather global tags from an entity XML.
* `update_body_pose(body_elem, yaml_entity)` sets `pos` and `quat` from YAML if present.
* `process_entity(entity)` returns the `<body>` subtree plus all collected globals for that entity and its nested children.
* `pretty_print_xml(elem)` normalizes indentation, inserts newlines, and ensures consistent block spacing.

Output:

* Writes `output.xml` in the project root.

### `load_env.py`

Utilities to load and run the generated model.

* Loads `output.xml` or a given MJCF path into MuJoCo or a wrapper such as mujoco-python or robosuite.
* Can be used to sanity check model validity, visualize the scene, and verify that assets resolve.

Typical usage:

```bash
python load_env.py --model output.xml
```

### `model_builder.py`

Programmatic model construction helpers.

* Builds MJCF elements in Python without starting from a hand-authored XML.
* Useful for templating robots and scene objects, generating parametrized variants, and emitting partial XMLs that you can later compose with `merge_xml.py`.

Typical usage:

```bash
python model_builder.py --out entity.xml --name cube --size 0.05
```

### `collision_map_generator.py`

Generates a 2D collision or occupancy grid from a 3D MuJoCo scene.

* Parses `output.xml`, reads body and geom placement, and rasterizes into a grid for planning.
* Supports exporting a matrix representation suitable for A\*, Dijkstra, BFS, or RRT.

Typical usage:

```bash
python collision_map_generator.py --model output.xml --res 0.02 --out obstacles.npy
```

## `config.yaml` schema

Minimal example:

```yaml
model_name: kitchen_scene
env_path: envs/base_env.xml

robot:
  name: panda
  xml_path: robots/franka_panda.xml
  pos: "0 0 0"
  quat: "1 0 0 0"
  bodies:
    - name: gripper
      xml_path: robots/panda_gripper.xml

Objects:
  - name: table
    xml_path: objects/table.xml
    pos: "0.8 0.0 0.0"
    quat: "1 0 0 0"

  - name: mug
    xml_path: objects/mug.xml
    pos: "0.7 0.1 0.75"
```

Fields:

* `model_name`: Name applied to the root `<mujoco>` tag.
* `env_path`: Base environment XML that gets included under the root. Usually defines compiler defaults and world settings.
* `robot`: One entity describing the robot assembly. Supports nested `bodies` for multi-part robots.
* `Objects`: List of scene objects. Each entry:

  * `name`: The `<body name="...">` to extract from the source XML.
  * `xml_path`: Path to the source MJCF containing that body.
  * `pos` and `quat` optional overrides for placement.
  * `bodies`: Optional nested entities that will be appended as children of this body.

Notes:

* `merge_xml.py` searches for `<body name="...">`. If `name` is omitted, it will take the first `<worldbody>/<body>` in that XML.
* Global sections from all entities are merged. Assets are de-duplicated by serialized element text.

## How the build works

1. Read `config.yaml`.
2. Create the root `<mujoco model=...>` and add:

   * `<compiler meshdir="assets">`
   * `<include file=env_path>`
3. For each entity in `robot` and `Objects`:

   * Parse the entity XML.
   * Extract the target body subtree, apply `pos` and `quat`, and recurse into `bodies`.
   * Accumulate global sections for assets, defaults, sensors, tendons, equalities, actuators.
4. Insert merged `<default>`, `<asset>`, and `<worldbody>` in that order, then append `<sensor>`, `<tendon>`, `<equality>`, `<actuator>` if present.
5. Pretty-print and write `output.xml`.

## Usage

Generate the model:

```bash
python merge_xml.py
```

Load and inspect:

```bash
python load_env.py --model output.xml
```

Create a collision map:

```bash
python collision_map_generator.py --model output.xml --res 0.02 --out obstacles.npy
```

Programmatically build an entity, then compose:

```bash
python model_builder.py --out objects/custom_block.xml --name block --size 0.1
# add it to config.yaml under Objects, then:
python merge_xml.py
```

## Troubleshooting

* Ensure `assets/` contains all meshes and textures referenced by any entity XML. The root compiler sets `meshdir="assets"`.
* If an entity XML lacks a matching `<body name="...">`, the script will raise an error. Check `name` and source XML.
* Defaults handling: only nested `<default>` children under the top-level `<default>` are collected.
* Asset de-duplication uses raw XML string equality. If two assets differ in attribute order or whitespace, they will be treated as distinct. Normalize upstream if needed.
