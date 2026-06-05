"""RoboCasa sibling adapter that produces LiberoDemo-shaped objects for LiberoRunner.

LIBERO and RoboCasa share the robomimic HDF5 schema (`model_file` attr,
`states` per-step) and the robosuite Panda naming convention, so `run_v2()`
consumes both with no further changes once this adapter normalises:

- the per-author MJCF asset paths (three known author machines so far);
- the PandaMobile arm slice (state vector has base + arm + fingers + objects);
- the manipulated-object allowlist read from `ep_meta["object_cfgs"]`
  (RoboCasa kitchens have hundreds of non-robot bodies otherwise).

See `docs/robocasa_integration.md` for the full integration plan.
"""

from planner.experiments.robocasa.adapter import (
    load_demo,
    materialise_mjcf,
    list_demos,
    read_ep_meta,
    object_allowlist_from_ep_meta,
)
from planner.experiments.robocasa.scene import build_scene_overrides

__all__ = [
    "load_demo",
    "materialise_mjcf",
    "list_demos",
    "read_ep_meta",
    "object_allowlist_from_ep_meta",
    "build_scene_overrides",
]
