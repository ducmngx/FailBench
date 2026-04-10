"""Scene randomization: sample new object poses for each trial."""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np


@dataclass
class ObjectPlacementConfig:
    """Placement bounds for a single free-joint object body."""
    body_name: str
    x_range: Tuple[float, float]  # (min, max) in world frame
    y_range: Tuple[float, float]  # (min, max) in world frame
    z: float                      # fixed world z for object center
    yaw_range: Tuple[float, float] = (-math.pi, math.pi)  # (min, max)


class SceneRandomizer:
    """Randomly re-place free-joint objects on the table every trial.

    Usage
    -----
    randomizer = SceneRandomizer(model, data, placements, min_spacing=0.06)
    randomizer.randomize(seed=42)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        placements: List[ObjectPlacementConfig],
        min_spacing: float = 0.06,
        max_attempts: int = 200,
    ):
        self.model = model
        self.data = data
        self.placements = placements
        self.min_spacing = min_spacing
        self.max_attempts = max_attempts

        # Resolve joint IDs once; skip objects not found in this model.
        self._entries: List[Tuple[ObjectPlacementConfig, int]] = []
        for cfg in placements:
            jnt_id = self._find_free_joint(cfg.body_name)
            if jnt_id < 0:
                continue
            self._entries.append((cfg, jnt_id))

    # ------------------------------------------------------------------

    def randomize(self, rng: Optional[random.Random] = None) -> bool:
        """Sample collision-free poses for all objects and write to data.qpos.

        Returns True if all objects were placed successfully, False if
        the max attempt budget was exhausted for any object.
        """
        if rng is None:
            rng = random.Random()

        placed_xys: List[Tuple[float, float]] = []
        success = True

        for cfg, jnt_id in self._entries:
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            x, y = self._sample_position(cfg, placed_xys, rng)
            yaw = rng.uniform(*cfg.yaw_range)

            # Free-joint qpos layout: [x, y, z, qw, qx, qy, qz]
            qw = math.cos(yaw / 2.0)
            qz = math.sin(yaw / 2.0)
            self.data.qpos[qpos_adr:qpos_adr + 7] = [x, y, cfg.z, qw, 0.0, 0.0, qz]
            self.data.qvel[self.model.jnt_dofadr[jnt_id]:
                           self.model.jnt_dofadr[jnt_id] + 6] = 0.0

            placed_xys.append((x, y))

        mujoco.mj_forward(self.model, self.data)
        return success

    # ------------------------------------------------------------------
    # Helpers

    def _sample_position(
        self,
        cfg: ObjectPlacementConfig,
        placed: List[Tuple[float, float]],
        rng: random.Random,
    ) -> Tuple[float, float]:
        """Rejection-sample (x, y) with minimum spacing from already placed objects."""
        for _ in range(self.max_attempts):
            x = rng.uniform(*cfg.x_range)
            y = rng.uniform(*cfg.y_range)
            if all(math.hypot(x - px, y - py) >= self.min_spacing
                   for px, py in placed):
                return x, y
        # Fallback: return centre of range (may overlap)
        return (
            (cfg.x_range[0] + cfg.x_range[1]) / 2.0,
            (cfg.y_range[0] + cfg.y_range[1]) / 2.0,
        )

    def _find_free_joint(self, body_name: str) -> int:
        """Return the joint ID of the free joint directly under body_name, or -1."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return -1
        for jnt_id in range(self.model.njnt):
            if (self.model.jnt_bodyid[jnt_id] == body_id and
                    self.model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE):
                return jnt_id
        return -1


# ---------------------------------------------------------------------------
# Pre-built placement configs for each scene
# ---------------------------------------------------------------------------

# Table surface z values:
#   simpleWoodTable / studyTable: 0.760 m  +  object half-height
#   ventionTable:                 0.824 m  +  object half-height

def kitchen_placements() -> List[ObjectPlacementConfig]:
    """Placement configs for scene_kitchen (simpleWoodTable, z_surface=0.760)."""
    z = 0.760
    region = dict(x_range=(-0.24, 0.24), y_range=(-0.53, -0.13))
    return [
        ObjectPlacementConfig("object3",           z=z + 0.025, **region),
        ObjectPlacementConfig("obstacle_coffeemug", z=z + 0.050, **region),
        ObjectPlacementConfig("obstacle_apple",     z=z + 0.037, **region),
        ObjectPlacementConfig("obstacle_banana",    z=z + 0.080, **region),
        ObjectPlacementConfig("obstacle_bowl",      z=z + 0.022, **region),
        ObjectPlacementConfig("obstacle_cup",       z=z + 0.055, **region),
        ObjectPlacementConfig("obstacle_knife",     z=z + 0.011, **region),
        ObjectPlacementConfig("obstacle_teapot",    z=z + 0.048, **region),
    ]


def workshop_placements() -> List[ObjectPlacementConfig]:
    """Placement configs for scene_workshop (ventionTable, z_surface=0.824)."""
    z = 0.824
    region = dict(x_range=(-0.24, 0.24), y_range=(-0.53, -0.13))
    return [
        ObjectPlacementConfig("object3",              z=z + 0.075, **region),
        ObjectPlacementConfig("obstacle_screwdriver", z=z + 0.085, **region),
        ObjectPlacementConfig("obstacle_bolt",        z=z + 0.028, **region),
        ObjectPlacementConfig("obstacle_wrench",      z=z + 0.010, **region),
        ObjectPlacementConfig("obstacle_hammer",      z=z + 0.026, **region),
        ObjectPlacementConfig("obstacle_pliers",      z=z + 0.020, **region),
    ]


def grocery_placements() -> List[ObjectPlacementConfig]:
    """Placement configs for scene_grocery (studyTable, z_surface=0.760)."""
    z = 0.760
    region = dict(x_range=(-0.24, 0.24), y_range=(-0.50, -0.10))
    return [
        ObjectPlacementConfig("object3",             z=z + 0.022, **region),
        ObjectPlacementConfig("obstacle_mustard",    z=z + 0.100, **region),
        ObjectPlacementConfig("obstacle_soup",       z=z + 0.060, **region),
        ObjectPlacementConfig("obstacle_crackers",   z=z + 0.077, **region),
        ObjectPlacementConfig("obstacle_sugar",      z=z + 0.085, **region),
        ObjectPlacementConfig("obstacle_pudding",    z=z + 0.080, **region),
        ObjectPlacementConfig("obstacle_gelatin",    z=z + 0.040, **region),
        ObjectPlacementConfig("obstacle_can_large",  z=z + 0.095, **region),
        ObjectPlacementConfig("obstacle_meatcan",    z=z + 0.065, **region),
    ]


def cluttered_placements() -> List[ObjectPlacementConfig]:
    """Placement configs for scene_cluttered (custom white table, z_surface=0.760)."""
    z = 0.760
    region = dict(x_range=(-0.19, 0.19), y_range=(-0.48, -0.13))
    return [
        ObjectPlacementConfig("object3",              z=z + 0.040, **region),
        ObjectPlacementConfig("obstacle_tape",        z=z + 0.028, **region),
        ObjectPlacementConfig("obstacle_stapler",     z=z + 0.040, **region),
        ObjectPlacementConfig("obstacle_duck",        z=z + 0.030, **region),
        ObjectPlacementConfig("obstacle_smallbox",    z=z + 0.030, **region),
        ObjectPlacementConfig("obstacle_marker",      z=z + 0.010, **region),
        ObjectPlacementConfig("obstacle_scissors",    z=z + 0.022, **region),
        ObjectPlacementConfig("obstacle_watch",       z=z + 0.005, **region),
        ObjectPlacementConfig("obstacle_phone",       z=z + 0.015, **region),
        ObjectPlacementConfig("obstacle_eraser",      z=z + 0.020, **region),
        ObjectPlacementConfig("obstacle_usb",         z=z + 0.012, **region),
        ObjectPlacementConfig("obstacle_pebble",      z=z + 0.018, **region),
        ObjectPlacementConfig("obstacle_coin",        z=z + 0.003, **region),
    ]


SCENE_PLACEMENTS = {
    "scene_kitchen":  kitchen_placements,
    "scene_workshop": workshop_placements,
    "scene_grocery":  grocery_placements,
    "scene_cluttered": cluttered_placements,
}
