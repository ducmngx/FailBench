import mujoco
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
import yaml
import xml.etree.ElementTree as ET
from collections import defaultdict
from io import StringIO
import copy
import os
import re
from planner.collision.geometry_utils import EnhancedXMLSeparator, SimpleXMLSeparator

class CollisionChecker:
    def __init__(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel) -> None:
        self.scene_model = scene_model
        self.robot_model = robot_model
        self.scene_data = mujoco.MjData(scene_model)
        self.robot_data = mujoco.MjData(robot_model)

        # xml extractor
        self._xml_extractor = SimpleXMLSeparator(
            scene_model=scene_model,
            robot_model=robot_model
        )
        self.collision_pairs = self._xml_extractor.get_collision_pairs()

    def set_robot_configuration_direct(self, robot_config: np.ndarray) -> None:
        if len(robot_config) > len(self.scene_data.qpos):
            raise ValueError(f"Config length {len(robot_config)} > scene qpos length {len(self.scene_data.qpos)}")

        self.scene_data.qpos[:len(robot_config)] = robot_config
        mujoco.mj_forward(self.scene_model, self.scene_data)

    def _check_collision(self, geom1_id: int, geom2_id: int, threshold: float) -> bool:
        """Check if two geometries are in collision."""
        fromto = np.zeros(6)
        dist = mujoco.mj_geomDistance(
            self.scene_model, self.scene_data, geom1_id, geom2_id,
            distmax=0.1, fromto=fromto
        )
        return np.abs(dist) <= threshold

    def _update_collision_pairs(self) -> None:
        """Update the collision pairs based on the current scene and robot models."""
        self.collision_pairs = self._xml_extractor.get_collision_pairs()

    def check_collisions(self, robot_config: Optional[np.ndarray] = None, threshold: float = 0.0) -> bool:
        """Check for collisions, optionally setting robot config first."""
        if robot_config is not None:
            self.set_robot_configuration_direct(robot_config)

        self._update_collision_pairs()

        for _, (group1, group2) in enumerate(self.collision_pairs):
            for geom1_id in group1:
                if geom1_id <= 0:
                    continue
                for geom2_id in group2:
                    if self._check_collision(geom1_id=geom1_id,
                                           geom2_id=geom2_id,
                                           threshold=threshold):
                        return True
        return False
