"""
Simple and reliable robot vs environment separator.
Works when robot.xml is included in scene.xml.
"""

import logging
import mujoco
import numpy as np
from typing import Dict, List, Set, Tuple
import yaml
import xml.etree.ElementTree as ET
from collections import defaultdict
from io import StringIO
import copy
import os
import re

logger = logging.getLogger(__name__)

class EnhancedXMLSeparator:
    """
    Your SimpleXMLSeparator with minor enhancements.
    Shows how to add the suggestions while keeping your excellent design.
    """
    
    def __init__(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel) -> None:
        """Your existing __init__ with minor additions."""
        self.scene_model = scene_model
        self.robot_model = robot_model
        self.scene_data = mujoco.MjData(self.scene_model)
        
        # print(f"Scene: {self.scene_model.ngeom} geoms, {self.scene_model.njnt} joints")
        # print(f"Robot: {self.robot_model.ngeom} geoms, {self.robot_model.njnt} joints")
        
        # Your existing code
        self.robot_names = self._get_robot_component_names()
        self.separation = self._separate_components()
        
        # Minor addition: Validate separation
        self._validate_separation()
        
        # Minor addition: Cache collision pairs
        self._cached_collision_pairs = None
    
    def _get_robot_component_names(self) -> Dict[str, Set[str]]:
        """Get all component names from robot.xml."""
        names = {
            'bodies': set(),
            'geoms': set(), 
            'joints': set()
        }
        
        # Get robot body names
        for body_id in range(self.robot_model.nbody):
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            if body_name:
                names['bodies'].add(body_name)
        
        # Get robot geom names
        for geom_id in range(self.robot_model.ngeom):
            geom_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name:
                names['geoms'].add(geom_name)
        
        # Get robot joint names
        for joint_id in range(self.robot_model.njnt):
            joint_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                names['joints'].add(joint_name)
        
        # print(f"Robot components: {len(names['bodies'])} bodies, {len(names['geoms'])} geoms, {len(names['joints'])} joints")
        return names
    
    def _separate_components(self) -> Dict[str, List[int]]:
        """Separate scene components into robot vs environment."""
        separation = {
            'robot_bodies': [],
            'robot_geoms': [],
            'robot_joints': [],
            'robot_collision_geoms': [],
            'environment_bodies': [],
            'environment_geoms': [],
            'environment_joints': [],
            'environment_collision_geoms': []
        }

        # Separate bodies
        for body_id in range(self.scene_model.nbody):
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            # print(f"Checking body {body_id}: {body_name} with current list of rname {self.robot_names['bodies']}")
            if body_name in self.robot_names['bodies']:
                separation['robot_bodies'].append(body_id)
            else:
                separation['environment_bodies'].append(body_id)
        
        # Separate geoms
        for geom_id in range(self.scene_model.ngeom):
            geom_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_id = self.scene_model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            # Check if geom belongs to robot (by name or by body)
            is_robot_geom = (geom_name in self.robot_names['geoms'] or 
                           body_name in self.robot_names['bodies'])
            
            if is_robot_geom:
                separation['robot_geoms'].append(geom_id)
                if self._is_collision_geom(geom_id):
                    separation['robot_collision_geoms'].append(geom_id)
            else:
                separation['environment_geoms'].append(geom_id)
                if self._is_collision_geom(geom_id):
                    separation['environment_collision_geoms'].append(geom_id)
        
        # Separate joints
        for joint_id in range(self.scene_model.njnt):
            joint_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            body_id = self.scene_model.jnt_bodyid[joint_id]
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            # Check if joint belongs to robot (by name or by body)
            is_robot_joint = (joint_name in self.robot_names['joints'] or 
                            body_name in self.robot_names['bodies'])
            
            if is_robot_joint:
                separation['robot_joints'].append(joint_id)
            else:
                separation['environment_joints'].append(joint_id)

        # print(f"Collision geoms: {len(separation['robot_collision_geoms'])} robot, {len(separation['environment_collision_geoms'])} environment")
        # print(f"Robot collision geoms: {separation['robot_collision_geoms']}")
        # print(f"Environment collision geoms: {separation['environment_collision_geoms']}")

        return separation
    
    def _is_collision_geom(self, geom_id: int) -> bool:
        """Check if geom is collision-capable."""
        group = self.scene_model.geom_group[geom_id]
        contype = self.scene_model.geom_contype[geom_id]
        conaffinity = self.scene_model.geom_conaffinity[geom_id]
        
        # Skip visual-only geoms
        if group == 2 and contype == 0 and conaffinity == 0:
            return False
        
        return True
    
    def _validate_separation(self):
        """Minor addition: Validate that separation makes sense."""
        robot_collision = len(self.separation['robot_collision_geoms'])
        env_collision = len(self.separation['environment_collision_geoms'])
        
        if robot_collision == 0:
            logger.warning("No robot collision geometries detected! "
                          "This might indicate a naming mismatch. "
                          "Consider checking robot part names with debug_name_matching() "
                          "or using fallback detection method.")

        if env_collision == 0:
            logger.warning("No environment collision geometries detected!")

        logger.info(f"Separation validation: {robot_collision} robot, {env_collision} env collision geoms")
    
    def get_collision_pairs(self) -> List[Tuple[List[int], List[int]]]:
        """Your existing method with optional caching."""
        if self._cached_collision_pairs is None:
            robot_collision = self.get_robot_geoms(collision_only=True)
            env_collision = self.get_environment_geoms(collision_only=True)
            
            pairs = []
            if robot_collision and env_collision:
                pairs.append((robot_collision, env_collision))
            
            self._cached_collision_pairs = pairs
        
        return self._cached_collision_pairs
    
    def get_fallback_robot_geoms(self):
        """Minor addition: Fallback detection using joint-body mapping."""
        logger.info("Using fallback robot geometry detection...")
        
        # Find bodies connected to robot joints
        robot_bodies = set()
        robot_joints = self.get_robot_joints()
        
        for joint_id in robot_joints:
            body_id = self.scene_model.jnt_bodyid[joint_id]
            robot_bodies.add(body_id)
        
        # Find geoms in these bodies
        fallback_geoms = []
        for geom_id in range(self.scene_model.ngeom):
            body_id = self.scene_model.geom_bodyid[geom_id]
            if body_id in robot_bodies and self._is_collision_geom(geom_id):
                fallback_geoms.append(geom_id)
        
        logger.info(f"Fallback detection found {len(fallback_geoms)} robot geoms")
        return fallback_geoms

class SimpleXMLSeparator:
    """Simple separator: Environment = Scene - Robot."""
    
    # def __init__(self, scene_xml_path: str, robot_xml_path: str):
    def __init__(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel):
        """
        Initialize separator.
        
        Args:
            scene_xml_path: Complete scene (robot + environment)
            robot_xml_path: Robot-only XML (should be included in scene)
        """
        # self.scene_xml_path = scene_xml_path
        # self.robot_xml_path = robot_xml_path
        
        # Load models
        # self.scene_model = mujoco.MjModel.from_xml_path(scene_xml_path)
        # self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        
        self.scene_model = scene_model
        self.robot_model = robot_model
        self.scene_data = mujoco.MjData(self.scene_model)
        
        # print(f"Scene: {self.scene_model.ngeom} geoms, {self.scene_model.njnt} joints")
        # print(f"Robot: {self.robot_model.ngeom} geoms, {self.robot_model.njnt} joints")
        
        # Get robot component names
        self.robot_names = self._get_robot_component_names()
        
        # Separate components
        self.separation = self._separate_components()
    
    def _get_robot_component_names(self) -> Dict[str, Set[str]]:
        """Get all component names from robot.xml."""
        names = {
            'bodies': set(),
            'geoms': set(), 
            'joints': set()
        }
        
        # Get robot body names
        for body_id in range(self.robot_model.nbody):
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            if body_name:
                names['bodies'].add(body_name)
        
        # Get robot geom names
        for geom_id in range(self.robot_model.ngeom):
            geom_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name:
                names['geoms'].add(geom_name)
        
        # Get robot joint names
        for joint_id in range(self.robot_model.njnt):
            joint_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                names['joints'].add(joint_name)
        
        # print(f"Robot components: {len(names['bodies'])} bodies, {len(names['geoms'])} geoms, {len(names['joints'])} joints")
        return names
    
    def _separate_components(self) -> Dict[str, List[int]]:
        """Separate scene components into robot vs environment."""
        separation = {
            'robot_bodies': [],
            'robot_geoms': [],
            'robot_joints': [],
            'robot_collision_geoms': [],
            'environment_bodies': [],
            'environment_geoms': [],
            'environment_joints': [],
            'environment_collision_geoms': []
        }

        # Separate bodies
        for body_id in range(self.scene_model.nbody):
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            # print(f"Checking body {body_id}: {body_name} with current list of rname {self.robot_names['bodies']}")
            if body_name in self.robot_names['bodies']:
                separation['robot_bodies'].append(body_id)
            else:
                separation['environment_bodies'].append(body_id)
        
        # Separate geoms
        for geom_id in range(self.scene_model.ngeom):
            geom_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_id = self.scene_model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            # Check if geom belongs to robot (by name or by body)
            is_robot_geom = (geom_name in self.robot_names['geoms'] or 
                           body_name in self.robot_names['bodies'])
            
            if is_robot_geom:
                separation['robot_geoms'].append(geom_id)
                if self._is_collision_geom(geom_id):
                    separation['robot_collision_geoms'].append(geom_id)
            else:
                separation['environment_geoms'].append(geom_id)
                if self._is_collision_geom(geom_id):
                    separation['environment_collision_geoms'].append(geom_id)
        
        # Separate joints
        for joint_id in range(self.scene_model.njnt):
            joint_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            body_id = self.scene_model.jnt_bodyid[joint_id]
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            # Check if joint belongs to robot (by name or by body)
            is_robot_joint = (joint_name in self.robot_names['joints'] or 
                            body_name in self.robot_names['bodies'])
            
            if is_robot_joint:
                separation['robot_joints'].append(joint_id)
            else:
                separation['environment_joints'].append(joint_id)

        # print(f"Collision geoms: {len(separation['robot_collision_geoms'])} robot, {len(separation['environment_collision_geoms'])} environment")
        # print(f"Robot collision geoms: {separation['robot_collision_geoms']}")
        # print(f"Environment collision geoms: {separation['environment_collision_geoms']}")

        return separation
    
    def _is_collision_geom(self, geom_id: int) -> bool:
        """Check if geom is collision-capable."""
        group = self.scene_model.geom_group[geom_id]
        contype = self.scene_model.geom_contype[geom_id]
        conaffinity = self.scene_model.geom_conaffinity[geom_id]
        
        # Skip visual-only geoms
        if group == 2 and contype == 0 and conaffinity == 0:
            return False
        
        return True
    
    def get_robot_geoms(self, collision_only: bool = True) -> List[int]:
        """Get robot geom IDs."""
        if collision_only:
            return self.separation['robot_collision_geoms']
        else:
            return self.separation['robot_geoms']
    
    def get_environment_geoms(self, collision_only: bool = True) -> List[int]:
        """Get environment geom IDs."""
        if collision_only:
            return self.separation['environment_collision_geoms']
        else:
            return self.separation['environment_geoms']
    
    def get_robot_joints(self) -> List[int]:
        """Get robot joint IDs."""
        return self.separation['robot_joints']
    
    def get_environment_joints(self) -> List[int]:
        """Get environment joint IDs."""
        return self.separation['environment_joints']
    
    def get_robot_joint_indices(self) -> List[int]:
        """Get qpos indices for robot joints."""
        indices = []
        for joint_id in self.get_robot_joints():
            qpos_addr = self.scene_model.jnt_qposadr[joint_id]
            indices.append(qpos_addr)
        return sorted(indices)
    
    def get_collision_pairs(self) -> List[Tuple[List[int], List[int]]]:
        """Generate collision pairs."""
        robot_collision = self.get_robot_geoms(collision_only=True)
        env_collision = self.get_environment_geoms(collision_only=True)

        pairs = []
    
        # Robot vs environment
        if robot_collision and env_collision:
            pairs.append((robot_collision, env_collision))
        
        return pairs
    
    def set_robot_configuration(self, robot_config: np.ndarray):
        """Set robot configuration in scene."""
        robot_joint_indices = self.get_robot_joint_indices()
        
        if len(robot_config) != len(robot_joint_indices):
            raise ValueError(f"Config length {len(robot_config)} != robot joints {len(robot_joint_indices)}")
        
        for i, qpos_idx in enumerate(robot_joint_indices):
            self.scene_data.qpos[qpos_idx] = robot_config[i]
        
        mujoco.mj_forward(self.scene_model, self.scene_data)
    
    def print_separation_report(self):
        """Print separation report."""
        logger.info("=" * 70)
        logger.info("SIMPLE XML SEPARATION REPORT")
        logger.info("=" * 70)
        
        sep = self.separation
        
        logger.info("FILES:")
        logger.info(f"  Scene XML: output.xml")
        logger.info(f"  Robot XML: panda.xml")

        logger.info("SEPARATION RESULTS:")
        logger.info(f"  Robot bodies: {len(sep['robot_bodies'])}")
        logger.info(f"  Robot geoms (all): {len(sep['robot_geoms'])}")
        logger.info(f"  Robot collision geoms: {len(sep['robot_collision_geoms'])}")
        logger.info(f"  Robot joints: {len(sep['robot_joints'])}")
        logger.info(f"  Environment bodies: {len(sep['environment_bodies'])}")
        logger.info(f"  Environment geoms (all): {len(sep['environment_geoms'])}")
        logger.info(f"  Environment collision geoms: {len(sep['environment_collision_geoms'])}")
        logger.info(f"  Environment joints: {len(sep['environment_joints'])}")

        logger.info("ROBOT COLLISION GEOMS:")
        for i, geom_id in enumerate(sep['robot_collision_geoms']):
            geom_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_id = self.scene_model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            logger.info(f"  {i+1:2d}. Geom {geom_id:3d}: {geom_name or 'unnamed':20s} in {body_name}")

        logger.info("ENVIRONMENT COLLISION GEOMS:")
        for i, geom_id in enumerate(sep['environment_collision_geoms']):
            geom_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_id = self.scene_model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            logger.info(f"  {i+1:2d}. Geom {geom_id:3d}: {geom_name or 'unnamed':20s} in {body_name}")

        logger.info("ROBOT JOINTS:")
        robot_joints = self.get_robot_joints()
        for i, joint_id in enumerate(robot_joints[:10]):
            joint_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qpos_addr = self.scene_model.jnt_qposadr[joint_id]
            logger.info(f"  {i+1:2d}. Joint {joint_id:2d}: {joint_name:20s} qpos[{qpos_addr}]")

        if len(robot_joints) > 10:
            logger.info(f"      ... and {len(robot_joints)-10} more")

    def test_collision_detection(self):
        """Test collision detection."""
        for _ in range(1000):
            mujoco.mj_step(self.scene_model, self.scene_data)

        collision_pairs = self.get_collision_pairs()

        logger.info("Testing collision detection...")
        for pair_idx, (group1, group2) in enumerate(collision_pairs):
            logger.info(f"  Pair {pair_idx + 1}: {len(group1)} vs {len(group2)} geoms")

            collisions = 0
            for geom1_id in group1[:5]:  # Test subset
                for geom2_id in group2[:5]:
                    fromto = np.zeros(6)
                    dist = mujoco.mj_geomDistance(
                        self.scene_model, self.scene_data, geom1_id, geom2_id,
                        distmax=0.1, fromto=fromto
                    )

                    if dist <= 0.1:  # 1cm threshold
                        logger.debug(f"    Collision: Geom {geom1_id} <-> Geom {geom2_id} (dist={dist:.7f})")
                        collisions += 1

            logger.info(f"    Found {collisions} close pairs")

    def debug_name_matching(self):
        """Debug what names are being matched."""
        logger.debug("=" * 50)
        logger.debug("DEBUG: NAME MATCHING")
        logger.debug("=" * 50)

        logger.debug("ROBOT BODY NAMES:")
        for name in sorted(self.robot_names['bodies']):
            logger.debug(f"  '{name}'")

        logger.debug("ROBOT GEOM NAMES:")
        for name in sorted(self.robot_names['geoms']):
            logger.debug(f"  '{name}'")

        logger.debug("ROBOT JOINT NAMES:")
        for name in sorted(self.robot_names['joints']):
            logger.debug(f"  '{name}'")

        logger.debug("SCENE BODIES WITH 'finger':")
        for body_id in range(self.scene_model.nbody):
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and 'finger' in body_name.lower():
                is_robot = body_name in self.robot_names['bodies']
                status = "ROBOT" if is_robot else "ENVIRONMENT"
                logger.debug(f"  '{body_name}' -> {status}")