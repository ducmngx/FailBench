import numpy as np
import mujoco
from typing import List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from enum import Enum
# Assuming imports from the IK module
# from inverse_kinematics import IKSolver, EndEffectorTarget, IKConfig, IKResult, solve_ik_for_planner
from planner.kinematics.inverse_kinematics import *
# from abstract_planner import *
from planner.algorithms.abstract_planner import *
# from collision_checker import CollisionChecker
from planner.collision.collision_checker import CollisionChecker

class JointSpaceRRT(AbstractRRTPlanner):
    """RRT planner operating in joint space."""
    
    def __init__(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel, ik_solver: IKSolver, collision_threshold: float, **kwargs):
        kwargs.pop('planning_space', None)
        super().__init__(scene_model, ik_solver, planning_space=PlanningSpace.JOINT_SPACE, **kwargs)
        # self.collision_checker = CollisionChecker(model)
        # Call to initialize collision checker
        self.collision_threshold = collision_threshold
        self.initialize_collision_checker(scene_model, robot_model)

    def initialize_collision_checker(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel):
        """Initialize the collision checker with scene and robot models."""
        self.collision_checker = CollisionChecker(scene_model, robot_model)
        
    def distance(self, config1: np.ndarray, config2: np.ndarray) -> float:
        """Euclidean distance in joint space."""
        # print(f"Calculating distance between {config1} and {config2}")
        return np.linalg.norm(config1 - config2)
    
    # Tree-Biased Sampling
    ##############################################################################
    def sample_random_config(self) -> Optional[np.ndarray]:
        """Sample with bias toward unexplored regions near existing tree."""
        if len(self.tree) <= 1 or random.random() < 0.3:
            # 30% completely random (global exploration)
            return self.ik_solver.get_random_valid_config()
        else:
            # 70% biased toward tree nodes (local exploration)
            base_node = random.choice(self.tree)
            
            # Add random noise around existing node
            noise = np.random.normal(0, self.step_size * 2, len(base_node.config))
            candidate = base_node.config + noise
            
            if self.ik_solver.is_config_valid(candidate):
                return candidate
            else:
                return self.ik_solver.get_random_valid_config()
    
    def is_valid_config(self, config):
        """
        FIXED: Your existing method with config parameter passed.
        """
        # ONLY CHANGE: Pass config to collision checker
        return not self.collision_checker.check_collisions(
            robot_config=config,  # ← ADD THIS LINE
            threshold=self.collision_threshold
        )
    
    def steer(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
        """Steer in joint space."""
        direction = to_config - from_config
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_config
        else:
            unit_direction = direction / distance
            return from_config + self.step_size * unit_direction
    
    def is_path_valid(self, from_config: np.ndarray, to_config: np.ndarray) -> bool:
        """Check if interpolated path in joint space is collision-free."""
        # Simple implementation: check intermediate points
        steps = int(np.linalg.norm(to_config - from_config) / (self.step_size * 0.1)) + 1
        for i in range(steps + 1):
            alpha = i / steps
            intermediate_config = (1 - alpha) * from_config + alpha * to_config
            if not self.is_valid_config(intermediate_config):
                return False
        return True 