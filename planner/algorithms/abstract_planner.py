import logging
import numpy as np
import mujoco
from typing import List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from enum import Enum
from planner.kinematics.inverse_kinematics import *
from planner.collision.collision_checker import CollisionChecker

logger = logging.getLogger(__name__)


@dataclass
class PlanningNode:
    """Node for tree-based planners."""
    config: np.ndarray
    parent: Optional['PlanningNode'] = None
    cost: float = 0.0
    count: int = -1

    def get_path_to_root(self) -> List['PlanningNode']:
        """Get path from this node back to root."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
class PlanningSpace(Enum):
    """Defines which space the RRT operates in."""
    JOINT_SPACE = "joint"
    TASK_SPACE = "task"

class AbstractRRTPlanner(ABC):
    """
    Abstract base class for RRT planners that can operate in joint space or task space.
    Supports both single-tree (RRT) and dual-tree (RRT-Connect) algorithms.
    """
    
    def __init__(
        self,
        scene_model: mujoco.MjModel,
        ik_solver: IKSolver,
        robot_model: Optional[mujoco.MjModel] = None,
        collision_threshold: float = 0.0,
        planning_space: PlanningSpace = PlanningSpace.JOINT_SPACE,
        max_iterations: int = 5000,
        step_size: float = 0.0005,
        goal_tolerance: float = 0.1,
        goal_bias: float = 0.7
    ):
        self.model = scene_model
        self.ik_solver = ik_solver
        self.collision_threshold = collision_threshold
        self.planning_space = planning_space
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.goal_bias = goal_bias

        # Create collision checker in the base class
        if robot_model is not None:
            self.collision_checker = CollisionChecker(scene_model, robot_model)

        # Single tree (for standard RRT)
        self.tree = []
        self.goal_node = None

        # Dual tree support (for RRT-Connect)
        self.tree_parents = {}  # For RRTConnect compatibility
        
    def distance(self, config1: np.ndarray, config2: np.ndarray) -> float:
        """Euclidean distance between two configurations."""
        return np.linalg.norm(config1 - config2)

    @abstractmethod
    def sample_random_config(self) -> Optional[np.ndarray]:
        """Sample a random valid configuration."""
        pass

    def is_valid_config(self, config: np.ndarray) -> bool:
        """Check if configuration is collision-free."""
        return not self.collision_checker.check_collisions(
            robot_config=config,
            threshold=self.collision_threshold
        )

    def steer(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
        """Steer from one configuration toward another by step_size."""
        direction = to_config - from_config
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            return to_config
        else:
            unit_direction = direction / distance
            return from_config + self.step_size * unit_direction

    def is_path_valid(self, from_config: np.ndarray, to_config: np.ndarray) -> bool:
        """Check if interpolated path between configurations is collision-free."""
        steps = int(np.linalg.norm(to_config - from_config) / (self.step_size * 0.1)) + 1
        for i in range(steps + 1):
            alpha = i / steps
            intermediate_config = (1 - alpha) * from_config + alpha * to_config
            if not self.is_valid_config(intermediate_config):
                return False
        return True
    
    def plan(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        frame_name: str = "gripper"
    ) -> Optional[List[np.ndarray]]:
        """
        Main RRT planning algorithm.
        
        Args:
            start_config: Starting configuration (joint or task space)
            goal_config: Goal configuration (joint or task space)
            frame_name: End-effector frame name (for task space planning)
            
        Returns:
            Path from start to goal or None if failed
        """
        # Convert inputs to appropriate space if needed
        start_config = self._ensure_planning_space(start_config, frame_name)
        goal_config = self._ensure_planning_space(goal_config, frame_name)
        
        if start_config is None or goal_config is None:
            return None
            
        # Initialize tree with start configuration
        self.tree = [PlanningNode(config=start_config.copy())]
        self.goal_node = None
        
        for iteration in range(self.max_iterations):
            # Sample random configuration (with goal bias)
            if random.random() < self.goal_bias:
                rand_config = goal_config.copy()
            else:
                rand_config = self.sample_random_config()
                
            if rand_config is None:
                continue
                
            # Find nearest node in tree
            nearest_node = self._find_nearest_node(rand_config)
            
            # Steer toward random configuration
            new_config = self.steer(nearest_node.config, rand_config)
            
            # Check if new configuration and path are valid
            if (self.is_valid_config(new_config) and 
                self.is_path_valid(nearest_node.config, new_config)):
                
                # Add new node to tree
                new_node = PlanningNode(
                    config=new_config,
                    parent=nearest_node,
                    cost=nearest_node.cost + self.distance(nearest_node.config, new_config)
                )
                self.tree.append(new_node)
                
                # Check if we reached the goal
                if self.distance(new_config, goal_config) < self.goal_tolerance:
                    self.goal_node = new_node
                    logger.info(f"Goal reached in {iteration + 1} iterations!")
                    break

        if self.goal_node is None:
            logger.warning(f"Failed to reach goal after {self.max_iterations} iterations")
            return None
            
        # Extract path and convert back to joint space if needed
        path_nodes = self.goal_node.get_path_to_root()
        path = [node.config for node in path_nodes]
        
        return self._convert_path_to_joint_space(path, frame_name)
    
    def _find_nearest_node(self, config: np.ndarray) -> PlanningNode:
        """Find the nearest node in the tree to the given configuration."""
        min_distance = float('inf')
        nearest_node = self.tree[0]
        
        for node in self.tree:
            dist = self.distance(node.config, config)
            if dist < min_distance:
                min_distance = dist
                nearest_node = node
                
        return nearest_node
    
    def _ensure_planning_space(self, config: np.ndarray, frame_name: str) -> Optional[np.ndarray]:
        """Convert configuration to the planning space if needed."""
        if self.planning_space == PlanningSpace.JOINT_SPACE:
            if len(config) == 3:  # Assume it's task space position
                # Convert task space to joint space
                target = EndEffectorTarget(position=config, frame_name=frame_name)
                solution, result = self.ik_solver.solve(target, np.zeros(self.ik_solver.model.nq))
                return solution if result == IKResult.SUCCESS else None
            else:
                return config  # Already joint space
        else:  # TASK_SPACE
            if len(config) > 3:  # Assume it's joint space
                # Convert joint space to task space
                self.ik_solver._data.qpos[:] = config
                mujoco.mj_forward(self.model, self.ik_solver._data)
                self.ik_solver._configuration.update(self.ik_solver._data.qpos)
                transform = self.ik_solver._configuration.get_transform_frame_to_world(frame_name, "site")
                return transform.translation()
            else:
                return config  # Already task space
    
    def _convert_path_to_joint_space(self, path: List[np.ndarray], frame_name: str) -> List[np.ndarray]:
        """Convert path to joint space configurations."""
        if self.planning_space == PlanningSpace.JOINT_SPACE:
            return path
            
        # Convert task space path to joint space
        joint_path = []
        for i, task_config in enumerate(path):
            if i == 0:
                # Use current joint configuration as seed for first waypoint
                seeds = [self.ik_solver._data.qpos.copy()]
            else:
                # Use previous joint configuration as seed
                seeds = [joint_path[-1]]
                
            target = EndEffectorTarget(position=task_config, frame_name=frame_name)
            solution = self.ik_solver.find_valid_solution(target, seeds)
            
            if solution is None:
                logger.warning(f"Failed to convert task space waypoint {i} to joint space")
                return None
                
            joint_path.append(solution)
            
        return joint_path