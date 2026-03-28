import numpy as np
import mujoco
from typing import List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from enum import Enum
import time
import logging

# Assuming imports from the IK module
# from inverse_kinematics import IKSolver, EndEffectorTarget, IKConfig, IKResult, solve_ik_for_planner
from planner.kinematics.inverse_kinematics import *
# from abstract_planner import *
from planner.algorithms.abstract_planner import *
from failure_injection.collision_estimation import CollisionEstimator
from failure_injection.optimized_collision_estimation import OptimizedCollisionEstimator

logger = logging.getLogger(__name__)

class JointSpaceRRT(AbstractRRTPlanner):
    """RRT planner operating in joint space."""

    def __init__(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel, ik_solver: IKSolver, collision_threshold: float, **kwargs):
        kwargs.pop('planning_space', None)
        super().__init__(scene_model, ik_solver, robot_model=robot_model,
                         collision_threshold=collision_threshold,
                         planning_space=PlanningSpace.JOINT_SPACE, **kwargs)

    def sample_random_config(self) -> Optional[np.ndarray]:
        """Sample with bias toward unexplored regions near existing tree."""
        if len(self.tree) <= 1 or random.random() < 0.3:
            return self.ik_solver.get_random_valid_config()
        else:
            base_node = random.choice(self.tree)
            noise = np.random.normal(0, self.step_size * 2, len(base_node.config))
            candidate = base_node.config + noise
            if self.ik_solver.is_config_valid(candidate):
                return candidate
            else:
                return self.ik_solver.get_random_valid_config()
    

class JointSpaceRRTConnect(AbstractRRTPlanner):
    """RRT-Connect planner operating in joint space - bidirectional RRT."""

    def __init__(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel,
                 ik_solver: IKSolver, collision_threshold: float, seed: int, **kwargs):
        kwargs.pop('planning_space', None)
        super().__init__(scene_model, ik_solver, robot_model=robot_model,
                         collision_threshold=collision_threshold,
                         planning_space=PlanningSpace.JOINT_SPACE, **kwargs)

        self.seed = seed
        self.rng = None   # Will be initialized in plan()

        # RRT-Connect specific: two trees
        self.goal_tree = []
        self.goal_tree_parents = {}

    def sample_random_config(self) -> Optional[np.ndarray]:
        """Sample with bias toward unexplored regions."""
        if len(self.tree) <= 1 or random.random() < 0.3:
            return self.ik_solver.get_random_valid_config(rng=self.rng)
        else:
            base_node = self.rng.choice(self.tree)
            noise = self.rng.normal(0, self.step_size * 2, len(base_node.config))
            candidate = base_node.config + noise
            if self.ik_solver.is_config_valid(candidate):
                return candidate
            else:
                return self.ik_solver.get_random_valid_config(rng=self.rng)
    
    def extend_tree(self, tree: List, parents: dict, target_config: np.ndarray) -> Tuple[str, Optional[int]]:
        """
        Extend tree toward target configuration.
        Returns: (status, node_index)
        status: 'reached', 'advanced', or 'trapped'
        """
        if not tree:
            return 'trapped', None
            
        # Find nearest node in tree
        nearest_idx = 0
        min_distance = float('inf')
        for i, node in enumerate(tree):
            dist = self.distance(node.config, target_config)
            if dist < min_distance:
                min_distance = dist
                nearest_idx = i
        
        nearest_node = tree[nearest_idx]
        
        # Steer toward target
        new_config = self.steer(nearest_node.config, target_config)
        
        # Check if new configuration and path are valid
        if not self.is_valid_config(new_config):
            return 'trapped', None
        
        if not self.is_path_valid(nearest_node.config, new_config):
            return 'trapped', None
        
        # Add new node to tree
        new_node = PlanningNode(new_config)
        tree.append(new_node)
        new_idx = len(tree) - 1
        parents[new_idx] = nearest_idx
        
        # Check if we reached the target exactly
        if self.distance(new_config, target_config) < self.goal_tolerance:
            return 'reached', new_idx
        else:
            return 'advanced', new_idx
    
    def reconstruct_path(self, tree1: List, tree2: List, parents1: dict, parents2: dict,
                        connect_idx1: int, connect_idx2: int, tree1_is_start: bool = True) -> List[np.ndarray]:
        """Reconstruct path from start to goal through connection point."""
        
        # Path from tree1 to connection point
        path1 = []
        current = connect_idx1
        while current is not None:
            path1.append(tree1[current].config.copy())
            current = parents1.get(current)
        
        # Path from tree2 to connection point
        path2 = []
        current = connect_idx2
        while current is not None:
            path2.append(tree2[current].config.copy())
            current = parents2.get(current)
        
        # Determine correct path direction based on which tree is start tree
        if tree1_is_start:
            # tree1 = start tree, tree2 = goal tree
            path1.reverse()  # Start to connection
            # path2 is already connection to goal (don't reverse)
            final_path = path1 + path2[1:] if len(path2) > 1 else path1
        else:
            # tree1 = goal tree, tree2 = start tree  
            path2.reverse()  # Start to connection
            # path1 is already connection to goal (don't reverse)
            final_path = path2 + path1[1:] if len(path1) > 1 else path2
        
        return final_path

    def densify_direct_path(self, start: np.ndarray, goal: np.ndarray, num_waypoints: int = 20) -> List[np.ndarray]:
        """Create smooth interpolated path for direct connections."""
        path = []
        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            config = (1 - alpha) * start + alpha * goal
            path.append(config)
        return path

    def plan(self, start_config: np.ndarray, goal_config: np.ndarray, 
            frame_name: str = "end_effector") -> Optional[List[np.ndarray]]:
        """
        Plan path using RRT-Connect algorithm with proper path reconstruction.
        """
        # print(f"🔄 Starting RRT-Connect planning...")
        # Initialize RNG with seed at start of planning
        self.rng = np.random.RandomState(self.seed)

        self._plan_start_config = start_config.copy()
        
        # Initialize trees
        self.tree = [PlanningNode(start_config)]
        self.tree_parents = {0: None}
        
        self.goal_tree = [PlanningNode(goal_config)]
        self.goal_tree_parents = {0: None}
        
        # Track which tree is the start tree (important for path reconstruction)
        tree_is_start = True  # True = self.tree is start tree, False = self.tree is goal tree
        
        # # Check if start and goal are directly connected
        # if self.is_path_valid(start_config, goal_config):
        #     print("✅ Direct path found, densifying...")
        #     # Create smooth interpolated path
        #     return self.densify_direct_path(start_config, goal_config, num_waypoints=3)
        
        for iteration in range(self.max_iterations):
            # Sample random configuration
            rand_config = self.sample_random_config()
            if rand_config is None:
                continue
            
            # Extend first tree toward random config
            status1, new_idx1 = self.extend_tree(self.tree, self.tree_parents, rand_config)
            
            if status1 != 'trapped':
                # Try to connect second tree to the new node
                new_config = self.tree[new_idx1].config
                
                # Keep extending second tree toward new node until trapped or reached
                while True:
                    status2, new_idx2 = self.extend_tree(self.goal_tree, self.goal_tree_parents, new_config)
                    
                    if status2 == 'reached':
                        # Trees connected!
                        # print(f"✅ Goal reached in {iteration + 1} iterations!")
                        return self.reconstruct_path(
                            self.tree, self.goal_tree, 
                            self.tree_parents, self.goal_tree_parents,
                            new_idx1, new_idx2, tree_is_start
                        )
                    elif status2 == 'trapped':
                        break
                    
                    # Check if the trees can be directly connected
                    if self.is_path_valid(self.tree[new_idx1].config, self.goal_tree[new_idx2].config):
                        # print(f"✅ Trees connected in {iteration + 1} iterations!")
                        return self.reconstruct_path(
                            self.tree, self.goal_tree,
                            self.tree_parents, self.goal_tree_parents,
                            new_idx1, new_idx2, tree_is_start
                        )
            
            # Swap trees for next iteration (alternate growth direction)
            self.tree, self.goal_tree = self.goal_tree, self.tree
            self.tree_parents, self.goal_tree_parents = self.goal_tree_parents, self.tree_parents
            tree_is_start = not tree_is_start  # Track which tree is start after swap
        
        logger.warning(f"RRT-Connect failed after {self.max_iterations} iterations")
        return None



class JointSpaceRRTConnectFailure(JointSpaceRRTConnect):
    """
    Overrides JointSpaceRRTConnect to integrate estimating impact of failure.
    """
    
    def __init__(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel, 
                 ik_solver: IKSolver, collision_threshold: float, seed: int, collision_estimator: CollisionEstimator, body_severity_table: np.ndarray,  **kwargs):
        super().__init__(scene_model, robot_model, ik_solver, collision_threshold, seed=seed, **kwargs)

        self.failure_weight = 0.01
        # self.failure_threshold = 10 # max_severity = 10, weight=0.01, 10*0.01 = 0.1
        self.collision_estimator = collision_estimator 
        self.count = 0                  # debugging
        self.total_fail_cost_time = 0   # debugging 
        self.body_severity_table = body_severity_table
    
    def get_trajectory_cost(self, path):
        mechanical_cost = []
        safety_cost = []

        for config in path:
            safety_cost.append(self.impact_of_failure(config))

        mechanical_cost.append(self.distance(self._plan_start_config, path[0]))
        for i in range(len(path)-1):
            mechanical_cost.append(self.distance(path[i], path[i+1]))

        mechanical_cost, safety_cost = np.array(mechanical_cost), np.array(safety_cost)

        # print(f"trajectory ({len(path)} waypoints) cost: {mechanical_cost.sum() + safety_cost.sum()}: (V: {mechanical_cost.sum() }, Safety: {safety_cost.sum()}, Safety weight: {self.failure_weight})")
        return mechanical_cost, safety_cost
    
    def impact_of_failure(self, config=None, calculate_grads=False):
        """
        Calculates \\sum_rj \\sum_ei {weight * P(x, rj, ei | F) * S(rj, ei) }  
        """
        begin_time = time.perf_counter()
        self.count += 1
        if config is not None: # similar to how collision_checker does it
            self.collision_estimator.forward_kinematics(config)

        body_id_pairs, prob_collision, grads = self.collision_estimator.estimate_bodies_in_collision(calculate_grads=calculate_grads)
        # body_id_pairs: (N, 2), prob_collision: (N,), grads: (N, njnt)
        # get severity of collision via LLM/table using body_id_pairs
        # for now assume severity = 1 for all pairs except for when cand_body_id = 13 (object2)
        # also filter out target objects (object1, object3) bids = 12, 14 i think
        # mask = ~np.isin(body_id_pairs[:,1], self.target_bids)
        # body_id_pairs = body_id_pairs[mask]
        # prob_collision = prob_collision[mask]
        # grads = grads[mask]

        severity = self.body_severity_table[body_id_pairs[:, 1]]

        # severity = np.ones(shape=len(body_id_pairs), dtype=np.float64)
        # severity[body_id_pairs[:,1] == 13] = 10 # max severity that LLM will give

        # print(f"impact_of_failure call {self.count}:")
        # print(f"Failure weight {self.failure_weight}, prob_collision: {prob_collision}, severity: {severity}")
        cost = self.failure_weight * (prob_collision * severity).sum()
        grads = self.failure_weight * (grads*severity[:, None]).sum(axis=0)
        elapsed_time = time.perf_counter() - begin_time
        self.total_fail_cost_time  += elapsed_time
        if calculate_grads:
            return cost, grads
        else:
            return cost