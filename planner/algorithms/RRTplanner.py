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
from failure_injection.collision_estimation import CollisionEstimator

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
    

class JointSpaceRRTConnect(AbstractRRTPlanner):
    """RRT-Connect planner operating in joint space - bidirectional RRT."""
    
    def __init__(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel, 
                 ik_solver: IKSolver, collision_threshold: float, seed: int, **kwargs):
        kwargs.pop('planning_space', None)
        super().__init__(scene_model, ik_solver, planning_space=PlanningSpace.JOINT_SPACE, **kwargs)
        
        self.collision_threshold = collision_threshold
        self.seed = seed  # Store the seed
        self.rng = None   # Will be initialized in plan()
        self.initialize_collision_checker(scene_model, robot_model)
        
        # RRT-Connect specific: two trees
        self.goal_tree = []
        self.goal_tree_parents = {}
        
    def initialize_collision_checker(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel):
        """Initialize the collision checker with scene and robot models."""
        self.collision_checker = CollisionChecker(scene_model, robot_model)
    
    def distance(self, config1: np.ndarray, config2: np.ndarray) -> float:
        """Euclidean distance in joint space."""
        return np.linalg.norm(config1 - config2)
    
    def sample_random_config(self) -> Optional[np.ndarray]:
        """Sample with bias toward unexplored regions."""
        if len(self.tree) <= 1 or random.random() < 0.3:
            # Pass RNG to IK solver
            return self.ik_solver.get_random_valid_config(rng=self.rng)
        else:
            base_node = self.rng.choice(self.tree)
            noise = self.rng.normal(0, self.step_size * 2, len(base_node.config))
            candidate = base_node.config + noise
            if self.ik_solver.is_config_valid(candidate):
                return candidate
            else:
                return self.ik_solver.get_random_valid_config(rng=self.rng)
    
    def is_valid_config(self, config):
        """Check if configuration is collision-free."""
        return not self.collision_checker.check_collisions(
            robot_config=config,
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
        steps = int(np.linalg.norm(to_config - from_config) / (self.step_size * 0.1)) + 1
        for i in range(steps + 1):
            alpha = i / steps
            intermediate_config = (1 - alpha) * from_config + alpha * to_config
            if not self.is_valid_config(intermediate_config):
                return False
        return True
    
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
        print(f"🔄 Starting RRT-Connect planning...")
        # Initialize RNG with seed at start of planning
        self.rng = np.random.RandomState(self.seed)
        
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
                        print(f"✅ Goal reached in {iteration + 1} iterations!")
                        return self.reconstruct_path(
                            self.tree, self.goal_tree, 
                            self.tree_parents, self.goal_tree_parents,
                            new_idx1, new_idx2, tree_is_start
                        )
                    elif status2 == 'trapped':
                        break
                    
                    # Check if the trees can be directly connected
                    if self.is_path_valid(self.tree[new_idx1].config, self.goal_tree[new_idx2].config):
                        print(f"✅ Trees connected in {iteration + 1} iterations!")
                        return self.reconstruct_path(
                            self.tree, self.goal_tree,
                            self.tree_parents, self.goal_tree_parents,
                            new_idx1, new_idx2, tree_is_start
                        )
            
            # Swap trees for next iteration (alternate growth direction)
            self.tree, self.goal_tree = self.goal_tree, self.tree
            self.tree_parents, self.goal_tree_parents = self.goal_tree_parents, self.tree_parents
            tree_is_start = not tree_is_start  # Track which tree is start after swap
        
        print(f"❌ RRT-Connect failed after {self.max_iterations} iterations")
        return None


class JointSpaceRRTConnectFailure(JointSpaceRRTConnect):
    """
    Overrides AbstractRRTPlanner _find_nearest_node to include failure cost.
    """
    def __init__(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel, 
                 ik_solver: IKSolver, collision_threshold: float, seed: int, **kwargs):
        super().__init__(scene_model, robot_model, ik_solver, collision_threshold, seed=seed, **kwargs)

        self.mjData = mujoco.MjData(self.model)
        
        self.failure_weight = 0.05
        failing_joints = [f"joint{i}" for i in range(1,8)]
        self.collision_estimator = CollisionEstimator(self.model, self.mjData, failing_joints=failing_joints)

    def safety_cost(self, config=None):
        """
        Calculates \\sum_rj \\sum_ei {weight * P(x, rj, ei | F) * S(rj, ei) }  
        """
        if config is not None: # similar to how collision_checker does it
            self.collision_checker.set_robot_configuration_direct(config)
        prob_collision_pairs = self.collision_estimator.estimate_bodies_in_collision()
        # prob_collision_pairs: (N, 2)
        # get severity of collision pairs
        # assume severity = 1 for all interactions
        safety_cost = self.failure_weight * prob_collision_pairs.shape[0] # simply the total count of how many estimated collisions there are
        
        print(f"Estimated {prob_collision_pairs.shape[0]} potential collisions, safety cost = {safety_cost}")
        print(f"Prob collision pairs: {prob_collision_pairs}\n")

        return safety_cost


    def extend_tree(self, tree: List, parents: dict, target_config: np.ndarray) -> Tuple[str, Optional[int]]:
        """
        Extend tree toward target configuration.
        Added safety cost to each node.
        Returns: (status, node_index)
        status: 'reached', 'advanced', or 'trapped'
        """
        if not tree:
            return 'trapped', None
            
        # Find nearest node in tree
        nearest_idx = 0
        min_distance = float('inf')
        for i, node in enumerate(tree):
            dist = self.distance(node.config, target_config) + node.cost
            print(f"Node {i} at {node.config} has distance {dist} (cost {node.cost})")
            
            if dist < min_distance:
                min_distance = dist
                nearest_idx = i
        
        nearest_node = tree[nearest_idx]
        
        # Steer toward target
        new_config = self.steer(nearest_node.config, target_config)
        
        # Check if new configuration and path are valid
        if not self.is_valid_config(new_config):
            return 'trapped', None
        
        # mData has the new_config as its ctrl
        new_config_safety_cost = self.safety_cost(config)

        if not self.is_path_valid(nearest_node.config, new_config):
            return 'trapped', None
        
        # Add new node to tree
        new_node = PlanningNode(new_config, cost=new_config_safety_cost)

        print(f"{new_node.config} has failure cost {new_node.cost}")

        tree.append(new_node)
        new_idx = len(tree) - 1
        parents[new_idx] = nearest_idx
        
        # Check if we reached the target exactly
        if self.distance(new_config, target_config) < self.goal_tolerance:
            return 'reached', new_idx
        else:
            return 'advanced', new_idx
        
    def plan(self, start_config: np.ndarray, goal_config: np.ndarray, 
            frame_name: str = "end_effector") -> Optional[List[np.ndarray]]:
        """
        Plan path using RRT-Connect algorithm with proper path reconstruction.
        Cost of each node in tree is the safety cost for that configuration.
        """
        print(f"🔄 Starting RRT-Connect planning...")

        self.rng = np.random.RandomState(self.seed)

        # Initialize trees
        self.tree = [PlanningNode(start_config, cost=self.safety_cost(start_config))]
        self.tree_parents = {0: None}
        
        self.goal_tree = [PlanningNode(goal_config, cost=self.safety_cost(goal_config))]
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
                        print(f"✅ Goal reached in {iteration + 1} iterations!")
                        return self.reconstruct_path(
                            self.tree, self.goal_tree, 
                            self.tree_parents, self.goal_tree_parents,
                            new_idx1, new_idx2, tree_is_start
                        )
                    elif status2 == 'trapped':
                        break
                    
                    # Check if the trees can be directly connected
                    if self.is_path_valid(self.tree[new_idx1].config, self.goal_tree[new_idx2].config):
                        print(f"✅ Trees connected in {iteration + 1} iterations!")
                        return self.reconstruct_path(
                            self.tree, self.goal_tree,
                            self.tree_parents, self.goal_tree_parents,
                            new_idx1, new_idx2, tree_is_start
                        )
            
            # Swap trees for next iteration (alternate growth direction)
            self.tree, self.goal_tree = self.goal_tree, self.tree
            self.tree_parents, self.goal_tree_parents = self.goal_tree_parents, self.tree_parents
            tree_is_start = not tree_is_start  # Track which tree is start after swap
        
        print(f"❌ RRT-Connect failed after {self.max_iterations} iterations")
        return None

