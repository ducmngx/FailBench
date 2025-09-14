import numpy as np
import mujoco
from typing import List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from enum import Enum
import time

# Assuming imports from the IK module
# from inverse_kinematics import IKSolver, EndEffectorTarget, IKConfig, IKResult, solve_ik_for_planner
from planner.kinematics.inverse_kinematics import *
# from abstract_planner import *
from planner.algorithms.abstract_planner import *
# from collision_checker import CollisionChecker
from planner.collision.collision_checker import CollisionChecker
from failure_injection.collision_estimation import CollisionEstimator
from failure_injection.optimized_collision_estimation import OptimizedCollisionEstimator

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
        # self.target_bids = [12, 14]     # hack
        # self.noisy_steer_std = 1e-3
        # self.failure_grad_lr = 1e-1

    # def plan(self, start_config: np.ndarray, goal_config: np.ndarray, 
    #         frame_name: str = "end_effector") -> Optional[List[np.ndarray]]:
    #     """
    #     Plan path using RRT-Connect algorithm with proper path reconstruction.
    #     Cost of each node in tree is the cost of failure at that configuration + distance to goal
    #     """
    #     print(f"🔄 Starting RRT-Connect planning...")
    #     self.collision_estimator.count = 0
    #     self.rng = np.random.RandomState(self.seed)

    #     # Initialize trees
        
    #     cost = self.distance(start_config, goal_config) + self.impact_of_failure(start_config)
    #     start_node = PlanningNode(start_config, cost=cost)
    #     self.tree = [start_node]
    #     self.tree_parents = {0: None}
        
    #     cost = self.distance(goal_config, start_config) + self.impact_of_failure(goal_config)
    #     goal_node = PlanningNode(goal_config, cost=cost)
    #     self.goal_tree = [goal_node]
    #     self.goal_tree_parents = {0: None}
        
    #     # Track which tree is the start tree (important for path reconstruction)
    #     tree_is_start = True  # True = self.tree is start tree, False = self.tree is goal tree
        
    #     # # Check if start and goal are directly connected
    #     # if self.is_path_valid(start_config, goal_config):
    #     #     print("✅ Direct path found, densifying...")
    #     #     # Create smooth interpolated path
    #     #     return self.densify_direct_path(start_config, goal_config, num_waypoints=3)
        
    #     for iteration in range(self.max_iterations):
    #         # Sample random configuration
    #         rand_config = self.sample_random_config()
    #         if rand_config is None:
    #             continue
            
    #         # Extend first tree toward random config
    #         status1, new_idx1 = self.extend_tree(self.tree, self.tree_parents, rand_config)
            
    #         if status1 != 'trapped':
    #             # sample tree nodes not based on the latest node, since its noisy, its not guaranteed to be the best node. 
    #             # sample_weights = np.array([1/(node.cost + 1e-9) for node in self.tree])
    #             # sample_weights = sample_weights / np.sum(sample_weights)
    #             # new_config = self.rng.choice(self.tree, p=sample_weights).config
    #             # Try to connect second tree to the new node
    #             new_config = self.tree[new_idx1].config
                
    #             # Keep extending second tree toward new node until trapped or reached
    #             while True:
    #                 status2, new_idx2 = self.extend_tree(self.goal_tree, self.goal_tree_parents, new_config)
                    
    #                 if status2 == 'reached':
    #                     # Trees connected!
    #                     print(f"✅ Goal reached in {iteration + 1} iterations!")
    #                     if self.count > 0:
    #                         print("number of calls to collision_estimator:", self.count)
    #                         print(f"avg time for impact_of_failure(): {self.total_fail_cost_time/self.count:.5f}s")
    #                     return self.reconstruct_path(
    #                         self.tree, self.goal_tree, 
    #                         self.tree_parents, self.goal_tree_parents,
    #                         new_idx1, new_idx2, tree_is_start
    #                     )
    #                 elif status2 == 'trapped':
    #                     break
                    
    #                 # Check if the trees can be directly connected
    #                 # if self.is_path_valid(self.tree[new_idx1].config, self.goal_tree[new_idx2].config):
    #                 #     print(f"✅ Trees connected in {iteration + 1} iterations!")
    #                 #     if self.count > 0:
    #                 #         print("number of calls to collision_estimator:", self.count)
    #                 #         print(f"avg time for impact_of_failure(): {self.total_fail_cost_time/self.count:.5f}s")
    #                 #     return self.reconstruct_path(
    #                 #         self.tree, self.goal_tree,
    #                 #         self.tree_parents, self.goal_tree_parents,
    #                 #         new_idx1, new_idx2, tree_is_start
    #                 #     )
            
    #         # Swap trees for next iteration (alternate growth direction)
    #         self.tree, self.goal_tree = self.goal_tree, self.tree
    #         self.tree_parents, self.goal_tree_parents = self.goal_tree_parents, self.tree_parents
    #         tree_is_start = not tree_is_start  # Track which tree is start after swap
        
    #     print(f"❌ RRT-Connect failed after {self.max_iterations} iterations")
    #     return None
    
    # def extend_tree(self, tree: List, parents: dict, target_config: np.ndarray) -> Tuple[str, Optional[int]]:
    #     """
    #     Extend tree toward target configuration using noisy steering.
    #     Added failure cost to each node.
    #     Returns: (status, node_index)
    #     status: 'reached', 'advanced', or 'trapped'
    #     """
    #     if not tree:
    #         return 'trapped', None
            
    #     # Find nearest node in tree
    #     nearest_idx = 0
    #     min_distance = float('inf')
    #     min_fail_cost = float('inf')
    #     for i, node in enumerate(tree):
    #         euc_dist = self.distance(node.config, target_config)
    #         fail_cost = self.impact_of_failure(node.config)
    #         print(f"Node {i} at {node.config} has failure cost {fail_cost} and euc_dist {euc_dist}")
    #         dist =  euc_dist + fail_cost
    #         if dist < min_distance:
    #             min_distance = dist
    #             min_euc = euc_dist
    #             nearest_idx = i
    #             min_fail_cost = fail_cost
        
    #     nearest_node = tree[nearest_idx]
    #     nearest_node.count += 1
    #     print(f"Nearest node {nearest_idx}/{len(tree)} at {nearest_node.config} has distance {min_euc}, total cost {min_distance} (failure cost {min_fail_cost})")
        
    #     # Steer toward target
    #     new_config = self.steer(nearest_node.config, target_config)
    #     # # steer with gradients from failure
    #     # new_config = self.steer_with_fail_grad(nearest_node.config)

    #     # Check if new configuration and path are valid
    #     if not self.is_valid_config(new_config):
    #         return 'trapped', None
        

    #     if not self.is_path_valid(nearest_node.config, new_config):
    #         return 'trapped', None
        
    #     new_config_cost_of_failure = self.impact_of_failure(new_config)
    #     new_config_dist_to_goal = self.distance(new_config, self.goal_tree[0].config)
    #     # Add new node to tree
    #     new_node = PlanningNode(new_config, cost=new_config_cost_of_failure+new_config_dist_to_goal)

    #     # print(f"new node - {new_node.config} has failure cost {new_node.cost}")

    #     tree.append(new_node)
    #     new_idx = len(tree) - 1
    #     parents[new_idx] = nearest_idx
        
    #     # Check if we reached the target exactly
    #     if self.distance(new_config, target_config) < self.goal_tolerance:
    #         return 'reached', new_idx
    #     else:
    #         return 'advanced', new_idx

    # def steer_with_fail_grad(self, config_to_steer_to: np.ndarray) -> np.ndarray:
    #     """Steer in joint space with derivatives of failure severity estimation"""
    #     _, grads = self.impact_of_failure(config_to_steer_to, calculate_grads=True) # grads (m, nv) m=no. of overlapping bodies, nv = degrees of freedom
    #     return config_to_steer_to - self.failure_grad_lr*grads
        
    # def noisy_steer(self, from_config: np.ndarray, to_config: np.ndarray, stuck_count: int) -> np.ndarray:
    #     """Steer in joint space with noise"""
    #     direction = to_config - from_config
    #     distance = np.linalg.norm(direction)
    #     if distance <= self.step_size:
    #         return to_config
    #     else:
    #         noise = self.rng.normal(0, self.noisy_steer_std*(2**min(stuck_count, 10)), direction.shape)
    #         noisy_direction = direction + noise
    #         unit_noisy_direction = noisy_direction / np.linalg.norm(noisy_direction)

    #         return from_config + self.step_size * unit_noisy_direction
        
    # def is_valid_config(self, config):
    #     """Check if configuration is collision-free AND safe"""
    #     is_collision_free = not self.collision_checker.check_collisions(
    #         robot_config=config,
    #         threshold=self.collision_threshold
    #     )
    #     # is_safe = self.impact_of_failure(config) < self.failure_threshold
    #     return is_collision_free #and is_safe 
    
    def get_trajectory_cost(self, path):
        mechanical_cost = []
        safety_cost = []

        for config in path:
            safety_cost.append(self.impact_of_failure(config))

        mechanical_cost.append(self.distance(self._plan_start_config, path[0]))
        for i in range(len(path)-1):
            mechanical_cost.append(self.distance(path[i], path[i+1]))

        mechanical_cost, safety_cost = np.array(mechanical_cost), np.array(safety_cost)

        print(f"trajectory ({len(path)} waypoints) cost: {mechanical_cost.sum() + safety_cost.sum()}: (V: {mechanical_cost.sum() }, Safety: {safety_cost.sum()}, Safety weight: {self.failure_weight})")
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