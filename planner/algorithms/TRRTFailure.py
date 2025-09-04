import numpy as np
import time
import mujoco
from typing import List, Optional, Tuple, Callable, Union, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from enum import Enum
# Assuming imports from the IK module
from planner.kinematics.inverse_kinematics import *
from planner.algorithms.abstract_planner import *
from planner.collision.collision_checker import CollisionChecker
from failure_injection.collision_estimation import CollisionEstimator

class JointSpaceTRRTOMPL(AbstractRRTPlanner):
    """
    T-RRT implementation exactly following OMPL C++ code with all proper constants.
    """
    def __init__(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel, 
                 ik_solver: IKSolver, collision_threshold: float, seed: int, 
                 collision_estimator: CollisionEstimator,
                 object_positions: Dict[int, np.ndarray],
                 # OMPL T-RRT default parameters (matching C++ exactly)
                 init_temperature: float = 50,#100.0,           # initTemperature_
                 temp_change_factor: float = 0.1,           # setTempChangeFactor(0.1) - NOTE: this is INCREASE factor
                 goal_bias: float = 0.05,                   # Standard RRT goal bias
                 frontier_threshold: float = None,          # Auto-set as 0.01 * workspace extent  
                 frontier_node_ratio: float = 0.1,          # frontierNodeRatio_ = 0.1
                 cost_threshold: float = float('inf'),      # costThreshold_
                 failure_weight: float = 2.0,               # Custom parameter
                 **kwargs):
        kwargs.pop('planning_space', None)
        super().__init__(scene_model, ik_solver, planning_space=PlanningSpace.JOINT_SPACE, **kwargs)
        
        self.collision_threshold = collision_threshold
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        
        self.initialize_collision_checker(scene_model, robot_model)
        self.collision_estimator = collision_estimator
        
        # OMPL T-RRT parameters (exact match to C++)
        self.init_temperature = init_temperature
        self.temp_change_factor = temp_change_factor  # IMPORTANT: In OMPL this INCREASES temperature
        self.goal_bias = goal_bias
        self.frontier_threshold = frontier_threshold
        self.frontier_node_ratio = frontier_node_ratio
        self.cost_threshold = cost_threshold
        self.failure_weight = failure_weight
        
        # Runtime variables (initialized in setup(), matching OMPL)
        self.temp = init_temperature
        self.frontier_count = 1      # init to 1 to prevent division by zero
        self.nonfrontier_count = 1   # init to 1 to prevent division by zero
        self.best_cost = float('inf')
        self.worst_cost = 0.0
        
        # Object information
        self.object_positions = object_positions
        print(f"Object positions: {self.object_positions}")
        self.target_bids = [12, 14]
        self.avoid_bids = [13]
        
        # Severity mapping for strong avoidance
        self.llm_severity_map = {
            12: 2,   # Target objects 
            13: 20,  # Avoid object - very high severity  
            14: 2    # Target objects
        }
        
        # Aggressive parameters for avoidance
        self.max_failure_prob = 0.98
        self.distance_decay_rate = 50.0
        
        # Statistics
        self.nodes_created = 0
        self.transition_tests_passed = 0
        self.transition_tests_failed = 0
        
    def initialize_collision_checker(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel):
        """Initialize the collision checker."""
        self.collision_checker = CollisionChecker(scene_model, robot_model)
        
    def setup(self):
        """Setup T-RRT parameters exactly following OMPL setup()."""
        print("Setting up T-RRT with OMPL parameters...")
        
        # Set frontier threshold if not provided (OMPL does this in setup)
        if self.frontier_threshold is None or self.frontier_threshold < 1e-10:
            # OMPL: frontierThreshold_ = si_->getMaximumExtent() * 0.01;
            # Simplified workspace extent estimation
            workspace_extent = 2.0  # Approximate for robot arm
            self.frontier_threshold = workspace_extent * 0.01
            print(f"Auto-set frontier threshold: {self.frontier_threshold}")
        
        # Setup TRRT specific variables (matching OMPL exactly)
        self.temp = self.init_temperature
        self.nonfrontier_count = 1
        self.frontier_count = 1
        self.best_cost = float('inf')
        self.worst_cost = 0.0
        
        print(f"T-RRT setup complete:")
        print(f"  Initial temperature: {self.init_temperature}")
        print(f"  Temp change factor: {self.temp_change_factor}")
        print(f"  Frontier threshold: {self.frontier_threshold}")
        print(f"  Frontier node ratio: {self.frontier_node_ratio}")
        print(f"  Goal bias: {self.goal_bias}")
        print(f"  Cost threshold: {self.cost_threshold}")
        
    def distance(self, config1: np.ndarray, config2: np.ndarray) -> float:
        """Euclidean distance in joint space."""
        return np.linalg.norm(config1 - config2)
    
    def get_end_effector_position(self, config: np.ndarray) -> np.ndarray:
        """Get end effector position from joint configuration."""
        temp_data = mujoco.MjData(self.model)
        temp_data.qpos[:min(len(config), temp_data.qpos.shape[0])] = config[:temp_data.qpos.shape[0]]
        mujoco.mj_forward(self.model, temp_data)
        
        try:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
            if site_id != -1:
                return temp_data.site_xpos[site_id].copy()
        except:
            pass
        
        return np.array([0, 0, 0.5])
    
    def smooth_severity_mapping(self, llm_severity: int) -> float:
        """Convert discrete LLM severity to smooth values."""
        severity_map = {
            1: 0.1, 2: 0.3, 3: 0.6, 5: 1.2, 8: 3.0,
            10: 8.0, 15: 20.0, 20: 50.0
        }
        
        severities = sorted(severity_map.keys())
        values = [severity_map[s] for s in severities]
        return np.interp(llm_severity, severities, values)
    
    def distance_based_failure_prob_2d(self, config: np.ndarray) -> Dict[int, float]:
        """Calculate failure probability using XY distance only."""
        ee_pos = self.get_end_effector_position(config)
        failure_probs = {}
        
        for obj_id, obj_pos in self.object_positions.items():
            if obj_id in self.target_bids:
                continue
                
            xy_distance = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
            prob = self.max_failure_prob * np.exp(-self.distance_decay_rate * xy_distance)
            failure_probs[obj_id] = max(0.0, min(prob, self.max_failure_prob))
        
        return failure_probs
    
    def contextual_severity_2d(self, obj_id: int, xy_distance: float) -> float:
        """Apply distance-based falloff to severity."""
        llm_severity = self.llm_severity_map.get(obj_id, 1)
        base_severity = self.smooth_severity_mapping(llm_severity)
        
        base_rad = 0.5
        if llm_severity >= 15:
            danger_radius = base_rad * 8
        elif llm_severity >= 10:
            danger_radius = base_rad * 4
        elif llm_severity >= 5:
            danger_radius = base_rad * 2
        else:
            danger_radius = base_rad
        
        falloff = np.exp(-xy_distance / danger_radius)
        return base_severity * falloff
    
    def state_cost(self, config: np.ndarray) -> float:
        """Calculate cost of a state (OMPL stateCost equivalent)."""
        distance_probs = self.distance_based_failure_prob_2d(config)
        total_cost = 0
        ee_pos = self.get_end_effector_position(config)
        
        for obj_id, failure_prob in distance_probs.items():
            obj_pos = self.object_positions.get(obj_id)
            if obj_pos is None:
                continue
                
            xy_distance = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
            severity = self.contextual_severity_2d(obj_id, xy_distance)
            total_cost += failure_prob * severity
        
        return total_cost * self.failure_weight
    
    def motion_cost(self, from_config: np.ndarray, to_config: np.ndarray) -> float:
        """Calculate cost of motion between two states (OMPL motionCost equivalent)."""
        from_cost = self.state_cost(from_config)
        to_cost = self.state_cost(to_config)
        distance = self.distance(from_config, to_config)
        
        avg_cost = (from_cost + to_cost) / 2.0
        return avg_cost * distance
    
    def transition_test(self, motion_cost: float) -> bool:
        """OMPL transition test - EXACT implementation from C++ code."""
        # Disallow any cost that is not better than the cost threshold
        if motion_cost >= self.cost_threshold:
            return False
        
        # Always accept if the cost is near or below zero
        if motion_cost < 1e-4:
            return True
        
        d_cost = motion_cost
        # OMPL: double transitionProbability = exp(-dCost / temp_);
        transition_probability = np.exp(-d_cost / self.temp)
        
        if transition_probability > 0.5:
            # Successful transition test. Decrease the temperature slightly
            cost_range = self.worst_cost - self.best_cost
            if abs(cost_range) > 1e-4:  # Do not divide by zero
                # OMPL: temp_ /= exp(dCost / (0.1 * costRange));
                self.temp /= np.exp(d_cost / (0.1 * cost_range))
            
            self.transition_tests_passed += 1
            return True
        
        # The transition failed. Increase the temperature (slightly)
        # OMPL: temp_ *= tempChangeFactor_;
        # NOTE: In OMPL, tempChangeFactor_ is set to 0.1 + 1.0 = 1.1 for INCREASE
        self.temp *= (1.0 + self.temp_change_factor)
        self.transition_tests_failed += 1
        return False
    
    def min_expansion_control(self, rand_motion_distance: float) -> bool:
        """OMPL minimum expansion control - EXACT implementation."""
        if rand_motion_distance > self.frontier_threshold:
            # participates in the tree expansion
            self.frontier_count += 1
            return True
        else:
            # participates in the tree refinement
            # check our ratio first before accepting it
            if self.nonfrontier_count / self.frontier_count > self.frontier_node_ratio:
                # reject this node as being too much refinement
                return False
            
            self.nonfrontier_count += 1
            return True
    
    def sample_random_config(self) -> Optional[np.ndarray]:
        """Sample configuration with goal bias."""
        if hasattr(self, 'goal_config') and self.rng.random() < self.goal_bias:
            return self.goal_config.copy()
        else:
            return self.ik_solver.get_random_valid_config(rng=self.rng)
    
    def is_valid_config(self, config: np.ndarray) -> bool:
        """Check if configuration is collision-free."""
        return not self.collision_checker.check_collisions(
            robot_config=config,
            threshold=self.collision_threshold
        )
    
    def steer(self, from_config: np.ndarray, to_config: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Steer from near config toward random config.
        Following OMPL interpolation logic.
        """
        direction = to_config - from_config
        distance = np.linalg.norm(direction)
        
        # OMPL uses maxDistance_ for step limiting
        max_distance = self.step_size
        
        if distance > max_distance:
            # OMPL: si_->getStateSpace()->interpolate(nearMotion->state, randState, 
            #       maxDistance_ / randMotionDistance, interpolatedState);
            t = max_distance / distance
            new_config = from_config + t * (to_config - from_config)
            actual_distance = max_distance
        else:
            # Random state is close enough - use it directly  
            new_config = to_config.copy()
            actual_distance = distance
        
        return new_config, actual_distance
    
    def is_path_valid(self, from_config: np.ndarray, to_config: np.ndarray, max_cost_threshold=20) -> bool:
        """Check if path between configs is collision-free (OMPL checkMotion)."""
        distance = np.linalg.norm(to_config - from_config)
        steps = max(10, int(distance / (self.step_size * 0.1)))
        
        for i in range(steps + 1):
            alpha = i / steps
            intermediate = (1 - alpha) * from_config + alpha * to_config
            if not self.is_valid_config(intermediate):
                return False
            # Add this cost check to avoid high-cost states
            if self.state_cost(intermediate) > max_cost_threshold:
                return False
        return True
    
    def plan(self, start_config: np.ndarray, goal_config: np.ndarray, 
             frame_name: str = "end_effector") -> Optional[List[np.ndarray]]:
        """
        T-RRT planning following OMPL solve() method exactly.
        """
        print("Starting OMPL-exact T-RRT planning...")
        
        # Setup (matching OMPL)
        self.setup()
        self.rng = np.random.RandomState(self.seed)
        self.goal_config = goal_config
        
        # Initialize tree with start state (matching OMPL)
        start_cost = self.state_cost(start_config)
        self.tree = [PlanningNode(start_config, cost=start_cost)]
        tree_parents = {0: None}
        
        # Initialize cost tracking (matching OMPL)
        if len(self.tree) == 1:  # First solve call
            self.best_cost = self.worst_cost = start_cost
        
        print(f"Start cost: {start_cost:.4f}")
        print(f"Goal cost: {self.state_cost(goal_config):.4f}")
        
        # Solution tracking (matching OMPL)
        solution_node_idx = None
        approx_solution_node_idx = None
        approx_difference = float('inf')
        
        for iteration in range(self.max_iterations):
            # I. Sample random state (with goal biasing) - OMPL logic
            rand_config = self.sample_random_config()
            if rand_config is None:
                continue
            
            # II. Find closest state in tree - OMPL nearestNeighbors_->nearest()
            nearest_idx = 0
            min_distance = float('inf')
            for i, node in enumerate(self.tree):
                dist = self.distance(node.config, rand_config)
                if dist < min_distance:
                    min_distance = dist
                    nearest_idx = i
            
            near_node = self.tree[nearest_idx]
            
            # III. Steer toward random state - OMPL interpolation logic
            new_config, rand_motion_distance = self.steer(near_node.config, rand_config)
            
            # IV. Check motion validity - OMPL si_->checkMotion()
            if not self.is_path_valid(near_node.config, new_config):
                continue  # try a new sample
            
            # V. Minimum Expansion Control - OMPL minExpansionControl()
            if not self.min_expansion_control(rand_motion_distance):
                continue  # give up on this one and try a new sample
            
            # VI. Calculate costs - OMPL opt_->stateCost() and opt_->motionCost()
            child_cost = self.state_cost(new_config)
            motion_cost_value = self.motion_cost(near_node.config, new_config)
            
            # VII. Transition test - OMPL transitionTest()
            if not self.transition_test(motion_cost_value):
                continue  # give up on this one and try a new sample
            
            # VIII. Create and add new motion - OMPL logic
            new_node = PlanningNode(new_config, cost=child_cost)
            new_idx = len(self.tree)
            self.tree.append(new_node)
            tree_parents[new_idx] = nearest_idx
            self.nodes_created += 1
            
            # Update cost bounds - OMPL logic
            if child_cost < self.best_cost:
                self.best_cost = child_cost
            if child_cost > self.worst_cost:
                self.worst_cost = child_cost
            
            # IX. Check if goal is reached - OMPL goal->isSatisfied()
            goal_distance = self.distance(new_config, goal_config)
            
            if goal_distance < self.goal_tolerance:
                solution_node_idx = new_idx
                approx_difference = goal_distance
                print(f"Goal reached in {iteration + 1} iterations!")
                break
            
            # Track best approximation - OMPL logic
            if goal_distance < approx_difference:
                approx_difference = goal_distance
                approx_solution_node_idx = new_idx
            
            # Progress reporting
            if iteration % 200 == 0:
                accept_rate = self.transition_tests_passed / max(1, self.transition_tests_passed + self.transition_tests_failed)
                print(f"Iter {iteration}: temp={self.temp:.3f}, tree={len(self.tree)}, "
                      f"accept_rate={accept_rate:.3f}, best_goal_dist={approx_difference:.4f}")
                print(f"  Frontier/Non-frontier ratio: {self.nonfrontier_count}/{self.frontier_count} = {self.nonfrontier_count/self.frontier_count:.3f}")
        
        # Solution processing - OMPL logic
        final_node_idx = solution_node_idx if solution_node_idx is not None else approx_solution_node_idx
        
        if final_node_idx is None:
            print("No solution found")
            return None
        
        # Build path - OMPL path reconstruction
        path = []
        current = final_node_idx
        while current is not None:
            path.append(self.tree[current].config.copy())
            current = tree_parents.get(current)
        
        path.reverse()
        
        # Statistics
        total_state_cost = sum(self.state_cost(config) for config in path)
        total_motion_cost = sum(
            self.motion_cost(path[i], path[i+1]) 
            for i in range(len(path)-1)
        ) if len(path) > 1 else 0
        
        print(f"\nOMPL T-RRT Solution:")
        print(f"  Created {self.nodes_created} states")
        print(f"  Path length: {len(path)} waypoints")
        print(f"  Total state cost: {total_state_cost:.4f}")
        print(f"  Total motion cost: {total_motion_cost:.4f}")
        print(f"  Final temperature: {self.temp:.4f}")
        print(f"  Frontier nodes: {self.frontier_count}")
        print(f"  Non-frontier nodes: {self.nonfrontier_count}")
        print(f"  Transition success rate: {self.transition_tests_passed/(self.transition_tests_passed + self.transition_tests_failed):.3f}")
        
        is_approximate = solution_node_idx is None
        if is_approximate:
            print(f"  Solution is approximate (distance to goal: {approx_difference:.4f})")
        
        return path
    
    def debug_transition_behavior(self):
        """Debug the transition test behavior."""
        print(f"\nTransition Test Debug:")
        print(f"  Current temperature: {self.temp:.4f}")
        print(f"  Initial temperature: {self.init_temperature:.4f}")
        print(f"  Temp change factor: {self.temp_change_factor:.4f}")
        print(f"  Cost range: {self.worst_cost - self.best_cost:.4f}")
        print(f"  Passed/Failed: {self.transition_tests_passed}/{self.transition_tests_failed}")
        
        # Test transition probabilities for different costs
        test_costs = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        print(f"  Acceptance probabilities:")
        for cost in test_costs:
            prob = np.exp(-cost / self.temp)
            print(f"    Cost {cost:6.1f}: P(accept) = {prob:.6f}")
        
        return self.temp
    
    def set_ompl_aggressive_parameters(self):
        """Set parameters for aggressive avoidance while maintaining OMPL structure."""
        print("Setting OMPL-compatible aggressive parameters...")
        
        # Increase cost sensitivity
        self.failure_weight = 5.0
        self.distance_decay_rate = 80.0
        
        # Lower initial temperature for more selectivity
        self.init_temperature = 50.0
        self.temp = 50.0
        
        # Smaller temperature increase factor (more selective)
        self.temp_change_factor = 0.05  # Will become 1.05 multiplier
        
        # Higher severity for object to avoid
        self.llm_severity_map[13] = 100
        
        print(f"New parameters:")
        print(f"  Failure weight: {self.failure_weight}")
        print(f"  Distance decay: {self.distance_decay_rate}")
        print(f"  Initial temp: {self.init_temperature}")
        print(f"  Temp change factor: {self.temp_change_factor}")
        print(f"  Object 2 severity: {self.llm_severity_map[13]}")


    ##### DEBUGGING METHODS #####

    def distance_based_failure_prob_2d_ee(self, ee_pos: np.ndarray) -> Dict[int, float]:
        """Calculate failure probability using XY distance only."""
        # ee_pos = self.get_end_effector_position(config)
        failure_probs = {}
        
        for obj_id, obj_pos in self.object_positions.items():
            if obj_id in self.target_bids:
                continue
                
            xy_distance = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
            prob = self.max_failure_prob * np.exp(-self.distance_decay_rate * xy_distance)
            failure_probs[obj_id] = max(0.0, min(prob, self.max_failure_prob))
        
        return failure_probs
    
    def state_cost_ee(self, ee_pos: np.ndarray) -> float:
        """Calculate cost of a state (OMPL stateCost equivalent)."""
        distance_probs = self.distance_based_failure_prob_2d_ee(ee_pos)
        total_cost = 0
        # ee_pos = self.get_end_effector_position(config)
        print(f"EE Position: {ee_pos}, Distance-based failure probs: {distance_probs}")
        for obj_id, failure_prob in distance_probs.items():
            obj_pos = self.object_positions.get(obj_id)
            if obj_pos is None:
                continue
                
            xy_distance = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
            severity = self.contextual_severity_2d(obj_id, xy_distance)
            total_cost += failure_prob * severity
            print(f"\t\tObj {obj_id}: prob={failure_prob:.3f}, sev={severity:.3f}, contrib={failure_prob*severity:.3f}")
        return total_cost * self.failure_weight

    def verify_cost_landscape_grid(self, height=0.15, resolution=0.05):
        """Sample costs in a grid pattern around objects."""
        obj2_pos = self.object_positions.get(13)
        if obj2_pos is None:
            print("Object 2 not found")
            return None
        
        # Define sampling grid
        minMaxRange = 1.0
        x_min, x_max = obj2_pos[0] - minMaxRange, obj2_pos[0] + minMaxRange
        y_min, y_max = obj2_pos[1] - minMaxRange, obj2_pos[1] + minMaxRange
        
        x_samples = np.arange(x_min, x_max, resolution)
        y_samples = np.arange(y_min, y_max, resolution)
        
        costs = []
        positions = []
        ik_failures = 0
        
        for x in x_samples:
            for y in y_samples:
                test_pos = np.array([x, y, height])
                cost = self.state_cost_ee(test_pos)
                costs.append(cost)
                positions.append([x, y, cost])
                
                # try:
                #     target = EndEffectorTarget(
                #         position=test_pos,
                #         frame_name="end_effector",
                #         frame_type="site"
                #     )
                #     for attempt in range(20):
                #         print(f"Attempting IK at ({x:.2f}, {y:.2f}), attempt {attempt+1}")
                #         seed = self.ik_solver.get_random_valid_config(rng=np.random.RandomState(42))
                #         solution, result = self.ik_solver.solve(target, seed)
                    
                #         if result == IKResult.SUCCESS:
                #             print(f"IK success at ({x:.2f}, {y:.2f}) on attempt {attempt+1}")
                #             cost = self.state_cost(solution)
                #             costs.append(cost)
                #             positions.append([x, y, cost])
                #     else:
                #         ik_failures += 1
                # except:
                #     ik_failures += 1
        
        if not costs:
            print("No valid IK solutions found in sampling region")
            return None
        
        costs = np.array(costs)
        positions = np.array(positions)
        
        # Analysis
        print(f"Cost Landscape Analysis:")
        print(f"  Samples: {len(costs)} valid, {ik_failures} IK failures")
        print(f"  Cost range: {costs.min():.4f} to {costs.max():.4f}")
        print(f"  Cost ratio: {costs.max()/costs.min():.2f}")
        print(f"  Mean cost: {costs.mean():.4f}")
        print(f"  Std dev: {costs.std():.4f}")
        
        # Find high-cost regions (potential barriers)
        high_cost_threshold = costs.mean() + 2 * costs.std()
        high_cost_points = positions[costs > high_cost_threshold]
        
        print(f"  High-cost barrier points: {len(high_cost_points)}")
        
        # Check if high-cost region surrounds object 2
        if len(high_cost_points) > 0:
            distances_to_obj2 = [np.linalg.norm(point[:2] - obj2_pos[:2]) 
                            for point in high_cost_points]
            avg_distance = np.mean(distances_to_obj2)
            print(f"  Avg distance of barriers from obj2: {avg_distance:.3f}m")
            
            if avg_distance < 0.3:
                print("  ✓ High-cost barriers surround object 2")
            else:
                print("  ⚠ High-cost barriers may be too far from object 2")
        
        return positions, costs

    def verify_radial_cost_profile(self, center_obj_id=13, max_radius=0.5, num_samples=20):
        """Sample costs at increasing distances from object center."""
        obj_pos = self.object_positions.get(center_obj_id)
        if obj_pos is None:
            return None
        
        distances = np.linspace(0.01, max_radius, num_samples)
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Sample in different directions
        
        print(f"Radial Cost Profile from Object {center_obj_id}:")
        
        all_costs = []
        for angle in angles:
            direction_costs = []
            print(f"\nDirection {angle*180/np.pi:.0f} degrees:")
            
            for dist in distances:
                x = obj_pos[0] + dist * np.cos(angle)
                y = obj_pos[1] + dist * np.sin(angle)
                test_pos = np.array([x, y, obj_pos[2] + 0.15])

                cost = self.state_cost_ee(test_pos)
                direction_costs.append(cost)
                
                # try:
                #     target = EndEffectorTarget(position=test_pos)
                #     solution, result = self.ik_solver.solve(target, np.zeros(7))
                    
                #     if result == IKResult.SUCCESS:
                #         cost = self.state_cost(solution)
                #         direction_costs.append(cost)
                #         print(f"  {dist:.3f}m: cost = {cost:.4f}")
                #     else:
                #         direction_costs.append(None)
                #         print(f"  {dist:.3f}m: IK failed")
                # except:
                #     direction_costs.append(None)
                #     print(f"  {dist:.3f}m: Error")
            
            all_costs.append(direction_costs)
        
        # Check for proper exponential decay
        valid_costs = [c for c in all_costs[0] if c is not None]  # Use first direction
        if len(valid_costs) >= 3:
            print(f"\nCost decay analysis (first direction):")
            for i in range(1, len(valid_costs)):
                if valid_costs[i-1] > 0:
                    decay_ratio = valid_costs[i] / valid_costs[i-1]
                    print(f"  Step {i}: decay ratio = {decay_ratio:.4f}")
            
            # Should see exponential decay pattern
            first_half_avg = np.mean(valid_costs[:len(valid_costs)//2])
            second_half_avg = np.mean(valid_costs[len(valid_costs)//2:])
            overall_decay = second_half_avg / first_half_avg if first_half_avg > 0 else 0
            
            print(f"  Overall decay (far/near): {overall_decay:.4f}")
            if overall_decay < 0.1:
                print("  ✓ Strong decay - good for avoidance")
            elif overall_decay < 0.3:
                print("  ✓ Moderate decay - should work")
            else:
                print("  ⚠ Weak decay - may not avoid effectively")
        
        return distances, all_costs
    
    def verify_path_cost_comparison(self):
        """Compare costs of direct vs avoidance paths."""
        obj1_pos = self.object_positions.get(12)
        obj2_pos = self.object_positions.get(13) 
        obj3_pos = self.object_positions.get(14)
        
        if any(x is None for x in [obj1_pos, obj2_pos, obj3_pos]):
            print("Not all object positions found")
            return

        
        # Define test paths
        paths = {
            "direct_over_obj2": [
                obj1_pos + np.array([0, 0, 0.15]),
                obj2_pos + np.array([0, 0, 0.20]),  # Over object 2
                obj3_pos + np.array([0, 0, 0.15])
            ],
            "around_obj2": [
                obj1_pos + np.array([0, 0, 0.15]),
                obj2_pos + np.array([0.1, -0.2, 0.15]),  # Around object 2
                obj3_pos + np.array([0, 0, 0.15])
            ],
            "wide_around": [
                obj1_pos + np.array([0, 0, 0.15]),
                obj2_pos + np.array([3, -2, 0.15]),  # Wide around object 2
                obj3_pos + np.array([0, 0, 0.15])
            ]
        }
        
        print("Path Cost Comparison:")
        path_costs = {}
        
        for path_name, waypoints in paths.items():
            total_cost = 0
            valid_path = True
            
            for waypoint in waypoints:
                cost = self.state_cost_ee(waypoint)
                total_cost += cost
                
                # try:
                #     target = EndEffectorTarget(position=waypoint)
                #     solution, result = self.ik_solver.solve(target, np.zeros(7))
                    
                #     if result == IKResult.SUCCESS:
                #         cost = self.state_cost(solution)
                #         total_cost += cost
                #     else:
                #         valid_path = False
                #         break
                # except:
                #     valid_path = False
                #     break
            
            if valid_path:
                path_costs[path_name] = total_cost
                print(f"  {path_name:15}: {total_cost:.4f}")
            else:
                print(f"  {path_name:15}: IK failed")
        
        # Analysis
        if len(path_costs) >= 2:
            direct_cost = path_costs.get("direct_over_obj2", 0)
            around_cost = path_costs.get("around_obj2", 0)
            
            if direct_cost > 0 and around_cost > 0:
                avoidance_benefit = direct_cost / around_cost
                print(f"\nAvoidance benefit ratio: {avoidance_benefit:.2f}")
                
                if avoidance_benefit > 5:
                    print("  ✓ Strong incentive to avoid object 2")
                elif avoidance_benefit > 2:
                    print("  ✓ Moderate incentive to avoid")
                else:
                    print("  ⚠ Weak avoidance incentive")
        
        return path_costs
    
    def verify_cost_gradients(self):
        """Check if cost gradients point away from object 2."""
        obj2_pos = self.object_positions.get(13)
        if obj2_pos is None:
            return
        
        # Sample points around object 2
        test_points = [
            obj2_pos + np.array([0.1, 0, 0.15]),
            obj2_pos + np.array([-0.1, 0, 0.15]),
            obj2_pos + np.array([0, 0.1, 0.15]),
            obj2_pos + np.array([0, -0.1, 0.15])
        ]
        
        print("Cost Gradient Analysis:")
        
        for i, center_pos in enumerate(test_points):
            print(f"\nGradient at point {i+1}:")
            
            # Calculate numerical gradient
            delta = 0.02
            gradients = []
            
            directions = [
                np.array([delta, 0, 0]),
                np.array([-delta, 0, 0]),
                np.array([0, delta, 0]),
                np.array([0, -delta, 0])
            ]
            
            try:
                # Center cost
                center_target = EndEffectorTarget(position=center_pos)
                center_solution, center_result = self.ik_solver.solve(center_target, np.zeros(7))
                
                if center_result != IKResult.SUCCESS:
                    continue
                    
                center_cost = self.state_cost(center_solution)
                
                for direction in directions:
                    test_pos = center_pos + direction
                    target = EndEffectorTarget(position=test_pos)
                    solution, result = self.ik_solver.solve(target, np.zeros(7))
                    
                    if result == IKResult.SUCCESS:
                        cost = self.state_cost(solution)
                        gradient = (cost - center_cost) / np.linalg.norm(direction)
                        gradients.append(gradient)
                        
                        # Check if gradient points away from object 2
                        to_obj2 = obj2_pos[:2] - center_pos[:2]
                        away_from_obj2 = direction[:2]
                        
                        dot_product = np.dot(to_obj2, away_from_obj2)
                        if dot_product < 0:  # Points away from object 2
                            direction_assessment = "away from obj2 ✓"
                        else:
                            direction_assessment = "toward obj2 ⚠"
                        
                        print(f"  Direction {direction[:2]}: gradient = {gradient:.4f} ({direction_assessment})")
            
            except Exception as e:
                print(f"  Error: {e}")