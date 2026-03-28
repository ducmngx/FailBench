import logging
import numpy as np
import time
import mujoco
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import scipy.ndimage
from planner.kinematics.inverse_kinematics import *
from planner.algorithms.abstract_planner import *
from planner.collision.collision_checker import CollisionChecker
from planner.costs.failure_cost import FailureCostModel, SeverityConfig
from failure_injection.collision_estimation import CollisionEstimator

logger = logging.getLogger(__name__)

class JointSpaceSTOMP():
    """
    STOMP (Stochastic Trajectory Optimization for Motion Planning) implementation.
    Uses trajectory-level optimization to minimize costs while avoiding obstacles.
    Standalone implementation not inheriting from AbstractRRTPlanner.
    """
    
    def __init__(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel,
                 ik_solver: IKSolver, collision_threshold: float, seed: int,
                 collision_estimator: CollisionEstimator,
                 object_positions: Dict[int, np.ndarray],
                 # STOMP-specific parameters
                 num_timesteps: int = 20,           # Number of waypoints in trajectory
                 num_rollouts: int = 50,            # Number of noisy trajectories per iteration
                 max_iterations: int = 100,         # STOMP iterations
                 control_cost_weight: float = 1e-4, # Smoothness regularization
                 noise_decay: float = 0.95,         # Noise reduction per iteration
                 learning_rate: float = 0.5,        # Update step size
                 failure_weight: float = 10.0,      # Safety cost weight
                 **kwargs):
        
        # Store scene and robot models
        self.model = scene_model
        self.robot_model = robot_model
        self.ik_solver = ik_solver
        self.collision_threshold = collision_threshold
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        
        self.initialize_collision_checker(scene_model, robot_model)
        self.collision_estimator = collision_estimator

        self.count = 0                  # debugging
        self.total_fail_cost_time = 0   # debugging 
        
        # STOMP parameters
        self.num_timesteps = num_timesteps
        self.num_rollouts = num_rollouts
        self.stomp_max_iterations = max_iterations  # Different from RRT max_iterations
        self.control_cost_weight = control_cost_weight
        self.noise_decay = noise_decay
        self.learning_rate = learning_rate
        self.failure_weight = failure_weight
        
        # Object information for cost calculation
        self.object_positions = object_positions
        self.target_bids = [12, 13]             
        self.avoid_bids = [bid for bid in object_positions.keys() if bid not in self.target_bids]       # avoid_bids = object_bids \ target_bids

        # Table avoidance parameters (similar to your object avoidance)
        self.max_table_cost = 25.0  # Maximum cost for being over table
        self.table_distance_decay_rate = 20.0  # How quickly cost decays with distance
        self.table_penalty_rate = 50.0  # Penalty rate for being inside table area
        self.table_penetration_rate = 10.0    
        
        # Build shared cost model for object avoidance
        self._severity_config = SeverityConfig(
            severity_map={12: 10, 13: 50, 14: 10, 0: 8},
            target_body_ids=self.target_bids,
            max_failure_prob=0.98,
            distance_decay_rate=50.0,
            failure_weight=failure_weight,
        )
        self.cost_model = FailureCostModel(
            scene_model=scene_model,
            object_positions=object_positions,
            config=self._severity_config,
        )
        
        # STOMP state variables
        self.current_noise_stddev = 0.1  # Initial noise level
        self.trajectory_dof = 7  # Joint space dimensionality
        
        logger.info(f"STOMP initialized with {num_timesteps} timesteps, {num_rollouts} rollouts")
        
    def initialize_collision_checker(self, scene_model: mujoco.MjModel, robot_model: mujoco.MjModel):
        """Initialize collision checker."""
        self.collision_checker = CollisionChecker(scene_model, robot_model)
    
    def get_end_effector_position(self, config: np.ndarray) -> np.ndarray:
        """Get end effector position from joint configuration."""
        return self.cost_model.get_end_effector_position(config)

    def contextual_severity_2d(self, obj_id: int, xy_distance: float) -> float:
        """Apply distance-based falloff to severity (delegates to cost model)."""
        return self.cost_model._contextual_severity(obj_id, xy_distance)
    
    def distance_based_table_cost_2d(self, config: np.ndarray, task_phase: str = "transit") -> tuple[float, float]:
        """Calculate table cost and distance - returns (probability, xy_distance)."""
        if task_phase != "transit":
            return 0.0, 0.0
            
        ee_pos = self.get_end_effector_position(config)
        xy_distance_to_table = self.calculate_xy_distance_to_table(ee_pos)
        
        if xy_distance_to_table <= 0:
            # Inside table zone
            penetration_depth = abs(xy_distance_to_table)
            prob = self.max_failure_prob * (1.0 - np.exp(-self.table_penetration_rate * penetration_depth))
            prob = max(prob, 0.8 * self.max_failure_prob)
            return prob, penetration_depth  # Return both probability and distance
        else:
            # Outside table zone
            prob = self.max_failure_prob * np.exp(-self.table_distance_decay_rate * xy_distance_to_table)
            return prob, xy_distance_to_table  # Return both probability and distance

    def table_cost(self, config: np.ndarray, task_phase: str = "transit") -> float:
        """Add cost for being near table surface during transit phases."""
        prob_table_hit, xy_dis = self.distance_based_table_cost_2d(config, task_phase)
        severity = self.contextual_severity_2d(0, xy_dis)  # Using 0 as table "object id"
        cost = prob_table_hit * severity
        return cost
    
    def calculate_xy_distance_to_table(self, ee_pos: np.ndarray) -> float:
        """
        Calculate XY distance to table boundary.
        - Positive distance: outside table (clearance)
        - Negative distance: inside table (penetration depth)
        """
        table_x_bounds = [-0.20, 0.20]
        table_y_bounds = [-0.53, -0.27]
        
        # Check if inside table bounds
        inside_x = table_x_bounds[0] <= ee_pos[0] <= table_x_bounds[1]
        inside_y = table_y_bounds[0] <= ee_pos[1] <= table_y_bounds[1]
        
        if inside_x and inside_y:
            # Inside table zone - calculate penetration depth (negative distance)
            x_penetration = min(ee_pos[0] - table_x_bounds[0], table_x_bounds[1] - ee_pos[0])
            y_penetration = min(ee_pos[1] - table_y_bounds[0], table_y_bounds[1] - ee_pos[1])
            # Use minimum penetration (closest to any edge)
            penetration_depth = min(x_penetration, y_penetration)
            return -penetration_depth  # Negative indicates inside
        else:
            # Outside table zone - calculate clearance distance (positive)
            x_dist = max(0, max(table_x_bounds[0] - ee_pos[0], ee_pos[0] - table_x_bounds[1]))
            y_dist = max(0, max(table_y_bounds[0] - ee_pos[1], ee_pos[1] - table_y_bounds[1]))
            
            if x_dist > 0 and y_dist > 0:
                # Outside in both directions - Euclidean distance to nearest corner
                return np.sqrt(x_dist**2 + y_dist**2)
            else:
                # Outside in only one direction
                return max(x_dist, y_dist)    
    
    def state_cost(self, config: np.ndarray, task_type: str = "transit") -> float:
        """Calculate safety cost for a single joint configuration."""
        # Object avoidance cost via shared model (already weighted)
        object_cost = self.cost_model.state_cost_from_config(config)

        # Add table avoidance cost (STOMP-specific)
        table_cost_val = self.table_cost(config, task_type)

        return object_cost + table_cost_val * self._severity_config.failure_weight
    
    def is_valid_config(self, config: np.ndarray) -> bool:
        """Check if configuration is collision-free."""
        return not self.collision_checker.check_collisions(
            robot_config=config,
            threshold=self.collision_threshold
        )
    
    def generate_initial_trajectory(self, start_config: np.ndarray, goal_config: np.ndarray) -> np.ndarray:
        """Generate initial trajectory using linear interpolation in joint space."""
        trajectory = np.zeros((self.num_timesteps, self.trajectory_dof))
        
        for t in range(self.num_timesteps):
            alpha = t / (self.num_timesteps - 1)
            trajectory[t] = (1 - alpha) * start_config[:self.trajectory_dof] + alpha * goal_config[:self.trajectory_dof]
        
        return trajectory
    
    def trajectory_cost(self, trajectory: np.ndarray, task_type: str) -> Tuple[float, np.ndarray]:
        """
        Calculate total cost of trajectory including safety and control costs.
        Returns (total_cost, cost_per_waypoint).
        """
        safety_costs = np.zeros(self.num_timesteps)
        control_costs = np.zeros(self.num_timesteps)
        
        # Safety cost at each waypoint
        for t in range(self.num_timesteps):
            config = trajectory[t]
            
            # Check collision first
            if not self.is_valid_config(config):
                safety_costs[t] = 1000.0  # High penalty for collision
            else:
                safety_costs[t] = self.state_cost(config, task_type)
        
        # Control cost (smoothness regularization)
        for t in range(1, self.num_timesteps - 1):
            # Second derivative (acceleration) penalty
            accel = trajectory[t+1] - 2*trajectory[t] + trajectory[t-1]
            control_costs[t] = self.control_cost_weight * np.sum(accel**2)
        
        # Total cost per waypoint
        total_costs_per_waypoint = safety_costs + control_costs
        
        return np.sum(total_costs_per_waypoint), total_costs_per_waypoint
    
    def generate_noisy_trajectories(self, reference_trajectory: np.ndarray) -> np.ndarray:
        """Generate noisy variations of the reference trajectory."""
        noisy_trajectories = np.zeros((self.num_rollouts, self.num_timesteps, self.trajectory_dof))
        
        for k in range(self.num_rollouts):
            # Add correlated noise to trajectory
            noise = np.zeros((self.num_timesteps, self.trajectory_dof))
            
            for joint in range(self.trajectory_dof):
                # Generate smooth noise using Gaussian filtering
                raw_noise = self.rng.normal(0, self.current_noise_stddev, self.num_timesteps)
                # Apply Gaussian filter for temporal correlation
                smooth_noise = scipy.ndimage.gaussian_filter1d(raw_noise, sigma=2.0)
                noise[:, joint] = smooth_noise
            
            # Keep start and end points fixed
            noise[0, :] = 0
            noise[-1, :] = 0
            
            noisy_trajectories[k] = reference_trajectory + noise
        
        return noisy_trajectories
    
    def compute_trajectory_update(self, reference_trajectory: np.ndarray, 
                                noisy_trajectories: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Compute trajectory update using probability-weighted averaging."""
        # Convert costs to probabilities (lower cost = higher probability)
        min_cost = np.min(costs)
        exp_costs = np.exp(-(costs - min_cost) / (np.std(costs) + 1e-8))
        probabilities = exp_costs / np.sum(exp_costs)
        
        # Compute probability-weighted average trajectory
        weighted_trajectory = np.zeros_like(reference_trajectory)
        for k in range(self.num_rollouts):
            weighted_trajectory += probabilities[k] * noisy_trajectories[k]
        
        # Update with learning rate
        delta = weighted_trajectory - reference_trajectory
        
        # Keep start and end points fixed
        delta[0, :] = 0
        delta[-1, :] = 0
        
        updated_trajectory = reference_trajectory + self.learning_rate * delta
        
        return updated_trajectory
    
    def plan(self, start_config: np.ndarray, goal_config: np.ndarray, 
             frame_name: str = "end_effector", task_type: str = "transit") -> Optional[List[np.ndarray]]:
        """
        Plan trajectory using STOMP algorithm.
        
        STOMP Algorithm:
        1. Initialize trajectory with linear interpolation
        2. For each iteration:
           a. Generate noisy trajectory variations
           b. Evaluate cost of each noisy trajectory
           c. Update trajectory toward lower-cost variations
           d. Reduce noise level
        3. Return optimized trajectory
        """
        logger.info(f"Starting STOMP planning...")
        logger.debug(f"Parameters: timesteps={self.num_timesteps}, rollouts={self.num_rollouts}, "
              f"iterations={self.stomp_max_iterations}")
        
        self.rng = np.random.RandomState(self.seed)
        
        # Initialize trajectory
        current_trajectory = self.generate_initial_trajectory(start_config, goal_config)
        initial_cost, _ = self.trajectory_cost(current_trajectory, task_type)
        logger.debug(f"Initial trajectory cost: {initial_cost:.4f}")
        
        best_trajectory = current_trajectory.copy()
        best_cost = initial_cost
        
        # Reset noise level
        self.current_noise_stddev = 0.1
        
        # STOMP optimization loop
        for iteration in range(self.stomp_max_iterations):
            # Generate noisy variations
            noisy_trajectories = self.generate_noisy_trajectories(current_trajectory)
            
            # Evaluate costs of all noisy trajectories
            costs = np.zeros(self.num_rollouts)
            for k in range(self.num_rollouts):
                costs[k], _ = self.trajectory_cost(noisy_trajectories[k], task_type)
            
            # Find best trajectory in this batch
            best_idx = np.argmin(costs)
            if costs[best_idx] < best_cost:
                best_cost = costs[best_idx]
                best_trajectory = noisy_trajectories[best_idx].copy()
            
            # Update trajectory using probability-weighted average
            current_trajectory = self.compute_trajectory_update(
                current_trajectory, noisy_trajectories, costs)
            
            # Evaluate current trajectory cost
            current_cost, cost_breakdown = self.trajectory_cost(current_trajectory, task_type)
            
            # Reduce noise level
            self.current_noise_stddev *= self.noise_decay
            
            # Progress reporting
            if iteration % 20 == 0:
                avg_cost = np.mean(costs)
                logger.info(f"Iter {iteration}: current_cost={current_cost:.4f}, "
                      f"best_cost={best_cost:.4f}, avg_cost={avg_cost:.4f}, "
                      f"noise_std={self.current_noise_stddev:.6f}")

                # Check cost distribution along trajectory
                max_waypoint_cost = np.max(cost_breakdown)
                logger.debug(f"  Max waypoint cost: {max_waypoint_cost:.4f}")
            
            # Early convergence check
            if self.current_noise_stddev < 1e-4:
                logger.info(f"Converged at iteration {iteration} (noise level too low)")
                break
        
        # Use best trajectory found
        final_trajectory = best_trajectory
        final_cost, final_breakdown = self.trajectory_cost(final_trajectory, task_type)
        
        logger.info(f"STOMP Results:")
        logger.info(f"  Initial cost: {initial_cost:.4f}")
        logger.info(f"  Final cost: {final_cost:.4f}")
        logger.info(f"  Improvement: {(initial_cost - final_cost):.4f} ({((initial_cost - final_cost)/initial_cost*100):.1f}%)")
        
        # Analyze trajectory safety
        high_cost_waypoints = np.sum(final_breakdown > 10.0)
        logger.debug(f"  High-cost waypoints: {high_cost_waypoints}/{self.num_timesteps}")
        
        # Check collision feasibility
        collision_free = all(self.is_valid_config(config) for config in final_trajectory)
        logger.info(f"  Collision-free: {collision_free}")

        if not collision_free:
            logger.warning("  Final trajectory contains collisions")
            return None
        
        # Convert to list format expected by caller
        return [config.copy() for config in final_trajectory]
    
    def debug_trajectory_costs(self, trajectory: np.ndarray):
        """Debug cost breakdown for each waypoint in trajectory."""
        logger.debug("Trajectory Cost Analysis:")
        logger.debug("Waypoint | EE Position (XY) | Safety Cost | Collision")
        logger.debug("-" * 55)
        
        for t, config in enumerate(trajectory):
            ee_pos = self.get_end_effector_position(config)
            safety_cost = self.state_cost(config)
            is_collision = not self.is_valid_config(config)
            
            logger.debug(f"  {t:2d}     | ({ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}) | {safety_cost:9.4f} | {is_collision}")
            
            # Highlight high-cost waypoints
            if safety_cost > 20:
                obj2_pos = self.object_positions.get(13, np.zeros(3))
                dist_to_obj2 = np.linalg.norm(ee_pos[:2] - obj2_pos[:2])
                logger.warning(f"    HIGH COST waypoint - distance to obj2: {dist_to_obj2:.3f}m")
    
    def set_aggressive_avoidance_parameters(self):
        """Set parameters for strong obstacle avoidance."""
        logger.info("Setting aggressive STOMP avoidance parameters...")
        self.control_cost_weight = 1e-5  # Lower smoothness weight, prioritize safety

        # Rebuild cost model with aggressive parameters
        self._severity_config = SeverityConfig(
            severity_map={12: 10, 13: 100, 14: 10, 0: 8},
            target_body_ids=self.target_bids,
            max_failure_prob=0.98,
            distance_decay_rate=100.0,
            failure_weight=50.0,
        )
        self.cost_model = FailureCostModel(
            scene_model=self.model,
            object_positions=self.object_positions,
            config=self._severity_config,
        )
        logger.debug(f"New parameters: failure_weight={self._severity_config.failure_weight}, "
              f"decay_rate={self._severity_config.distance_decay_rate}")
    
    def visualize_trajectory_costs(self, trajectory: np.ndarray):
        """Analyze and visualize cost distribution along trajectory."""
        costs = []
        ee_positions = []
        
        for config in trajectory:
            cost = self.state_cost(config)
            ee_pos = self.get_end_effector_position(config)
            costs.append(cost)
            ee_positions.append(ee_pos[:2])  # XY only
        
        costs = np.array(costs)
        ee_positions = np.array(ee_positions)
        
        obj2_pos = self.object_positions.get(13, np.array([0, 0, 0]))
        
        logger.debug(f"Trajectory Analysis:")
        logger.debug(f"  Cost range: {costs.min():.4f} to {costs.max():.4f}")
        logger.debug(f"  Mean cost: {costs.mean():.4f}")
        logger.debug(f"  Std dev: {costs.std():.4f}")
        
        # Find closest approach to object 2
        distances_to_obj2 = [np.linalg.norm(pos - obj2_pos[:2]) for pos in ee_positions]
        min_distance = min(distances_to_obj2)
        closest_idx = np.argmin(distances_to_obj2)
        
        logger.debug(f"  Closest approach to obj2: {min_distance:.3f}m at waypoint {closest_idx}")
        logger.debug(f"  Cost at closest point: {costs[closest_idx]:.4f}")

        if min_distance < 0.1:
            logger.warning("  Trajectory passes very close to object 2")
        elif min_distance > 0.2:
            logger.debug("  Trajectory maintains good clearance from object 2")
        
        return costs, ee_positions, distances_to_obj2