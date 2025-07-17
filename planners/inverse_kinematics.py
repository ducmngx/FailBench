"""
Motion Planner-friendly IK interface using Mink.
Designed for integration with sampling-based planners (RRT, PRM, etc.)
"""

import numpy as np
import mujoco
import mink
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class IKResult(Enum):
    """IK solving results for motion planners."""
    SUCCESS = "success"
    FAILED_TO_CONVERGE = "failed_to_converge"
    INVALID_TARGET = "invalid_target"
    JOINT_LIMITS_VIOLATED = "joint_limits_violated"


@dataclass
class IKConfig:
    """Configuration for IK solving."""
    solver: str = "osqp"
    max_iterations: int = 100
    dt: float = 0.01
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    position_cost: float = 1.0
    orientation_cost: float = 1.0
    posture_cost: float = 1e-2
    damping: float = 5e-3
    check_joint_limits: bool = True
    
    
@dataclass 
class EndEffectorTarget:
    """End-effector target specification."""
    position: np.ndarray
    orientation: Optional[np.ndarray] = None  # quaternion [w,x,y,z]
    frame_name: str = "end_effector"
    frame_type: str = "site"
    position_cost: Optional[float] = None
    orientation_cost: Optional[float] = None


class IKSolver:
    """
    Stateless IK solver optimized for motion planning integration.
    
    Key features for motion planners:
    - Fast solving with configurable tolerances
    - Multiple seed support for finding different solutions
    - Joint limit checking
    - Clear success/failure status
    - Minimal memory allocation
    """
    
    def __init__(self, model: mujoco.MjModel, config: Optional[IKConfig] = None):
        """
        Initialize IK solver.
        
        Args:
            model: MuJoCo model (should be thread-safe for parallel planning)
            config: IK configuration
        """
        self.model = model
        self.config = config or IKConfig()
        
        # Pre-allocate arrays for efficiency
        self._data = mujoco.MjData(model)
        self._configuration = mink.Configuration(model)
        
        # Joint limits for validation
        if self.config.check_joint_limits:
            self.joint_limits_lower = model.jnt_range[:, 0].copy()
            self.joint_limits_upper = model.jnt_range[:, 1].copy()
        else:
            self.joint_limits_lower = None
            self.joint_limits_upper = None
    
    def solve(
        self, 
        targets: Union[EndEffectorTarget, List[EndEffectorTarget]],
        seed_config: np.ndarray,
        config_override: Optional[IKConfig] = None
    ) -> Tuple[np.ndarray, IKResult]:
        """
        Solve IK for given targets starting from seed configuration.
        
        Args:
            targets: Single target or list of targets
            seed_config: Starting joint configuration
            config_override: Override default config for this solve
            
        Returns:
            Tuple of (solution_config, result_status)
        """
        cfg = config_override or self.config
        
        # Ensure targets is a list
        if isinstance(targets, EndEffectorTarget):
            targets = [targets]
        
        # Validate inputs
        if len(seed_config) != self.model.nq:
            return seed_config.copy(), IKResult.INVALID_TARGET
        
        # Check joint limits on seed
        if cfg.check_joint_limits and not self._check_joint_limits(seed_config):
            return seed_config.copy(), IKResult.JOINT_LIMITS_VIOLATED
        
        # Setup configuration
        self._data.qpos[:] = seed_config
        mujoco.mj_forward(self.model, self._data)
        self._configuration.update(self._data.qpos)
        
        # Create tasks
        tasks = []
        
        # Frame tasks
        for target in targets:
            frame_task = mink.FrameTask(
                frame_name=target.frame_name,
                frame_type=target.frame_type,
                position_cost=target.position_cost or cfg.position_cost,
                orientation_cost=target.orientation_cost or cfg.orientation_cost,
                lm_damping=1.0,
            )
            
            # Set target pose
            if target.orientation is not None:
                target_transform = mink.SE3.from_rotation_and_translation(
                    target.orientation, target.position 
                )
            else:
                # Keep current orientation
                current_transform = self._configuration.get_transform_frame_to_world(
                    target.frame_name, target.frame_type
                )
                target_transform = mink.SE3.from_rotation_and_translation(
                    current_transform.rotation(), target.position 
                )
            
            frame_task.set_target(target_transform)
            tasks.append(frame_task)
        
        # Posture task for regularization
        posture_task = mink.PostureTask(model=self.model, cost=cfg.posture_cost)
        posture_task.set_target_from_configuration(self._configuration)
        tasks.append(posture_task)
        
        # Iterative solving
        for iteration in range(cfg.max_iterations):
            # Solve one step
            velocity = mink.solve_ik(
                self._configuration, tasks, cfg.dt, cfg.solver, cfg.damping
            )
            self._configuration.integrate_inplace(velocity, cfg.dt)
            
            # Check joint limits
            if cfg.check_joint_limits and not self._check_joint_limits(self._configuration.q):
                return seed_config.copy(), IKResult.JOINT_LIMITS_VIOLATED
            
            # Check convergence
            if self._check_convergence(targets, cfg):
                return self._configuration.q.copy(), IKResult.SUCCESS
        
        # Failed to converge
        return self._configuration.q.copy(), IKResult.FAILED_TO_CONVERGE
    
    def solve_with_multiple_seeds(
        self,
        targets: Union[EndEffectorTarget, List[EndEffectorTarget]],
        seed_configs: List[np.ndarray],
        config_override: Optional[IKConfig] = None
    ) -> List[Tuple[np.ndarray, IKResult]]:
        """
        Solve IK with multiple seed configurations.
        Useful for finding different solutions or increasing success rate.
        
        Args:
            targets: Target specification
            seed_configs: List of seed configurations to try
            config_override: Override default config
            
        Returns:
            List of (solution, result) tuples for each seed
        """
        results = []
        for seed in seed_configs:
            solution, result = self.solve(targets, seed, config_override)
            results.append((solution, result))
        return results
    
    def find_valid_solution(
        self,
        targets: Union[EndEffectorTarget, List[EndEffectorTarget]],
        seed_configs: List[np.ndarray],
        config_override: Optional[IKConfig] = None
    ) -> Optional[np.ndarray]:
        """
        Find first valid solution from multiple seeds.
        
        Args:
            targets: Target specification
            seed_configs: List of seed configurations to try
            config_override: Override default config
            
        Returns:
            First successful solution or None if all fail
        """
        for seed in seed_configs:
            solution, result = self.solve(targets, seed, config_override)
            if result == IKResult.SUCCESS:
                return solution
        return None
    
    def is_target_reachable(
        self,
        targets: Union[EndEffectorTarget, List[EndEffectorTarget]],
        seed_configs: List[np.ndarray],
        max_attempts: int = 5
    ) -> bool:
        """
        Quick reachability check for motion planners.
        
        Args:
            targets: Target specification
            seed_configs: Seed configurations to try
            max_attempts: Maximum number of seeds to try
            
        Returns:
            True if target is reachable from any seed
        """
        # Use fast config for reachability check
        fast_config = IKConfig(
            max_iterations=20,
            position_tolerance=0.01,
            orientation_tolerance=0.1
        )
        
        for i, seed in enumerate(seed_configs[:max_attempts]):
            _, result = self.solve(targets, seed, fast_config)
            if result == IKResult.SUCCESS:
                return True
        return False
    
    def get_random_valid_config(self, num_attempts: int = 100) -> Optional[np.ndarray]:
        """
        Generate a random valid configuration within joint limits.
        Useful for sampling-based planners.
        
        Args:
            num_attempts: Maximum attempts to find valid config
            
        Returns:
            Random valid configuration or None
        """
        if not self.config.check_joint_limits:
            # No joint limits defined, sample from reasonable range
            return np.random.uniform(-np.pi, np.pi, self.model.nq)
        
        for _ in range(num_attempts):
            # Sample within joint limits
            config = np.random.uniform(
                self.joint_limits_lower, 
                self.joint_limits_upper
            )
            
            # Check if configuration is valid (no collisions, etc.)
            if self._is_config_valid(config):
                return config
        
        return None
    
    def interpolate_configs(
        self,
        config1: np.ndarray,
        config2: np.ndarray,
        num_points: int = 10
    ) -> List[np.ndarray]:
        """
        Interpolate between two joint configurations.
        Useful for local planning and trajectory generation.
        
        Args:
            config1: Start configuration
            config2: End configuration  
            num_points: Number of interpolation points
            
        Returns:
            List of interpolated configurations
        """
        # Linear interpolation in joint space
        # For more sophisticated robots, you might want to use SLERP for quaternions
        alphas = np.linspace(0, 1, num_points)
        configs = []
        
        for alpha in alphas:
            config = (1 - alpha) * config1 + alpha * config2
            configs.append(config)
        
        return configs
    
    def _check_joint_limits(self, config: np.ndarray) -> bool:
        """Check if configuration satisfies joint limits."""
        if self.joint_limits_lower is None:
            return True
        
        return np.all(config >= self.joint_limits_lower) and np.all(config <= self.joint_limits_upper)
    
    def _check_convergence(self, targets: List[EndEffectorTarget], config: IKConfig) -> bool:
        """Check if all targets are reached within tolerance."""
        for target in targets:
            current_transform = self._configuration.get_transform_frame_to_world(
                target.frame_name, target.frame_type
            )
            
            # Position error
            pos_error = np.linalg.norm(
                current_transform.translation() - target.position
            )
            
            if pos_error > config.position_tolerance:
                return False
            
            # Orientation error (if specified)
            if target.orientation is not None:
                current_quat = current_transform.as_quaternion_xyzw()
                target_quat = target.orientation
                # Convert to [x,y,z,w] format if needed
                if len(target_quat) == 4 and target_quat[0] > 0.7:  # Likely [w,x,y,z]
                    target_quat = np.array([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
                
                ori_error = min(
                    np.linalg.norm(current_quat - target_quat),
                    np.linalg.norm(current_quat + target_quat)
                )
                
                if ori_error > config.orientation_tolerance:
                    return False
        
        return True
    
    def _is_config_valid(self, config: np.ndarray) -> bool:
        """
        Check if a configuration is valid (no self-collisions, etc.).
        Override this method to add collision checking.
        """
        # Basic implementation - just check joint limits
        return self._check_joint_limits(config)


# Convenience functions for common motion planning use cases
def solve_ik_for_planner(
    model: mujoco.MjModel,
    target_position: np.ndarray,
    seed_config: np.ndarray,
    target_orientation: Optional[np.ndarray] = None,
    frame_name: str = "end_effector",
    fast_mode: bool = False
) -> Optional[np.ndarray]:
    """
    Simple IK solve optimized for motion planners.
    
    Args:
        model: MuJoCo model
        target_position: Target end-effector position
        seed_config: Starting configuration
        target_orientation: Target orientation (optional)
        frame_name: Name of end-effector frame
        fast_mode: Use faster, less accurate solving
        
    Returns:
        Solution configuration or None if failed
    """
    config = IKConfig(
        max_iterations=20 if fast_mode else 100,
        position_tolerance=0.01 if fast_mode else 0.005,
        orientation_tolerance=0.1 if fast_mode else 0.05
    )
    
    solver = IKSolver(model, config)
    target = EndEffectorTarget(
        position=target_position,
        orientation=target_orientation,
        frame_name=frame_name
    )
    
    solution, result = solver.solve(target, seed_config)
    return solution if result == IKResult.SUCCESS else None


def batch_ik_solve(
    model: mujoco.MjModel,
    targets_and_seeds: List[Tuple[np.ndarray, np.ndarray]],
    frame_name: str = "end_effector",
    fast_mode: bool = True
) -> List[Optional[np.ndarray]]:
    """
    Solve IK for multiple targets efficiently.
    Useful for validating many nodes in a planning tree.
    
    Args:
        model: MuJoCo model
        targets_and_seeds: List of (target_position, seed_config) pairs
        frame_name: End-effector frame name
        fast_mode: Use fast solving
        
    Returns:
        List of solutions (None for failures)
    """
    config = IKConfig(
        max_iterations=20 if fast_mode else 50,
        position_tolerance=0.01 if fast_mode else 0.005,
    )
    
    solver = IKSolver(model, config)
    results = []
    
    for target_pos, seed in targets_and_seeds:
        target = EndEffectorTarget(position=target_pos, frame_name=frame_name)
        solution, result = solver.solve(target, seed)
        results.append(solution if result == IKResult.SUCCESS else None)
    
    return results