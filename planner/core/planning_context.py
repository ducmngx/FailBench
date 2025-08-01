"""
core/planning_context.py

Central state management for RRT planning.
This is the hub that all other modules use to access MuJoCo models and data.
"""

import numpy as np
import mujoco
from typing import Tuple, Optional
from pathlib import Path


class PlanningContext:
    """
    Central context that manages all MuJoCo models and data.
    All planning components share this context for synchronized state management.
    """
    
    def __init__(self, scene_xml_path: str, robot_xml_path: str):
        """
        Initialize the planning context with models and data.
        
        Args:
            scene_xml_path: Path to complete scene XML
            robot_xml_path: Path to robot-only XML
        """
        self.scene_xml_path = Path(scene_xml_path)
        self.robot_xml_path = Path(robot_xml_path)
        
        # Validate paths
        if not self.scene_xml_path.exists():
            raise FileNotFoundError(f"Scene XML not found: {scene_xml_path}")
        if not self.robot_xml_path.exists():
            raise FileNotFoundError(f"Robot XML not found: {robot_xml_path}")
        
        # Load models
        self.scene_model = mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        self.robot_model = mujoco.MjModel.from_xml_path(str(self.robot_xml_path))
        
        # Create data contexts
        self.simulation_data = mujoco.MjData(self.scene_model)  # For actual simulation
        self.planning_data = mujoco.MjData(self.scene_model)    # For planning/collision checking
        self.robot_data = mujoco.MjData(self.robot_model)       # For robot-specific operations
        
        # Robot configuration properties
        self.robot_dof = self.robot_model.njnt
        self.current_robot_config = np.zeros(self.robot_dof)
        
        # Extract joint limits
        self.joint_lower_limits = self.robot_model.jnt_range[:, 0].copy()
        self.joint_upper_limits = self.robot_model.jnt_range[:, 1].copy()
        
        # Initialize to neutral position
        self._initialize_neutral_state()
        
        print(f"PlanningContext initialized:")
        print(f"  Scene: {self.scene_model.ngeom} geoms, {self.scene_model.njnt} joints")
        print(f"  Robot: {self.robot_model.ngeom} geoms, {self.robot_dof} DOF")
        print(f"  Joint limits: [{self.joint_lower_limits[:3]}, {self.joint_upper_limits[:3]}...]")
    
    def _initialize_neutral_state(self):
        """Initialize both simulation and planning data to neutral state."""
        # Set to middle of joint ranges where possible
        neutral_config = np.zeros(self.robot_dof)
        for i in range(self.robot_dof):
            if self.joint_lower_limits[i] != self.joint_upper_limits[i]:
                # Joint has limits, use middle
                neutral_config[i] = (self.joint_lower_limits[i] + self.joint_upper_limits[i]) / 2
        
        # Apply to both contexts
        self.simulation_data.qpos[:self.robot_dof] = neutral_config
        self.planning_data.qpos[:self.robot_dof] = neutral_config
        
        # Update physics
        mujoco.mj_forward(self.scene_model, self.simulation_data)
        mujoco.mj_forward(self.scene_model, self.planning_data)
        
        self.current_robot_config = neutral_config.copy()
    
    def set_robot_configuration_for_planning(self, config: np.ndarray):
        """
        Set robot configuration in PLANNING context only.
        This doesn't affect the simulation state.
        
        Args:
            config: Joint configuration for the robot
            
        Raises:
            ValueError: If config length doesn't match robot DOF
        """
        if len(config) != self.robot_dof:
            raise ValueError(f"Config length {len(config)} != robot DOF {self.robot_dof}")
        
        # Update planning data only
        self.planning_data.qpos[:self.robot_dof] = config
        mujoco.mj_forward(self.scene_model, self.planning_data)
        
        # Store current planning config
        self.current_robot_config = config.copy()
    
    def set_robot_configuration_for_simulation(self, config: np.ndarray):
        """
        Set robot configuration in SIMULATION context.
        Use this for actual robot control and visualization.
        
        Args:
            config: Joint configuration for the robot
            
        Raises:
            ValueError: If config length doesn't match robot DOF
        """
        if len(config) != self.robot_dof:
            raise ValueError(f"Config length {len(config)} != robot DOF {self.robot_dof}")
        
        # Update simulation data
        self.simulation_data.qpos[:self.robot_dof] = config
        mujoco.mj_forward(self.scene_model, self.simulation_data)
    
    def copy_simulation_to_planning(self):
        """
        Copy current simulation state to planning context.
        Call this before planning to ensure planning uses current environment state.
        """
        self.planning_data.qpos[:] = self.simulation_data.qpos[:]
        self.planning_data.qvel[:] = self.simulation_data.qvel[:]
        mujoco.mj_forward(self.scene_model, self.planning_data)
        self.current_robot_config = self.simulation_data.qpos[:self.robot_dof].copy()
    
    def copy_planning_to_simulation(self):
        """
        Copy planning state to simulation context.
        Use this to execute a planned configuration.
        """
        self.simulation_data.qpos[:] = self.planning_data.qpos[:]
        self.simulation_data.qvel[:] = self.planning_data.qvel[:]
        mujoco.mj_forward(self.scene_model, self.simulation_data)
    
    def get_current_robot_config(self) -> np.ndarray:
        """Get current robot configuration (copy)."""
        return self.current_robot_config.copy()
    
    def get_simulation_robot_config(self) -> np.ndarray:
        """Get current simulation robot configuration."""
        return self.simulation_data.qpos[:self.robot_dof].copy()
    
    def is_config_within_limits(self, config: np.ndarray) -> bool:
        """
        Check if configuration is within joint limits.
        
        Args:
            config: Joint configuration to check
            
        Returns:
            True if all joints are within limits, False otherwise
        """
        if len(config) != self.robot_dof:
            return False
        
        # Check each joint
        for i, q in enumerate(config):
            lower, upper = self.joint_lower_limits[i], self.joint_upper_limits[i]
            # Only check joints that have actual limits
            if lower != upper and (q < lower or q > upper):
                return False
        
        return True
    
    def clamp_config_to_limits(self, config: np.ndarray) -> np.ndarray:
        """
        Clamp configuration to joint limits.
        
        Args:
            config: Joint configuration to clamp
            
        Returns:
            Configuration with all joints within limits
        """
        return np.clip(config, self.joint_lower_limits, self.joint_upper_limits)
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joint limits.
        
        Returns:
            Tuple of (lower_limits, upper_limits)
        """
        return self.joint_lower_limits.copy(), self.joint_upper_limits.copy()
    
    def get_random_config(self) -> np.ndarray:
        """
        Generate random configuration within joint limits.
        
        Returns:
            Random valid joint configuration
        """
        return np.random.uniform(self.joint_lower_limits, self.joint_upper_limits)
    
    def step_simulation(self, control_input: Optional[np.ndarray] = None):
        """
        Step the simulation forward by one timestep.
        
        Args:
            control_input: Optional control input to apply
        """
        if control_input is not None:
            if len(control_input) <= len(self.simulation_data.ctrl):
                self.simulation_data.ctrl[:len(control_input)] = control_input
        
        mujoco.mj_step(self.scene_model, self.simulation_data)
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.scene_model, self.simulation_data)
        mujoco.mj_forward(self.scene_model, self.simulation_data)
    
    def get_end_effector_pose(self, config: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get end-effector pose for given configuration.
        
        Args:
            config: Joint configuration (None = use current planning state)
            
        Returns:
            End-effector pose [x, y, z, qx, qy, qz, qw]
        """
        if config is not None:
            self.set_robot_configuration_for_planning(config)
        
        # TODO: Implement based on your robot's end-effector body/site
        # This is a placeholder - you'll need to identify your robot's end-effector
        
        # Example for common robots:
        # site_id = mujoco.mj_name2id(self.scene_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        # if site_id >= 0:
        #     pos = self.planning_data.site_xpos[site_id]
        #     mat = self.planning_data.site_xmat[site_id].reshape(3, 3)
        #     quat = np.zeros(4)
        #     mujoco.mju_mat2Quat(quat, mat.flatten())
        #     return np.concatenate([pos, quat])
        
        # Placeholder: return zeros
        return np.zeros(7)
    
    def __str__(self) -> str:
        """String representation."""
        return (f"PlanningContext(scene={self.scene_xml_path.name}, "
                f"robot={self.robot_xml_path.name}, dof={self.robot_dof})")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"PlanningContext(\n"
                f"  scene_xml='{self.scene_xml_path}',\n"
                f"  robot_xml='{self.robot_xml_path}',\n"
                f"  robot_dof={self.robot_dof},\n"
                f"  scene_geoms={self.scene_model.ngeom},\n"
                f"  scene_joints={self.scene_model.njnt}\n"
                f")")


# Factory function for easy creation
def create_planning_context(scene_xml_path: str, robot_xml_path: str) -> PlanningContext:
    """
    Factory function to create a PlanningContext.
    
    Args:
        scene_xml_path: Path to scene XML file
        robot_xml_path: Path to robot XML file
        
    Returns:
        Initialized PlanningContext
    """
    return PlanningContext(scene_xml_path, robot_xml_path)