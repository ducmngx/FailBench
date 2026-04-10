#!/usr/bin/env python3
"""
Panda Pick and Place Demo
Uses your existing RRT planner to pick up object3 and place it in a new location.
Includes full tendon gripper support for physics-based grasping.
"""
import numpy as np
import mujoco
import mujoco.viewer
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple, Callable, Union
# Import existing modules
from planner.algorithms.RRTplanner import JointSpaceRRT, JointSpaceRRTConnect, JointSpaceRRTConnectFailure
from planner.algorithms.TRRTFailure import JointSpaceTRRTOMPL
from planner.algorithms.STOMP import JointSpaceSTOMP
from planner.collision.collision_checker import CollisionChecker  
from planner.kinematics.inverse_kinematics import IKSolver, EndEffectorTarget, IKResult
from planner.algorithms.abstract_planner import PlanningSpace
from failure_injection.collision_estimation import CollisionEstimator
from planner.utils.traj_saver import ExperimentTrajectoryManager

import argparse
import os

class PandaPickAndPlace:
    """
    Pick and place demo using your existing planning modules.
    Supports both individual finger actuators and tendon-based grippers.
    """
    
    def __init__(self, scene_xml_path: str, robot_xml_path: str, seed: int = 42):
        """Initialize with enhanced tendon gripper support."""
        
        print("🤖 Initializing Panda Pick and Place Demo")
        print("=" * 50)
        
        # Load models
        print("🔍 Loading MuJoCo models...")
        self.scene_model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        self.scene_data = mujoco.MjData(self.scene_model)
        self.seed = seed
        
        print(f"✅ Models loaded:")
        print(f"   Scene: {self.scene_model.ngeom} geoms, {self.scene_model.njnt} joints")
        print(f"   Robot: {self.robot_model.ngeom} geoms, {self.robot_model.njnt} joints")
        
        # Initialize your planning modules
        self.ik_solver = IKSolver(self.robot_model)
        self.collision_checker = CollisionChecker(self.scene_model, self.robot_model)
        failing_joints = [f"joint{i}" for i in range(1,8)]
        # failing_joints += ['finger_joint1', 'finger_joint2']
        self.collision_estimator = CollisionEstimator(self.scene_model, inflation_radius=0, failing_joints=failing_joints, robot_joints=failing_joints)

        # self.rrt_planner = JointSpaceTRRTFailure(
        #     scene_model=self.scene_model,
        #     robot_model=self.robot_model,
        #     ik_solver=self.ik_solver,
        #     collision_threshold=0.0000005,  # 1cm threshold
        #     seed=self.seed,
        #     collision_estimator=self.collision_estimator,
        #     planning_space=PlanningSpace.JOINT_SPACE,
        #     step_size= 0.05, #0.008,
        #     goal_bias=0.7
        # )

        # self.rrt_planner = JointSpaceTRRTOMPL(
        #     scene_model=self.scene_model,
        #     robot_model=self.robot_model,
        #     ik_solver=self.ik_solver,
        #     collision_threshold=0.0000005,
        #     seed=self.seed,
        #     collision_estimator=self.collision_estimator,
        #     object_positions={
        #         12: self.get_object_position("object1"),
        #         13: self.get_object_position("object2"), 
        #         14: self.get_object_position("object3")
        #     },
        #     step_size=0.05,
        #     goal_bias=0.15,
        #     failure_weight=0.3  # Start low and tune up
        # )

        # self.rrt_planner = JointSpaceTRRTOMPL(
        #     scene_model=self.scene_model,
        #     robot_model=self.robot_model,
        #     ik_solver=self.ik_solver,
        #     collision_threshold=0.0000005,
        #     seed=self.seed,
        #     collision_estimator=self.collision_estimator,
        #     step_size=0.05,
        #     object_positions={
        #         12: self.get_object_position("object1"),
        #         13: self.get_object_position("object2"), 
        #         14: self.get_object_position("object3")
        #     },
        #     failure_weight=5.0,
        #     init_temperature=50.0,
        #     temp_change_factor=1.1
        # )
                
        self.robot_dof = self.robot_model.njnt
        self.current_path = None
        self.current_viewer = None
        
        # Set initial pose
        self._set_home_position()

        # self.rrt_planner = JointSpaceSTOMP(
        #     scene_model=self.scene_model,
        #     robot_model=self.robot_model,
        #     ik_solver=self.ik_solver,
        #     collision_threshold=0.0000005,
        #     seed=self.seed,
        #     collision_estimator=self.collision_estimator,
        #     step_size=0.05,
        #     object_positions={
        #         12: self.get_object_position("object1"),
        #         13: self.get_object_position("object2"), 
        #         14: self.get_object_position("object3")
        #     },
        #     failure_weight=20.0,
        #     init_temperature=10.0,
        #     temp_change_factor=1.1
        # )

        self.rrt_planner = JointSpaceSTOMP(
            scene_model=self.scene_model,
            robot_model=self.robot_model,
            ik_solver=self.ik_solver,
            collision_threshold=0.0000005,
            seed=self.seed,
            collision_estimator=self.collision_estimator,
            object_positions={
                12: self.get_object_position("object1"),
                13: self.get_object_position("object2"), 
                14: self.get_object_position("object3")
            }
        )

        # path = self.stomp_planner.plan(start_config, goal_config)
        
        # # ENHANCED GRIPPER DETECTION INCLUDING TENDON-BASED
        # print("\n🔍 Enhanced gripper detection (including tendon-based)...")
        # if not self.quick_fix_gripper_indices():
        #     print("⚠️ Could not auto-detect gripper. Will need manual setup.")
        #     self.gripper_actuator_indices = []
        #     self.gripper_joint_indices = []
        #     self.gripper_type = "none"
        # else:
        #     print(f"✅ Gripper detected: {self.gripper_type}")
        #     print(f"   Actuator indices: {self.gripper_actuator_indices}")
        #     print(f"   Joint indices: {self.gripper_joint_indices}")
            
    def _set_home_position(self):
        """Set robot to home position."""
        home_config = np.zeros(self.robot_dof)
        self.scene_data.qpos[:self.robot_dof] = home_config
        mujoco.mj_forward(self.scene_model, self.scene_data)
        self.collision_estimator.forward_kinematics(home_config)
        self.collision_estimator.post_mj_forward_init()
    
    def get_current_config(self) -> np.ndarray:
        """Get current robot configuration."""
        return self.scene_data.qpos[:self.robot_dof].copy()
    
    def get_object_position(self, object_name: str) -> Optional[np.ndarray]:
        """Get current position of an object in the scene."""
        try:
            body_id = mujoco.mj_name2id(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            if body_id == -1:
                print(f"❌ Object '{object_name}' not found")
                return None
            
            obj_pos = self.scene_data.xpos[body_id].copy()
            print(f"📍 {object_name} position: {obj_pos}")
            return obj_pos
        except Exception as e:
            print(f"❌ Error getting {object_name} position: {e}")
            return None
    
    def get_end_effector_position(self) -> Optional[np.ndarray]:
        """Get current end-effector position."""
        try:
            site_id = mujoco.mj_name2id(self.scene_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
            if site_id == -1:
                return None
            return self.scene_data.site_xpos[site_id].copy()
        except:
            return None
        
    def densify_path(self, path: List[np.ndarray], max_joint_step: float = 0.05) -> List[np.ndarray]:
        """Densify path with special handling for large jumps."""
        if len(path) < 2:
            return path
        
        smooth_path = [path[0]]
        
        for i in range(1, len(path)):
            start_config = path[i-1]
            end_config = path[i]
            
            joint_diff = np.abs(end_config - start_config)
            max_change = np.max(joint_diff)
            
            if max_change > 1.0:  # Detect huge jumps
                print(f"⚠️ Huge jump detected ({max_change:.2f} rad), adding many intermediate points")
                num_steps = int(max_change / max_joint_step) + 1
            else:
                num_steps = int(max_change / max_joint_step) + 1
            
            for step in range(1, num_steps + 1):
                alpha = step / num_steps
                intermediate = (1 - alpha) * start_config + alpha * end_config
                smooth_path.append(intermediate)
        
        return smooth_path
    
    def plan_to_config(self, target_config: np.ndarray, use_downward_constraint: bool = False) -> bool:
        """Plan to end-effector pose using your IK + RRT."""
        
        print(f"🎯 Planning to config Position: {target_config}")
        # Create target with optional orientation constraint
        # if use_downward_constraint:
        #     print("   🔽 Using downward orientation constraint")
        #     # Create downward orientation quaternion [w, x, y, z]
        #     downward_rotation = R.from_euler('x', 180, degrees=True)
        #     quat_scipy = downward_rotation.as_quat()  # scipy format [x,y,z,w]
            
        #     # Convert to [w,x,y,z] format for EndEffectorTarget
        #     orientation_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
            
        #     try:
        #         target = EndEffectorTarget(
        #             position=target_pos,
        #             orientation=orientation_quat,
        #             frame_name="end_effector",
        #             frame_type="site"
        #         )
        #         print(f"   ✅ EndEffectorTarget created successfully")
        #     except Exception as e:
        #         print(f"   ❌ Error creating EndEffectorTarget: {e}")
        #         # Try without orientation as fallback
        #         print("   🔄 Falling back to position-only target")
        #         target = EndEffectorTarget(
        #             position=target_pos,
        #             frame_name="end_effector",
        #             frame_type="site"
        #         )
        # else:
        #     # Use YOUR original target (position only)
        #     target = EndEffectorTarget(
        #         position=target_pos,
        #         frame_name="end_effector",
        #         frame_type="site"
        #     )
        
        # Try multiple IK seeds (increased attempts for constrained cases)
        # max_attempts = 20 if use_downward_constraint else 10
        goal_configs = [target_config]
        # In your plan_to_ee_pose, when finding IK solutions:
        # for attempt in range(max_attempts):
        #     if attempt == 0:
        #         # First attempt: use current configuration as seed
        #         seed = self.get_current_config()
        #     else:
        #         # Other attempts: random seeds
        #         seed = self.ik_solver.get_random_valid_config(rng=np.random.RandomState(self.seed))
                
        #     if seed is None:
        #         continue
            
        #     solution, result = self.ik_solver.solve(target, seed) 
        #     if result == IKResult.SUCCESS:
        #         if not self.collision_checker.check_collisions(solution):
        #             goal_configs.append(solution)
        #             print(f"   Found IK solution {len(goal_configs)}")
        #             if len(goal_configs) >= 10:
        #                 break
        
        # if not goal_configs:
        #     print("❌ No valid IK solutions found")
        #     # If constrained planning failed, try unconstrained as fallback
        #     if use_downward_constraint:
        #         print("🔄 Trying fallback without orientation constraint...")
        #         return self.plan_to_ee_pose(target_pos, use_downward_constraint=False)
        #     return False
        
        # Try RRT to each goal
        start_config = self.get_current_config()
        
        for i, goal_config in enumerate(goal_configs):
            print(f"   Trying RRT to solution {i+1}/{len(goal_configs)}")
            
            path = self.rrt_planner.plan(
                start_config=start_config,
                goal_config=goal_config,
                frame_name="end_effector"
            )
            
            if path:
                self.current_path = path
                print(f"✅ Planning successful!")
                print(f"   Waypoints: {len(path)}")
                return True
        
        print("❌ RRT failed to reach any IK solution")
        return False
        
    def plan_to_ee_pose(self, target_pos: np.ndarray, task_type: str,use_downward_constraint: bool = False) -> bool:
        """Plan to end-effector pose using your IK + RRT."""
        # print(f"🎯 Planning to EE Position: {target_pos}")
        # Create target with optional orientation constraint
        if use_downward_constraint:
            # print("   🔽 Using downward orientation constraint")
            # Create downward orientation quaternion [w, x, y, z]
            downward_rotation = R.from_euler('x', 180, degrees=True)
            quat_scipy = downward_rotation.as_quat()  # scipy format [x,y,z,w]
            
            # Convert to [w,x,y,z] format for EndEffectorTarget
            orientation_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
            
            try:
                target = EndEffectorTarget(
                    position=target_pos,
                    orientation=orientation_quat,
                    frame_name="end_effector",
                    frame_type="site"
                )
                print(f"   ✅ EndEffectorTarget created successfully")
            except Exception as e:
                print(f"   ❌ Error creating EndEffectorTarget: {e}")
                # Try without orientation as fallback
                print("   🔄 Falling back to position-only target")
                target = EndEffectorTarget(
                    position=target_pos,
                    frame_name="end_effector",
                    frame_type="site"
                )
        else:
            # Use YOUR original target (position only)
            target = EndEffectorTarget(
                position=target_pos,
                frame_name="end_effector",
                frame_type="site"
            )        
        # Try multiple IK seeds (increased attempts for constrained cases).
        # Use a single RNG that advances across attempts so each gets a
        # different random seed config (the old code re-created the RNG with
        # the same seed every iteration, making attempts 1-N identical).
        max_attempts = 20 if use_downward_constraint else 10
        goal_configs = []
        ik_rng = np.random.RandomState(self.seed)
        for attempt in range(max_attempts):
            if attempt == 0:
                # First attempt: use current configuration as seed
                seed = self.get_current_config()
            else:
                # Other attempts: advancing random seeds
                seed = self.ik_solver.get_random_valid_config(rng=ik_rng)
                
            if seed is None:
                continue
            
            solution, result = self.ik_solver.solve(target, seed) 
            if result == IKResult.SUCCESS:
                if not self.collision_checker.check_collisions(solution):
                    goal_configs.append(solution)
                    print(f"   Found IK solution {len(goal_configs)}")
                    if len(goal_configs) >= 3:
                        break
        
        if not goal_configs:
            print("❌ No valid IK solutions found")
            # If constrained planning failed, try unconstrained as fallback
            if use_downward_constraint:
                print("🔄 Trying fallback without orientation constraint...")
                return self.plan_to_ee_pose(target_pos, task_type=task_type, use_downward_constraint=False)
            return False
        
        # Try RRT to each goal
        start_config = self.get_current_config()

        # print(f"Start config: {start_config} -- Trying {len(goal_configs)} goal configs")
        
        for i, goal_config in enumerate(goal_configs):
            print(f"   Trying RRT to solution {i+1}/{len(goal_configs)}")
            
            path = self.rrt_planner.plan(
                start_config=start_config,
                goal_config=goal_config,
                frame_name="end_effector",
                # task_type = task_type for STOMP im guessing
            )
            # print(f"Path first waypoint: {path[0]} -- Path last waypoint: {path[-1]}")
            # print(f"Start config: {start_config} -- Goal config: {goal_config} \n\n")
            if path:
                self.current_path = path
                # print(f"✅ Planning successful!")
                # print(f"   Waypoints: {len(path)}")
                return True
        
        print("❌ RRT failed to reach any IK solution")
        return False
    
    def execute_path(self, speed: float = 0.5, use_physics: bool = True, isGrasping: bool = False):
        """Execute planned path with optional physics simulation."""

        if self.current_path is None:
            print("No path to execute")
            return
        
        if use_physics:
            print(f"🚀 Executing path with PHYSICS: {len(self.current_path)} waypoints")
            self._execute_path_physics(speed, isGrasping = isGrasping)
        else:
            print(f"🚀 Executing path kinematically: {len(self.current_path)} waypoints")
            self._execute_path_kinematic(speed)
    
    def _execute_path_physics(self, speed: float, isGrasping: bool = False):
        """Execute path using physics-based position control."""
        
        # Control parameters
        settle_time = int(100 / speed)  # Physics steps to wait for convergence
        position_tolerance = 0.01       # Radians tolerance for "reached"
        
        for i, target_config in enumerate(self.current_path):
            print(f" 🎮 Physics waypoint {i+1}/{len(self.current_path)}")
            
            # Set position control targets (only for arm joints)
            self.scene_data.ctrl[:7] = target_config[:7]

            if isGrasping:
                self.close_gripper_gentle(target_force=5000.0)

            # Let the controller reach the target
            converged = False
            for step in range(settle_time):
                # Run physics simulation
                mujoco.mj_step(self.scene_model, self.scene_data)
                
                # Sync viewer
                if self.current_viewer is not None:
                    self.current_viewer.sync()
                
                # Check convergence
                current_pos = self.scene_data.qpos[:7]
                error = np.linalg.norm(current_pos - target_config[:7])
                
                if error < position_tolerance:
                    if not converged:
                        print(f"   ✅ Reached target (error: {error:.4f})")
                        converged = True
                
                time.sleep(0.001)  # 1ms delay
            
            if not converged:
                current_pos = self.scene_data.qpos[:7]
                error = np.linalg.norm(current_pos - target_config[:7])
                print(f"   ⚠️ Timeout (error: {error:.4f})")
            
            # Brief pause between waypoints
            time.sleep(0.1 / speed)
        
        print("✅ Physics path execution completed")
    
    def _execute_path_kinematic(self, speed: float):
        """Execute path using kinematic positioning (original method)."""
        
        for i, config in enumerate(self.current_path):
            print(f" Waypoint {i+1}/{len(self.current_path)}")
            
            # Set configuration directly
            self.scene_data.qpos[:self.robot_dof] = config
            mujoco.mj_forward(self.scene_model, self.scene_data)
            
            # Sync viewer
            if self.current_viewer is not None:
                self.current_viewer.sync()
            
            time.sleep(0.2 / speed)
        
        print("✅ Kinematic path execution completed")
    
    # =======================================================================
    # UNIFIED PHYSICS-BASED GRIPPER CONTROL METHODS
    # =======================================================================
        

        # Simple gripper control methods for your 8th actuator (index 7) tendon gripper

    def open_gripper(self):
        """Open the tendon gripper using actuator 7 (0-255 range)."""
        print("🤏 Opening tendon gripper...")
        
        # Set actuator 7 to fully open (255 = open, 0 = closed)
        self.scene_data.ctrl[7] = 255.0
        
        # Run physics until gripper opens
        for step in range(500):
            mujoco.mj_step(self.scene_model, self.scene_data)
            
            # Sync viewer if available
            if self.current_viewer is not None:
                self.current_viewer.sync()
            
            time.sleep(0.001)  # 1ms per step
        
        print(f"   ✅ Gripper opened (control value: {self.scene_data.ctrl[7]:.1f})")

    def close_gripper(self):
        """Close the tendon gripper using actuator 7 with basic force control."""
        print("✋ Closing tendon gripper...")
        
        # Start from current position and gradually close
        initial_control = self.scene_data.ctrl[7]
        print(f"   Initial control value: {initial_control:.1f}")
        
        # Gradually reduce control value (close gripper)
        for step in range(1000):
            # Calculate target control value (gradually closing)
            progress = step / 1000
            target_control = initial_control * (1 - progress * 0.9)  # Close to 10% of initial
            
            # Set control command
            self.scene_data.ctrl[7] = target_control
            
            # Step physics
            mujoco.mj_step(self.scene_model, self.scene_data)
            
            # Check for contact forces with object3
            max_contact_force = 0
            object_in_contact = False
            
            for i in range(self.scene_data.ncon):
                contact = self.scene_data.contact[i]
                geom1_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                
                # Check for finger-object contact
                if ((geom1_name and ('finger' in geom1_name.lower() or 'pad' in geom1_name.lower()) and 
                    geom2_name and 'object3' in geom2_name.lower()) or 
                    (geom2_name and ('finger' in geom2_name.lower() or 'pad' in geom2_name.lower()) and 
                    geom1_name and 'object3' in geom1_name.lower())):
                    
                    object_in_contact = True
                    force_mag = np.linalg.norm(contact.f[:3])
                    max_contact_force = max(max_contact_force, force_mag)
            
            # Stop if sufficient force is reached
            if object_in_contact and max_contact_force > 10.0:  # 10N target force
                print(f"   ✅ Object grasped! Force: {max_contact_force:.2f}N")
                break
            
            # Stop if gripper nearly closed
            if target_control < 5.0:  # Control value below 5
                print(f"   ⚠️ Gripper nearly closed (control: {target_control:.1f})")
                break
            
            # Sync viewer occasionally
            if step % 20 == 0 and self.current_viewer is not None:
                self.current_viewer.sync()
            
            time.sleep(0.002)  # 2ms per step
        
        # Hold final position
        final_control = self.scene_data.ctrl[7]
        for _ in range(100):
            self.scene_data.ctrl[7] = final_control
            mujoco.mj_step(self.scene_model, self.scene_data)
            if self.current_viewer is not None:
                self.current_viewer.sync()
            time.sleep(0.001)
        
        print(f"   Final control value: {final_control:.1f}")

    def close_gripper_gentle(self, target_force):
        """Gentle close for testing - lower force to avoid pushing object away."""
        print("✋ Closing tendon gripper (gentle)...")
        
        initial_control = self.scene_data.ctrl[7]
        max_step = 15 #100

        for step in range(max_step):
            # Slower, more gentle closing
            progress = step / max_step
            target_control = initial_control * (1 - progress * 0.8)  # Only close to 20% of initial
            
            self.scene_data.ctrl[7] = target_control
            mujoco.mj_step(self.scene_model, self.scene_data)
            
            # Check for lighter contact
            max_contact_force = 0
            object_in_contact = False
            
            for i in range(self.scene_data.ncon):
                contact = self.scene_data.contact[i]
                geom1_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                
                if ((geom1_name and ('finger' in geom1_name.lower() or 'pad' in geom1_name.lower()) and 
                    geom2_name and 'object3' in geom2_name.lower()) or 
                    (geom2_name and ('finger' in geom2_name.lower() or 'pad' in geom2_name.lower()) and 
                    geom1_name and 'object3' in geom1_name.lower())):
                    
                    object_in_contact = True
                    force_mag = np.linalg.norm(contact.f[:3])
                    max_contact_force = max(max_contact_force, force_mag)
            
            # Stop at lower force for gentle grasp
            if object_in_contact and max_contact_force > target_force:
                print(f"   ✅ Gentle grasp! Force: {max_contact_force:.2f}N")
                break
            
            if step % 20 == 0 and self.current_viewer is not None:
                self.current_viewer.sync()
            
            time.sleep(0.05)  # Slower for gentle approach
        
        # Hold position
        final_control = self.scene_data.ctrl[7]
        for _ in range(100):
            self.scene_data.ctrl[7] = final_control
            mujoco.mj_step(self.scene_model, self.scene_data)
            if self.current_viewer is not None:
                self.current_viewer.sync()
            time.sleep(0.001)
        
        print(f"   Final control value: {final_control:.1f}")

    def test_gripper_simple(self):
        """Simple test of gripper open/close cycle."""
        print("\n🧪 SIMPLE GRIPPER TEST")
        print("=" * 25)
        
        input("Press Enter to open gripper...")
        self.open_gripper()
        
        input("Press Enter to close gripper (gentle)...")
        self.close_gripper_gentle(target_force=3.0)  # Very gentle for testing
        
        input("Press Enter to open gripper again...")
        self.open_gripper()
        
        print("✅ Gripper test completed!")

    # Usage example in your pick and place:
    def simple_pick_and_place_example(self):
        """Simple pick and place using the basic gripper methods."""
        
        # Step 1: Open gripper
        self.open_gripper()
        
        # Step 2: Move to object (your existing planning code)
        # ... your planning and movement code here ...
        
        # Step 3: Close gripper to grasp
        self.close_gripper()  # or use close_gripper_gentle() for lighter objects
        
        # Step 4: Move to place location (your existing planning code)
        # ... your planning and movement code here ...
        
        # Step 5: Open gripper to release
        self.open_gripper()


    # Key points for your tendon gripper:
    """
    CONTROL MAPPING:
    - Actuator index: 7 (8th actuator)
    - Control range: 0 (closed) to 255 (open)
    - Your XML: <general name="actuator8" tendon="split" ctrlrange="0 255" .../>

    USAGE:
    1. self.scene_data.ctrl[7] = 255.0  # Fully open
    2. self.scene_data.ctrl[7] = 0.0    # Fully closed
    3. self.scene_data.ctrl[7] = 127.5  # Half open

    PHYSICS STEPS:
    - Always call mujoco.mj_step() after setting control values
    - Use viewer.sync() to update visualization
    - Add time.sleep() for realistic motion timing
    """

    # =======================================================================
    # UNIFIED PHYSICS-BASED GRIPPER CONTROL METHODS
    # =======================================================================

    def pick_and_place_object3_full_physics(self) -> bool:
        """Complete pick and place demo using physics simulation throughout."""
        
        print("\n🎯 Physics-Based Pick and Place Demo: Object3")
        print("=" * 50)
        
        # Get object positions
        obj_pos = self.get_object_position("object3")
        if obj_pos is None:
            return False
        
        obj1_pos = self.get_object_position("object1")
        if obj1_pos is None:
            return False
        
        # Calculate approach positions
        approach_height = 0.12
        grasp_height = 0.01  # Slightly higher for physics-based approach
        
        approach_pos = obj_pos.copy()
        approach_pos[2] = obj_pos[2] + approach_height
        
        grasp_pos = obj_pos.copy()
        grasp_pos[2] = obj_pos[2] + grasp_height
        
        place_approach_pos = obj1_pos.copy()
        place_approach_pos[2] = obj1_pos[2] + approach_height
        
        place_pos = obj1_pos.copy()
        place_pos[2] = obj1_pos[2] + grasp_height + 0.08
        
        print(f"📋 Physics Pick and Place Plan:")
        print(f"  Object3 at: {obj_pos}")
        print(f"  Approach:   {approach_pos}")
        print(f"  Grasp:      {grasp_pos}")
        print(f"  Place approach: {place_approach_pos}")
        print(f"  Place:      {place_pos}")
        
        # Phase 1: Open gripper with physics
        print("\n" + "="*30)
        print("PHASE 1: PREPARATION")
        print("="*30)
        # # input("Press Enter to open gripper with physics...")
        # # self.open_gripper()
        
        # Phase 2: Approach object
        print("\n" + "="*30)
        print("PHASE 2: APPROACH OBJECT")
        print("="*30)
        # input("Press Enter to move to approach position...")
        if not self.plan_to_ee_pose(approach_pos, use_downward_constraint=True, task_type = "transit"):
            print("❌ Failed to plan to approach position")
            return False
        
        # if self.current_path and len(self.current_path) < 15:
        #     smooth_path = self.densify_path(self.current_path, max_joint_step=0.04)
        #     print(f"✅ Path densified from {self.current_path} to {len(smooth_path)} waypoints")
        #     self.current_path = smooth_path
        
        self.execute_path(speed=0.1, use_physics=False)
        
        # Phase 3: Move to grasp position
        print("\n" + "="*30)
        print("PHASE 3: POSITION FOR GRASPING")
        print("="*30)
        # input("Press Enter to move to grasp position...")
        if not self.plan_to_ee_pose(grasp_pos, use_downward_constraint=True, task_type = "pick"):
            print("❌ Failed to plan to grasp position")
            return False
        
        # if self.current_path and len(self.current_path) < 15:
        #     smooth_path = self.densify_path(self.current_path, max_joint_step=0.03)
        #     print(f"✅ Path densified from {self.current_path} to {len(smooth_path)} waypoints")
        #     self.current_path = smooth_path
        
        self.open_gripper()
        self.execute_path(speed=0.5, use_physics=True)  # Slower for precision
        
        # Phase 4: Physics-based grasping
        print("\n" + "="*30)
        print("PHASE 4: PHYSICS-BASED GRASPING")
        print("="*30)
        # input("Press Enter to grasp object with full physics...")
        
        self.close_gripper_gentle(target_force=5000.0)
        
        # Phase 5: Lift with physics verification
        print("\n" + "="*30)
        print("PHASE 5: LIFT OBJECT")
        print("="*30)
        # input("Press Enter to lift object...")
        
        # Plan lift motion
        lift_pos = approach_pos.copy()
        lift_pos[2] += 0.05  # Extra height for safety
        
        if not self.plan_to_ee_pose(lift_pos, use_downward_constraint=False, task_type = "pick"):
            print("❌ Failed to plan lift motion")
            return False
        
        # Execute lift with physics (object should follow if grasped)
        self.execute_path(speed=0.5, use_physics=True, isGrasping=True)
        
        # Verify object is still grasped after lift
        current_obj_pos = self.get_object_position("object3")
        if current_obj_pos is not None:
            height_gained = current_obj_pos[2] - obj_pos[2]
            print(f"   Object height gained: {height_gained*100:.1f}cm")
        
        # Phase 6: Transport to place location
        print("\n" + "="*30)
        print("PHASE 6: TRANSPORT TO PLACE")
        print("="*30)
        # input("Press Enter to move to place location...")
        
        if not self.plan_to_ee_pose(place_approach_pos, use_downward_constraint=True, task_type = "transit"):
            print("❌ Failed to plan transport motion")
            return False
        
        self.execute_path(speed=0.5, use_physics=True, isGrasping=True)
        
        # Phase 7: Lower to place position
        print("\n" + "="*30)
        print("PHASE 7: PLACE OBJECT")
        print("="*30)
        # input("Press Enter to lower object to place position...")
        
        if not self.plan_to_ee_pose(place_pos, use_downward_constraint=True, task_type = "place"):
            print("❌ Failed to plan to place position")
            return False
        
        self.execute_path(speed=0.5, use_physics=True, isGrasping=True)  # Slow and careful
        
        # Phase 8: Release with physics
        print("\n" + "="*30)
        print("PHASE 8: RELEASE OBJECT")
        print("="*30)
        # input("Press Enter to release object with physics...")
        
        self.open_gripper()
        
        # Give time for object to settle
        print("   Allowing object to settle...")
        for _ in range(100):  # 100 physics steps
            mujoco.mj_step(self.scene_model, self.scene_data)
            if self.current_viewer is not None:
                self.current_viewer.sync()
            time.sleep(0.005)
        
        # Verify placement
        final_obj_pos = self.get_object_position("object3")
        if final_obj_pos is not None:
            placement_distance = np.linalg.norm(final_obj_pos[:2] - obj1_pos[:2])  # XY distance to target
            print(f"   Placement accuracy: {placement_distance*100:.1f}cm from target")
            
            if placement_distance < 0.05:  # Within 5cm
                print("   ✅ Excellent placement!")
            elif placement_distance < 0.10:  # Within 10cm
                print("   ✅ Good placement!")
            else:
                print("   ⚠️ Placement could be better")
        
        # Phase 9: Retreat
        print("\n" + "="*30)
        print("PHASE 9: RETREAT")
        print("="*30)
        # input("Press Enter to retreat from object...")
        
        retreat_pos = place_approach_pos.copy()
        retreat_pos[2] += 0.05  # Extra clearance
        
        if not self.plan_to_ee_pose(retreat_pos, use_downward_constraint=False, task_type = "transit"):
            print("❌ Failed to plan retreat motion")
            return False
        
        self.execute_path(speed=0.5, use_physics=True)
        
        # Phase 10: Return home
        print("\n" + "="*30)
        print("PHASE 10: RETURN HOME")
        print("="*30)
        # input("Press Enter to return to home position...")
        
        home_config = np.zeros(self.robot_dof)
        # self.current_path = [self.get_current_config(), home_config]
        # input("Press Enter to move to grasp position...")
        if not self.plan_to_config(home_config, use_downward_constraint=False):
            print("❌ Failed to plan to grasp position")
            return False
        # smooth_path = self.densify_path(self.current_path, max_joint_step=0.05)
        # smooth_path = self.densify_path(self.current_path, max_joint_step=0.03)
        # print(f"✅ Path densified from {self.current_path} to {len(smooth_path)} waypoints")
        # self.current_path = smooth_path
        
        self.execute_path(speed=0.5, use_physics=True)
        
        print("\n" + "🎉"*20)
        print("PHYSICS-BASED PICK AND PLACE COMPLETED!")
        print("🎉"*20)
        
        return True

    # Update your run_demo method to use the full physics version:
    def run_demo_full_physics(self):
        """Run the full physics pick and place demo."""
        
        print("\n🎬 Starting FULL PHYSICS Pick and Place Demo")
        print("=" * 50)
        
        with mujoco.viewer.launch_passive(self.scene_model, self.scene_data) as viewer:
            self.current_viewer = viewer
            
            print("\n📋 This PHYSICS demo will:")
            print("1. Use physics simulation for ALL gripper operations")
            print("2. Monitor contact forces during grasping")
            print("3. Verify grasp quality with lift tests")
            print("4. Handle grasp failures with recovery strategies")
            print("5. Use physics-based position control throughout")
            print("\nEach phase requires Enter to proceed...")
            
            input("\nPress Enter to start full physics demo...")
            
            # Run full physics pick and place
            success = self.pick_and_place_object3_full_physics()
            
            if success:
                print("\n✅ Full physics pick and place demo completed successfully!")
            else:
                print("\n❌ Full physics pick and place demo failed")
            
            self.current_viewer = None
            input("\nPress Enter to exit...")
 
    def debug_ik_issue(self, target_pos: np.ndarray):
        """Debug why IK fails at this position."""
        print(f"\n🔍 Debugging IK at position: {target_pos}")
        
        # Test 1: IK without collision checking
        target = EndEffectorTarget(position=target_pos, frame_name="end_effector", frame_type="site")
        seed = np.zeros(self.robot_dof)
        
        print("1. Testing IK without collision...")
        solution, result = self.ik_solver.solve(target, seed)
        print(f"   IK result: {result}")
        
        if result == IKResult.SUCCESS:
            print("2. Testing collision detection on solution...")
            collision = self.collision_checker.check_collisions(solution)
            print(f"   Collision detected: {collision}")
            
            if collision:
                print("3. Checking what's colliding...")
                self.debug_collision_details(solution)
                
        # Test 2: Try multiple seeds
        print("4. Testing with 5 different seeds...")
        for i in range(5):
            seed = self.ik_solver.get_random_valid_config()
            if seed is not None:
                solution, result = self.ik_solver.solve(target, seed)
                collision = self.collision_checker.check_collisions(solution) if result == IKResult.SUCCESS else "N/A"
                print(f"   Seed {i+1}: IK={result}, Collision={collision}")
                
                # Show collision details for first successful seed with collision
                if result == IKResult.SUCCESS and collision and i == 0:
                    print(f"   Collision details for seed {i}:")
                    self.debug_collision_details(solution)

    def debug_collision_details(self, config: np.ndarray):
        """Debug what specific geometries are colliding."""
        print("   🔍 Collision Analysis:")
        
        # Set the configuration
        self.scene_data.qpos[:self.robot_dof] = config
        mujoco.mj_forward(self.scene_model, self.scene_data)
        
        # Check all collision pairs
        for i in range(self.scene_data.ncon):
            contact = self.scene_data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # Get geometry names
            geom1_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            
            print(f"     Collision: {geom1_name} <-> {geom2_name}")
            print(f"     Distance: {contact.dist:.4f}m")

    def test_position_reachability(self):
        """Test if current grasp position is reachable."""
        obj_pos = self.get_object_position("object3")
        if obj_pos is None:
            return
        
        approach_pos = obj_pos.copy()
        approach_pos[2] = obj_pos[2] + 0.15
        
        grasp_pos = obj_pos.copy() 
        grasp_pos[2] = obj_pos[2] + 0.04
        
        print("Testing approach position:")
        self.debug_ik_issue(approach_pos)
        
        print("\nTesting grasp position:")
        self.debug_ik_issue(grasp_pos)

    def landscape_test(self):
                # In your demo, before planning:
        print("Verifying cost landscape...")
        print("\nTest 1: Cost landscape grid")
        self.rrt_planner.verify_cost_landscape_grid()

        print("\nTest 2: Radial Cost Profile")
        self.rrt_planner.verify_radial_cost_profile()

        print("\nTest 3: Path Cost Comparison")
        self.rrt_planner.verify_path_cost_comparison()
        # self.rrt_planner.verify_cost_gradients()
    
    # def run_demo(self):
    #     """Run the pick and place demo with MuJoCo viewer."""
        
    #     print("\n🎬 Starting Pick and Place Demo")
    #     print("=" * 40)
        
    #     with mujoco.viewer.launch_passive(self.scene_model, self.scene_data) as viewer:
    #         self.current_viewer = viewer
            
    #         print("\n📋 This demo will:")
    #         print("1. Locate object3 in the scene")
    #         print("2. Plan approach trajectory (🔽 downward constraint)")
    #         print("3. Grasp the object (🔽 downward constraint)")
    #         print("4. Move it to place approach (🔽 downward constraint)")
    #         print("5. Place and release (🔽 downward constraint)")
    #         print("6. Retreat and return home (unconstrained)")
    #         print("\nEach step requires Enter to proceed...")
            
    #         input("\nPress Enter to start pick and place demo...")

    #         self.test_position_reachability()
            
    #         # Run pick and place
    #         success = self.pick_and_place_object3()
            
    #         if success:
    #             print("\n✅ Pick and place demo completed successfully!")
    #         else:
    #             print("\n❌ Pick and place demo failed")
            
    #         self.current_viewer = None
    #         input("\nPress Enter to exit...")


class PandaPickAndPlace_L2(PandaPickAndPlace):
    def __init__(self, scene_xml_path: str, robot_xml_path: str, seed: int = 42):
        """Initialize with enhanced tendon gripper support."""
        
        print("🤖 Initializing Panda Pick and Place Demo")
        print("=" * 50)
        
        # Load models
        print("🔍 Loading MuJoCo models...")
        self.scene_name = os.path.splitext(os.path.basename(scene_xml_path))[0]
        self.scene_model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        self.scene_data = mujoco.MjData(self.scene_model)
        self.seed = seed
        
        print(f"✅ Models loaded from {scene_xml_path}:")
        print(f"   Scene: {self.scene_model.ngeom} geoms, {self.scene_model.njnt} joints")
        print(f"   Robot: {self.robot_model.ngeom} geoms, {self.robot_model.njnt} joints")
        
        # Initialize your planning modules
        self.ik_solver = IKSolver(self.robot_model)
        self.collision_checker = CollisionChecker(self.scene_model, self.robot_model)
        failing_joints = ['finger_joint1', 'finger_joint2']
        self.collision_estimator = CollisionEstimator(self.scene_model, inflation_radius=0, failing_joints=failing_joints, robot_joints=failing_joints)

        self.exp_traj_manager = ExperimentTrajectoryManager()
        

        self.robot_dof = self.robot_model.njnt
        self.current_path = None
        self.current_viewer = None
        
        # Set initial pose
        self._set_home_position()

        self.severity_table = self.generate_severity_table()

        self.rrt_planner = JointSpaceRRTConnectFailure(
            collision_estimator=self.collision_estimator,
            body_severity_table=self.severity_table,
            scene_model=self.scene_model,
            robot_model=self.robot_model,
            ik_solver=self.ik_solver,
            collision_threshold=0.0000005, 
            planning_space=PlanningSpace.JOINT_SPACE,
            step_size=0.005,
            goal_bias=0.7,
            seed=self.seed)
        
        # self.rrt_planner = JointSpaceRRTConnect(
        #     scene_model=self.scene_model,
        #     robot_model=self.robot_model,
        #     ik_solver=self.ik_solver,
        #     collision_threshold=0.01,  # 1cm threshold
        #     planning_space=PlanningSpace.JOINT_SPACE,
        #     step_size=0.005,
        #     goal_bias=0.7,
        #     seed=self.seed
        # )

    def generate_multiple_trajectories(self):
        """Run the full physics pick and place demo."""
        
        print("\n🎬 Starting FULL PHYSICS Pick and Place Demo")
        print("=" * 50)
        
        obj_pos = self.get_object_position("object3")
        if obj_pos is None:
            return False
        
        obj1_pos = self.get_object_position("object1")
        if obj1_pos is None:
            return False
        
        self.object3_init_qpos = np.append(obj_pos.copy(),[0,0,0,1])

        # Calculate approach positions
        approach_height = 0.12
        obj_pos[2] += approach_height
        obj1_pos[2] += approach_height

        waypoint_pos = (obj_pos + obj1_pos) / 2
        
        # y_padding = 0.1

        # obj_pos[1] += y_padding
        # obj1_pos[1] -= y_padding

        # waypoint is somewhere in between 
        rng = np.random.RandomState(self.seed)

        num_samples = 1
        # waypoint_pos = (obj_pos - obj1_pos) + obj1_pos #rng.random(size=(num_samples, 3)) * (obj_pos - obj1_pos) + obj1_pos
        with mujoco.viewer.launch_passive(self.scene_model, self.scene_data) as viewer:
            self.current_viewer = viewer
            
            print("\n📋 This PHYSICS demo will:")
            print("1. Use physics simulation for ALL gripper operations")
            print("2. Monitor contact forces during grasping")
            print("3. Verify grasp quality with lift tests")
            print("4. Handle grasp failures with recovery strategies")
            print("5. Use physics-based position control throughout")
            print("\nEach phase requires Enter to proceed...")
            
            # input("\nPress Enter to start full physics demo...")
            time.sleep(3)

            for i in range(num_samples):
                # Run full physics pick and place
                try:
                    self.reset_object3("object3")
                    self._set_home_position()
                    success = self.pick_and_place_generate_traj(waypoint_pos, i)
                except Exception as e:
                    print(e)
                    continue

            if success:
                print("\n✅ Full physics pick and place demo completed successfully!")
            else:
                print("\n❌ Full physics pick and place demo failed")
            
            self.current_viewer = None
            time.sleep(5)
            # input("\nPress Enter to exit...")
    
    def play_saved_trajectories(self, trajectory_files):
        """Run the full physics pick and place demo."""
        print("\n🎬 Starting FULL PHYSICS Pick and Place Demo")
        print("=" * 50)
        
        obj_pos = self.get_object_position("object3")
        if obj_pos is None:
            return False
        self.object3_init_qpos = np.append(obj_pos.copy(),[0,0,0,1])

        with mujoco.viewer.launch_passive(self.scene_model, self.scene_data) as viewer:
            self.current_viewer = viewer
            
            # print("\n📋 This PHYSICS demo will:")
            # print("1. Use physics simulation for ALL gripper operations")
            # print("2. Monitor contact forces during grasping")
            # print("3. Verify grasp quality with lift tests")
            # print("4. Handle grasp failures with recovery strategies")
            # print("5. Use physics-based position control throughout")
            # print("\nEach phase requires Enter to proceed...")

            # print("Playing saved trajectories in folder")
            
            # input("\nPress Enter to start full physics demo...")
            time.sleep(3)

            for sample_i, trajectory_file in enumerate(trajectory_files):
                print("\n\nPlaying trajectory file: "+os.path.basename(trajectory_file))
                input("Start....")

                self.exp_traj_manager.load_from_file(trajectory_file)
                saved_trajectories = None
                if "baseline" in trajectory_file:
                    # no waypoint
                    waypoint_pos = None
                    saved_trajectories = {"phase7": self.exp_traj_manager.trajectories[self.scene_name]['baseline']['trajectory']}
                else:
                    waypoint_pos = self.exp_traj_manager.trajectories[self.scene_name]['phase6']['goal_pos']
                    saved_trajectories = {}
                    saved_trajectories["phase6"] = self.exp_traj_manager.trajectories[self.scene_name]['phase6']['trajectory']
                    saved_trajectories["phase7"] = self.exp_traj_manager.trajectories[self.scene_name]['phase7']['trajectory']


                # Run full physics pick and place
                try:
                    self.reset_object3("object3")
                    self._set_home_position()
                    success = self.pick_and_place_generate_traj(waypoint_pos, sample_i, saved_trajectories)
                except Exception as e:
                    print("huh?")
                    continue

                if success:
                    print("\n✅ Full physics pick and place demo completed successfully!")
                else:
                    print("\n❌ Full physics pick and place demo failed")
                    break
            
            self.current_viewer = None
            time.sleep(5)
            # input("\nPress Enter to exit...")
    
    def pick_and_place_generate_traj(self, waypoint_pos=None, sample_i=None, saved_trajectories=None) -> bool:
        """Complete pick and place demo using physics simulation throughout."""
        
        print("\n🎯 Physics-Based Pick and Place Demo: Object3")
        print("=" * 50)
        
        # Get object positions
        obj_pos = self.get_object_position("object3")
        if obj_pos is None:
            return False
        
        obj1_pos = self.get_object_position("object1")
        if obj1_pos is None:
            return False
        
        # Calculate approach positions
        approach_height = 0.12
        grasp_height = 0.01  # Slightly higher for physics-based approach
        
        approach_pos = obj_pos.copy()
        approach_pos[2] = obj_pos[2] + approach_height
        
        grasp_pos = obj_pos.copy()
        grasp_pos[2] = obj_pos[2] + grasp_height
        
        place_approach_pos = obj1_pos.copy()
        place_approach_pos[2] = obj1_pos[2] + approach_height
        
        place_pos = obj1_pos.copy()
        place_pos[2] = obj1_pos[2] + grasp_height + 0.08
        
        # print(f"📋 Physics Pick and Place Plan:")
        # print(f"  Object3 at: {obj_pos}")
        # print(f"  Approach:   {approach_pos}")
        # print(f"  Grasp:      {grasp_pos}")
        # print(f"  Place approach: {place_approach_pos}")
        # print(f"  Place:      {place_pos}")
        
        # Phase 1: Open gripper with physics
        # print("\n" + "="*30)
        # print("PHASE 1: PREPARATION")
        # print("="*30)
        # # input("Press Enter to open gripper with physics...")
        # # self.open_gripper()
        
        # Phase 2: Approach object
        # print("\n" + "="*30)
        # print("PHASE 2: APPROACH OBJECT")
        # print("="*30)
        # input("Press Enter to move to approach position...")
        if not self.plan_to_ee_pose(approach_pos, use_downward_constraint=True, task_type = "transit"):
            print("❌ Failed to plan to approach position")
            return False
        
        self.execute_path(speed=0.5, use_physics=False)
        
        # Phase 3: Move to grasp position
        # print("\n" + "="*30)
        # print("PHASE 3: POSITION FOR GRASPING")
        # print("="*30)
        # input("Press Enter to move to grasp position...")
        if not self.plan_to_ee_pose(grasp_pos, use_downward_constraint=True, task_type = "pick"):
            print("❌ Failed to plan to grasp position")
            return False
        
        self.open_gripper()
        self.execute_path(speed=0.5, use_physics=True)  # Slower for precision
        
        # Phase 4: Physics-based grasping
        # print("\n" + "="*30)
        # print("PHASE 4: PHYSICS-BASED GRASPING")
        # print("="*30)
        # input("Press Enter to grasp object with full physics...")
        
        self.close_gripper_gentle(target_force=5000.0)
        
        # Phase 5: Lift with physics verification
        # print("\n" + "="*30)
        # print("PHASE 5: LIFT OBJECT")
        # print("="*30)
        # # input("Press Enter to lift object...")
        
        # Plan lift motion
        lift_pos = approach_pos.copy()
        lift_pos[2] += 0.05  # Extra height for safety
        
        if not self.plan_to_ee_pose(lift_pos, use_downward_constraint=False, task_type = "pick"):
            print("❌ Failed to plan lift motion")
            return False
        
        # Execute lift with physics (object should follow if grasped)
        self.execute_path(speed=0.5, use_physics=True, isGrasping=True)
        
        # Verify object is still grasped after lift
        current_obj_pos = self.get_object_position("object3")
        if current_obj_pos is not None:
            height_gained = current_obj_pos[2] - obj_pos[2]
            print(f"   Object height gained: {height_gained*100:.1f}cm")
        

        # optional Phase 6: Transport to waypoint
        if waypoint_pos is not None:
            print("\n" + "="*30)
            print("PHASE 6: TRANSPORT TO WAYPOINT")
            print("="*30)
            # input("Press Enter to move to place location...")
            if saved_trajectories is None:
                print(f"Hello I am here with waypoint {waypoint_pos}")
                if not self.plan_to_ee_pose(waypoint_pos, use_downward_constraint=True, task_type = "transit"):
                    print("❌ Failed to plan transport motion")
                    return False
                
                print("Have a plan")
                # print("storing trajectory")
                me_cost, safety_cost = self.rrt_planner.get_trajectory_cost(self.current_path)
                self.exp_traj_manager.store_trajectory(self.scene_name, "phase6", self.current_path, goal_pos=waypoint_pos, me_cost=me_cost, safety_cost=safety_cost)
            else:
                print("   Moving to start of saved trajectory")
                self.current_path = [saved_trajectories['phase6'][0]]
                self.execute_path(speed=0.5, use_physics=True, isGrasping=True)
                print("   Allowing robot to settle...")
                for _ in range(10):  # 100 physics steps
                    mujoco.mj_step(self.scene_model, self.scene_data)
                    if self.current_viewer is not None:
                        self.current_viewer.sync()
                    time.sleep(0.005)
                
                self.current_path = saved_trajectories['phase6']
            print("    Executing saved trajectory")
            # GET SITE POSITION HERE AND TRACK ITS TRAJECTORY WHILE PATH IS GETTING EXECUTED
            self.execute_path(speed=0.5, use_physics=True, isGrasping=True)


        # Phase 7: Transport to place location
        print("\n" + "="*30)
        print("PHASE 7: TRANSPORT TO FINAL PLACE")
        print("="*30)
        # input("Press Enter to move to place location...")
    
        if saved_trajectories is None:
            if not self.plan_to_ee_pose(place_approach_pos, use_downward_constraint=True, task_type = "transit"):
                print("❌ Failed to plan transport motion")
                return False

            # print("storing trajectory")
            me_cost, safety_cost = self.rrt_planner.get_trajectory_cost(self.current_path)
            print(f"")
            self.exp_traj_manager.store_trajectory(self.scene_name, "baseline", self.current_path, goal_pos=place_approach_pos, me_cost=me_cost, safety_cost=safety_cost)
            # self.exp_traj_manager.save_to_file(self.scene_name+"_RRTConnect_sample_"+str(sample_i)+".pkl")
            self.exp_traj_manager.save_to_file(self.scene_name+"_RRTConnect_new_baseline.pkl")

        else:
            print("   Moving to start of saved trajectory")
            self.current_path = [saved_trajectories['phase7'][0]]
            self.execute_path(speed=0.5, use_physics=True, isGrasping=True)
            print("   Allowing robot to settle...")
            for _ in range(10):  # 100 physics steps
                mujoco.mj_step(self.scene_model, self.scene_data)
                if self.current_viewer is not None:
                    self.current_viewer.sync()
                time.sleep(0.005)
            self.current_path = saved_trajectories['phase7']
        print("    Executing saved trajectory")
        # GET SITE POSITION HERE AND TRACK ITS TRAJECTORY WHILE PATH IS GETTING EXECUTED
        self.execute_path(speed=0.5, use_physics=True, isGrasping=True)
        
        # Phase 7: Lower to place position
        print("\n" + "="*30)
        print("PHASE 8: PLACE OBJECT")
        print("="*30)
        # input("Press Enter to lower object to place position...")
        
        if not self.plan_to_ee_pose(place_pos, use_downward_constraint=True, task_type = "place"):
            print("❌ Failed to plan to place position")
            return False
        
        self.execute_path(speed=0.5, use_physics=True, isGrasping=True)  # Slow and careful
        
        # Phase 8: Release with physics
        print("\n" + "="*30)
        print("PHASE 9: RELEASE OBJECT")
        print("="*30)
        # input("Press Enter to release object with physics...")
        
        self.open_gripper()
        
        # Give time for object to settle
        print("   Allowing object to settle...")
        for _ in range(100):  # 100 physics steps
            mujoco.mj_step(self.scene_model, self.scene_data)
            if self.current_viewer is not None:
                self.current_viewer.sync()
            time.sleep(0.005)
        
        # Verify placement
        final_obj_pos = self.get_object_position("object3")
        if final_obj_pos is not None:
            placement_distance = np.linalg.norm(final_obj_pos[:2] - obj1_pos[:2])  # XY distance to target
            print(f"   Placement accuracy: {placement_distance*100:.1f}cm from target")
            
            if placement_distance < 0.05:  # Within 5cm
                print("   ✅ Excellent placement!")
            elif placement_distance < 0.10:  # Within 10cm
                print("   ✅ Good placement!")
            else:
                print("   ⚠️ Placement could be better")
        
        # Phase 9: Retreat
        # print("\n" + "="*30)
        # print("PHASE 10: RETREAT")
        # print("="*30)
        # # input("Press Enter to retreat from object...")
        
        # retreat_pos = place_approach_pos.copy()
        # retreat_pos[2] += 0.05  # Extra clearance
        
        # if not self.plan_to_ee_pose(retreat_pos, use_downward_constraint=False, task_type = "transit"):
        #     print("❌ Failed to plan retreat motion")
        #     return False
        
        # self.execute_path(speed=0.5, use_physics=False)
        
        # # Phase 10: Return home
        # print("\n" + "="*30)
        # print("PHASE 10: RETURN HOME")
        # print("="*30)
        # # input("Press Enter to return to home position...")
        
        # home_config = np.zeros(self.robot_dof)
        # # self.current_path = [self.get_current_config(), home_config]
        # # input("Press Enter to move to grasp position...")
        # if not self.plan_to_config(home_config, use_downward_constraint=False):
        #     print("❌ Failed to plan to grasp position")
        #     return False
        # # smooth_path = self.densify_path(self.current_path, max_joint_step=0.05)
        # # smooth_path = self.densify_path(self.current_path, max_joint_step=0.03)
        # # print(f"✅ Path densified from {self.current_path} to {len(smooth_path)} waypoints")
        # # self.current_path = smooth_path
        
        # self.execute_path(speed=0.5, use_physics=False)
        
        print("\n" + "🎉"*20)
        print("PHYSICS-BASED PICK AND PLACE COMPLETED!")
        print("🎉"*20)
        
        return True

    
    def reset_object3(self, body_name: str):
        model = self.scene_model
        data = self.scene_data
        
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jid = model.body_jntadr[bid]      # assume first joint is free
        qpos_adr = model.jnt_qposadr[jid] # index into qpos
        # copy 7 values (3 pos + 4 quat) from keyframe 0
        data.qpos[qpos_adr:qpos_adr+7] = self.object3_init_qpos
        mujoco.mj_forward(model, data) 

    def generate_severity_table(self):
        """
        rules: 
            red blocks named "obstacle_hard_xyz" -> severity = 10
            white blocks named "obstacles_soft_xyz" -> severity = 2
            table -> severity = 1
            otherwise -> severity = 0
        """
        nbodies = self.scene_model.nbody
        body_names = [mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, bid) for bid in np.arange(nbodies)]
        severity_table = np.zeros(nbodies)
        for i, name in enumerate(body_names):
            if "obstacle_hard" in name:
                severity_table[i] = 10
            elif "obstacle_soft" in name:
                severity_table[i] = 2
            elif "table" in name:
                severity_table[i] = 1
        return severity_table


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", default="scene_level2.xml", type=str)
    args = parser.parse_args()

    # Update these paths to your XML files
    XML_PATH = "/home/aaron/workspace/FailBench/"
    scene_xml_path = XML_PATH + "franka_emika_panda/scene_level2.xml"
    scene_xml_path = XML_PATH + "franka_emika_panda/"+args.f
    robot_xml_path = XML_PATH + "franka_emika_panda/panda.xml"

    collected_traj_folder = "scene2_trajs"
    # if "2" in args.f:
    #     collected_traj_folder = "scene2_trajs"
    # elif "3" in args.f:
    #     collected_traj_folder = "scene3_trajs"
    # else:
    #     collected_traj_folder = "scene1_trajs"

    print("PLAYING ALL SAVED TRAJECTORIES IN FOLDER "+collected_traj_folder)

    repo_path = os.getcwd()
    collected_traj_folder = os.path.join(repo_path, "collected_trajs", collected_traj_folder)
    saved_trajs = [os.path.join(collected_traj_folder, f) for f in os.listdir(collected_traj_folder)]

    try:
        # Create pick and place demo
        demo = PandaPickAndPlace_L2(scene_xml_path, robot_xml_path, seed=15)
        # demo = PandaPickAndPlace(scene_xml_path, robot_xml_path, seed=15)

        # demo.run_demo_full_physics()

        '''
        Baseline RRT good seed: 15, 29, 49
        '''
        
        # demo.rrt_planner.set_aggressive_avoidance_parameters()

        # Run tests
        # demo.landscape_test()

        # Run the demo
        # demo.generate_multiple_trajectories()


        # demo.exp_traj_manager.report()
        demo.play_saved_trajectories(saved_trajs)
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please update the XML file paths in main() function")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all your modules are in the correct paths")


if __name__ == "__main__":
    print("🤖 Panda Pick and Place Demo")
    print("=" * 40)
    print("Using your existing planning modules:")
    print("  ✅ RRTplanner.py (JointSpaceRRT)")
    print("  ✅ collision_checker.py (CollisionChecker)")
    print("  ✅ inverse_kinematics.py (IKSolver)")
    print("  ✅ abstract_planner.py (AbstractRRTPlanner)")
    print()
    
    main()