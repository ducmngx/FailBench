#!/usr/bin/env python3
"""
Complete Example Using Your Existing Modules
- Uses your RRTplanner.py
- Uses your collision_checker.py  
- Uses your inverse_kinematics.py
- Uses your abstract_planner.py
- Uses your module.py (SimpleXMLSeparator)

Just integrates them with MuJoCo visualization.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import threading
from typing import List, Optional
from tabulate import tabulate

# Import YOUR existing modules (adjust paths as needed)
from planner.algorithms.RRTplanner import JointSpaceRRT
from planner.collision.collision_checker import CollisionChecker  
from planner.kinematics.inverse_kinematics import IKSolver, EndEffectorTarget, IKResult
from planner.algorithms.abstract_planner import PlanningSpace
from failure_injection.collision_estimation import CollisionEstimator


class PandaPlanningDemo:
    """
    Demo using ALL your existing modules with MuJoCo visualization.
    No changes to your code - just integration!
    """
    
    def __init__(self, scene_xml_path: str, robot_xml_path: str):
        """Initialize with your existing modules."""
        
        print("🚀 Initializing Panda Planning Demo")
        print("=" * 50)
        
        # Load models
        print("🔍 Loading MuJoCo models...")
        print(f"   Scene XML: {scene_xml_path}")
        print(f"   Robot XML: {robot_xml_path}")
        
        # Load MuJoCo models (adjust paths as needed)
        self.scene_model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        self.scene_data = mujoco.MjData(self.scene_model)
        
        print(f"✅ Models loaded:")
        print(f"   Scene: {self.scene_model.ngeom} geoms, {self.scene_model.njnt} joints")
        print(f"   Robot: {self.robot_model.ngeom} geoms, {self.robot_model.njnt} joints")
        
        # Create YOUR components (using your existing code)
        print("\n🔧 Initializing your modules...")
        
        # 1. Your IK Solver
        self.ik_solver = IKSolver(self.robot_model)
        print("✅ IKSolver (inverse_kinematics.py) initialized")
        
        # 2. Your Collision Checker  
        self.collision_checker = CollisionChecker(self.scene_model, self.robot_model)
        print("✅ CollisionChecker (collision_checker.py) initialized")
        
        # 3. Your RRT Planner (using your AbstractRRTPlanner)
        self.rrt_planner = JointSpaceRRT(
            scene_model=self.scene_model,
            robot_model=self.robot_model,
            ik_solver=self.ik_solver,
            collision_threshold=0.03,
            planning_space=PlanningSpace.JOINT_SPACE,
            step_size=0.1,
            goal_bias=0.1
        )
        print("✅ JointSpaceRRT (RRTplanner.py) initialized")
        
        # Robot configuration
        self.robot_dof = self.robot_model.njnt
        self.current_path = None
        self.executing = False
        
        # Set initial pose
        self._set_home_position()

        # Collision Estimation upon total failure
        self.collision_estimator = CollisionEstimator(self.scene_model, self.scene_data)
        
        print(f"\n🎯 Demo ready! Robot DOF: {self.robot_dof}")
    
    def _set_home_position(self):
        """Set robot to home position."""
        home_config = np.zeros(self.robot_dof)
        # Use your collision checker's method to set configuration
        self.collision_checker.set_robot_configuration_direct(home_config)
        
        # Copy to scene data for visualization
        self.scene_data.qpos[:self.robot_dof] = home_config
        mujoco.mj_forward(self.scene_model, self.scene_data)
    
    def get_current_config(self) -> np.ndarray:
        """Get current robot configuration."""
        return self.scene_data.qpos[:self.robot_dof].copy()
    
    def set_robot_config(self, config: np.ndarray):
        """Set robot configuration for visualization."""
        self.scene_data.qpos[:self.robot_dof] = config
        mujoco.mj_forward(self.scene_model, self.scene_data)

    def execute_robot_config(self, config: np.ndarray):
        """Set robot configuration for visualization."""
        # self.scene_data.qpos[:self.robot_dof] = config
        print(f"Setting qpos to: {config[:9]}...")  # Print first 3 values
        print(f"Before: {self.scene_data.qpos[:9]}")
        self.scene_data.qpos[:self.robot_dof] = config
        print(f"After:  {self.scene_data.qpos[:9]}")
        mujoco.mj_forward(self.scene_model, self.scene_data)  # Use mj_forward instead of mj_step
        
        # CRITICAL: Sync viewer if available
        if hasattr(self, 'current_viewer') and self.current_viewer is not None:
            self.current_viewer.sync()
        
        time.sleep(0.5)  # Longer sleep to see movement

    def test_basic_movement(self):
        print("=== TESTING BASIC MOVEMENT ===")
        
        # Save current state
        original_qpos = self.scene_data.qpos.copy()
        print(f"Original qpos[0:9]: {original_qpos[:9]}")
        
        # Test: Set joint 1 to a large angle
        print("\n1. Setting joint1 to 1.5 radians...")
        self.scene_data.qpos[0] = 1.5
        mujoco.mj_forward(self.scene_model, self.scene_data)
        print(f"After setting: {self.scene_data.qpos[:9]}")
        input("Press Enter to continue...")
        
        # Test: Reset and try joint 2
        print("\n2. Resetting and trying joint2...")
        self.scene_data.qpos[:] = original_qpos
        self.scene_data.qpos[1] = -1.0
        mujoco.mj_forward(self.scene_model, self.scene_data)
        print(f"After setting: {self.scene_data.qpos[:9]}")
        input("Press Enter to continue...")
        
        # Reset
        self.scene_data.qpos[:] = original_qpos
        mujoco.mj_forward(self.scene_model, self.scene_data)

    def check_viewer(self):
        print("=== CHECKING VIEWER ===")
        print(f"Do you have a viewer? {hasattr(self, 'viewer')}")
        if hasattr(self, 'viewer'):
            print(f"Viewer type: {type(self.viewer)}")
            print(f"Viewer is None? {self.viewer is None}")

    def test_with_viewer_sync(self):
        print("=== TESTING WITH VIEWER SYNC ===")
        
        # Move joint and force viewer update
        self.scene_data.qpos[0] = 1.0
        mujoco.mj_forward(self.scene_model, self.scene_data)
        
        # Try different ways to update viewer
        if hasattr(self, 'viewer') and self.viewer is not None:
            if hasattr(self.viewer, 'sync'):
                self.viewer.sync()
            if hasattr(self.viewer, 'render'):
                self.viewer.render()
            if hasattr(self.viewer, 'update'):
                self.viewer.update()
        
        # time.sleep(0.2)

    def debug_model_state(self):
        print("=== MODEL STATE DEBUG ===")
        print(f"Model nq (total qpos): {self.scene_model.nq}")
        print(f"Model nv (total qvel): {self.scene_model.nv}")
        print(f"Data qpos shape: {self.scene_data.qpos.shape}")
        print(f"Data qvel shape: {self.scene_data.qvel.shape}")
        
        # Check if data matches model
        print(f"Data qpos size matches model? {len(self.scene_data.qpos) == self.scene_model.nq}")

    def check_joint_limits(self):
        print("=== JOINT LIMITS DEBUG ===")
        for i in range(7):  # First 7 robot joints
            joint_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_range = self.scene_model.jnt_range[i]
            limited = self.scene_model.jnt_limited[i]
            print(f"Joint {i} ({joint_name}): limited={limited}, range={joint_range}")

    def quick_diagnostic(self):
        print("=== QUICK DIAGNOSTIC ===")
        print(f"Model file: {getattr(self.scene_model, 'file', 'unknown')}")
        print(f"Data timestep: {self.scene_data.time}")
        print(f"First 9 qpos: {self.scene_data.qpos[:9]}")

        # Try extreme movement
        old_qpos = self.scene_data.qpos[0]
        self.scene_data.qpos[0] = 2.0  # Very large angle
        mujoco.mj_forward(self.scene_model, self.scene_data)
        print(f"After setting joint 0 to 2.0: {self.scene_data.qpos[0]}")

        # Check if it was clamped by limits
        if abs(self.scene_data.qpos[0] - 2.0) > 0.01:
            print("⚠️ Joint was clamped - check joint limits!")
        else:
            print("✅ Joint value was set correctly")
    
    def test_your_modules(self):
        """Test all your modules individually."""
        
        print("\n🧪 Testing Your Modules")
        print("=" * 30)
        
        # Test 1: IK Solver
        print("1. Testing IKSolver...")
        random_config = self.ik_solver.get_random_valid_config()
        if random_config is not None:
            print(f"   ✅ Generated random config: {random_config[:3]}...")
            is_valid = self.ik_solver.is_config_valid(random_config)
            print(f"   ✅ Config validation: {is_valid}")
        else:
            print("   ❌ Failed to generate random config")
        
        # Test 2: Collision Checker
        print("\n2. Testing CollisionChecker...")
        test_config = np.zeros(self.robot_dof)
        
        # Test the FIXED version with robot_config parameter
        has_collision = self.collision_checker.check_collisions(
            robot_config=test_config,  # ← Using your fixed method
            threshold=0.03
        )
        print(f"   ✅ Home position collision: {has_collision}")
        
        # Test collision checker's XML separator
        robot_geoms = self.collision_checker._xml_extractor.get_robot_geoms(collision_only=True)
        env_geoms = self.collision_checker._xml_extractor.get_environment_geoms(collision_only=True)
        print(f"   ✅ Detected {len(robot_geoms)} robot geoms, {len(env_geoms)} env geoms")
        
        # Test 3: RRT Planner  
        print("\n3. Testing RRTPlanner...")
        
        # Test validation (this uses your fixed is_valid_config)
        start_config = np.zeros(self.robot_dof)
        is_start_valid = self.rrt_planner.is_valid_config(start_config)
        print(f"   ✅ Start config validation: {is_start_valid}")
        
        # Test sampling
        sample = self.rrt_planner.sample_random_config()
        if sample is not None:
            print(f"   ✅ Random sampling: {sample[:3]}...")
        else:
            print("   ⚠️  Random sampling failed (might be normal)")
        
        # Test distance metric
        if sample is not None:
            dist = self.rrt_planner.distance(start_config, sample)
            print(f"   ✅ Distance calculation: {dist:.3f}")
        
        print("\n✅ All module tests completed!")
    
    def plan_to_random_config(self) -> bool:
        """Plan to a random valid configuration using your RRT."""
        
        print("\n🎯 Planning to Random Configuration")
        print("-" * 35)
        
        # Get start configuration
        start_config = self.get_current_config()
        print(f"Start: {start_config}")
        
        # Generate random goal
        goal_config = None
        for attempt in range(50):
            candidate = self.ik_solver.get_random_valid_config()
            if candidate is not None and self.rrt_planner.is_valid_config(candidate):
                goal_config = candidate
                break
        
        if goal_config is None:
            print("❌ Failed to find valid random goal")
            return False
        
        print(f"Goal:  {goal_config}")
        
        # Plan using YOUR RRT planner
        print("Planning with your RRT...")
        start_time = time.time()
        
        path = self.rrt_planner.plan(
            start_config=start_config,
            goal_config=goal_config,
            frame_name="end_effector"  # For your abstract planner
        )
        
        planning_time = time.time() - start_time
        
        if path:
            self.current_path = path
            print(f"✅ Planning successful!")
            print(f"   Time: {planning_time:.3f}s")
            print(f"   Waypoints: {len(path)}")
            print(f"   Tree size: {len(self.rrt_planner.tree)}")
            print(f"   Start: {path[0][:3]}, Goal: {path[-1][:3]}")
            # print(f"   Path: {path}")
            return True
        else:
            print(f"❌ Planning failed after {planning_time:.3f}s")
            return False
    
    def plan_to_ee_pose(self, target_pos: np.ndarray) -> bool:
        """Plan to end-effector pose using your IK + RRT."""
        
        print(f"\n🎯 Planning to EE Position: {target_pos}")
        print("-" * 40)
        
        # Use YOUR IK solver to find goal configurations
        target = EndEffectorTarget(
            position=target_pos,
            frame_name="end_effector",  # Adjust to your robot's EE frame
            frame_type="site"
        )
        
        # Try multiple IK seeds using your IK solver
        goal_configs = []
        for attempt in range(10):
            seed = self.ik_solver.get_random_valid_config()
            if seed is None:
                continue
            
            solution, result = self.ik_solver.solve(target, seed)
            
            if result == IKResult.SUCCESS:
                # Check collision using your collision checker
                if not self.collision_checker.check_collisions(solution):
                    goal_configs.append(solution)
                    print(f"   Found IK solution {len(goal_configs)}")
                    if len(goal_configs) >= 3:
                        break
        
        if not goal_configs:
            print("❌ No valid IK solutions found")
            return False
        
        # Try RRT to each goal using YOUR RRT planner
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
                print(f"✅ Planning successful via IK solution {i+1}")
                print(f"   Waypoints: {len(path)}")
                return True
        
        print("❌ RRT failed to reach any IK solution")
        return False
    
    def visualize_bounding_spheres(self, viewer, geom_ids: np.ndarray, duration=10.0):
        """
        Starts a thread that visualizes the bounding spheres for duration seconds.
        """
        def _show_spheres_temporarily(viewer, geom_ids, model, data, duration=10.0):
            ngeom = 0
            for i in geom_ids:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[model.geom_rbound[i], 0, 0],
                    pos=data.geom_xpos[i],
                    mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 0.3],
                )
                ngeom += 1
            viewer.user_scn.ngeom = ngeom
            viewer.sync()
            time.sleep(duration)
            viewer.user_scn.ngeom = 0
            viewer.sync()

        # When C is pressed:
        threading.Thread(
            target=_show_spheres_temporarily,
            args=(viewer, geom_ids, self.scene_model, self.scene_data, duration),
            daemon=True
        ).start()


    def check_total_failure_collisions(self):
        print("=== CHECK TOTAL FAILURE COLLISIONS ===")
        robot_joint_names = [f"joint{i}" for i in range(1,8)]
        collision_pair_body_ids = self.collision_estimator.estimate_bodies_in_collision(robot_joint_names, remove_world_body=True)
        robot_body_names = [
            mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, bid)
            for bid in collision_pair_body_ids[:, 0]
        ]
        world_body_names = [
            mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, bid)
            for bid in collision_pair_body_ids[:, 1]
        ]
        table_data = [[rb, wb] for rb, wb in zip(robot_body_names, world_body_names)]
        print(tabulate(table_data, headers=["🤖 Robot Body Part", "🌍 Collides With"], tablefmt="fancy_grid"))
        print("\n✅ Total failure collision check complete.")

    # def execute_path(self, speed: float = 1.0):
    #     """Execute planned path with visualization."""
        
    #     if self.current_path is None:
    #         print("No path to execute")
    #         return
        
    #     print(f"\n🚀 Executing path: {len(self.current_path)} waypoints")
        
    #     self.executing = True
        
    #     def execute_thread():
    #         for i, config in enumerate(self.current_path):
    #             if not self.executing:
    #                 break
                
    #             print(f"   Waypoint {i+1}/{len(self.current_path)}")
    #             self.set_robot_config(config)
    #             time.sleep(0.2 / speed)  # Control speed
            
    #         self.executing = False
    #         print("✅ Execution completed")
        
    #     # Run in thread so it doesn't block
    #     exec_thread = threading.Thread(target=execute_thread)
    #     exec_thread.daemon = True
    #     exec_thread.start()
    
    def execute_path_with_viewer(self, speed: float = 1.0, viewer=None):
        """Execute planned path with viewer synchronization."""
        if self.current_path is None:
            print("No path to execute")
            return
        
        print(f"\n🚀 Executing path: {len(self.current_path)} waypoints")
        self.executing = True
        
        for i, config in enumerate(self.current_path):
            if not self.executing:
                break
            print(f" Waypoint {i+1}/{len(self.current_path)}")
            
            # Set configuration
            self.scene_data.qpos[:self.robot_dof] = config
            mujoco.mj_forward(self.scene_model, self.scene_data)
            
            # CRITICAL: Sync viewer to show changes
            if viewer is not None:
                viewer.sync()
            
            time.sleep(0.2 / speed)
        
        self.executing = False
        print("✅ Execution completed")
    
    def execute_path(self, speed: float = 1.0):
        if self.current_path is None:
            print("No path to execute")
            return
        
        print(f"\n🚀 Executing path: {len(self.current_path)} waypoints")
        self.executing = True
        
        for i, config in enumerate(self.current_path):
            if not self.executing:
                break
            print(f" Waypoint {i+1}/{len(self.current_path)}")
            self.execute_robot_config(config)
            time.sleep(0.2 / speed)
        
        self.executing = False
        print("✅ Execution completed")

    def stop_execution(self):
        """Stop path execution."""
        self.executing = False
        print("⏹️ Execution stopped")
    
    def run_complete_demo(self):
        """Run complete demo with MuJoCo viewer."""
        
        print("\n🎬 Starting Complete Demo")
        print("=" * 40)
        
        with mujoco.viewer.launch_passive(self.scene_model, self.scene_data) as viewer:
            self.current_viewer = viewer  # Store viewer reference
            
            print("\n📋 Demo Sequence:")
            print("1. Test all your modules")
            print("2. Plan to random configuration")  
            print("3. Execute path")
            print("4. Plan to EE pose")
            print("5. Execute path")
            print("\nPress Enter to continue through each step...")
            
            # Step 1: Test modules
            input("\nPress Enter to test your modules...")
            self.test_your_modules()
            
            # Step 2: Plan to random config
            input("\nPress Enter to plan to random configuration...")
            if self.plan_to_random_config():
                
                # Step 3: Execute with viewer sync
                input("Press Enter to execute planned path...")
                self.execute_path_with_viewer(speed=2.0, viewer=viewer)
            
            # Step 4: Plan to EE pose
            input("\nPress Enter to plan to EE pose...")
            target_positions = [
                np.array([0.5, 0.2, 0.3]),
                np.array([0.3, -0.2, 0.4]),
                np.array([0.6, 0.0, 0.2])
            ]
            
            for i, target_pos in enumerate(target_positions):
                print(f"\nTarget {i+1}: {target_pos}")
                
                if self.plan_to_ee_pose(target_pos):
                    input("Press Enter to execute...")
                    self.execute_path_with_viewer(speed=1.5, viewer=viewer)
                else:
                    print("Skipping due to planning failure")
                
                if i < len(target_positions) - 1:
                    input("Press Enter for next target...")
            
            self.current_viewer = None  # Clean up viewer reference
            print("\n🎉 Demo completed! Press Enter to exit...")
            input()
    
    def run_interactive_mode(self):
        """Run interactive mode with manual controls."""
        
        print("\n🎮 Interactive Mode")
        print("=" * 20)
        
        with mujoco.viewer.launch_passive(self.scene_model, self.scene_data) as viewer:
            self.current_viewer = viewer  # Store viewer reference
            
            print("\nCommands:")
            print("  h - Home position")
            print("  r - Plan to random config")
            print("  p - Plan to EE pose [0.5, 0.2, 0.3]")
            print("  e - Execute current path")
            print("  s - Stop execution")
            print("  t - Test modules")
            print("  d - Debug movement")
            print("  c - Estimate total failure collisions")
            print("  q - Quit")
            
            while viewer.is_running():
                try:
                    cmd = input("\nEnter command: ").strip().lower()
                    
                    if cmd == 'h':
                        print("Going to home position...")
                        self._set_home_position()
                        viewer.sync()  # Sync after movement
                    
                    elif cmd == 'r':
                        self.plan_to_random_config()
                    
                    elif cmd == 'p':
                        target = np.array([0.5, 0.2, 0.3])
                        self.plan_to_ee_pose(target)
                    
                    elif cmd == 'e':
                        self.execute_path_with_viewer(viewer=viewer)
                    
                    elif cmd == 's':
                        self.stop_execution()
                    
                    elif cmd == 't':
                        self.test_your_modules()
                    
                    elif cmd == 'd':
                        self.test_basic_movement()
                        self.check_viewer()
                        self.debug_model_state()
                        self.check_joint_limits()
                        self.quick_diagnostic()

                    elif cmd == 'c':
                        self.check_total_failure_collisions()
                        self.visualize_bounding_spheres(viewer, np.unique(self.collision_estimator.current_collision_pairs.reshape(-1)))
                        
                    elif cmd == 'q':
                        break
                    
                    else:
                        print("Unknown command")
                    
                    # Update viewer
                    mujoco.mj_step(self.scene_model, self.scene_data)
                    viewer.sync()
                    
                except KeyboardInterrupt:
                    break
            
            self.current_viewer = None  # Clean up viewer reference
        
        print("👋 Interactive mode ended")


def main():
    """Main function - update paths and run demo."""
    
    # 🔧 UPDATE THESE PATHS TO YOUR XML FILES
    # scene_xml_path = "path/to/your/scene.xml"      # Scene with Panda + environment
    # robot_xml_path = "path/to/your/panda.xml"      # Panda robot only
    scene_xml_path="/Users/saghani/Workspace/Research/GenAISim/franka_emika_panda/scene.xml"
    robot_xml_path="/Users/saghani/Workspace/Research/GenAISim/franka_emika_panda/panda.xml"
    
    try:
        # Create demo using your existing modules
        demo = PandaPlanningDemo(scene_xml_path, robot_xml_path)
        
        # Choose demo mode
        print("\n🎯 Choose Demo Mode:")
        print("1. Complete automated demo")
        print("2. Interactive manual mode")
        print("3. Just test modules")
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == '1':
            demo.run_complete_demo()
        elif choice == '2':
            demo.run_interactive_mode()
        elif choice == '3':
            demo.test_your_modules()
        else:
            print("Invalid choice, running complete demo...")
            demo.run_complete_demo()
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please update the XML file paths in main() function")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all your modules are in the correct paths")


if __name__ == "__main__":
    print("🚀 Complete Example Using Your Existing Modules")
    print("=" * 60)
    print("This demo uses:")
    print("  ✅ Your RRTplanner.py (JointSpaceRRT)")
    print("  ✅ Your collision_checker.py (CollisionChecker)")
    print("  ✅ Your inverse_kinematics.py (IKSolver)")
    print("  ✅ Your abstract_planner.py (AbstractRRTPlanner)")
    print("  ✅ Your module.py (SimpleXMLSeparator)")
    print()
    
    main()