#!/usr/bin/env python3
"""
Panda Pick and Place Demo
Uses your existing RRT planner to pick up object3 and place it in a new location.
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from typing import Optional

# Import YOUR existing modules
from planner.algorithms.RRTplanner import JointSpaceRRT
from planner.collision.collision_checker import CollisionChecker  
from planner.kinematics.inverse_kinematics import IKSolver, EndEffectorTarget, IKResult
from planner.algorithms.abstract_planner import PlanningSpace


class PandaPickAndPlace:
    """
    Pick and place demo using your existing planning modules.
    """
    
    def __init__(self, scene_xml_path: str, robot_xml_path: str):
        """Initialize with your existing modules."""
        
        print("🤖 Initializing Panda Pick and Place Demo")
        print("=" * 50)
        
        # Load models
        print("🔍 Loading MuJoCo models...")
        self.scene_model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        self.scene_data = mujoco.MjData(self.scene_model)
        
        print(f"✅ Models loaded:")
        print(f"   Scene: {self.scene_model.ngeom} geoms, {self.scene_model.njnt} joints")
        print(f"   Robot: {self.robot_model.ngeom} geoms, {self.robot_model.njnt} joints")
        
        # Initialize your planning modules
        self.ik_solver = IKSolver(self.robot_model)
        self.collision_checker = CollisionChecker(self.scene_model, self.robot_model)
        self.rrt_planner = JointSpaceRRT(
            scene_model=self.scene_model,
            robot_model=self.robot_model,
            ik_solver=self.ik_solver,
            collision_threshold=0.03,
            planning_space=PlanningSpace.JOINT_SPACE,
            step_size=0.1,
            goal_bias=0.1
        )
        
        self.robot_dof = self.robot_model.njnt
        self.current_path = None
        self.current_viewer = None
        
        # Set initial pose
        self._set_home_position()
        
        print(f"✅ Pick and place demo ready! Robot DOF: {self.robot_dof}")
    
    def _set_home_position(self):
        """Set robot to home position."""
        home_config = np.zeros(self.robot_dof)
        self.scene_data.qpos[:self.robot_dof] = home_config
        mujoco.mj_forward(self.scene_model, self.scene_data)
    
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
    
    def plan_to_ee_pose(self, target_pos: np.ndarray) -> bool:
        """Plan to end-effector pose using your IK + RRT."""
        
        print(f"🎯 Planning to EE Position: {target_pos}")
        
        # Use YOUR IK solver to find goal configurations
        target = EndEffectorTarget(
            position=target_pos,
            frame_name="end_effector",
            frame_type="site"
        )
        
        # Try multiple IK seeds
        goal_configs = []
        for attempt in range(10):
            seed = self.ik_solver.get_random_valid_config()
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
            return False
        
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
    
    def execute_path(self, speed: float = 1.0):
        """Execute planned path with viewer synchronization."""
        if self.current_path is None:
            print("No path to execute")
            return
        
        print(f"🚀 Executing path: {len(self.current_path)} waypoints")
        
        for i, config in enumerate(self.current_path):
            print(f" Waypoint {i+1}/{len(self.current_path)}")
            
            # Set configuration
            self.scene_data.qpos[:self.robot_dof] = config
            mujoco.mj_forward(self.scene_model, self.scene_data)
            
            # Sync viewer
            if self.current_viewer is not None:
                self.current_viewer.sync()
            
            time.sleep(0.2 / speed)
        
        print("✅ Path execution completed")
    
    def open_gripper(self):
        """Open the Panda gripper."""
        print("🤏 Opening gripper...")
        self.scene_data.qpos[7] = 0.04  # finger_joint1
        self.scene_data.qpos[8] = 0.04  # finger_joint2
        mujoco.mj_forward(self.scene_model, self.scene_data)
        
        if self.current_viewer is not None:
            self.current_viewer.sync()
        time.sleep(1.0)
    
    def close_gripper(self):
        """Close the Panda gripper."""
        print("✋ Closing gripper...")
        self.scene_data.qpos[7] = 0.0   # finger_joint1
        self.scene_data.qpos[8] = 0.0   # finger_joint2
        mujoco.mj_forward(self.scene_model, self.scene_data)
        
        if self.current_viewer is not None:
            self.current_viewer.sync()
        time.sleep(1.0)
    
    def pick_and_place_object3(self) -> bool:
        """Complete pick and place demo for object3."""
        
        print("\n🎯 Pick and Place Demo: Object3")
        print("=" * 40)
        
        # Step 1: Get object3 position
        obj_pos = self.get_object_position("object3")
        if obj_pos is None:
            return False
        
        # Step 2: Calculate approach positions
        approach_height = 0.15  # 15cm above object
        grasp_height = 0.02     # 2cm above table surface
        place_offset = np.array([0.3, 0.0, 0.0])  # 30cm to the side
        
        approach_pos = obj_pos.copy()
        approach_pos[2] = obj_pos[2] + approach_height
        
        grasp_pos = obj_pos.copy()
        grasp_pos[2] = obj_pos[2] + grasp_height
        
        place_approach_pos = obj_pos + place_offset
        place_approach_pos[2] = obj_pos[2] + approach_height
        
        place_pos = obj_pos + place_offset
        place_pos[2] = obj_pos[2] + grasp_height
        
        print(f"📋 Pick and Place Plan:")
        print(f"  Object3 at: {obj_pos}")
        print(f"  Approach:   {approach_pos}")
        print(f"  Grasp:      {grasp_pos}")
        print(f"  Place approach: {place_approach_pos}")
        print(f"  Place:      {place_pos}")
        
        # Step 3: Open gripper
        input("\nPress Enter to open gripper...")
        self.open_gripper()
        
        # Step 4: Move to approach position
        input("Press Enter to move to approach position...")
        if not self.plan_to_ee_pose(approach_pos):
            print("❌ Failed to plan to approach position")
            return False
        self.execute_path(speed=1.0)
        
        # Step 5: Move down to grasp position
        input("Press Enter to move to grasp position...")
        if not self.plan_to_ee_pose(grasp_pos):
            print("❌ Failed to plan to grasp position")
            return False
        self.execute_path(speed=0.5)
        
        # Step 6: Close gripper to grasp object
        input("Press Enter to close gripper and grasp object...")
        self.close_gripper()
        
        # Step 7: Lift object (back to approach height)
        input("Press Enter to lift object...")
        if not self.plan_to_ee_pose(approach_pos):
            print("❌ Failed to plan lift motion")
            return False
        self.execute_path(speed=0.5)
        
        # Step 8: Move to place approach position
        input("Press Enter to move to place location...")
        if not self.plan_to_ee_pose(place_approach_pos):
            print("❌ Failed to plan to place approach")
            return False
        self.execute_path(speed=1.0)
        
        # Step 9: Lower to place position
        input("Press Enter to lower object...")
        if not self.plan_to_ee_pose(place_pos):
            print("❌ Failed to plan to place position")
            return False
        self.execute_path(speed=0.5)
        
        # Step 10: Open gripper to release object
        input("Press Enter to release object...")
        self.open_gripper()
        
        # Step 11: Retreat
        input("Press Enter to retreat...")
        if not self.plan_to_ee_pose(place_approach_pos):
            print("❌ Failed to plan retreat motion")
            return False
        self.execute_path(speed=1.0)
        
        # Step 12: Return home
        input("Press Enter to return home...")
        home_config = np.zeros(self.robot_dof)
        self.current_path = [self.get_current_config(), home_config]
        self.execute_path(speed=1.0)
        
        print("\n🎉 Pick and Place Completed Successfully!")
        return True
    
    def run_demo(self):
        """Run the pick and place demo with MuJoCo viewer."""
        
        print("\n🎬 Starting Pick and Place Demo")
        print("=" * 40)
        
        with mujoco.viewer.launch_passive(self.scene_model, self.scene_data) as viewer:
            self.current_viewer = viewer
            
            print("\n📋 This demo will:")
            print("1. Locate object3 in the scene")
            print("2. Plan approach trajectory using your RRT")
            print("3. Grasp the object with gripper control")
            print("4. Move it to a new location")
            print("5. Release and return home")
            print("\nEach step requires Enter to proceed...")
            
            input("\nPress Enter to start pick and place demo...")
            
            # Run pick and place
            success = self.pick_and_place_object3()
            
            if success:
                print("\n✅ Pick and place demo completed successfully!")
            else:
                print("\n❌ Pick and place demo failed")
            
            self.current_viewer = None
            input("\nPress Enter to exit...")


def main():
    """Main function."""
    
    # Update these paths to your XML files
    scene_xml_path = "/home/aaron/workspace/mujoco-arena/franka_emika_panda/scene.xml"
    robot_xml_path = "/home/aaron/workspace/mujoco-arena/franka_emika_panda/panda.xml"
    
    try:
        # Create pick and place demo
        demo = PandaPickAndPlace(scene_xml_path, robot_xml_path)
        
        # Run the demo
        demo.run_demo()
    
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