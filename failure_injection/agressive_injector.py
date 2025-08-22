import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

class AggressiveFailureInjector:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Store original values for restoration
        self.original_gains = model.actuator_gainprm.copy()
        self.original_biastype = model.actuator_biastype.copy()
        self.original_gaintype = model.actuator_gaintype.copy()
        self.original_stiffness = model.jnt_stiffness.copy()
        self.original_damping = model.dof_damping.copy()
        self.original_ranges = model.jnt_range.copy()
        self.original_frictionloss = model.dof_frictionloss.copy()
        
        # Track failed joints
        self.failed_joints = set()
        
    def turn_off_joint_aggressive(self, joint_name):
        """
        AGGRESSIVELY turn off a joint - removes ALL control authority
        """
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        
        print(f"🔥 AGGRESSIVELY FAILING {joint_name}...")
        
        # Method 1: Zero ALL actuator parameters
        for actuator_id in range(self.model.nu):
            if self.model.actuator_trnid[actuator_id, 0] == joint_id:
                print(f"   Disabling actuator {actuator_id}")
                
                # Zero ALL gains
                self.model.actuator_gainprm[actuator_id, :] = 0.0
                
                # Zero control signal
                self.data.ctrl[actuator_id] = 0.0
                
                # Change actuator type to completely passive
                self.model.actuator_gaintype[actuator_id] = 0  # No gain processing <- incorrect. gaintype = 0 means fixed gain. there is some process, gain*ctrl, but the gain is fixed.
                self.model.actuator_biastype[actuator_id] = 0  # No bias processing 
        
        # Method 2: Remove joint stiffness (no internal springs) and damping
        self.model.jnt_stiffness[joint_id] = 0.0
        self.model.dof_damping[joint_id] = 0.0  # Remove damping too - let it swing freely
        
        # Method 3: Zero friction so joint moves freely
        self.model.dof_frictionloss[joint_id] = 0.0
        
        # Method 4: Set very loose joint limits -> quadrupling the range
        original_range = self.model.jnt_range[joint_id, :].copy()
        range_center = np.mean(original_range)
        range_span = original_range[1] - original_range[0]
        self.model.jnt_range[joint_id, 0] = range_center - range_span * 2  # Very loose 
        self.model.jnt_range[joint_id, 1] = range_center + range_span * 2
        
        # Method 5: Apply small downward bias force to encourage falling
        self.data.qfrc_applied[joint_id] = -0.1  # Small downward force
        # is this necessary? 

        self.failed_joints.add(joint_name)
        print(f"   ✅ {joint_name} completely disabled - should fall freely!")
    
    def turn_off_multiple_joints(self, joint_names):
        """Turn off multiple joints aggressively"""
        for joint_name in joint_names:
            self.turn_off_joint_aggressive(joint_name)
    
    def turn_off_all_joints(self):
        """Turn off ALL joints - complete system failure"""
        print("💀 COMPLETE SYSTEM SHUTDOWN - TURNING OFF ALL JOINTS")
        all_joints = []
        for i in range(1, 8):  # Panda has joints 1-7
            joint_name = f"joint{i}"
            try:
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                all_joints.append(joint_name)
            except:
                pass
        
        self.turn_off_multiple_joints(all_joints)
        print("💀 ALL JOINTS DISABLED - ARM SHOULD DROP LIKE A DEAD WEIGHT!")
    
    def restore_joint(self, joint_name):
        """Restore joint to original functionality"""
        if joint_name not in self.failed_joints:
            print(f"⚠️  {joint_name} is not currently failed")
            return
            
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        
        print(f"🔧 Restoring {joint_name}...")
        
        # Restore original parameters
        self.model.jnt_stiffness[joint_id] = self.original_stiffness[joint_id]
        self.model.dof_damping[joint_id] = self.original_damping[joint_id]
        self.model.dof_frictionloss[joint_id] = self.original_frictionloss[joint_id]
        self.model.jnt_range[joint_id, :] = self.original_ranges[joint_id, :]
        
        # Remove applied forces
        self.data.qfrc_applied[joint_id] = 0.0
        # this may not be desirable as it may interfere in planning/whatever the robot was doing before the failure happened

        # Restore actuator parameters
        for actuator_id in range(self.model.nu):
            if self.model.actuator_trnid[actuator_id, 0] == joint_id:
                self.model.actuator_gainprm[actuator_id, :] = self.original_gains[actuator_id, :]
                self.model.actuator_gaintype[actuator_id] = self.original_gaintype[actuator_id]
                self.model.actuator_biastype[actuator_id] = self.original_biastype[actuator_id]
        
        self.failed_joints.remove(joint_name)
        print(f"   ✅ {joint_name} restored")
    
    def restore_all_joints(self):
        """Restore all failed joints"""
        failed_list = list(self.failed_joints)
        for joint_name in failed_list:
            self.restore_joint(joint_name)
        print("🎉 ALL JOINTS RESTORED!")

def test_guaranteed_drop():
    """
    Test that GUARANTEES the arm will drop
    """
    robot_xml_path = "/home/aaron/workspace/mujoco-arena/franka_emika_panda/panda.xml"
    model = mujoco.MjModel.from_xml_path(robot_xml_path)
    data = mujoco.MjData(model)
    
    # FORCE enable gravity with strong setting
    model.opt.gravity[:] = [0, 0, -9.81]
    
    # Reduce any existing damping that might slow falling
    model.dof_damping[:] *= 0.1  # Reduce global damping
    
    print(f"🌍 Gravity: {model.opt.gravity}")
    print(f"🎯 Reduced global damping for faster falling")
    
    injector = AggressiveFailureInjector(model, data)
    
    # Extended pose that WILL fall when failed
    target_pose = {
        "joint1": 0.0,       # Base: straight ahead
        "joint2": 0.0,       # Shoulder: horizontal (90 degrees from vertical)
        "joint3": 0.0,       # Arm rotation: neutral
        "joint4": -1.57,     # Elbow: straight out (90 degrees)
        "joint5": 0.0,       # Forearm rotation: neutral
        "joint6": 0.0,       # Wrist: straight
        "joint7": 0.0        # Hand: neutral
    }
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        phase = "moving_to_pose"
        pose_reached = False
        failure_triggered = False
        
        print("\n" + "="*60)
        print("GUARANTEED ARM DROP TEST")
        print("="*60)
        print("🎯 Step 1: Moving to extended horizontal pose...")
        print("🎯 Step 2: Will aggressively fail shoulder joint")
        print("🎯 Step 3: Arm WILL drop - guaranteed!")
        print("\nControls:")
        print("  [F] = Fail shoulder joint (after pose reached)")
        print("  [A] = Fail ALL joints immediately")
        print("  [R] = Restore all joints")
        print("  [Q] = Quit")
        print("="*60)
        
        def keyboard_input():
            nonlocal failure_triggered, phase, pose_reached
            
            while True:
                try:
                    key = input("Press [F]=fail shoulder, [A]=fail all, [R]=restore, [Q]=quit: ").strip().lower()
                    
                    if key == 'f':
                        if pose_reached and not failure_triggered:
                            print("\n💥 FAILING SHOULDER JOINT - WATCH IT DROP!")
                            injector.turn_off_joint_aggressive("joint2")
                            failure_triggered = True
                        elif not pose_reached:
                            print("⚠️  Wait for pose to be reached first")
                        else:
                            print("⚠️  Already failed - press R to restore first")
                    
                    elif key == 'a':
                        print("\n💀 FAILING ALL JOINTS - COMPLETE SYSTEM DEATH!")
                        injector.turn_off_all_joints()
                        failure_triggered = True
                    
                    elif key == 'r':
                        print("\n🔧 RESTORING ALL JOINTS")
                        injector.restore_all_joints()
                        failure_triggered = False
                        # Don't reset pose - let it stay in current position
                    
                    elif key == 'q':
                        break
                        
                except (EOFError, KeyboardInterrupt):
                    break
        
        # Start keyboard input thread
        input_thread = threading.Thread(target=keyboard_input, daemon=True)
        input_thread.start()
        
        while viewer.is_running():
            step_start = time.time()
            current_time = data.time
            
            if phase == "moving_to_pose":
                # Move to extended pose with strong control
                pose_error = 0
                
                for joint_name, target_pos in target_pose.items():
                    if joint_name not in injector.failed_joints:  # Only control non-failed joints
                        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                        current_pos = data.qpos[joint_id]
                        error = abs(target_pos - current_pos)
                        pose_error += error
                        
                        # Find actuator and apply strong control
                        for actuator_id in range(model.nu):
                            if model.actuator_trnid[actuator_id, 0] == joint_id:
                                data.ctrl[actuator_id] = target_pos
                                break
                
                # Check if pose reached
                if pose_error < 0.3 and current_time > 3.0:
                    if not pose_reached:
                        print("\n✅ EXTENDED POSE REACHED!")
                        print("🎯 Arm is now extended horizontally")
                        print("💥 Press F to fail the shoulder joint and watch it DROP!")
                        print("⚡ Press A to fail ALL joints for complete collapse!")
                        pose_reached = True
                    phase = "ready_for_failure"
            
            elif phase == "ready_for_failure":
                # Hold pose only for non-failed joints
                if not failure_triggered:
                    for joint_name, target_pos in target_pose.items():
                        if joint_name not in injector.failed_joints:
                            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                            
                            for actuator_id in range(model.nu):
                                if model.actuator_trnid[actuator_id, 0] == joint_id:
                                    data.ctrl[actuator_id] = target_pos
                                    break
                
                # Status updates
                if int(current_time * 2) % 10 == 0:  # Every 5 seconds
                    if failure_triggered:
                        failed_count = len(injector.failed_joints)
                        print(f"💀 {failed_count} joints failed - observing gravity effects...")
                        print(f"📍 Current positions: {data.qpos[:7].round(3)}")
                    else:
                        print("📍 Holding extended pose - ready for failure test")
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Maintain real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def test_immediate_drop():
    """
    Immediate drop test - no waiting, just instant failure
    """
    robot_xml_path = "/home/aaron/workspace/mujoco-arena/franka_emika_panda/panda.xml"
    model = mujoco.MjModel.from_xml_path(robot_xml_path)
    data = mujoco.MjData(model)
    
    # Maximum gravity and minimum damping
    model.opt.gravity[:] = [0, 0, -9.81]
    model.dof_damping[:] = 0.01  # Very low damping
    
    injector = AggressiveFailureInjector(model, data)
    
    print("\n💀 IMMEDIATE COMPLETE FAILURE TEST")
    print("🎯 All joints will be disabled immediately")
    print("🎯 Robot should drop like a dead weight")
    print("="*50)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set extended pose immediately
        data.qpos[0] = 0.0      # joint1
        data.qpos[1] = 0.0      # joint2 - horizontal
        data.qpos[2] = 0.0      # joint3
        data.qpos[3] = -1.57    # joint4 - extended
        data.qpos[4] = 0.0      # joint5
        data.qpos[5] = 0.0      # joint6
        data.qpos[6] = 0.0      # joint7
        
        mujoco.mj_forward(model, data)  # Update kinematics
        
        print("📍 Initial extended pose set")
        print("⏰ Waiting 2 seconds, then COMPLETE FAILURE...")
        
        step_count = 0
        failure_time = 200  # 2 seconds at 100Hz
        
        while viewer.is_running():
            if step_count == failure_time:
                print("💀 COMPLETE SYSTEM FAILURE - ALL JOINTS OFF!")
                injector.turn_off_all_joints()
                print("👀 Watch the arm drop!")
            
            # No control after failure
            if step_count < failure_time:
                # Hold extended pose briefly
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0      # horizontal
                data.ctrl[2] = 0.0
                data.ctrl[3] = -1.57    # extended
                data.ctrl[4] = 0.0
                data.ctrl[5] = 0.0
                data.ctrl[6] = 0.0
            
            # Monitor positions
            if step_count % 100 == 0:  # Every second
                positions = data.qpos[:7]
                print(f"t={step_count/100:.1f}s: positions = {positions.round(3)}")
            
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
            
            step_count += 1

if __name__ == "__main__":
    print("Choose aggressive failure test:")
    print("1. Guaranteed drop test (interactive)")
    print("2. Immediate complete failure (automatic)")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        test_guaranteed_drop()
    elif choice == "2":
        test_immediate_drop()
    else:
        print("Running guaranteed drop test...")
        test_guaranteed_drop()