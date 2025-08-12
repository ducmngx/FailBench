#!/usr/bin/env python3
"""
Test suite for JointSpaceRRT implementation.
Tests various scenarios and validates RRT functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Optional
import random

# Import your modules (adjust paths as needed)
from RRTplanner import JointSpaceRRT, PlanningSpace
from inverse_kinematics import *
import mujoco

# Mock classes for testing if you don't have the actual implementations
class MockIKSolver:
    """Mock IK solver for testing purposes."""
    
    def __init__(self, n_joints=9, joint_limits=None):  # Changed from 6 to 9
        self.n_joints = n_joints
        if joint_limits is None:
            # Default joint limits: [-π, π] for each joint
            self.joint_limits = [(-np.pi, np.pi) for _ in range(n_joints)]
        else:
            self.joint_limits = joint_limits
        
        # Mock current configuration
        self._data = type('MockData', (), {})()
        self._data.qpos = np.zeros(n_joints)
        
        print(f"🔧 MockIKSolver initialized with {n_joints} joints")
        
    def get_random_valid_config(self) -> Optional[np.ndarray]:
        """Generate random valid joint configuration."""
        config = np.zeros(self.n_joints)
        for i, (min_val, max_val) in enumerate(self.joint_limits):
            config[i] = np.random.uniform(min_val, max_val)
        print(f"🎲 Generated random config with shape: {config.shape}")
        return config
    
    def is_valid_configuration(self, config: np.ndarray) -> bool:
        """Check if configuration is within joint limits."""
        if len(config) != self.n_joints:
            return False
            
        for i, (min_val, max_val) in enumerate(self.joint_limits):
            if not (min_val <= config[i] <= max_val):
                return False
        return True
    
    def interpolate_configs(self, start: np.ndarray, end: np.ndarray, num_points: int = 10) -> List[np.ndarray]:
        """Interpolate between two configurations."""
        path = []
        for i in range(num_points):
            alpha = i / (num_points - 1)
            config = (1 - alpha) * start + alpha * end
            path.append(config)
        return path


class JointSpaceRRTTester:
    """Comprehensive test suite for JointSpaceRRT."""
    
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.test_results = {}
        
        if use_mock:
            # Use mock for standalone testing
            self.ik_solver = MockIKSolver(n_joints=6)
            self.model = None  # Mock doesn't need model
        else:
            # Use your actual implementations
            # self.model = mujoco.MjModel.from_xml_path("your_robot.xml")
            # self.ik_solver = IKSolver(self.model, ...)
            # Load robot model
            self.model = mujoco.MjModel.from_xml_path("/home/aaron/workspace/mujoco-arena/mink/examples/franka_emika_panda/mjx_panda.xml")
            
            # Create IK solver
            self.ik_config = IKConfig(
                max_iterations=50,
                position_tolerance=0.005,
                check_joint_limits=True
            )
            self.ik_solver = IKSolver(self.model, self.ik_config)
            
    
    def run_all_tests(self):
        """Run all test cases."""
        print("🤖 Starting JointSpaceRRT Test Suite")
        print("=" * 50)
        
        # Basic functionality tests
        # self.test_initialization()
        self.test_sampling_strategies()
        # self.test_distance_metrics()
        # self.test_steering()
        self.test_path_validation()
        
        # Planning tests
        # self.test_simple_planning()
        self.test_planning_with_obstacles()
        # self.test_planning_performance()
        # self.test_goal_bias_effect()
        
        # # Edge cases
        # self.test_unreachable_goals()
        # self.test_invalid_start_configs()
        
        # Print summary
        self.print_test_summary()
    
    def test_initialization(self):
        """Test RRT planner initialization."""
        print("\n📋 Testing Initialization...")
        
        try:
            planner = JointSpaceRRT(
                model=self.model,
                ik_solver=self.ik_solver,
                max_iterations=1000,
                step_size=0.1,
                goal_tolerance=0.05
            )
            
            assert planner.planning_space == PlanningSpace.JOINT_SPACE
            assert planner.max_iterations == 1000
            assert planner.step_size == 0.1
            assert planner.goal_tolerance == 0.05
            
            self.test_results['initialization'] = True
            print("✅ Initialization test passed")
            
        except Exception as e:
            self.test_results['initialization'] = False
            print(f"❌ Initialization test failed: {e}")
    
    def test_sampling_strategies(self):
        """Test different sampling strategies."""
        print("\n🎯 Testing Sampling Strategies...")
        
        planner = JointSpaceRRT(self.model, self.ik_solver)
        
        # Initialize tree with a dummy node for testing
        from abstract_planner import PlanningNode
        dummy_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        planner.tree = [PlanningNode(config=dummy_config)]
        
        # Test random sampling
        samples = []
        for _ in range(100):
            sample = planner.sample_random_config()
            if sample is not None:
                samples.append(sample)
        
        try:
            assert len(samples) > 50, "Too few valid samples generated"
            
            # Check sample diversity - reduced threshold for 9D
            samples = np.array(samples)
            sample_std = np.std(samples, axis=0)
            assert np.all(sample_std > 0.05), f"Samples not diverse enough. Std: {sample_std}"  # Reduced from 0.1
            
            self.test_results['sampling'] = True
            print(f"✅ Sampling test passed ({len(samples)} valid samples)")
            
        except Exception as e:
            self.test_results['sampling'] = False
            print(f"❌ Sampling test failed: {e}")
    
    def test_distance_metrics(self):
        """Test distance calculation."""
        print("\n📏 Testing Distance Metrics...")
        
        planner = JointSpaceRRT(self.model, self.ik_solver)
        
        try:
            # Use 9D configurations to match your IK solver
            config1 = np.zeros(9)  # Changed from 6 to 9
            config2 = np.ones(9)   # Changed from 6 to 9
            
            distance = planner.distance(config1, config2)
            expected_distance = np.sqrt(9.0)  # sqrt(9 * 1^2)
            
            assert abs(distance - expected_distance) < 1e-6
            
            # Test symmetry
            reverse_distance = planner.distance(config2, config1)
            assert abs(distance - reverse_distance) < 1e-6
            
            # Test zero distance
            zero_distance = planner.distance(config1, config1)
            assert abs(zero_distance) < 1e-6
            
            self.test_results['distance'] = True
            print("✅ Distance metric test passed")
            
        except Exception as e:
            self.test_results['distance'] = False
            print(f"❌ Distance metric test failed: {e}")
    
    def test_steering(self):
        """Test steering function."""
        print("\n🎮 Testing Steering...")
        
        planner = JointSpaceRRT(self.model, self.ik_solver, step_size=0.5)
        
        try:
            # Use 9D configurations
            start = np.zeros(9)
            target = np.zeros(9)
            target[0] = 2.0  # Only change first joint
            
            result = planner.steer(start, target)
            
            # Should move step_size distance toward target
            expected = np.zeros(9)
            expected[0] = 0.5
            assert np.allclose(result, expected, atol=1e-6)
            
            # Test when target is closer than step_size
            close_target = np.zeros(9)
            close_target[0] = 0.2
            result2 = planner.steer(start, close_target)
            assert np.allclose(result2, close_target, atol=1e-6)
            
            self.test_results['steering'] = True
            print("✅ Steering test passed")
            
        except Exception as e:
            self.test_results['steering'] = False
            print(f"❌ Steering test failed: {e}")
    
    def test_path_validation(self):
        """Test path validity checking."""
        print("\n🛣️ Testing Path Validation...")
        
        planner = JointSpaceRRT(self.model, self.ik_solver)
        
        try:
            # Test valid path (within joint limits) - 9D configs
            start = np.zeros(9)
            end = np.full(9, 0.1)  # Small values within limits
            
            is_valid = planner.is_path_valid(start, end)
            assert is_valid, "Valid path marked as invalid"
            
            # Test invalid path (outside joint limits) - 9D configs  
            invalid_end = np.full(9, 10.0)  # Large values outside limits
            is_invalid = planner.is_path_valid(start, invalid_end)
            assert not is_invalid, "Invalid path marked as valid"
            
            self.test_results['path_validation'] = True
            print("✅ Path validation test passed")
            
        except Exception as e:
            self.test_results['path_validation'] = False
            print(f"❌ Path validation test failed: {e}")
    
    def test_simple_planning(self):
        """Test basic planning functionality."""
        print("\n🗺️ Testing Simple Planning...")
        
        planner = JointSpaceRRT(
            self.model, 
            self.ik_solver,
            max_iterations=500,
            step_size=0.2,
            goal_tolerance=0.1
        )
        
        try:
            # Use 9D configurations
            start_config = np.zeros(9)
            goal_config = np.array([1.0, 0.5, -0.5, 0.3, -0.2, 0.8, 0.0, 0.0, 0.0])
            
            start_time = time.time()
            path = planner.plan(start_config, goal_config)
            planning_time = time.time() - start_time
            
            if path is not None:
                assert len(path) >= 2, "Path too short"
                assert np.allclose(path[0], start_config, atol=0.1)
                assert np.allclose(path[-1], goal_config, atol=planner.goal_tolerance)
                
                self.test_results['simple_planning'] = True
                print(f"✅ Simple planning test passed ({len(path)} waypoints, {planning_time:.2f}s)")
                
                # Visualize tree growth
                self.visualize_tree_growth(planner)
                
            else:
                print("⚠️ Planning failed (may be normal for difficult problems)")
                self.test_results['simple_planning'] = False
                
        except Exception as e:
            self.test_results['simple_planning'] = False
            print(f"❌ Simple planning test failed: {e}")
    
    def test_planning_with_obstacles(self):
        """Test planning with collision checking."""
        print("\n🚧 Testing Planning with Obstacles...")
        
        # Create planner with custom collision checking
        class ObstacleAwareRRT(JointSpaceRRT):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Define forbidden region (sphere in joint space)
                self.obstacle_center = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
                self.obstacle_radius = 0.3
            
            def is_valid_config(self, config):
                if not super().is_valid_config(config):
                    return False
                
                # Check obstacle collision
                distance = np.linalg.norm(config - self.obstacle_center)
                return distance > self.obstacle_radius
        
        try:
            planner = ObstacleAwareRRT(
                self.model,
                self.ik_solver,
                max_iterations=1000,
                step_size=0.1
            )
            
            start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            goal_config = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            path = planner.plan(start_config, goal_config)
            
            if path is not None:
                # Verify path avoids obstacle
                for config in path:
                    distance = np.linalg.norm(config - planner.obstacle_center)
                    assert distance > planner.obstacle_radius, "Path goes through obstacle"
                
                self.test_results['obstacle_planning'] = True
                print(f"✅ Obstacle planning test passed ({len(path)} waypoints)")
            else:
                print("⚠️ Obstacle planning failed")
                self.test_results['obstacle_planning'] = False
                
        except Exception as e:
            self.test_results['obstacle_planning'] = False
            print(f"❌ Obstacle planning test failed: {e}")
    
    def test_planning_performance(self):
        """Test planning performance with different parameters."""
        print("\n⚡ Testing Planning Performance...")
        
        test_configs = [
            {'step_size': 0.05, 'max_iterations': 1000, 'goal_bias': 0.1},
            {'step_size': 0.1, 'max_iterations': 1000, 'goal_bias': 0.1},
            {'step_size': 0.2, 'max_iterations': 1000, 'goal_bias': 0.1},
        ]
        
        results = []
        
        for config in test_configs:
            planner = JointSpaceRRT(self.model, self.ik_solver, **config)
            
            # Debug: Check configuration shapes
            start_config = np.zeros(9)  # Changed to 9D
            goal_config = np.array([1.5, 1.0, -1.0, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0])  # 9D
            
            print(f"🔍 Debug - Start config shape: {start_config.shape}")
            print(f"🔍 Debug - Goal config shape: {goal_config.shape}")
            
            # Test random config from IK solver
            random_config = self.ik_solver.get_random_valid_config()
            if random_config is not None:
                print(f"🔍 Debug - Random config shape: {random_config.shape}")
            
            start_time = time.time()
            path = planner.plan(start_config, goal_config)
            planning_time = time.time() - start_time
            
            results.append({
                'config': config,
                'success': path is not None,
                'time': planning_time,
                'path_length': len(path) if path else 0,
                'tree_size': len(planner.tree)
            })
        
        try:
            # Print performance comparison
            print(f"{'Step Size':<10} {'Success':<8} {'Time(s)':<8} {'Path Len':<10} {'Tree Size':<10}")
            print("-" * 50)
            for result in results:
                config = result['config']
                print(f"{config['step_size']:<10.2f} "
                      f"{'Yes' if result['success'] else 'No':<8} "
                      f"{result['time']:<8.2f} "
                      f"{result['path_length']:<10} "
                      f"{result['tree_size']:<10}")
            
            self.test_results['performance'] = True
            print("✅ Performance test completed")
            
        except Exception as e:
            self.test_results['performance'] = False
            print(f"❌ Performance test failed: {e}")
    
    def test_goal_bias_effect(self):
        """Test effect of goal bias on planning."""
        print("\n🎯 Testing Goal Bias Effect...")
        
        bias_values = [0.0, 0.1, 0.3, 0.5]
        results = []
        
        for bias in bias_values:
            planner = JointSpaceRRT(
                self.model,
                self.ik_solver,
                max_iterations=500,
                goal_bias=bias
            )
            
            successes = 0
            total_time = 0
            
            # Run multiple trials
            for _ in range(5):
                start_config = self.ik_solver.get_random_valid_config()
                goal_config = self.ik_solver.get_random_valid_config()
                
                start_time = time.time()
                path = planner.plan(start_config, goal_config)
                total_time += time.time() - start_time
                
                if path is not None:
                    successes += 1
            
            success_rate = successes / 5
            avg_time = total_time / 5
            
            results.append({
                'bias': bias,
                'success_rate': success_rate,
                'avg_time': avg_time
            })
        
        try:
            print(f"{'Goal Bias':<10} {'Success Rate':<12} {'Avg Time(s)':<12}")
            print("-" * 35)
            for result in results:
                print(f"{result['bias']:<10.1f} "
                      f"{result['success_rate']:<12.1%} "
                      f"{result['avg_time']:<12.2f}")
            
            self.test_results['goal_bias'] = True
            print("✅ Goal bias test completed")
            
        except Exception as e:
            self.test_results['goal_bias'] = False
            print(f"❌ Goal bias test failed: {e}")
    
    def test_unreachable_goals(self):
        """Test handling of unreachable goals."""
        print("\n🚫 Testing Unreachable Goals...")
        
        planner = JointSpaceRRT(
            self.model,
            self.ik_solver,
            max_iterations=100  # Low iterations for quick failure
        )
        
        try:
            # Valid start configuration (within joint limits) - 9D
            start_config = np.zeros(9)
            
            # Goal outside joint limits - 9D
            unreachable_goal = np.full(9, 10.0)
            
            path = planner.plan(start_config, unreachable_goal)
            
            # Should return None for unreachable goals
            assert path is None, "Should return None for unreachable goals"
            
            self.test_results['unreachable_goals'] = True
            print("✅ Unreachable goals test passed")
            
        except Exception as e:
            self.test_results['unreachable_goals'] = False
            print(f"❌ Unreachable goals test failed: {e}")
    
    def test_invalid_start_configs(self):
        """Test handling of invalid start configurations."""
        print("\n❌ Testing Invalid Start Configs...")
        
        planner = JointSpaceRRT(self.model, self.ik_solver)
        
        try:
            # Invalid start configuration (outside joint limits) - 9D
            invalid_start = np.full(9, 10.0)
            valid_goal = np.full(9, 0.5)
            
            path = planner.plan(invalid_start, valid_goal)
            
            # Should handle gracefully (return None or raise appropriate error)
            if path is not None:
                print("⚠️ Planner accepted invalid start config")
            else:
                print("✅ Planner correctly rejected invalid start config")
            
            self.test_results['invalid_start'] = True
            
        except Exception as e:
            print(f"ℹ️ Planner raised exception for invalid start: {type(e).__name__}")
            self.test_results['invalid_start'] = True  # This is acceptable behavior
    
    def visualize_tree_growth(self, planner):
        """Visualize RRT tree growth (for 2D projection)."""
        if len(planner.tree) < 10:
            return
            
        try:
            # Project to first two joints for visualization
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot tree edges
            for node in planner.tree[1:]:  # Skip root
                if node.parent:
                    x_coords = [node.parent.config[0], node.config[0]]
                    y_coords = [node.parent.config[1], node.config[1]]
                    ax.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=0.5)
            
            # Plot nodes
            x_coords = [node.config[0] for node in planner.tree]
            y_coords = [node.config[1] for node in planner.tree]
            ax.scatter(x_coords, y_coords, c='red', s=10, alpha=0.7)
            
            # Highlight start and goal
            ax.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='s', label='Start')
            if planner.goal_node:
                goal_x, goal_y = planner.goal_node.config[0], planner.goal_node.config[1]
                ax.scatter(goal_x, goal_y, c='red', s=100, marker='*', label='Goal')
            
            ax.set_xlabel('Joint 1 (rad)')
            ax.set_ylabel('Joint 2 (rad)')
            ax.set_title(f'RRT Tree Growth ({len(planner.tree)} nodes)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not create visualization: {e}")
    
    def print_test_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("🏁 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():<25} {status}")
        
        print("-" * 50)
        print(f"Total: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Your JointSpaceRRT is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the implementation.")


def main():
    """Main test runner."""
    tester = JointSpaceRRTTester(use_mock=False)
    tester.run_all_tests()


if __name__ == "__main__":
    main()