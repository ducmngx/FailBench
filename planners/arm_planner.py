"""
Examples of integrating the IK solver with different motion planners.
"""

import numpy as np
import mujoco
from typing import List, Optional, Tuple
from dataclasses import dataclass
import random

# Assuming imports from the IK module
from inverse_kinematics import IKSolver, EndEffectorTarget, IKConfig, IKResult, solve_ik_for_planner


@dataclass
class PlanningNode:
    """Node for tree-based planners."""
    config: np.ndarray
    parent: Optional['PlanningNode'] = None
    cost: float = 0.0
    

class RRTMotionPlanner:
    """
    Example RRT planner using the IK solver.
    Shows how to integrate IK with sampling-based planning.
    """
    
    def __init__(
        self, 
        model: mujoco.MjModel,
        ik_solver: IKSolver,
        step_size: float = 0.1,
        max_iterations: int = 1000
    ):
        self.model = model
        self.ik_solver = ik_solver
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tree: List[PlanningNode] = []
    
    def plan_to_pose(
        self,
        start_config: np.ndarray,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        frame_name: str = "gripper"
    ) -> Optional[List[np.ndarray]]:
        """
        Plan from start configuration to target end-effector pose.
        
        Returns:
            Path as list of configurations, or None if failed
        """
        # Initialize tree
        self.tree = [PlanningNode(start_config.copy())]
        target = EndEffectorTarget(
            position=target_position,
            orientation=target_orientation,
            frame_name=frame_name
        )
        
        for iteration in range(self.max_iterations):
            # Sample random configuration
            random_config = self.ik_solver.get_random_valid_config()
            if random_config is None:
                continue
            
            # Find nearest node in tree
            nearest_node = self._find_nearest_node(random_config)
            
            # Steer towards random config
            new_config = self._steer(nearest_node.config, random_config)
            
            # Check if new config is valid and reachable
            if self._is_valid_edge(nearest_node.config, new_config):
                new_node = PlanningNode(new_config, parent=nearest_node)
                self.tree.append(new_node)
                
                # Check if we can reach target from this config
                solution, result = self.ik_solver.solve(target, new_config)
                if result == IKResult.SUCCESS:
                    # Found path to target!
                    target_node = PlanningNode(solution, parent=new_node)
                    return self._extract_path(target_node)
        
        return None  # Planning failed
    
    def plan_through_waypoints(
        self,
        start_config: np.ndarray,
        waypoint_positions: List[np.ndarray],
        frame_name: str = "gripper"
    ) -> Optional[List[np.ndarray]]:
        """
        Plan through a sequence of waypoint poses.
        
        Returns:
            Complete path through all waypoints, or None if failed
        """
        full_path = [start_config.copy()]
        current_config = start_config.copy()
        
        for i, waypoint_pos in enumerate(waypoint_positions):
            print(f"Planning to waypoint {i+1}/{len(waypoint_positions)}")
            
            # Plan from current position to waypoint
            segment = self.plan_to_pose(current_config, waypoint_pos, frame_name=frame_name)
            
            if segment is None:
                print(f"Failed to reach waypoint {i+1}")
                return None
            
            # Add segment to full path (skip first point to avoid duplication)
            full_path.extend(segment[1:])
            current_config = segment[-1].copy()
        
        return full_path
    
    def _find_nearest_node(self, config: np.ndarray) -> PlanningNode:
        """Find nearest node in tree to given configuration."""
        min_dist = float('inf')
        nearest = self.tree[0]
        
        for node in self.tree:
            dist = np.linalg.norm(node.config - config)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
        """Steer from one config towards another by step_size."""
        direction = to_config - from_config
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_config.copy()
        else:
            unit_direction = direction / distance
            return from_config + self.step_size * unit_direction
    
    def _is_valid_edge(self, config1: np.ndarray, config2: np.ndarray) -> bool:
        """Check if edge between two configs is valid (no collisions)."""
        # Interpolate between configs and check validity
        interpolated = self.ik_solver.interpolate_configs(config1, config2, num_points=5)
        
        for config in interpolated:
            if not self.ik_solver._is_config_valid(config):
                return False
        
        return True
    
    def _extract_path(self, goal_node: PlanningNode) -> List[np.ndarray]:
        """Extract path from root to goal node."""
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.config.copy())
            current = current.parent
        
        return list(reversed(path))


class CartesianPathPlanner:
    """
    Cartesian path planner that plans in end-effector space.
    Uses IK to convert Cartesian waypoints to joint space.
    """
    
    def __init__(self, model: mujoco.MjModel, ik_solver: IKSolver):
        self.model = model
        self.ik_solver = ik_solver
    
    def plan_straight_line(
        self,
        start_config: np.ndarray,
        end_position: np.ndarray,
        end_orientation: Optional[np.ndarray] = None,
        frame_name: str = "gripper",
        num_waypoints: int = 20
    ) -> Optional[List[np.ndarray]]:
        """
        Plan straight line in Cartesian space.
        
        Returns:
            Joint space path or None if any waypoint unreachable
        """
        # Get start pose
        self.ik_solver._data.qpos[:] = start_config
        mujoco.mj_forward(self.model, self.ik_solver._data)
        self.ik_solver._configuration.update(self.ik_solver._data.qpos)
        
        start_transform = self.ik_solver._configuration.get_transform_frame_to_world(
            frame_name, "site"
        )
        start_position = start_transform.translation()
        start_orientation = start_transform.as_quaternion_xyzw()
        
        # Generate Cartesian waypoints
        cartesian_waypoints = []
        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            
            # Interpolate position
            waypoint_pos = (1 - alpha) * start_position + alpha * end_position
            
            # Interpolate orientation (SLERP would be better for quaternions)
            if end_orientation is not None:
                waypoint_ori = (1 - alpha) * start_orientation + alpha * end_orientation
                waypoint_ori /= np.linalg.norm(waypoint_ori)  # Normalize
            else:
                waypoint_ori = start_orientation
            
            cartesian_waypoints.append((waypoint_pos, waypoint_ori))
        
        # Convert to joint space using IK
        joint_path = [start_config.copy()]
        current_config = start_config.copy()
        
        for waypoint_pos, waypoint_ori in cartesian_waypoints[1:]:
            target = EndEffectorTarget(
                position=waypoint_pos,
                orientation=waypoint_ori,
                frame_name=frame_name
            )
            
            solution, result = self.ik_solver.solve(target, current_config)
            
            if result != IKResult.SUCCESS:
                print(f"IK failed for waypoint at {waypoint_pos}")
                return None
            
            joint_path.append(solution.copy())
            current_config = solution  # Use as seed for next waypoint
        
        return joint_path
    
    def plan_arc(
        self,
        start_config: np.ndarray,
        center: np.ndarray,
        end_angle: float,
        radius: float,
        normal: np.ndarray = np.array([0, 0, 1]),
        frame_name: str = "gripper",
        num_waypoints: int = 20
    ) -> Optional[List[np.ndarray]]:
        """
        Plan circular arc in Cartesian space.
        
        Args:
            start_config: Starting joint configuration
            center: Center of arc
            end_angle: End angle in radians
            radius: Arc radius
            normal: Normal vector to arc plane
            frame_name: End-effector frame name
            num_waypoints: Number of waypoints along arc
            
        Returns:
            Joint space path or None if failed
        """
        # Generate arc waypoints
        angles = np.linspace(0, end_angle, num_waypoints + 1)
        
        # Create orthonormal basis for arc plane
        normal = normal / np.linalg.norm(normal)
        # Pick arbitrary vector not parallel to normal
        if abs(normal[0]) < 0.9:
            u = np.array([1, 0, 0])
        else:
            u = np.array([0, 1, 0])
        
        # Gram-Schmidt to create orthonormal basis
        u = u - np.dot(u, normal) * normal
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # Generate waypoints
        cartesian_waypoints = []
        for angle in angles:
            waypoint_pos = center + radius * (np.cos(angle) * u + np.sin(angle) * v)
            cartesian_waypoints.append(waypoint_pos)
        
        # Convert to joint space
        joint_path = [start_config.copy()]
        current_config = start_config.copy()
        
        for waypoint_pos in cartesian_waypoints[1:]:
            target = EndEffectorTarget(position=waypoint_pos, frame_name=frame_name)
            solution, result = self.ik_solver.solve(target, current_config)
            
            if result != IKResult.SUCCESS:
                return None
            
            joint_path.append(solution.copy())
            current_config = solution
        
        return joint_path


class BiRRTPlanner:
    """
    Bi-directional RRT planner for complex motion planning.
    Grows trees from both start and goal configurations.
    """
    
    def __init__(self, model: mujoco.MjModel, ik_solver: IKSolver):
        self.model = model
        self.ik_solver = ik_solver
        self.step_size = 0.1
        self.max_iterations = 2000
    
    def plan(
        self,
        start_config: np.ndarray,
        goal_position: np.ndarray,
        goal_orientation: Optional[np.ndarray] = None,
        frame_name: str = "gripper"
    ) -> Optional[List[np.ndarray]]:
        """
        Plan using bi-directional RRT.
        
        Returns:
            Path from start to goal, or None if failed
        """
        # Find valid goal configuration using IK
        target = EndEffectorTarget(
            position=goal_position,
            orientation=goal_orientation,
            frame_name=frame_name
        )
        
        # Try multiple seeds to find goal config
        seeds = [start_config]  # Start with current config
        for _ in range(10):
            random_seed = self.ik_solver.get_random_valid_config()
            if random_seed is not None:
                seeds.append(random_seed)
        
        goal_config = self.ik_solver.find_valid_solution(target, seeds)
        if goal_config is None:
            print("Goal pose unreachable")
            return None
        
        # Initialize trees
        start_tree = [PlanningNode(start_config.copy())]
        goal_tree = [PlanningNode(goal_config.copy())]
        
        for iteration in range(self.max_iterations):
            # Extend start tree
            if self._extend_tree(start_tree):
                # Check if we can connect to goal tree
                connection = self._find_connection(start_tree, goal_tree)
                if connection:
                    return self._build_path(connection)
            
            # Extend goal tree
            if self._extend_tree(goal_tree):
                # Check if we can connect to start tree
                connection = self._find_connection(goal_tree, start_tree)
                if connection:
                    return self._build_path(connection, reverse_second=True)
        
        return None
    
    def _extend_tree(self, tree: List[PlanningNode]) -> bool:
        """Extend tree by one node. Returns True if successful."""
        random_config = self.ik_solver.get_random_valid_config()
        if random_config is None:
            return False
        
        # Find nearest node
        nearest = min(tree, key=lambda n: np.linalg.norm(n.config - random_config))
        
        # Steer towards random config
        direction = random_config - nearest.config
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            new_config = random_config
        else:
            new_config = nearest.config + self.step_size * direction / distance
        
        # Check validity
        if self.ik_solver._is_config_valid(new_config):
            tree.append(PlanningNode(new_config, parent=nearest))
            return True
        
        return False
    
    def _find_connection(
        self, 
        tree1: List[PlanningNode], 
        tree2: List[PlanningNode]
    ) -> Optional[Tuple[PlanningNode, PlanningNode]]:
        """Find connection between two trees."""
        connection_threshold = 0.2
        
        for node1 in tree1:
            for node2 in tree2:
                if np.linalg.norm(node1.config - node2.config) < connection_threshold:
                    return (node1, node2)
        
        return None
    
    def _build_path(
        self, 
        connection: Tuple[PlanningNode, PlanningNode], 
        reverse_second: bool = False
    ) -> List[np.ndarray]:
        """Build complete path from connection between trees."""
        node1, node2 = connection
        
        # Extract path from first tree (to root)
        path1 = []
        current = node1
        while current is not None:
            path1.append(current.config.copy())
            current = current.parent
        path1.reverse()
        
        # Extract path from second tree
        path2 = []
        current = node2
        while current is not None:
            path2.append(current.config.copy())
            current = current.parent
        
        if not reverse_second:
            path2.reverse()
        
        # Combine paths
        return path1 + path2


class TaskSpacePlanner:
    """
    Plans directly in task space (end-effector positions) and uses IK to map to joint space.
    Good for manipulation tasks with known workspace constraints.
    """
    
    def __init__(self, model: mujoco.MjModel, ik_solver: IKSolver):
        self.model = model
        self.ik_solver = ik_solver
    
    def plan_pick_and_place(
        self,
        start_config: np.ndarray,
        pick_position: np.ndarray,
        place_position: np.ndarray,
        approach_height: float = 0.1,
        frame_name: str = "gripper"
    ) -> Optional[List[np.ndarray]]:
        """
        Plan a pick-and-place motion sequence.
        
        Args:
            start_config: Starting joint configuration
            pick_position: Object pickup position
            place_position: Object placement position  
            approach_height: Height above pick/place for approach/retreat
            frame_name: End-effector frame name
            
        Returns:
            Complete pick-and-place path or None if failed
        """
        # Define task space waypoints
        pick_approach = pick_position + np.array([0, 0, approach_height])
        place_approach = place_position + np.array([0, 0, approach_height])
        
        waypoints = [
            ("approach_pick", pick_approach),
            ("pick", pick_position),
            ("retreat_pick", pick_approach),
            ("approach_place", place_approach),
            ("place", place_position),
            ("retreat_place", place_approach)
        ]
        
        # Convert each waypoint to joint space
        full_path = [start_config.copy()]
        current_config = start_config.copy()
        
        for waypoint_name, waypoint_pos in waypoints:
            print(f"Planning to {waypoint_name}")
            
            target = EndEffectorTarget(position=waypoint_pos, frame_name=frame_name)
            
            # Try current config first, then multiple seeds if needed
            seeds = [current_config]
            for _ in range(5):
                random_seed = self.ik_solver.get_random_valid_config()
                if random_seed is not None:
                    seeds.append(random_seed)
            
            solution = self.ik_solver.find_valid_solution(target, seeds)
            
            if solution is None:
                print(f"Failed to reach {waypoint_name} at {waypoint_pos}")
                return None
            
            # Add interpolated path to waypoint
            segment = self.ik_solver.interpolate_configs(current_config, solution, num_points=10)
            full_path.extend(segment[1:])  # Skip first point to avoid duplication
            current_config = solution
        
        return full_path
    
    def plan_constrained_motion(
        self,
        start_config: np.ndarray,
        constraint_func,
        goal_position: np.ndarray,
        frame_name: str = "gripper",
        max_iterations: int = 100
    ) -> Optional[List[np.ndarray]]:
        """
        Plan motion subject to task space constraints.
        
        Args:
            start_config: Starting configuration
            constraint_func: Function that returns True if position satisfies constraints
            goal_position: Goal end-effector position
            frame_name: End-effector frame name
            max_iterations: Maximum planning iterations
            
        Returns:
            Constrained path or None if failed
        """
        path = [start_config.copy()]
        current_config = start_config.copy()
        
        for iteration in range(max_iterations):
            # Get current end-effector position
            self.ik_solver._data.qpos[:] = current_config
            mujoco.mj_forward(self.model, self.ik_solver._data)
            self.ik_solver._configuration.update(self.ik_solver._data.qpos)
            
            current_transform = self.ik_solver._configuration.get_transform_frame_to_world(
                frame_name, "site"
            )
            current_position = current_transform.translation()
            
            # Check if we've reached the goal
            if np.linalg.norm(current_position - goal_position) < 0.01:
                return path
            
            # Sample next position that satisfies constraints
            for attempt in range(50):
                # Sample direction towards goal with some randomness
                direction = goal_position - current_position
                direction = direction / np.linalg.norm(direction)
                
                # Add some randomness
                random_component = np.random.normal(0, 0.1, 3)
                direction = direction + random_component
                direction = direction / np.linalg.norm(direction)
                
                next_position = current_position + 0.05 * direction
                
                # Check constraints
                if constraint_func(next_position):
                    # Try IK for this position
                    target = EndEffectorTarget(position=next_position, frame_name=frame_name)
                    solution, result = self.ik_solver.solve(target, current_config)
                    
                    if result == IKResult.SUCCESS:
                        path.append(solution.copy())
                        current_config = solution
                        break
            else:
                print("Failed to find valid next step")
                return None
        
        print("Max iterations reached")
        return None


# Example usage and integration
def example_rrt_planning():
    """Example of using RRT with IK solver."""
    
    # Load robot model
    model = mujoco.MjModel.from_xml_path("robot.xml")
    
    # Create IK solver
    ik_config = IKConfig(
        max_iterations=50,
        position_tolerance=0.005,
        check_joint_limits=True
    )
    ik_solver = IKSolver(model, ik_config)
    
    # Create RRT planner
    rrt = RRTMotionPlanner(model, ik_solver)
    
    # Plan to target pose
    start_config = np.zeros(model.nq)
    target_position = np.array([0.5, 0.2, 0.8])
    
    path = rrt.plan_to_pose(start_config, target_position)
    
    if path:
        print(f"Found path with {len(path)} waypoints")
        return path
    else:
        print("Planning failed")
        return None


def example_cartesian_planning():
    """Example of Cartesian space planning."""
    
    model = mujoco.MjModel.from_xml_path("robot.xml")
    ik_solver = IKSolver(model)
    cartesian_planner = CartesianPathPlanner(model, ik_solver)
    
    start_config = np.zeros(model.nq)
    end_position = np.array([0.4, 0.3, 0.6])
    
    # Plan straight line
    straight_path = cartesian_planner.plan_straight_line(
        start_config, end_position, num_waypoints=30
    )
    
    if straight_path:
        print("Straight line path found")
    
    # Plan arc
    center = np.array([0.3, 0.0, 0.5])
    arc_path = cartesian_planner.plan_arc(
        start_config, center, np.pi/2, radius=0.2, num_waypoints=20
    )
    
    if arc_path:
        print("Arc path found")
    
    return straight_path, arc_path


def example_pick_and_place():
    """Example of task-space pick and place planning."""
    
    model = mujoco.MjModel.from_xml_path("/home/aaron/workspace/mujoco-arena/mink/examples/franka_emika_panda/mjx_panda.xml")
    ik_solver = IKSolver(model)
    task_planner = TaskSpacePlanner(model, ik_solver)
    
    start_config = np.zeros(model.nq)
    pick_pos = np.array([0.4, 0.2, 0.1])
    place_pos = np.array([0.2, -0.3, 0.15])
    
    path = task_planner.plan_pick_and_place(
        start_config, pick_pos, place_pos, approach_height=0.1
    )
    
    if path:
        print(f"Path is {path}")
        print(f"Pick and place path with {len(path)} waypoints")
        return path
    else:
        print("Pick and place planning failed")
        return None


def example_constrained_motion():
    """Example of motion planning with constraints."""
    
    def table_constraint(position):
        """Constraint to keep end-effector above table height."""
        return position[2] > 0.05  # At least 5cm above table
    
    def workspace_constraint(position):
        """Constraint to keep end-effector in workspace."""
        return (np.linalg.norm(position[:2]) < 0.8 and  # Within reach
                position[2] > 0.05 and position[2] < 1.0)  # Height limits
    
    def combined_constraint(position):
        """Combine multiple constraints."""
        return table_constraint(position) and workspace_constraint(position)
    
    model = mujoco.MjModel.from_xml_path("/home/aaron/workspace/mujoco-arena/mink/examples/franka_emika_panda/mjx_panda.xml")
    ik_solver = IKSolver(model)
    task_planner = TaskSpacePlanner(model, ik_solver)
    
    start_config = np.zeros(model.nq)
    goal_position = np.array([0.5, 0.4, 0.3])
    
    path = task_planner.plan_constrained_motion(
        start_config, combined_constraint, goal_position
    )
    
    if path:
        print("Constrained motion path found")
        return path
    else:
        print("Constrained motion planning failed")
        return None


if __name__ == "__main__":
    # Run examples
    # rrt_path = example_rrt_planning()
    # cartesian_paths = example_cartesian_planning()
    pick_place_path = example_pick_and_place()
    # constrained_path = example_constrained_motion()