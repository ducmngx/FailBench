# Motion Planner IK Interface

A stateless, high-performance inverse kinematics solver built on Mink, designed specifically for integration with sampling-based motion planners like RRT and PRM.

## Overview

This IK solver provides a clean, efficient interface for motion planning applications where you need to:
- Quickly validate target reachability
- Generate multiple solutions for the same target
- Handle batch IK solving for planning trees
- Maintain clear separation between kinematics and collision checking

## Core Components

### IKResult Enum

Enumeration of possible IK solving outcomes for clear error handling.

```python
class IKResult(Enum):
    SUCCESS = "success"                    # IK solved successfully
    FAILED_TO_CONVERGE = "failed_to_converge"  # Reached max iterations
    INVALID_TARGET = "invalid_target"      # Invalid input parameters
    JOINT_LIMITS_VIOLATED = "joint_limits_violated"  # Solution violates limits
```

### IKConfig Dataclass

Configuration parameters for IK solving behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solver` | str | "osqp" | Optimization solver backend |
| `max_iterations` | int | 100 | Maximum IK iterations |
| `dt` | float | 0.01 | Integration time step |
| `position_tolerance` | float | 0.005 | Position convergence threshold (meters) |
| `orientation_tolerance` | float | 0.05 | Orientation convergence threshold (radians) |
| `position_cost` | float | 1.0 | Weight for position objectives |
| `orientation_cost` | float | 1.0 | Weight for orientation objectives |
| `posture_cost` | float | 1e-2 | Weight for posture regularization |
| `damping` | float | 5e-3 | Numerical damping factor |
| `check_joint_limits` | bool | True | Enable joint limit validation |

#### Example Usage
```python
# Fast solving for planning
fast_config = IKConfig(
    max_iterations=20,
    position_tolerance=0.01,
    orientation_tolerance=0.1
)

# High precision for final execution
precise_config = IKConfig(
    max_iterations=200,
    position_tolerance=0.001,
    orientation_tolerance=0.01
)
```

### EndEffectorTarget Dataclass

Specification for end-effector target poses.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position` | np.ndarray | Required | Target position [x, y, z] |
| `orientation` | Optional[np.ndarray] | None | Target quaternion [w, x, y, z] |
| `frame_name` | str | "end_effector" | MuJoCo frame name |
| `frame_type` | str | "site" | Frame type (site/body/geom) |
| `position_cost` | Optional[float] | None | Override position cost |
| `orientation_cost` | Optional[float] | None | Override orientation cost |

#### Example Usage
```python
# Position-only target
target = EndEffectorTarget(
    position=np.array([0.5, 0.2, 0.8]),
    frame_name="gripper_tip"
)

# Full pose target
target = EndEffectorTarget(
    position=np.array([0.5, 0.2, 0.8]),
    orientation=np.array([1, 0, 0, 0]),  # [w, x, y, z]
    frame_name="gripper_tip",
    position_cost=2.0,  # Higher priority on position
    orientation_cost=0.5  # Lower priority on orientation
)
```

## IKSolver Class

The main IK solver class optimized for motion planning integration.

### Constructor

```python
def __init__(self, model: mujoco.MjModel, config: Optional[IKConfig] = None)
```

**Parameters:**
- `model`: MuJoCo model (should be thread-safe for parallel planning)
- `config`: IK configuration (uses defaults if None)

**Features:**
- Pre-allocates data structures for efficiency
- Extracts joint limits for validation
- Minimal memory allocation during solving

### Core Methods

#### solve()

Primary IK solving method for single or multiple targets.

```python
def solve(
    self, 
    targets: Union[EndEffectorTarget, List[EndEffectorTarget]],
    seed_config: np.ndarray,
    config_override: Optional[IKConfig] = None
) -> Tuple[np.ndarray, IKResult]
```

**Parameters:**
- `targets`: Single target or list of simultaneous targets
- `seed_config`: Starting joint configuration
- `config_override`: Temporary config override for this solve

**Returns:**
- `Tuple[np.ndarray, IKResult]`: Solution configuration and result status

**Example:**
```python
solver = IKSolver(model)
target = EndEffectorTarget(position=np.array([0.5, 0.2, 0.8]))
solution, result = solver.solve(target, current_joints)

if result == IKResult.SUCCESS:
    robot.move_to_joints(solution)
else:
    print(f"IK failed: {result}")
```

#### solve_with_multiple_seeds()

Try IK with multiple seed configurations to find different solutions.

```python
def solve_with_multiple_seeds(
    self,
    targets: Union[EndEffectorTarget, List[EndEffectorTarget]],
    seed_configs: List[np.ndarray],
    config_override: Optional[IKConfig] = None
) -> List[Tuple[np.ndarray, IKResult]]
```

**Use Case:** Finding alternative solutions or increasing success rate.

**Example:**
```python
seeds = [current_joints, random_config1, random_config2]
results = solver.solve_with_multiple_seeds(target, seeds)

# Find all successful solutions
solutions = [sol for sol, result in results if result == IKResult.SUCCESS]
```

#### find_valid_solution()

Find the first valid solution from multiple seeds.

```python
def find_valid_solution(
    self,
    targets: Union[EndEffectorTarget, List[EndEffectorTarget]],
    seed_configs: List[np.ndarray],
    config_override: Optional[IKConfig] = None
) -> Optional[np.ndarray]
```

**Returns:** First successful solution or None if all fail.

**Example:**
```python
seeds = generate_random_seeds(10)
solution = solver.find_valid_solution(target, seeds)

if solution is not None:
    # Use the solution
    execute_motion(solution)
```

#### is_target_reachable()

Quick reachability check for motion planners.

```python
def is_target_reachable(
    self,
    targets: Union[EndEffectorTarget, List[EndEffectorTarget]],
    seed_configs: List[np.ndarray],
    max_attempts: int = 5
) -> bool
```

**Features:**
- Uses fast, low-precision solving
- Early termination on first success
- Optimized for planning algorithms

**Example:**
```python
# Check if target is reachable before expensive planning
if solver.is_target_reachable(target, [current_config]):
    path = planner.plan_to_target(target)
else:
    print("Target unreachable, skipping planning")
```

### Utility Methods

#### get_random_valid_config()

Generate random valid configurations for sampling-based planners.

```python
def get_random_valid_config(self, num_attempts: int = 100) -> Optional[np.ndarray]
```

**Features:**
- Respects joint limits
- Can be extended for collision checking
- Essential for RRT-style planners

#### interpolate_configs()

Linear interpolation between joint configurations.

```python
def interpolate_configs(
    self,
    config1: np.ndarray,
    config2: np.ndarray,
    num_points: int = 10
) -> List[np.ndarray]
```

**Use Cases:**
- Local planning between nearby configurations
- Trajectory generation
- Edge validation in motion planning

**Example:**
```python
# Generate smooth path between two configurations
waypoints = solver.interpolate_configs(start_config, end_config, num_points=20)
for config in waypoints:
    robot.move_to_joints(config)
    time.sleep(0.1)
```

## Convenience Functions

### solve_ik_for_planner()

Simple function interface for basic motion planning needs.

```python
def solve_ik_for_planner(
    model: mujoco.MjModel,
    target_position: np.ndarray,
    seed_config: np.ndarray,
    target_orientation: Optional[np.ndarray] = None,
    frame_name: str = "end_effector",
    fast_mode: bool = False
) -> Optional[np.ndarray]
```

**Example:**
```python
# Quick IK solve in a planning loop
solution = solve_ik_for_planner(
    model, target_pos, current_joints, fast_mode=True
)
if solution is not None:
    add_to_planning_tree(solution)
```

### batch_ik_solve()

Efficient batch processing for multiple IK problems.

```python
def batch_ik_solve(
    model: mujoco.MjModel,
    targets_and_seeds: List[Tuple[np.ndarray, np.ndarray]],
    frame_name: str = "end_effector",
    fast_mode: bool = True
) -> List[Optional[np.ndarray]]
```

**Use Case:** Validating many nodes in a planning tree efficiently.

**Example:**
```python
# Validate multiple potential tree nodes
target_seed_pairs = [(pos1, seed1), (pos2, seed2), (pos3, seed3)]
solutions = batch_ik_solve(model, target_seed_pairs, fast_mode=True)

valid_nodes = [sol for sol in solutions if sol is not None]
```

## Integration Patterns

### With RRT Planners

```python
class RRTWithIK:
    def __init__(self, model):
        self.ik_solver = IKSolver(model)
    
    def plan_to_pose(self, target_position):
        # Check reachability first
        if not self.ik_solver.is_target_reachable(target, [current_config]):
            return None
        
        # Use IK in tree expansion
        for iteration in range(max_iterations):
            random_config = self.ik_solver.get_random_valid_config()
            # ... RRT logic using IK solver
```

### With Task Planning

```python
def execute_pick_and_place():
    pick_target = EndEffectorTarget(position=pick_pos)
    place_target = EndEffectorTarget(position=place_pos)
    
    # Generate approach sequence
    for target in [pick_target, place_target]:
        solution = solver.find_valid_solution(target, seed_configs)
        if solution:
            execute_motion(solution)
        else:
            raise PlanningError(f"Cannot reach target at {target.position}")
```

## Performance Considerations

### Fast vs. Accurate Modes

| Mode | Iterations | Tolerance | Use Case |
|------|------------|-----------|----------|
| Fast | 20 | 0.01m, 0.1rad | Tree expansion, reachability |
| Standard | 100 | 0.005m, 0.05rad | General planning |
| Precise | 200 | 0.001m, 0.01rad | Final execution |

### Memory Efficiency

- Pre-allocated data structures minimize allocation overhead
- Reuses MuJoCo data objects across solves
- No dynamic memory allocation in solve loops

### Collision Checking Integration

The solver provides hooks for collision checking:

```python
class CollisionAwareIKSolver(IKSolver):
    def _is_config_valid(self, config):
        # Add your collision checking here
        return (super()._is_config_valid(config) and 
                not self.collision_checker.in_collision(config))
```

## Best Practices

1. **Use appropriate tolerances** for your application - tighter for final execution, looser for planning
2. **Provide multiple seeds** when high success rate is important
3. **Check reachability** before expensive planning operations
4. **Use fast mode** for tree expansion and validation
5. **Batch process** when validating many nodes simultaneously
6. **Override `_is_config_valid()`** to add collision checking specific to your robot

## Error Handling

Always check `IKResult` values:

```python
solution, result = solver.solve(target, seed)

match result:
    case IKResult.SUCCESS:
        execute_motion(solution)
    case IKResult.FAILED_TO_CONVERGE:
        try_different_seed()
    case IKResult.JOINT_LIMITS_VIOLATED:
        adjust_target_or_seed()
    case IKResult.INVALID_TARGET:
        validate_input_parameters()
```

This ensures robust handling of different failure modes in your motion planning pipeline.