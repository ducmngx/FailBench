"""Configuration dataclasses for the experiment pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class FailureMode(Enum):
    GRIPPER_OPEN = "gripper_open"
    SINGLE_JOINT = "single_joint"
    MULTI_JOINT = "multi_joint"
    ALL_JOINTS = "all_joints"
    SLIPPERY_GRIP = "slippery_grip"


@dataclass
class FailureConfig:
    """Configuration for a single failure mode."""
    mode: FailureMode
    probability: float = 1.0
    joint_names: Optional[List[str]] = None
    grip_value: float = 255.0  # for SLIPPERY_GRIP (0=closed, 255=open)


def _default_failures() -> List[FailureConfig]:
    """Default failure configs with real-world-inspired probabilities."""
    return [
        FailureConfig(mode=FailureMode.GRIPPER_OPEN, probability=0.35),
        FailureConfig(mode=FailureMode.SLIPPERY_GRIP, grip_value=180.0, probability=0.25),
        FailureConfig(mode=FailureMode.SINGLE_JOINT, joint_names=["joint4"], probability=0.15),
        FailureConfig(mode=FailureMode.SINGLE_JOINT, joint_names=["joint6"], probability=0.10),
        FailureConfig(mode=FailureMode.MULTI_JOINT, joint_names=["joint4", "joint6"], probability=0.10),
        FailureConfig(mode=FailureMode.ALL_JOINTS, probability=0.05),
    ]


@dataclass
class ExperimentConfig:
    """Full configuration for a single experiment trial."""
    scene_xml_path: str
    robot_xml_path: str
    trajectory_file: str
    seed: int = 42

    failure_configs: List[FailureConfig] = field(default_factory=_default_failures)

    # Task identity
    task_id: str = "unknown"
    traj_id: int = 0

    # Where along the trajectory to inject failure
    fail_fraction: Optional[float] = None   # fraction [0,1] of trajectory; None = random from canonical set
    canonical_fail_fractions: List[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    )

    # Trajectory interpolation
    interp_points_per_segment: int = 100
    interp_method: str = "cubic"       # "cubic" or "linear"
    steps_per_interp_point: int = 8    # sim steps per dense point (more = better tracking)

    # Failure selection: "all" runs every mode, "sample" picks probabilistically
    failure_sample_mode: str = "all"
    num_failure_samples: int = 1

    # Capture params
    image_width: int = 640
    image_height: int = 480
    camera_name: Optional[str] = "front_cam"
    # Free camera fallback params (used if camera_name is None)
    camera_lookat: Optional[List[float]] = None
    camera_distance: Optional[float] = None
    camera_azimuth: Optional[float] = None
    camera_elevation: Optional[float] = None

    # Additional cameras (e.g., ["ee_cam"] for end-effector view)
    extra_cameras: Optional[List[str]] = None

    # Post-failure settle
    post_failure_settle_steps: int = 500

    # Grasped object body name (must have a free joint and <name>_geom)
    grasped_object_name: str = "object3"

    experiment_id: Optional[str] = None
