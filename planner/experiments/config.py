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

    # Where along the trajectory to inject failure
    fail_phase: Optional[int] = None   # 6 or 7; None = random
    fail_step_offset: Optional[int] = None  # step within phase; None = random
    fail_duration: int = 40

    # Capture params
    image_width: int = 640
    image_height: int = 480
    camera_name: Optional[str] = "overhead_cam"
    # Free camera fallback params (used if camera_name is None)
    camera_lookat: Optional[List[float]] = None
    camera_distance: Optional[float] = None
    camera_azimuth: Optional[float] = None
    camera_elevation: Optional[float] = None

    # Post-failure settle
    post_failure_settle_steps: int = 500

    experiment_id: Optional[str] = None
