# planner/__init__.py - Main package interface
from enum import Enum

from .algorithms.abstract_planner import AbstractRRTPlanner, PlanningNode, PlanningSpace
from .algorithms.RRTplanner import JointSpaceRRT, JointSpaceRRTConnect, JointSpaceRRTConnectFailure
from .algorithms.TRRTFailure import JointSpaceTRRTOMPL
from .algorithms.STOMP import JointSpaceSTOMP
from .collision.collision_checker import CollisionChecker
from .kinematics.inverse_kinematics import IKSolver, EndEffectorTarget, IKResult
from .collision.geometry_utils import EnhancedXMLSeparator, SimpleXMLSeparator
from .core.planning_context import PlanningContext
from .costs.failure_cost import FailureCostModel, SeverityConfig
from .utils.traj_saver import ExperimentTrajectoryManager, TrajectoryRecord
from . import experiments

__all__ = [
    # Planners
    "AbstractRRTPlanner",
    "JointSpaceRRT",
    "JointSpaceRRTConnect",
    "JointSpaceRRTConnectFailure",
    "JointSpaceTRRTOMPL",
    "JointSpaceSTOMP",
    # Core types
    "PlanningNode",
    "PlanningSpace",
    "PlanningContext",
    "PlannerType",
    # Collision
    "CollisionChecker",
    "EnhancedXMLSeparator",
    "SimpleXMLSeparator",
    # Kinematics
    "IKSolver",
    "EndEffectorTarget",
    "IKResult",
    # Costs
    "FailureCostModel",
    "SeverityConfig",
    # Utilities
    "ExperimentTrajectoryManager",
    "TrajectoryRecord",
    # Factory
    "create_planner",
    # Experiments
    "experiments",
]


class PlannerType(Enum):
    """Available planner algorithms."""
    RRT = "rrt"
    RRT_CONNECT = "rrt_connect"
    RRT_CONNECT_FAILURE = "rrt_connect_failure"
    TRRT = "trrt"
    STOMP = "stomp"


def create_planner(scene_xml: str, robot_xml: str, planner_type: PlannerType = PlannerType.RRT, **kwargs) -> AbstractRRTPlanner:
    """Factory function for creating planners.

    Args:
        scene_xml: Path to scene XML file.
        robot_xml: Path to robot XML file.
        planner_type: Which planner algorithm to use.
        **kwargs: Additional keyword arguments passed to the planner constructor.

    Returns:
        A configured planner instance.
    """
    context = PlanningContext(scene_xml, robot_xml)
    ik_solver = IKSolver(context.robot_model)
    collision_threshold = kwargs.pop("collision_threshold", 0.03)

    if planner_type == PlannerType.RRT:
        return JointSpaceRRT(context.scene_model, context.robot_model, ik_solver, collision_threshold, **kwargs)
    elif planner_type == PlannerType.RRT_CONNECT:
        seed = kwargs.pop("seed", 42)
        return JointSpaceRRTConnect(context.scene_model, context.robot_model, ik_solver, collision_threshold, seed=seed, **kwargs)
    elif planner_type == PlannerType.RRT_CONNECT_FAILURE:
        seed = kwargs.pop("seed", 42)
        collision_estimator = kwargs.pop("collision_estimator")
        body_severity_table = kwargs.pop("body_severity_table")
        return JointSpaceRRTConnectFailure(
            context.scene_model, context.robot_model, ik_solver, collision_threshold,
            seed=seed, collision_estimator=collision_estimator,
            body_severity_table=body_severity_table, **kwargs
        )
    elif planner_type == PlannerType.TRRT:
        seed = kwargs.pop("seed", 42)
        collision_estimator = kwargs.pop("collision_estimator")
        object_positions = kwargs.pop("object_positions")
        return JointSpaceTRRTOMPL(
            context.scene_model, context.robot_model, ik_solver, collision_threshold,
            seed=seed, collision_estimator=collision_estimator,
            object_positions=object_positions, **kwargs
        )
    elif planner_type == PlannerType.STOMP:
        return JointSpaceSTOMP(context.scene_model, context.robot_model, ik_solver, **kwargs)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")
