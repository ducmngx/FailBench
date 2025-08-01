# planner/__init__.py - Main package interface
from .algorithms.RRTplanner import *
from .collision.collision_checker import *
from .kinematics.inverse_kinematics import *
from .collision.geometry_utils import *
from .core.planning_context import *
from .examples.panda_planner_demo import *

# High-level interface
def create_planner(scene_xml, robot_xml):
    context = PlanningContext(scene_xml, robot_xml)
    collision_checker = CollisionChecker(context.scene_model, context.robot_model)
    ik_solver = IKSolver(context.robot_model)
    return JointSpaceRRT(context.scene_model, context.robot_model, ik_solver, 0.03)