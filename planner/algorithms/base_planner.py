import numpy as np
import mujoco
from abc import ABC, abstractmethod
from typing import List, Optional, Callable


from .mapping.map import Map

class AbstractBasePlanner(ABC):
    """
    Abstract Base Planner that is the parent of all planners.
    """
    def __init__(self, model: mujoco.MjModel):
        self.model = model
    
    @abstractmethod
    def plan(
        self, 
        start_config: Optional[np.ndarray]=None, 
        end_config: Optional[np.ndarray]=None, 
        start_pos: Optional[np.ndarray]=None,
        end_pos: Optional[np.ndarray]=None,
        end_pose: Optional[np.ndarray]=None
        ) -> Optional[List[np.ndarray]]:
        """
        Plan using underlying planner. 

        Args:
            start_config: Starting joint configuration
            end_config: Ending joint configuration 
            start_pos: Starting translational configuration
            end_pos: Ending translational configuration
            end_pose: Ending rotational configuration

        Returns:
            Path from start to end, or None if failed
        """
        pass


class AbstractGraphPlanner(AbstractBasePlanner, ABC):

    def __init__(self, model: mujoco.MjModel, map: Map):
        super().__init__(model)
        self.map = map
    
    def _euclidean_dist(self, cur: np.ndarray, goal: np.ndarray):
        return np.sqrt(((goal - cur)**2).sum(-1))

    def drop_path_points(self, plan: List[np.ndarray], skip: int):
        """
        Given a list of coordinates (map or world), drop the coordinates and decrease the resolution of the plan according to scale.
        Note: the path is already in the user resolution of the map. This function will further decrease the resolution.
        New path will be size 1/skip * |plan|.
        """
        return plan[slice(None,None, skip)]


