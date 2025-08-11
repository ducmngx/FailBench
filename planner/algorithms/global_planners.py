from typing import List, Optional, Callable
import numpy as np

import heapq
import itertools

from .base_planner import AbstractGraphPlanner
from .mapping.map import Map

class A_StarPlanner(AbstractGraphPlanner):

    def __init__(self, model, map: Map, heuristic_fn: Optional[Callable[..., float]]):
        super().__init__(model, map)
        self.heuristic_fn = heuristic_fn if heuristic_fn is not None else self._euclidean_dist
        self.counter = itertools.count()

    def plan(
        self, 
        start_config: Optional[np.ndarray]=None, 
        end_config: Optional[np.ndarray]=None, 
        start_pos: Optional[np.ndarray]=None,
        end_pos: Optional[np.ndarray]=None,
        end_pose: Optional[np.ndarray]=None
        ) -> Optional[List[np.ndarray]]:

        assert start_pos is not None and end_pos is not None
        heap = []
        visited = []
        end_pos_ = (end_pos[0], end_pos[1])
        heapq.heappush(heap, (0, next(self.counter), [(start_pos[0].item(), start_pos[1].item())]))

        while(len(heap) > 0):
            cost, _, path = heapq.heappop(heap)
            cur_pos = path[-1]
            if cur_pos in visited:
                continue
            if end_pos_ == cur_pos:
                return path
            neighbors = self.map.getNeighbors(cur_pos)
            for n_ in neighbors:
                heapq.heappush(heap, (cost+1 + self.heuristic_fn(n_, end_pos), next(self.counter), path + [n_]))
            
            visited.append(cur_pos)

        return None            

    
def test():
    """
    Testing code for A_StarPlanner, Mapper, and Map.
    """
    import mujoco
    from .mapping.generate_map import Mapper
    xml_path = "/Users/saghani/Workspace/Research/GenAISim/assets/environments/dumped_kitchen_noassets.xml"
    mjmodel = mujoco.MjModel.from_xml_path(xml_path)
    mjdata = mujoco.MjData(mjmodel)
    mapper = Mapper(mjmodel, mjdata, xml_path, static_scale=100, robot_max_height=5)

    map = mapper.get_map()
    map.change_resolution(5)        # downscale x100/5 (x20)
    # map.visualize()

    planner = A_StarPlanner(mjmodel, map, heuristic_fn=None)

    path = planner.plan(start_pos=np.array([20,5]), end_pos=np.array([20, 50]))
    # print(path)

    path_decresed_resolution = planner.decrease_path_resolution(path, 3)
    # print(path_decresed_resolution)

    path_env_coords = map.convert_to_world_coordinates(path_decresed_resolution)
    print(path_env_coords)

    map.visualize_path(path_env_coords)

if __name__ == "__main__":
    test()