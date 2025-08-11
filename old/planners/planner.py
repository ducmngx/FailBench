class BasePlanner:
    def __init__(self):
        pass


class AStarSearch:
    """
    This planner basically takes in the map and plans a path from start -> goal using the search method.
    It has a heuristic function that helps in planning.
    After the plan in map coordinates is done, it should produce a sequence of local goals in mjc world coordinates, 
    or relative coordinates so that the controller can do inverse kinematics on that sequence of local goals to get to the global goal.
    
    """

    def __init__(self, map):
        self.map = map
    
    def heuristic_function(self):
        pass

    def search(self, start_coordinate, goal_coordinate):
        pass