import mujoco

class Mapper:

    """
    All measurements are in cm.
    """
    def __init__(self, mjmodel: mujoco.mjmodel, mjdata: mujoco.mjdata, static_map: bool, static_resolution: int, robot_max_height: int):
        self.mjmodel = mjmodel
        self.mjdata = mjdata

        self.is_map_static = static_map
        self.resolution_of_static_map = static_resolution

        self.map = None

        self.initializeMap()

        self.fillMap()


    """
    This function initializes the map according to the parameters and environment
    """
    def initializeMap(self):
        # two methods to do this - pure xml or python object. For now lets hard code pure xml
        
        pass


    """
    The core function that takes in all the parameters and runs the mapping algorithm. 
    """
    def fillMap(self):

        pass


    def getMap(self):
        return self.map