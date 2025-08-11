
class Robot:
    """
    """
    def __init__(self, name):
        self.name = name
        self.joints = None
        self.controller = None
        self.sensors = None


    def setJoints(self, list_of_actuators: list):
        raise NotImplementedError
    
    def setController(self, controller: object):
        raise NotImplementedError
    
    def setSensors(self, list_of_sensors: list):
        raise NotImplementedError
    
    def injectFailure(self):
        raise NotImplementedError
    
    