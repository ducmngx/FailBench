# provides a similar interface to ROS differential drive package - http://wiki.ros.org/diff_drive_controller
# mathematical equations taken from https://en.wikipedia.org/wiki/Differential_wheeled_robot
# abstract class Controller is created so we can use polymorphism with other controllers

from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    @abstractmethod
    def control(self, *args):
        pass

class DifferentialDriveController(BaseController):

    # TODO: need to include velocity and acceleration limits
    # left_wheel, right_wheel is similar to ROS left/right_wheel, ie. it can be a list
    def __init__(self, wheel_radius, vehicle_width, left_wheel_ctrl, right_wheel_ctrl):
        self.wheel_radius = wheel_radius
        self.vehicle_width = vehicle_width
        self.left_wheel = left_wheel_ctrl
        self.right_wheel = right_wheel_ctrl


    def control(self, *args):
        assert len(args) == 2, "Differential Drive Controller expects 2 arguments - linear velocity and angular velocity"
        linear = args[0]
        angular = args[1]

        w_r = (linear - angular*(self.vehicle_width/2))/self.wheel_radius
        w_l = (linear + angular*(self.vehicle_width/2))/self.wheel_radius

        v_r = self.wheel_radius*w_r
        v_l = self.wheel_radius*w_l

        if isinstance(self.left_wheel, np.ndarray):
            for i in range(len(self.left_wheel)):
                self.left_wheel[i] = v_l
        else:
            self.left_wheel = v_l

        if isinstance(self.right_wheel, np.ndarray):
            for i in range(len(self.right_wheel)):
                self.right_wheel[i] = v_r
        else:
            self.right_wheel = v_r
    
