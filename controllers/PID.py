'''
incomplete class.
'''

import numpy as np
from typing import List

class PID:
    def __init__(self, robot, Kps: List[float], Kis: List[float], Kds: List[float], dt: float = 0.01):
        self.Kps = Kps
        self.Kis = Kis
        self.Kds = Kds
        self.dt = dt

    def run(self, reference, output):
        error = reference - output
        proportional = self.Kps * error

        derivative = error / self.dt