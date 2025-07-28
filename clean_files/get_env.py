from robocasa.environments.kitchen.kitchen import Kitchen
from robosuite.environments.manipulation.pick_place import PickPlace
import os
import mujoco
from mujoco.viewer import launch


model = mujoco.MjModel.from_xml_path("output.xml")
data = mujoco.MjData(model)
launch(model, data)