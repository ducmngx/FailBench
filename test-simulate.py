from robocasa.environments.kitchen.kitchen import Kitchen
import mujoco
from mujoco.viewer import launch
import os

env = Kitchen(
    layout_ids=8,
    style_ids=5
)

xml_str = env.sim.model.get_xml()
model = mujoco.MjModel.from_xml_string(xml_str)
output_path = "dumped_kitchen.xml"
mujoco.mj_saveLastXML(output_path, model)
print(f"Model saved to: {os.path.abspath(output_path)}")

'''
data = mujoco.MjData(model)
viewer = launch(model, data)
'''
