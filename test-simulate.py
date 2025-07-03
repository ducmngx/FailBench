from robocasa.environments.kitchen.kitchen import Kitchen
import mujoco
from mujoco.viewer import launch

env = Kitchen(
    layout_ids=8,
    style_ids=5
)

xml_str = env.sim.model.get_xml()
model = mujoco.MjModel.from_xml_string(xml_str)
data = mujoco.MjData(model)

viewer = launch(model, data)