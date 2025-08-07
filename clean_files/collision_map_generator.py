import mujoco
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
import trimesh
fig, ax = plt.subplots(figsize=(8, 8), dpi=75)

class CollisionMapper:

    x = ["front", "left", "right", "main", "island"]
    y = ["1", "2", "3"]

    SKIP_KEYWORDS = [
        "room_g0", "front_group_g0",
        *[("toaster_" + i + "_group_g") for i in x],
        *[("sink_" + i + "_group_g") for i in x],
        *[("plant_" + i + "_group_g") for i in x],
        *[("coffee_machine_" + i + "_group_g") for i in x],
        *[("dishwasher_" + i + "_group_g") for i in x],
        *[("fridge_" + i + "_group_g") for i in x],
        *[("oven_" + i + "_group_g") for i in x],
        *[("stovetop_" + i + "_group_g") for i in x],
        *[("stool_" + k + "_island_group_g0") for k in y],
        *[("stool_" + k + "_stool_group_g0") for k in y],
        "window",
        *[("microwave_" + i + "_group_g") for i in x],
        "counter_1_left_group_top_left_visual", "counter_1_left_group_left_left_visual","counter_1_left_group_top_right_visual", "counter_1_left_left_group_top_right_visual",
        "counter_1_left_group_top_visual", "counter_1_right_group_top_left_visual", "counter_1_right_group_top_right_visual", "counter_1_right_group_top_visual", "counter_2_right_group_top_visual",
        "counter_main_main_group_top_visual", "counter_right_main_group_top_visual", "counter_left_group_top_left_visual",
        "counter_corner_right_group_top_1",
        "shelves_right_group_shelf_0_shelf", "shelves_right_group_shelf_1_shelf", "shelves_left_group_shelf_0_shelf", "shelves_left_group_shelf_1_shelf",
        "side_bottom_right_group_3_right_door_door",
        "island_left_group_top_visual",
        *[("hood_" + i + "_group_g") for i in x],
        "bottom_left_group_1_door_door", "bottom_left_group_2_door_door", "bottom_right_group_1_door_door", "bottom_right_group_2_door_door",
        "cab_",
        "top_main_group_right_door_door", "top_main_group_left_door_door", 
        *[("knife_block_" + i + "_group_g") for i in x],
        *[("paper_towel_" + i + "_group_g") for i in x],
        *[("utensil_holder_" + i + "_group_g") for i in x]
    ]

    STACK_KEYWORDS = [
        "_main_group_1_left_door_door",
        "_main_group_1_right_door_door",
        "_main_group_1_door_door",
        "_island_group_1_right_door_door",
        "_island_group_1_left_door_door",
        "_right_group_1_left_door_door",
        "_right_group_1_right_door_door",
        "_right_group_1_door_door",
        "_left_group_1_left_door_door",
        "_left_group_1_right_door_door",
        "_left_group_1_door_door"
    ]

    def __init__(self, mjmodel, mjdata, xml_string, static_scale=100, robot_max_height=5, floor_padding = 1):
        self.mjmodel = mjmodel
        self.mjdata = mjdata
        self.static_scale = static_scale
        self.robot_max_height = robot_max_height
        self.floor_padding = floor_padding
        self.map, self.mesh_dir = None

        if xml_string:
            self.mesh_files = self.parse_mesh_files_from_xml(xml_string)
        self.geoms_info = self._collect_geoms()

        self._compute_bounds_and_fill_maps()

    def parse_mesh_files_from_xml(self, xml_string):
        mesh_dict = {}
        root = ET.fromstring(xml_string)
        for mesh in root.findall(".//mesh"):
            name = mesh.attrib.get("name")
            file = mesh.attrib.get("file")
            if name and file:
                mesh_dict[name] = file
        return mesh_dict
    
    def extract_mesh_bounds(self, mesh_name):
        if mesh_name not in self.mesh_files:
            print(f"Mesh '{mesh_name}' not found in model.")
            return None
        file_path = self.mesh_files[mesh_name]

        if self.mesh_dir and not os.path.isabs(file_path):
            file_path = os.path.join(self.mesh_dir, file_path)

        try:
            mesh = trimesh.load(file_path, force='mesh')
            if len(mesh.vertices) > 1e6:
                print(f"Skipping mesh '{mesh_name}' due to too many vertices.")
                return None
            bounds = mesh.bounds
            size = bounds[1] - bounds[0]
            return size.tolist()
        except Exception as e:
            print(f"Error loading mesh '{mesh_name}': {e}")
            return None
        
    def _should_include_geom(self, name):
        if not name:
            return False
        name = name.lower()
        return any(kw in name for kw in self.SKIP_KEYWORDS) or ("stack_" in name and any(kw in name for kw in self.STACK_KEYWORDS))
    
    def _collect_geoms(self):
        mujoco.mj_forward(self.mjmodel, self.mjdata)
        geoms_info = []
        for i in range(self.mjmodel.ngeom):
            name = mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_GEOM, i)
            if not (self._should_include_geom(name) or (name and "floor" in name.lower())):
                continue
            geom_type = self.mjmodel.geom_type[i]
            center = [float(s) for s in self.mjdata.geom_xpos[i]]
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = self.mjmodel.geom_dataid[i]
                mesh_name = mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
                size = self.extract_mesh_bounds(mesh_name)
                if size is None:
                    print(f"Skipping geom '{name}' with mesh '{mesh_name}' due to failed mesh loading.")
                    continue
            else:
                size = [float(s) for s in self.mjmodel.geom_size[i]]
            if(len(size) == 1):
                size = [size[0], size[0], size[0]]
            if(len(size) == 2):
                size = [size[0], size[1], size[1]]
            info = {
                'id': i,
                'name': name,
                'center': center,
                'size': size,
                'type': geom_type,
            }
            geoms_info.append(info)
        return geoms_info
    
    def _compute_bounds_and_fill_maps(self):
        xs, ys = [], []
        self.patches = []
        self.patch_to_name = {}
        for geom in self.geoms_info:
            if ("floor" in geom['name']):
                center, size = geom['center'], geom['size']
                xs.extend([center[0] - size[0], center[0] + size[0] + 2 * self.floor_padding])
                ys.extend([center[1] - size[1], center[1] + size[1] + 2 * self.floor_padding])

        if not xs or not ys:
            raise ValueError(":x: No valid geoms found for mapping.")
        
        self.x_min = min(xs)
        self.x_max = max(xs)
        self.y_min = min(ys)
        self.y_max = max(ys)
        self.width = int(np.ceil((self.x_max - self.x_min) * 2 * self.static_scale))
        self.height = int(np.ceil((self.y_max - self.y_min) * 2 * self.static_scale))
        self.map = np.zeros((self.height, self.width), dtype=bool)

        for geom in self.geoms_info:
            center, size = geom['center'], geom['size']
            if "floor" in geom['name']:
                x0, x1 = center[0] - size[0], center[0] + size[0] + 2 * self.floor_padding
                y0, y1 = center[1] - size[1], center[1] + size[1] + 2 * self.floor_padding
            else:
                x0, x1 = center[0] - size[0] + self.floor_padding, center[0] + size[0] + self.floor_padding
                y0, y1 = center[1] - size[1] + self.floor_padding, center[1] + size[1] + self.floor_padding

            col0 = int(np.floor((x0 - self.x_min) * 2 * self.static_scale))
            col1 = int(np.ceil((x1 - self.x_min) * 2 * self.static_scale))
            row0 = int(np.floor((y0 - self.y_min) * 2 * self.static_scale))
            row1 = int(np.ceil((y1 - self.y_min) * 2 * self.static_scale))

            if(col1 - col0)<(self.static_scale/5) or (row1 - row0)<(self.static_scale/5):
                continue

            col0, col1 = max(0, col0), min(col1, self.width)
            row0, row1 = max(0, row0), min(row1, self.height)

            if row1 > row0 and col1 > col0:
                self.map[row0:row1, col0:col1] = True

                if "floor" in geom['name']:
                    rect = patches.Rectangle((col0, row0), col1-col0, row1-row0, fill=False, edgecolor='red', linewidth=0.5)
                else:
                    rect = patches.Rectangle((col0, row0), col1-col0, row1-row0, edgecolor='red', facecolor='none', linewidth=0.5)
                    
                ax.add_patch(rect)
                self.patches.append(rect)
                self.patch_to_name[rect] = geom['name']
                
    def get_map(self):
        return self.map

from robocasa.environments.kitchen.kitchen import Kitchen

env = Kitchen(
    robots=[],
    layout_ids=8,
    style_ids=5
)

xml_str = env.sim.model.get_xml()
mjmodel = mujoco.MjModel.from_xml_string(xml_str)
mjdata = mujoco.MjData(mjmodel)

mapper = CollisionMapper(mjmodel, mjdata, xml_str, static_scale=100, robot_max_height=5)
ax.imshow(mapper.get_map(), cmap="gray_r", origin="lower", interpolation="nearest")
ax.set_title("Occupancy Map with Floor Outline")
ax.set_xlabel("X (grid cells)")
ax.set_ylabel("Y (grid cells)")
plt.tight_layout()
plt.savefig("fixture_map_with_floor_box.png")

annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(rect, name, event):
    annot.xy = (event.xdata, event.ydata)
    annot.set_text(name)
    annot.get_bbox_patch().set_facecolor('yellow')
    annot.set_visible(True)
def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        for rect in mapper.patches:
            contains, _ = rect.contains(event)
            if contains:
                name = mapper.patch_to_name[rect]
                if "floor" in name.lower():
                    continue
                update_annot(rect, name, event)
                fig.canvas.draw_idle()
                return
    if vis:
        annot.set_visible(False)
        fig.canvas.draw_idle()

current_box = {"rect": None}

def on_key(event):
    if event.inaxes == ax:
        if event.key == 'i':
            x = int(event.xdata)
            y = int(event.ydata)
            print(f"Clicked at pixel: ({x}, {y})")
            
            world_x = mapper.x_min + (x / (2 * mapper.static_scale))
            world_y = mapper.y_min + (y / (2 * mapper.static_scale))
            print(f"Clicked at world coordinates: ({world_x:.3f}, {world_y:.3f})")

            if current_box["rect"] is not None:
                current_box["rect"].remove()

            box_size = 100
            rect = patches.Rectangle((x - box_size // 2, y - box_size // 2), box_size, box_size,
                            linewidth=1.5, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
            current_box["rect"] = rect
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()
