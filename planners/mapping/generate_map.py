import mujoco
import numpy as np
import os

import xml.etree.ElementTree as ET
import trimesh

from .map import Map


class Mapper:
    INCLUDE_KEYWORDS = [
        
    ]

    SKIP_KEYWORDS = [
        "room_g0", "front_group_g0", "toaster_main_group_g0", "sink_left_group_g0", "plant_left_group_g1", "window",
        "coffee_machine_main_group_g", "group_3_right", "dishwasher_left_group_g", "fridge_right_group_g1", "oven_right_group_g",
        "microwave_right_group_g", "stovetop_main_group_g", "counter_1_left_group_top_left_visual", "counter_1_left_group_top_right_visual",
        "hood_main_group_g", "bottom_right_group_1_door_door", "bottom_right_group_2_door_door", "cab_"
    ]

    STACK_KEYWORDS = [
        "_main_group_1_left_door_door",
        "_main_group_1_right_door_door",
        "_main_group_1_door_door"
    ]

    REGION_LABELS = {
        "counter": 2,
        "sink": 3,
        "stove": 4,
        "cabinet": 5,
        "drawer": 6,
        "fridge": 7,
        "oven": 8,
        "table": 9,
    }

    def __init__(self, mjmodel, mjdata, xml_path, static_scale=100, robot_max_height=5):
        self.mjmodel = mjmodel
        self.mjdata = mjdata
        self.static_scale = static_scale *2
        self.robot_max_height = robot_max_height
        self.map = Map(type="grid", init_resolution=self.static_scale)
        self.label_map = None
        self.object_poses = {}
        self.mesh_dir = None

        if xml_path:
            self.mesh_files = self.parse_mesh_files_from_xml(xml_path)
        self.geoms_info = self._collect_geoms()
        self._compute_bounds_and_fill_maps()
        self._detect_objects()

    def parse_mesh_files_from_xml(self, xml_path):
        mesh_dict = {}
        tree = ET.parse(xml_path)
        root = tree.getroot()
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
            bounds = mesh.bounds  # (min, max)
            size = bounds[1] - bounds[0]
            return size.tolist()
        except Exception as e:
            print(f"Error loading mesh '{mesh_name}': {e}")
            return None
        
    def _should_include_geom(self, name):
        if not name:
            return False
        name = name.lower()
        if any(kw in name for kw in self.SKIP_KEYWORDS) or ("stack_" in name and any(kw in name for kw in self.STACK_KEYWORDS)):
            return False
        if any(kw in name for kw in self.INCLUDE_KEYWORDS):
            return True
        return True
    
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
            #if center[2] - size[2] > self.robot_max_height:
            #    continue
            label_id = 1
            for keyword, region_id in self.REGION_LABELS.items():
                if keyword in name.lower():
                    label_id = region_id
                    break
            info = {
                'id': i,
                'name': name,
                'center': center,
                'size': size,
                'type': geom_type,
                'label_id': label_id
            }
            geoms_info.append(info)
        return geoms_info
    
    def _compute_bounds_and_fill_maps(self):
        xs, ys = [], []
        # --- Store patches and patch-to-name mapping ---
        self.patches = []
        self.patch_to_name = {}
        for geom in self.geoms_info:
            if ("floor" in geom['name']):
                center, size = geom['center'], geom['size']
                xs.extend([center[0] - size[0], center[0] + size[0]])
                ys.extend([center[1] - size[1], center[1] + size[1]])
        if not xs or not ys:
            raise ValueError(":x: No valid geoms found for mapping.")
        self.x_min = min(xs)
        self.x_max = max(xs)
        self.y_min = min(ys)
        self.y_max = max(ys)
        self.width = int(np.ceil((self.x_max - self.x_min) * self.static_scale))
        self.height = int(np.ceil((self.y_max - self.y_min) * self.static_scale))

        print(f"generate_map max_mins: x:({self.x_min}, {self.x_max}), y:({self.y_min}, {self.y_max})")
        self.map.set_size(self.height, self.width) 
        self.map.set_world_mins(self.x_min, self.y_min)
        self.label_map = np.zeros((self.height, self.width), dtype=np.uint8)

        for geom in self.geoms_info:
            if ("floor" in geom['name']):
                continue
            label_id = geom['label_id']
            center, size = geom['center'], geom['size']
            x0, x1 = center[0] - size[0], center[0] + size[0]
            y0, y1 = center[1] - size[1], center[1] + size[1]
            col0 = int(np.floor((x0 - self.x_min) * self.static_scale))
            col1 = int(np.ceil((x1 - self.x_min) * self.static_scale))
            row0 = int(np.floor((y0 - self.y_min) * self.static_scale))
            row1 = int(np.ceil((y1 - self.y_min) * self.static_scale))
            if(col1 - col0)<20 or (row1 - row0)<20:
                continue
            col0, col1 = max(0, col0), min(col1, self.width)
            row0, row1 = max(0, row0), min(row1, self.height)
            if row1 > row0 and col1 > col0:

                self.map.set_rect_obstacles(col0, row0, col1, row1)

                # self.map[row0:row1, col0:col1] = True
                self.label_map[row0:row1, col0:col1] = label_id
                # --- Store the rectangle patch and map it to geom name ---
                # rect = patches.Rectangle((col0, row0), col1-col0, row1-row0,
                #                         edgecolor='red', facecolor='none', linewidth=0.5)
                # ax.add_patch(rect)
                # self.patches.append(rect)
                # self.patch_to_name[rect] = geom['name']

    def _detect_objects(self):
        for i in range(self.mjmodel.nbody):
            name = mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.lower().startswith("cup"):
                pos = self.mjdata.xpos[i]
                self.object_poses[name] = tuple(pos)

    def get_map(self):
        return self.map
    
    def get_label_map(self):
        return self.label_map
    
    def get_region_label(self, px, py):
        if 0 <= py < self.label_map.shape[0] and 0 <= px < self.label_map.shape[1]:
            return self.label_map[py, px]
        return 0
    
    def get_object_pose(self, name):
        return self.object_poses.get(name, None)
    



if __name__ == "__main__":
    xml_path = "/Users/saghani/Workspace/Research/GenAISim/assets/environments/dumped_kitchen_noassets.xml"
    mjmodel = mujoco.MjModel.from_xml_path(xml_path)
    mjdata = mujoco.MjData(mjmodel)
    mapper = Mapper(mjmodel, mjdata, xml_path, static_scale=200, robot_max_height=5)

    mapper.get_map().visualize()

    