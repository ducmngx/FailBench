import mujoco
import numpy as np
from typing import Dict, List, Set, Tuple
import yaml
import xml.etree.ElementTree as ET
from collections import defaultdict
from io import StringIO
import copy
import os
import re

class SimpleXMLSeparator:    
    def __init__(self, scene_xml_path: str, robot_xml_path: str):

        self.scene_xml_path = scene_xml_path
        self.robot_xml_path = robot_xml_path
        
        self.scene_model = mujoco.MjModel.from_xml_string(scene_xml_path)
        self.robot_model = mujoco.MjModel.from_xml_string(robot_xml_path)
        self.scene_data = mujoco.MjData(self.scene_model)
        
        print(f"Scene: {self.scene_model.ngeom} geoms, {self.scene_model.njnt} joints")
        print(f"Robot: {self.robot_model.ngeom} geoms, {self.robot_model.njnt} joints")
        
        self.robot_names = self._get_robot_component_names()
        
        self.separation = self._separate_components()
    
    def _get_robot_component_names(self) -> Dict[str, Set[str]]:
        names = {
            'bodies': set(),
            'geoms': set(), 
            'joints': set()
        }
        
        for body_id in range(self.robot_model.nbody):
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            if body_name:
                names['bodies'].add(body_name)
        
        for geom_id in range(self.robot_model.ngeom):
            geom_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name:
                names['geoms'].add(geom_name)
        
        for joint_id in range(self.robot_model.njnt):
            joint_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                names['joints'].add(joint_name)
        
        print(f"Robot components: {len(names['bodies'])} bodies, {len(names['geoms'])} geoms, {len(names['joints'])} joints")
        return names
    
    def _separate_components(self) -> Dict[str, List[int]]:
        separation = {
            'robot_bodies': [],
            'robot_geoms': [],
            'robot_joints': [],
            'robot_collision_geoms': [],
            'environment_bodies': [],
            'environment_geoms': [],
            'environment_joints': [],
            'environment_collision_geoms': []
        }

        for body_id in range(self.scene_model.nbody):
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name in self.robot_names['bodies']:
                separation['robot_bodies'].append(body_id)
            else:
                separation['environment_bodies'].append(body_id)
        
        for geom_id in range(self.scene_model.ngeom):
            geom_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_id = self.scene_model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            is_robot_geom = (geom_name in self.robot_names['geoms'] or 
                           body_name in self.robot_names['bodies'])
            
            if is_robot_geom:
                separation['robot_geoms'].append(geom_id)
                if self._is_collision_geom(geom_id):
                    separation['robot_collision_geoms'].append(geom_id)
            else:
                separation['environment_geoms'].append(geom_id)
                if self._is_collision_geom(geom_id):
                    separation['environment_collision_geoms'].append(geom_id)
        
        for joint_id in range(self.scene_model.njnt):
            joint_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            body_id = self.scene_model.jnt_bodyid[joint_id]
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            
            is_robot_joint = (joint_name in self.robot_names['joints'] or 
                            body_name in self.robot_names['bodies'])
            
            if is_robot_joint:
                separation['robot_joints'].append(joint_id)
            else:
                separation['environment_joints'].append(joint_id)

        return separation
    
    def _is_collision_geom(self, geom_id: int) -> bool:
        group = self.scene_model.geom_group[geom_id]
        
        if group == 1 or group == 2:
            return False
        
        return True
    
    def get_robot_geoms(self, collision_only: bool = True) -> List[int]:
        if collision_only:
            return self.separation['robot_collision_geoms']
        else:
            return self.separation['robot_geoms']
    
    def get_environment_geoms(self, collision_only: bool = True) -> List[int]:
        if collision_only:
            return self.separation['environment_collision_geoms']
        else:
            return self.separation['environment_geoms']
    
    def get_robot_joints(self) -> List[int]:
        return self.separation['robot_joints']
    
    def get_environment_joints(self) -> List[int]:
        return self.separation['environment_joints']
    
    def get_robot_joint_indices(self) -> List[int]:
        indices = []
        for joint_id in self.get_robot_joints():
            qpos_addr = self.scene_model.jnt_qposadr[joint_id]
            indices.append(qpos_addr)
        return sorted(indices)
    
    def get_collision_pairs(self) -> List[Tuple[List[int], List[int]]]:
        robot_collision = self.get_robot_geoms(collision_only=True)
        env_collision = self.get_environment_geoms(collision_only=True)

        pairs = []
        
        if robot_collision and env_collision:
            pairs.append((robot_collision, env_collision))
        
        return pairs
    
    def set_robot_configuration(self, robot_config: np.ndarray):
        robot_joint_indices = self.get_robot_joint_indices()
        
        if len(robot_config) != len(robot_joint_indices):
            raise ValueError(f"Config length {len(robot_config)} != robot joints {len(robot_joint_indices)}")
        
        for i, qpos_idx in enumerate(robot_joint_indices):
            self.scene_data.qpos[qpos_idx] = robot_config[i]
        
        mujoco.mj_forward(self.scene_model, self.scene_data)
    
    def print_separation_report(self):
        print("\n" + "="*70)
        print("SIMPLE XML SEPARATION REPORT")
        print("="*70)
        
        sep = self.separation
        
        print(f"\nFILES:")
        print(f"  Scene XML: output.xml")
        print(f"  Robot XML: panda.xml")
        
        print(f"\nSEPARATION RESULTS:")
        print(f"  Robot bodies: {len(sep['robot_bodies'])}")
        print(f"  Robot geoms (all): {len(sep['robot_geoms'])}")
        print(f"  Robot collision geoms: {len(sep['robot_collision_geoms'])}")
        print(f"  Robot joints: {len(sep['robot_joints'])}")
        print(f"  Environment bodies: {len(sep['environment_bodies'])}")
        print(f"  Environment geoms (all): {len(sep['environment_geoms'])}")
        print(f"  Environment collision geoms: {len(sep['environment_collision_geoms'])}")
        print(f"  Environment joints: {len(sep['environment_joints'])}")
        
        print(f"\nROBOT COLLISION GEOMS:")
        for i, geom_id in enumerate(sep['robot_collision_geoms'][:10]):
            geom_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_id = self.scene_model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"  {i+1:2d}. Geom {geom_id:3d}: {geom_name or 'unnamed':20s} in {body_name}")
        
        if len(sep['robot_collision_geoms']) > 10:
            print(f"      ... and {len(sep['robot_collision_geoms'])-10} more")
        
        print(f"\nENVIRONMENT COLLISION GEOMS:")
        for i, geom_id in enumerate(sep['environment_collision_geoms'][:8]):
            geom_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_id = self.scene_model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"  {i+1:2d}. Geom {geom_id:3d}: {geom_name or 'unnamed':20s} in {body_name}")
        
        if len(sep['environment_collision_geoms']) > 8:
            print(f"      ... and {len(sep['environment_collision_geoms'])-8} more")
        
        print(f"\nROBOT JOINTS:")
        robot_joints = self.get_robot_joints()
        for i, joint_id in enumerate(robot_joints[:10]):
            joint_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qpos_addr = self.scene_model.jnt_qposadr[joint_id]
            print(f"  {i+1:2d}. Joint {joint_id:2d}: {joint_name:20s} qpos[{qpos_addr}]")
        
        if len(robot_joints) > 10:
            print(f"      ... and {len(robot_joints)-10} more")
    
    def test_collision_detection(self):
        collision_pairs = self.get_collision_pairs()
        
        mujoco.mj_forward(self.scene_model, self.scene_data)
        mujoco.mj_collision(self.scene_model, self.scene_data)

        from mujoco.viewer import launch
        launch(self.scene_model)

        print(f"\nTesting collision detection...")
        for pair_idx, (group1, group2) in enumerate(collision_pairs):
            print(f"  Pair {pair_idx + 1}: {len(group1)} vs {len(group2)} geoms")
            collisions = 0

            for geom1 in group1:
                g1_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
                print(g1_name)

            for i in range(self.scene_data.ncon):
                contact = self.scene_data.contact[i]
                geom1 = contact.geom1
                geom2 = contact.geom2

                if geom1 in group1 and geom2 in group2 or geom1 in group2 and geom2 in group1:
                    print(f"    Contact: Geom {geom1} <-> Geom {geom2}")
                    collisions += 1

            print(f"    Found {collisions} contacts")
    
    def debug_name_matching(self):
        print(f"\n" + "="*50)
        print("DEBUG: NAME MATCHING")
        print("="*50)
        
        print(f"\nROBOT BODY NAMES:")
        for name in sorted(self.robot_names['bodies']):
            print(f"  '{name}'")
        
        print(f"\nROBOT GEOM NAMES:")
        for name in sorted(self.robot_names['geoms']):
            print(f"  '{name}'")
        
        print(f"\nROBOT JOINT NAMES:")
        for name in sorted(self.robot_names['joints']):
            print(f"  '{name}'")
        
        print(f"\nSCENE BODIES WITH 'finger':")
        for body_id in range(self.scene_model.nbody):
            body_name = mujoco.mj_id2name(self.scene_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and 'finger' in body_name.lower():
                is_robot = body_name in self.robot_names['bodies']
                status = "ROBOT" if is_robot else "ENVIRONMENT"
                print(f"  '{body_name}' -> {status}")

class RunTest:
    def __init__(self, scene_xml, robot_xml):
        try:
            print("SIMPLE XML SEPARATOR TEST")
            print("=" * 50)
            
            separator = SimpleXMLSeparator(scene_xml, robot_xml)
            
            separator.debug_name_matching()
            separator.print_separation_report()
            separator.test_collision_detection()
            
            print(f"\n" + "="*70)
            print("USAGE EXAMPLES")
            print("="*70)
            
            robot_geoms = separator.get_robot_geoms(collision_only=True)
            env_geoms = separator.get_environment_geoms(collision_only=True)
            robot_joints = separator.get_robot_joints()
            collision_pairs = separator.get_collision_pairs()
            
            print(f"\n# Get components:")
            print(f"robot_collision_geoms = {robot_geoms[:5]}...  # {len(robot_geoms)} total")
            print(f"env_collision_geoms = {env_geoms[:3]}...  # {len(env_geoms)} total")
            print(f"robot_joints = {robot_joints[:5]}...  # {len(robot_joints)} total")            
            print(f"\n✓ Simple separation completed!")

            body_name = mujoco.mj_id2name(separator.scene_model, mujoco.mjtObj.mjOBJ_BODY, 0)

            print(f"Selected body in scene: {body_name}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

class MergeXML:

    def __init__(self):

        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        root = ET.Element('mujoco', model=config['model_name'])
        root.append(ET.Element('compiler', meshdir="assets"))
        root.append(ET.Element('include', file=config['env_path']))
        all_assets = []
        all_defaults = []
        all_tendons = []
        all_equalities = []
        all_sensors = []
        all_actuators = []
        worldbody = ET.Element('worldbody')
        entities = [config["robot"]] if "robot" in config else []
        entities += config.get("Objects", [])

        for ent in entities:
            body, assets, defaults, sensors, tendons, equalities, actuators = self.process_entity(ent)
            worldbody.append(body)
            all_assets.extend(assets)
            all_defaults.extend(defaults)
            all_tendons.extend(tendons)
            all_equalities.extend(equalities)
            all_sensors.extend(sensors)
            all_actuators.extend(actuators)

        if all_defaults:
            defaults_tag = ET.Element("default")
            for elem, src in all_defaults:
                defaults_tag.append(elem)
            root.insert(2, defaults_tag)

        if all_assets:
            assets_tag = ET.Element("asset")
            assets_dict = {ET.tostring(elem, encoding="unicode"): elem for elem, _ in all_assets}
            for elem in assets_dict.values():
                assets_tag.append(elem)
            root.insert(3, assets_tag)
        root.insert(4, worldbody)

        if all_sensors:
            sensor_tag = ET.Element("sensor")
            for elem, src in all_sensors:
                sensor_tag.append(elem)
            root.append(sensor_tag)
        if all_tendons:
            tendon_tag = ET.Element("tendon")
            for elem, src in all_tendons:
                tendon_tag.append(elem)
            root.append(tendon_tag)
        if all_equalities:
            equality_tag = ET.Element("equality")
            for elem, src in all_equalities:
                equality_tag.append(elem)
            root.append(equality_tag)
        if all_actuators:
            actuator_tag = ET.Element("actuator")
            for elem, src in all_actuators:
                actuator_tag.append(elem)
            root.append(actuator_tag)

        formated_xml = self.format_xml(root)
        with open("output.xml", "w") as f:
            f.write(formated_xml)

    def find_body_by_name(self, xml_root, target_name=None):
        if target_name is not None:
            for body in xml_root.findall(".//body"):
                if body.attrib.get("name") == target_name:
                    return copy.deepcopy(body)
        worldbody = xml_root.find(".//worldbody")
        if worldbody is not None and len(worldbody) > 0:
            for body in worldbody.findall("body"):
                return copy.deepcopy(body)
        return None
    
    def collect_defaults(self, xml_root):
        top_defaults = xml_root.find('default')
        if top_defaults is None:
            return []
        collected = []
        for child in top_defaults:
            if child.tag == "default":
                collected.append(copy.deepcopy(child))
        return collected
    
    def collect_tendons(self, xml_root):
        tendons = []
        for tendons_tag in xml_root.findall(".//tendon"):
            for child in tendons_tag:
                tendons.append(copy.deepcopy(child))
        return tendons
    
    def collect_equalities(self, xml_root):
        equalities = []
        for equalities_tag in xml_root.findall(".//equality"):
            for child in equalities_tag:
                equalities.append(copy.deepcopy(child))
        return equalities
    
    def collect_assets(self, xml_root):
        assets = []
        for assets_tag in xml_root.findall(".//asset"):
            for child in assets_tag:
                assets.append(copy.deepcopy(child))
        return assets
    
    def update_body_pose(self, body_elem, yaml_entity):
        if "pos" in yaml_entity:
            body_elem.set("pos", str(yaml_entity["pos"]))
        if "quat" in yaml_entity:
            body_elem.set("quat", str(yaml_entity["quat"]))

    def process_entity(self, entity):
        xml_path = entity["xml_path"]
        assert os.path.exists(xml_path), f"{xml_path} not found"
        tree = ET.parse(xml_path)
        xml_root = tree.getroot()
        body = self.find_body_by_name(xml_root, entity["name"])

        if body is None:
            raise Exception(f"Body {entity['name']} not found in {xml_path}")
        
        self.update_body_pose(body, entity)
        assets = [(copy.deepcopy(a), xml_path) for a in self.collect_assets(xml_root)]
        defaults = [(copy.deepcopy(d), xml_path) for d in self.collect_defaults(xml_root)]
        sensors = []
        tendons = [(copy.deepcopy(t), xml_path) for t in self.collect_tendons(xml_root)]
        equalities = [(copy.deepcopy(e), xml_path) for e in self.collect_equalities(xml_root)]
        actuators = []
        sensor_tag = xml_root.find("sensor")

        if sensor_tag is not None:
            for child in sensor_tag:
                sensors.append((copy.deepcopy(child), xml_path))

        for actuator_tag in xml_root.findall(".//actuator"):
            for child in actuator_tag:
                actuators.append((copy.deepcopy(child), xml_path))

        for subent in entity.get("bodies", []):
            sub_body, sub_assets, sub_defaults, sub_sensors, sub_tendons, sub_equalities, sub_actuators = self.process_entity(subent)
            body.append(sub_body)
            assets += sub_assets
            defaults += sub_defaults
            tendons += sub_tendons
            equalities += sub_equalities
            sensors += sub_sensors
            actuators += sub_actuators

        return body, assets, defaults, sensors, tendons, equalities, actuators
    
    def format_xml(self, elem):
        ET.indent(elem, space="  ", level=0)
        xml_str = ET.tostring(elem, encoding="unicode")
        xml_str = xml_str.replace('><', '>\n<')

        for tag in ['asset', 'default', 'worldbody', 'sensor', 'tendon', 'equality', 'actuator']:
            xml_str = xml_str.replace(f'</{tag}>', f'</{tag}>\n')
        match = re.search(r'(<worldbody>.*?</worldbody>)', xml_str, re.DOTALL)

        if match:
            wb_block = match.group(1)
            wb_block_new = re.sub(r'(</body>)\n(\s*)(<body )', r'\1\n\n\2\3', wb_block)
            xml_str = xml_str.replace(wb_block, wb_block_new)
        xml_str = xml_str.replace('?>\n<assets', '?>\n\n<assets')
        xml_str = re.sub(r'\n{3,}', '\n\n', xml_str)

        if not xml_str.endswith('\n'):
            xml_str += '\n'
        return xml_str
    
    def get_xml(self):
        return "output.xml"

class main:
    def __init__(self):
        merge = MergeXML()

        scene_xml = self.rename_geoms_with_mesh(merge.get_pretty_xml(), "_in_scene")
        robot_xml = self.rename_geoms_with_mesh("panda.xml", "_in_robot")

        RunTest(scene_xml, robot_xml)

    def rename_geoms_with_mesh(self, input_file, suffix):
        tree = ET.parse(input_file)
        root = tree.getroot()

        name_counts = defaultdict(int)
        used_names = set()

        for geom in root.findall(".//geom"):
            geom_class = geom.get("class")
            if geom_class == "visual" or geom_class == None:
                continue
            if geom.get("name") != None:
                continue

            if geom.get("mesh"):
                prefix = geom.get("mesh")
            elif geom_class != "collision" and "collision" in geom_class:
                prefix = geom_class
            else:
                prefix = None
            
            base_name = f"{prefix}{suffix}"
            new_name = base_name

            count = name_counts[base_name]
            while new_name in used_names:
                count += 1
                new_name = f"{base_name}_{count}"
            name_counts[base_name] = count
            used_names.add(new_name)

            geom.set("name", new_name)

        xml_str = ET.tostring(root, encoding="unicode")

        return xml_str

main()
