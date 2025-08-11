import yaml
import xml.etree.ElementTree as ET
import copy
import os
import re

# Returns deep copy of body in XML, using passed in name
def find_body_by_name(xml_root, target_name=None):
    if target_name is not None:
        for body in xml_root.findall(".//body"):
            if body.attrib.get("name") == target_name:
                return copy.deepcopy(body)
    worldbody = xml_root.find(".//worldbody")
    if worldbody is not None and len(worldbody) > 0:
        for body in worldbody.findall("body"):
            return copy.deepcopy(body)
    return None

# Collects all asset tags in XML
def collect_assets(xml_root):
    assets = []
    for assets_tag in xml_root.findall(".//asset"):
        for child in assets_tag:
            assets.append(copy.deepcopy(child))
    return assets

# Collects all defaults in XML
def collect_defaults(xml_root):
    top_defaults = xml_root.find('default')
    if top_defaults is None:
        return []

    collected = []
    for child in top_defaults:
        if child.tag == "default":
            collected.append(copy.deepcopy(child))
    return collected

# Collects all tendons in XML
def collect_tendons(xml_root):
    tendons = []
    for tendons_tag in xml_root.findall(".//tendon"):
        for child in tendons_tag:
            tendons.append(copy.deepcopy(child))
    return tendons

# Collects all equalities in XML
def collect_equalities(xml_root):
    equalities = []
    for equalities_tag in xml_root.findall(".//equality"):
        for child in equalities_tag:
            equalities.append(copy.deepcopy(child))
    return equalities

# Updates body's pose and quaternion rotation if specified in YAML
def update_body_pose(body_elem, yaml_entity):
    if "pos" in yaml_entity:
        body_elem.set("pos", str(yaml_entity["pos"]))
    if "quat" in yaml_entity:
        body_elem.set("quat", str(yaml_entity["quat"]))

# Process all the assets/defaults/sensors/etc. in an XML file
def process_entity(entity):
    xml_path = entity["xml_path"]
    assert os.path.exists(xml_path), f"{xml_path} not found"
    tree = ET.parse(xml_path)
    xml_root = tree.getroot()

    body = find_body_by_name(xml_root, entity["name"])
    if body is None:
        raise Exception(f"Body {entity['name']} not found in {xml_path}")
    update_body_pose(body, entity)

    assets = [(copy.deepcopy(a), xml_path) for a in collect_assets(xml_root)]
    defaults = [(copy.deepcopy(d), xml_path) for d in collect_defaults(xml_root)]
    sensors = []
    tendons = [(copy.deepcopy(t), xml_path) for t in collect_tendons(xml_root)]
    equalities = [(copy.deepcopy(e), xml_path) for e in collect_equalities(xml_root)]
    actuators = []
    sensor_tag = xml_root.find("sensor")
    if sensor_tag is not None:
        for child in sensor_tag:
            sensors.append((copy.deepcopy(child), xml_path))
    for actuator_tag in xml_root.findall(".//actuator"):
        for child in actuator_tag:
            actuators.append((copy.deepcopy(child), xml_path))
    # Recursively process sub-bodies
    for subent in entity.get("bodies", []):
        sub_body, sub_assets, sub_defaults, sub_sensors, sub_tendons, sub_equalities, sub_actuators = process_entity(subent)
        body.append(sub_body)
        assets += sub_assets
        defaults += sub_defaults
        tendons += sub_tendons
        equalities += sub_equalities
        sensors += sub_sensors
        actuators += sub_actuators

    return body, assets, defaults, sensors, tendons, equalities, actuators

# Formats XML for correct indentation, tag sequence, and spacing
def pretty_print_xml(elem):
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

    xml_str = xml_str.replace('?>\n<asset', '?>\n\n<asset')
    xml_str = re.sub(r'\n{3,}', '\n\n', xml_str)

    if not xml_str.endswith('\n'):
        xml_str += '\n'
    return xml_str

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

root = ET.Element('mujoco', model=config['model_name'])
root.append(ET.Element('compiler', meshdir="assets"))
root.append(ET.Element('include', file=config['env_path']))

# Collect everything
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
    body, assets, defaults, sensors, tendons, equalities, actuators = process_entity(ent)
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

pretty_xml = pretty_print_xml(root)

with open("output.xml", "w") as f:
    f.write(pretty_xml)
