
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom
import mujoco
from mujoco.viewer import launch
import xml.etree.ElementTree as ET
import re

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

tree = ET.parse('full_scene.xml')
root = tree.getroot()
worldbody = root.find('.//worldbody')

def create_body(params):
    """
    Recursively build a body element from params.
    If 'xml_path' is present, add an <include> INSIDE this <body>.
    """
    body_elem = ET.Element('body', name=params['name'])
    for k, v in params.items():
        if k not in ['bodies', 'xml_path']:
            body_elem.set(k, str(v))
    # If xml_path, add <include> as a child (at the top; could also append at the end)
    if 'xml_path' in params:
        include_elem = ET.Element('include', file=params['xml_path'])
        body_elem.append(include_elem)
    # Recursively add nested bodies
    if 'bodies' in params:
        for sub_body in params['bodies']:
            child = create_body(sub_body)
            body_elem.append(child)
    return body_elem

for obj in config['bodies']:
    worldbody.append(create_body(obj))

def smart_indent(elem, level=0):
    """
    Indent all XML except the root element, which stays flush left.
    Closing tags line up with their opening tags.
    """
    i = "\n" + level * "  "
    j = "\n" + (level-1) * "  " if level > 0 else "\n"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for idx, child in enumerate(elem):
            smart_indent(child, level + 1)
            if idx < len(elem) - 1:
                if not child.tail or not child.tail.strip():
                    child.tail = i + "  "
            else:
                if not child.tail or not child.tail.strip():
                    child.tail = i
        if level == 0:
            elem.tail = "\n"
    else:
        if not elem.text or not elem.text.strip():
            elem.text = None
        if level != 0 and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
        elif level == 0:
            elem.tail = "\n"

def write_clean_xml(root, filename):
    smart_indent(root, 0)
    xml_str = ET.tostring(root, encoding="unicode")
    xml_str = re.sub(r'<body([^>]*)\s*/>', r'<body\1></body>', xml_str)
    xml_str = re.sub(r' \>', '>', xml_str)
    xml_str = re.sub(r'\n\s*\n', '\n', xml_str)
    xml_str = xml_str.strip()

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_str)

write_clean_xml(root, "output.xml")
