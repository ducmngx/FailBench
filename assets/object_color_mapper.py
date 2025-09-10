import xml.etree.ElementTree as ET
import random
from typing import Dict, Tuple

# Mapping cost → color
COST_COLORS = {
    1: (1.0, 1.0, 0.698),
    2: (1.0, 0.878, 0.4),
    3: (1.0, 0.78, 0.0),
    4: (1.0, 0.678, 0.2),
    5: (1.0, 0.522, 0.2),
    6: (1.0, 0.353, 0.212),
    7: (1.0, 0.180, 0.180),
    8: (0.847, 0.0, 0.0),
    9: (0.639, 0.0, 0.0),
    10:(0.4, 0.0, 0.0)
}

def assign_obstacle_costs(xml_file: str, output_file: str) -> Tuple[Dict[str,int], str]:
    """
    Parses a MuJoCo XML file, finds all bodies containing 'obstacle', assigns
    a random cost 1-10 and corresponding color, writes updated XML, and returns
    mapping of body name -> cost and path to generated XML.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find all <body> elements with 'obstacle' in their name
    obstacle_bodies = []
    for body in root.findall(".//body"):
        name = body.get("name", "")
        if "obstacle" in name:
            obstacle_bodies.append(body)
    
    body_cost_mapping = {}
    
    for body in obstacle_bodies:
        cost = random.randint(1,10)
        color = COST_COLORS[cost]
        body_cost_mapping[body.get("name")] = cost
        
        # Find or create <geom> element to assign color
        geom = body.find("geom")
        if geom is None:
            geom = ET.SubElement(body, "geom")
            geom.set("type", "box")
            geom.set("size", "0.03 0.03 0.03")
        # geom.set("rgba", color + " 1")  # add alpha
        geom.set("rgba", f"{color[0]} {color[1]} {color[2]} 1.0")
        
    # Save updated XML
    tree.write(output_file)
    
    return body_cost_mapping, output_file


if __name__ == "__main__":
    # Example usage
    xml_file = "/Users/saghani/Workspace/Research/GenAISim/franka_emika_panda/scene_level2.xml"        # input XML
    output_file = "scene_cost.xml"  # output XML
    mapping, saved_file = assign_obstacle_costs(xml_file, output_file)
    print("Body cost mapping:", mapping)
    print("Saved XML file:", saved_file)
