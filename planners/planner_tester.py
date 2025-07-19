import numpy as np
import mujoco

from .mapping.generate_map import Mapper

def test():
    """
    Testing code for A_StarPlanner, Mapper, and Map.
    """
    xml_path = "/Users/saghani/Workspace/Research/GenAISim/assets/environments/dumped_kitchen_noassets.xml"
    mjmodel = mujoco.MjModel.from_xml_path(xml_path)
    mjdata = mujoco.MjData(mjmodel)
    mapper = Mapper(mjmodel, mjdata, xml_path, static_scale=100, robot_max_height=5)

    map = mapper.get_map()
    map.change_resolution(5)        # downscale x100/5 (x20)
    # map.visualize()

    planner = A_StarPlanner(mjmodel, map, heuristic_fn=None)

    path = planner.plan(start_pos=np.array([20,5]), end_pos=np.array([20, 50]))
    # print(path)

    path_decresed_resolution = planner.decrease_path_resolution(path, 3)
    # print(path_decresed_resolution)

    path_env_coords = map.convert_to_world_coordinates(path_decresed_resolution)
    print(path_env_coords)

    map.visualize_path(path_env_coords)


def play_waypoints(waypoints, xml_path="/home/aaron/workspace/mujoco-arena/mink/examples/franka_emika_panda/mjx_panda.xml", speed=1.0):
    """
    Play waypoints in MuJoCo viewer.
    
    Args:
        waypoints: List of 55 joint configurations from your planner
        xml_path: MuJoCo XML file path
        speed: Playback speed (higher = faster)
    """
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    print(f"Playing {len(waypoints)} waypoints")
    print("Press ESC to exit, SPACE to pause/resume")
    
    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        # Set nice camera view
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0, 0, 0.5]
        
        paused = False
        
        while viewer.is_running():
            for i, config in enumerate(waypoints):
                if not viewer.is_running():
                    break
                
                # Handle pause (if your viewer supports it)
                # while paused and viewer.is_running():
                #     time.sleep(0.01)
                #     viewer.sync()
                
                # Set robot configuration
                data.qpos[:len(config)] = config
                
                # Update MuJoCo
                mujoco.mj_forward(model, data)
                viewer.sync()
                
                # Control playback speed
                time.sleep(0.1 / speed)
                
                # Show progress
                if i % 10 == 0:
                    print(f"Waypoint {i+1}/{len(waypoints)}")
            
            print("Animation complete. Restarting...")
            
if __name__ == "__main__":
    test()