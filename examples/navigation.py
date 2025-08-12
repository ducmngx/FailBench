import numpy as np
import mujoco
import time
from mujoco import viewer

from planners.mapping.generate_map import Mapper
from planner.algorithms.global_planners import A_StarPlanner

from scipy.spatial.transform import Rotation as R

def test_planner_on_map():
    """
    Testing code for A_StarPlanner, Mapper, and Map.
    """
    xml_path = "/Users/saghani/Workspace/Research/GenAISim/assets/environments/dumped_kitchen_noassets.xml"
    mjmodel = mujoco.MjModel.from_xml_path(xml_path)
    mjdata = mujoco.MjData(mjmodel)
    mapper = Mapper(mjmodel, mjdata, xml_path, static_scale=100, robot_max_height=5)

    map = mapper.get_map()
    # map.visualize()
    map.change_resolution(5)        # downscale x100/5 (x20)
    # map.visualize()
    map.inflate_obstacles(np.array([0.21,0.155])) # inflation is done at the raw_map level so it doesnt matter whether you change resolution first or inflate first.
    # map.visualize()
    

    planner = A_StarPlanner(mjmodel, map, heuristic_fn=None)

    path = planner.plan(start_pos=np.array([20,5]), end_pos=np.array([20, 50]))
    # # print(path)

    path_decresed_resolution = planner.decrease_path_resolution(path, 3)
    # # print(path_decresed_resolution)

    path_env_coords = map.convert_to_world_coordinates(path_decresed_resolution)
    print(path_env_coords)

    map.visualize_path(path_env_coords)


def test(xml_path="/Users/saghani/Workspace/Research/GenAISim/assets/environments/jackal_in_kitchen.xml", speed=0.1):
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
    
    # have to manually put in jackal's dimensions
    JACKAL_DIMENSIONS = [0.43, 0.508]
   
    # the static_scale becomes x2 since generally things in mujoco are 'half_height/length/width' etc.
    mapper = Mapper(model, data, xml_path, static_scale=100, robot_max_height=5)
    map = mapper.get_map()
    map.inflate_obstacles(JACKAL_DIMENSIONS, half_sizes=False)
    map.change_resolution(10)        # downscale x200/10 (x20)

    planner = A_StarPlanner(model, map, heuristic_fn=None)
    path = planner.plan(start_pos=np.array([20,5]), end_pos=np.array([20, 50]))
    path_env_coords = map.convert_to_world_coordinates(path)
    print(path_env_coords)

    # get jackal body to calculate current pose
    jackal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jackal")
    jackal_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jackal_free_joint")
    jackal_qpos_adr = model.jnt_qposadr[jackal_joint_id]

    with viewer.launch_passive(model=model, data=data) as viewer_:
        # Set nice camera view
        viewer_.cam.azimuth = 45
        viewer_.cam.elevation = -20
        viewer_.cam.distance = 20
        viewer_.cam.lookat[:] = [0, 0, 0.5]

        
        while viewer_.is_running():
            for i, waypoint in enumerate(path_env_coords):
                if not viewer_.is_running():
                    break
                
                # get robot state information (pos, rotation)
                cur_pos = data.xpos[jackal_body_id][:2]
                cur_quat = R.from_matrix(data.xmat[jackal_body_id].reshape(3, 3)).as_quat()
                r_robot = R.from_quat(cur_quat) # world->robot transformation

                # transform the waypoint into robot frame:
                direction = waypoint - cur_pos
                direction = np.append(direction, 0) # adding 0 as the following transformations expect a 3-dimensional vector 
                r_direction = r_robot.inv().apply(direction)

                # calculate desired angle in robot frame
                angle = np.arctan2(r_direction[1], r_direction[0])
                desired_quat = R.from_euler('z', angle, degrees=False).as_quat()

                # data.qpos[3:7] (the quat. part) due to jackal's mesh messing with its frame, we need to apply the desired quat on top of the current one to
                # visually achieve the desired quat.
                r_cur = R.from_quat(cur_quat)
                r_desired = R.from_quat(desired_quat)
                composed_quat = (r_cur * r_desired).as_quat()  # still in [x, y, z, w]

                # converting back to mujoco quat coordinates [w, x, y, z]
                mj_quat = np.array([composed_quat[3], composed_quat[0], composed_quat[1], composed_quat[2]])
                data.qpos[jackal_qpos_adr+3:jackal_qpos_adr+7] = mj_quat

                # Set robot position
                data.qpos[jackal_qpos_adr:jackal_qpos_adr+2] = waypoint
                
                # Update MuJoCo
                mujoco.mj_forward(model, data)
                viewer_.sync()
                
                # Control playback speed
                time.sleep(0.1 / speed)
            
            print("Animation complete. Restarting...")
            
if __name__ == "__main__":
    test()