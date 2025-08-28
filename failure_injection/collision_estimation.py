import mujoco
import numpy as np
from failure_injection.collision_utils import *
from tabulate import tabulate
import itertools

class CollisionEstimator:

    def __init__(self, model, data, failing_joints, method_type="AABB", inflation_radius=0, robot_root_name="link0", robot_joints=[f"joint{i}" for i in range(1,8)]):
        self.model = model
        self.data = data
        self.method_type = method_type
        self.inflation_radius = inflation_radius
        self.collision_function = None
        self.robot_root_name = robot_root_name
        self._init_non_robot_geoms()
        self._init_robot_geoms(robot_joints)
        self.all_joint_ids = self._convert_to_joint_ids(failing_joints)
        # if self.method_type == "AABB":
        #     self._init_aabb_corners()

    def estimate_bodies_in_collision(self, failing_joints="all", failure_type="aggressive", remove_world_body=True):
        """
        The main method that will estimate the non-robot bodies that are estimated to be in collision upon failure_type failure at joint failing_joint.
        failing_joints: the joint(s) that fails. string or list of strings.
        failure_type: fixed to the only failure we have, can be enum in the future. (string)
        """
        assert failure_type == "aggressive", "No other failure type is supported."
        if failing_joints == "all":
            failing_joint_ids = self.all_joint_ids
        else:
            failing_joint_ids = self._convert_to_joint_ids(failing_joints)

        failing_geoms = np.unique(np.concatenate([self.robot_joint_geoms[joint_id] for joint_id in failing_joint_ids]))
        # print("robot geom xpos:", self.data.geom_xpos[failing_geoms])

        non_robot_geoms = self.non_robot_geoms
        if remove_world_body:
            body_names = np.array([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) for bid in self.model.geom_bodyid[non_robot_geoms]])
            non_robot_geoms = non_robot_geoms[body_names != "world"]

        # get robot-candidate geom pairs to check possible collision
        # preprocess this in case its slow
        colliding_id_pairs = self._get_colliding_geom_pairs(failing_geoms, non_robot_geoms)

        
        if self.method_type == "AABB":
            # check for collisions using AABB - not implemented but could be a faster and more finer method.
            estimated_collisions, collision_area, total_cand_area, total_robot_area = self._axis_aligned_bounding_box_method(colliding_id_pairs)
        elif self.method_type == "bounding_sphere":
            # check for collisions using bounding spheres - easier but a more course method.
            _, _, _ = self._bounding_sphere_method(colliding_id_pairs)
        else:
            raise NotImplementedError()
        
        # save current estimate as reference
        self.current_collision_pairs = estimated_collisions 
        self.current_collision_area = collision_area
        self.current_total_cand_area = total_cand_area
        self.current_total_robot_area = total_robot_area 

        # given geom id pairs and collision probability of each pair, find the bodies the geoms belong to
        # for each body pair (robot-object) prob of collision =  total sum of intersected area / total sum of geom area
        # prob = max(intersection/object_area, intersection/robot_area)
        estimated_body_id_pairs = self.model.geom_bodyid[estimated_collisions]
        sorting_idx = np.lexsort((estimated_body_id_pairs[:,1], estimated_body_id_pairs[:,0]))
        sorted_collision_pairs_bids = estimated_body_id_pairs[sorting_idx]
        sorted_merged_cand_areas = np.stack((collision_area[sorting_idx], total_cand_area[sorting_idx]), axis=1)
        sorted_merged_robot_areas = np.stack((collision_area[sorting_idx], total_robot_area[sorting_idx]), axis=1)

        unique_body_pairs, unique_idx = np.unique(sorted_collision_pairs_bids, return_index=True, axis=0)

        prob_of_cand_collision_over_geoms_in_body = np.fromiter((np.power(intersect_area.sum(axis=0), [1,-1]).prod() for intersect_area in  np.split(sorted_merged_cand_areas, unique_idx[1:])), np.float64, count=len(unique_body_pairs))
        prob_of_robot_collision_over_geoms_in_body = np.fromiter((np.power(intersect_area.sum(axis=0), [1,-1]).prod() for intersect_area in  np.split(sorted_merged_robot_areas, unique_idx[1:])), np.float64, count=len(unique_body_pairs))
        prob_of_collision_over_geoms_in_body = np.max((prob_of_cand_collision_over_geoms_in_body, prob_of_robot_collision_over_geoms_in_body), 0)
        
        return unique_body_pairs, prob_of_collision_over_geoms_in_body

    def forward_kinematics(self, config):
        """
        Very similar to planner.collision.collision_checker.set_robot_configuration_direct, 
            except this calls mj_kinematics (only updates xpos and xmat of all bodies/geoms, ie. stage 2 of mujoco pipeline)
            while that runs forward (bunch of other things, stages 2-22 of mujoco pipeline)
        """
        if len(config) > len(self.data.qpos):
            raise ValueError(f"Config length {len(config)} > scene qpos length {len(self.data.qpos)}")
        
        self.data.qpos[:len(config)] = config
        mujoco.mj_kinematics(self.model, self.data)

    def _init_non_robot_geoms(self):
        non_robot_bodies = get_non_robot_bodies(self.model, self.robot_root_name)
        self.non_robot_geoms = get_bodies_geoms(self.model, non_robot_bodies, keep_seperate=False)

    def _init_robot_geoms(self, all_robot_joints):
        """
        Instead of finding geoms everytime estimate_bodies_in_collision is called, pre-process it and store in a dictionary
        """
        self.robot_joint_geoms = {joint:[] for joint in all_robot_joints}
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, failing_joint) for failing_joint in all_robot_joints]
        child_bids  = self.model.jnt_bodyid[joint_ids]
        parent_bids = self.model.body_parentid[child_bids]

        # get geoms in bodies
        child_geoms = get_bodies_geoms(self.model, child_bids, keep_seperate=True) 
        parent_geoms = get_bodies_geoms(self.model, parent_bids, keep_seperate=True)
        self.robot_joint_ids = {all_robot_joints[i]:joint_ids[i] for i in range(len(joint_ids))}
        self.robot_joint_geoms = {joint_ids[i]: np.unique(np.concatenate((child_geoms[i], parent_geoms[i]))) for i in range(len(joint_ids)) }

    def _get_colliding_geom_pairs(self, robot_geoms, candidate_geoms):
        """
        Given robot_geoms and candidate_geoms, check against their contypes and conaffinities to filter out geoms that cannot collide against each other.
        Useful to reduce the size of the problem for collision estimating.
        """
        non_robot_geom_contypes = self.model.geom_contype[candidate_geoms]
        robot_geom_contypes = self.model.geom_contype[robot_geoms]

        non_robot_geom_conaffinities = self.model.geom_conaffinity[candidate_geoms]
        robot_geom_conaffinities = self.model.geom_conaffinity[robot_geoms]
        
        robot_geom_length = len(robot_geoms)
        contypes = np.concatenate((robot_geom_contypes, non_robot_geom_contypes))
        conaffinities = np.concatenate((robot_geom_conaffinities, non_robot_geom_conaffinities))

        first_test = contypes[:, None] & conaffinities[None, :] !=0 
        second_test = conaffinities[:, None] & contypes[None, :] !=0
        collision_test = first_test | second_test
        np.fill_diagonal(collision_test, False)
        lower_half_test = np.tril(collision_test)
        candidate_idxs, robot_idxs = np.where(lower_half_test[robot_geom_length:, :robot_geom_length])
        geom_id_pairs = np.stack((robot_geoms[robot_idxs], candidate_geoms[candidate_idxs]), axis=1)
        return geom_id_pairs

    def _bounding_sphere_method(self, geom_pairs):
        raise NotImplementedError("probability of collision not implemented. use method_type=AABB for now.")
        # get bounding sphere and world coordinates of robot geoms
        robot_geom_coords = self.data.geom_xpos[geom_pairs[:,0]]
        robot_geom_bounding_radius = self.model.geom_rbound[geom_pairs[:,0]] + self.inflation_radius

        # get bounding sphere and world coordinates of candidate geoms
        candidate_geom_coords = self.data.geom_xpos[geom_pairs[:,1]]
        candidate_geom_bounding_radius = self.model.geom_rbound[geom_pairs[:,1]] 

        # in collision if candidate geom's bounding sphere is under the bounding sphere of the robot's geom
        dx = (candidate_geom_coords[:,0] -robot_geom_coords[:, 0])
        dy = (candidate_geom_coords[:,1] -robot_geom_coords[:, 1])
        cond = (dx**2 + dy**2 <= (robot_geom_bounding_radius + candidate_geom_bounding_radius)**2) & (candidate_geom_coords[:, 2] - candidate_geom_bounding_radius <= robot_geom_coords[:,2]) 
        
        return geom_pairs[cond], np.ones()

    def _convert_to_joint_ids(self, joint_names):
        if isinstance(joint_names, list):
            return [self.robot_joint_ids[joint_name] for joint_name in joint_names]
        else:
            assert isinstance(joint_names, str) 
            return [self.robot_joint_ids[joint_names]]
    
    # def _init_aabb_corners(self):
    #     robot_aabb = 

    def _axis_aligned_bounding_box_method(self, geom_pairs):
        # precompute these
        robot_geom_aabb = self.model.geom_aabb[geom_pairs[:,0]]
        corner_scale = np.array(list(itertools.product([-1,1], repeat=3)))[None]
        robot_geom_corners = robot_geom_aabb[:, None, :3] + (corner_scale*robot_geom_aabb[:, None, 3:])

        robot_geom_coor = self.data.geom_xpos[geom_pairs[:,0]]
        robot_geom_rot_mat = self.data.geom_xmat[geom_pairs[:,0]].reshape(-1, 3, 3)

        geom_world_coor = robot_geom_corners @ robot_geom_rot_mat.transpose(0,2,1) + robot_geom_coor[:, None]
        robot_geom_mins = geom_world_coor.min(axis=1)
        robot_geom_maxs = geom_world_coor.max(axis=1)
        

        cand_geom_aabb = self.model.geom_aabb[geom_pairs[:,1]]
        cand_geom_corners = cand_geom_aabb[:, None, :3] + (corner_scale*cand_geom_aabb[:, None, 3:])

        cand_geom_coor = self.data.geom_xpos[geom_pairs[:,1]]
        cand_geom_rot_mat = self.data.geom_xmat[geom_pairs[:,1]].reshape(-1, 3, 3)

        geom_world_coor = cand_geom_corners @ cand_geom_rot_mat.transpose(0,2,1) + cand_geom_coor[:, None]
        cand_geom_mins = geom_world_coor.min(axis=1)
        cand_geom_maxs = geom_world_coor.max(axis=1)

        # prob of interaction = area of intersection in x-y axis / area of cand geom
        x_overlap = np.clip(np.min([robot_geom_maxs[:, 0], cand_geom_maxs[:, 0]], axis=0) - np.max([robot_geom_mins[:,0], cand_geom_mins[:, 0]], axis=0), min=0)
        y_overlap = np.clip(np.min([robot_geom_maxs[:, 1], cand_geom_maxs[:, 1]], axis=0) - np.max([robot_geom_mins[:,1], cand_geom_mins[:, 1]], axis=0), min=0)

        intersection = x_overlap*y_overlap
        
        # cand_under_robot_geom = robot_geom_mins[:,2] + self.inflation_radius > cand_geom_maxs[:, 2]
        # intersection = intersection*cand_under_robot_geom
        # area of cand geom
        diff = (cand_geom_maxs - cand_geom_mins)[:, :2]
        cand_area = np.prod(diff, axis=1)

        # area of robot geoms
        diff = (robot_geom_maxs - robot_geom_mins)[:, :2]
        robot_area = np.prod(diff, axis=1)

        # prob = intersection / area
        possible_collision = intersection > 0
        return geom_pairs[possible_collision], intersection[possible_collision], cand_area[possible_collision], robot_area[possible_collision]
        




def fancy_test_collision_estimator():
    """Physics-based test of collision estimation for Franka joints."""

    print("\n🎯 Collision Estimation Test")
    print("=" * 50)

    # Phase 1: Load model
    print("\n" + "="*30)
    print("PHASE 1: LOAD MODEL")
    print("="*30)

    robot_xml_path = "/Users/saghani/Workspace/Research/GenAISim/franka_emika_panda/scene.xml"
    try:
        model = mujoco.MjModel.from_xml_path(robot_xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)  # Update kinematics
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Phase 2: Initialize estimator
    print("\n" + "="*30)
    print("PHASE 2: INITIALIZE ESTIMATOR")
    print("="*30)

    try:
        failing_joints = [f"joint{i}" for i in range(1,8)]
        estimator = CollisionEstimator(model, data, failing_joints=failing_joints)
        print("✅ Collision estimator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize estimator: {e}")
        return False

    # Phase 3: Single joint collision test
    print("\n" + "="*30)
    print("PHASE 3: SINGLE JOINT COLLISION TEST")
    print("="*30)

    try:
        collision_pair_body_ids = estimator.estimate_bodies_in_collision("joint1", remove_world_body=False)
        robot_body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            for bid in collision_pair_body_ids[:, 0]
        ]
        world_body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            for bid in collision_pair_body_ids[:, 1]
        ]

        if len(robot_body_names) == 0:
            print("⚠️ No collisions detected for joint1")
        else:
            table_data = [[rb, wb] for rb, wb in zip(robot_body_names, world_body_names)]
            print(tabulate(table_data, headers=["🤖 Robot Body Part", "🌍 Collides With"], tablefmt="fancy_grid"))

        print("✅ Single joint collision test completed")
    except Exception as e:
        print(f"❌ Single joint collision estimation failed: {e}")
        return False

    # Phase 4: Multiple joint collision test
    print("\n" + "="*30)
    print("PHASE 4: MULTIPLE JOINT COLLISION TEST")
    print("="*30)

    try:
        all_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        collision_pair_body_ids = estimator.estimate_bodies_in_collision(all_joints, remove_world_body=True)
        robot_body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            for bid in collision_pair_body_ids[:, 0]
        ]
        world_body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            for bid in collision_pair_body_ids[:, 1]
        ]

        if len(robot_body_names) == 0:
            print("⚠️ No collisions detected for multiple joints")
        else:
            table_data = [[rb, wb] for rb, wb in zip(robot_body_names, world_body_names)]
            print(tabulate(table_data, headers=["🤖 Robot Body Part", "🌍 Collides With"], tablefmt="fancy_grid"))

        print("✅ Multiple joints collision test completed")
    except Exception as e:
        print(f"❌ Multiple joint collision estimation failed: {e}")
        return False

    # Success
    print("\n🎉 Collision Estimator Test Completed Successfully!")
    return True


if __name__ == "__main__":
    fancy_test_collision_estimator()
