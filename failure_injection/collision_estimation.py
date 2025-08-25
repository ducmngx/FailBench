import mujoco
import numpy as np
from failure_injection.collision_utils import *

class CollisionEstimator:

    def __init__(self, model, data, method_type="bounding_sphere", inflation_radius=0, robot_root_name="link0", robot_joints=[f"joint{i}" for i in range(1,8)]):
        self.model = model
        self.data = data
        self.method_type = method_type
        self.inflation_radius = inflation_radius
        self.collision_function = None
        self.robot_root_name = robot_root_name
        self._init_non_robot_geoms()
        self._init_robot_geoms(robot_joints)
    
    def estimate_bodies_in_collision(self, failing_joints, failure_type="aggressive"):
        """
        The main method that will estimate the non-robot bodies that are estimated to be in collision upon failure_type failure at joint failing_joint.
        failing_joints: the joint(s) that fails. string or list of strings.
        failure_type: fixed to the only failure we have, can be enum in the future. (string)
        """
        assert failure_type == "aggressive", "No other failure type is supported."

        failing_joint_ids = self._convert_to_joint_ids(failing_joints)

        if isinstance(failing_joints, list):
            failing_geoms = np.unique(np.concatenate([self.robot_joint_geoms[joint_id] for joint_id in failing_joint_ids]))
            return self._estimate_bodies_in_collision(failing_geoms)
        else:
            failing_geoms = self.robot_joint_geoms[failing_joint_ids]
            return self._estimate_bodies_in_collision(failing_geoms)

    def _init_non_robot_geoms(self):
        non_robot_bodies = get_non_robot_bodies(self.model, self.robot_root_name)
        self.non_robot_geoms = get_bodies_geoms(self.model, non_robot_bodies)
        return self.non_robot_geoms

    def _init_robot_geoms(self, all_robot_joints):
        """
        Instead of finding geoms everytime estimate_bodies_in_collision is called, pre-process it and store in a dictionary
        """
        self.robot_joint_geoms = {joint:[] for joint in all_robot_joints}
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, failing_joint) for failing_joint in all_robot_joints]
        child_bids  = self.model.jnt_bodyid[joint_ids]
        parent_bids = self.model.body_parentid[child_bids]

        # get geoms in bodies
        child_geoms = get_bodies_geoms(self.model, child_bids) 
        parent_geoms = get_bodies_geoms(self.model, parent_bids)
        self.robot_joint_ids = {all_robot_joints[i]:joint_ids[i] for i in range(len(joint_ids))}
        self.robot_joint_geoms = {joint_ids[i]: np.unique(np.concatenate((child_geoms[i], parent_geoms[i]))) for i in range(len(joint_ids)) }


    def _estimate_bodies_in_collision(self, failing_geoms):
        """
        Helper function for single and multiple failures.
        """
        # get robot-candidate geom pairs to check possible collision
        colliding_id_pairs = self._get_colliding_geom_pairs(failing_geoms, self.non_robot_geoms)

        if self.method_type == "bounding_sphere":
            # check for collisions using bounding spheres - easier but a more course method.
            estimated_collisions = self._bounding_sphere_method(colliding_id_pairs)
        elif self.method_type == "AABB":
            # check for collisions using AABB - not implemented but could be a faster and more finer method.
            estimated_collisions = self._axis_aligned_bounding_box_method(colliding_id_pairs)
        else:
            raise NotImplementedError()

        # given geom collision pairs, get the non-robot ones and return their body ids. 
        estimated_collision_geom_ids = estimated_collisions[:, 1]
        estimated_collision_body_ids = np.unique(self.model.geom_bodyid[estimated_collision_geom_ids])  # to get rid of duplicates
        return estimated_collision_body_ids

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
        collision_test = first_test & second_test
        np.fill_diagonal(collision_test, False)
        lower_half_test = np.tril(collision_test)
        candidate_idxs, robot_idxs = np.where(lower_half_test[robot_geom_length:, :robot_geom_length])
        geom_id_pairs = np.stack((robot_geoms[robot_idxs], candidate_geoms[candidate_idxs]), axis=1)
        return geom_id_pairs

    def _bounding_sphere_method(self, geom_pairs):
        """
        Given (N,2) geom_pairs with the first column representing geoms of the failing bodies and the second column representing the candidate geoms that the failing geoms will collide with,
        this method will return the pairs of geoms that will collide using bounding spheres.
        Essentially, if a candidate geom's bounding sphere is below the failing geom's inflated bounding sphere, then the two are estimated to be in collision, given failure.
        """

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
        return geom_pairs[cond]

    def _convert_to_joint_ids(self, joint_names):
        if isinstance(joint_names, list):
            return [self.robot_joint_ids[joint_name] for joint_name in joint_names]
        else:
            assert isinstance(joint_names, str) 
            return self.robot_joint_ids[joint_names]
    
    def _axis_aligned_bounding_box_method(self, geom_pairs):
        raise NotImplementedError()


def test():
    robot_xml_path = "/Users/saghani/Workspace/Research/GenAISim/franka_emika_panda/scene.xml"
    model = mujoco.MjModel.from_xml_path(robot_xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)  # Update kinematics

    estimator = CollisionEstimator(model, data)
    body_ids = estimator.estimate_bodies_in_collision("joint1") # the function I want to test
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in body_ids]
    print(body_ids, body_names)

    all_joints = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
    
    body_ids = estimator.estimate_bodies_in_collision(all_joints) # the function I want to test
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in body_ids]
    print(body_ids, body_names)

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
        estimator = CollisionEstimator(model, data)
        print("✅ Collision estimator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize estimator: {e}")
        return False

    # Phase 3: Test single failure case
    print("\n" + "="*30)
    print("PHASE 3: SINGLE FAILURE TEST")
    print("="*30)

    try:
        body_ids = estimator.estimate_bodies_in_collision("joint1")
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) for bid in body_ids]
        print(f"   Colliding bodies (joint1): {body_names}")
    except Exception as e:
        print(f"❌ Single failure estimation failed: {e}")
        return False

    # Phase 4: Test multiple failure case
    print("\n" + "="*30)
    print("PHASE 4: MULTIPLE FAILURES TEST")
    print("="*30)

    try:
        all_joints = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
        body_ids = estimator.estimate_bodies_in_collision(all_joints)
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) for bid in body_ids]
        print(f"   Colliding bodies (all joints): {body_names}")
    except Exception as e:
        print(f"❌ Multiple failure estimation failed: {e}")
        return False

    # Success
    print("\n🎉 Collision Estimator Test Completed Successfully!")
    return True


if __name__ == "__main__":
    fancy_test_collision_estimator()
