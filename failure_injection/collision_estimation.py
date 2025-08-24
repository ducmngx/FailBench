import mujoco
import numpy as np
from failure_injection.collision_utils import *

class CollisionEstimator:

    def __init__(self, model, data, method_type="bounding_sphere", inflation_radius=0, robot_root_name="link0"):
        self.model = model
        self.data = data
        self.method_type = method_type
        self.inflation_radius = inflation_radius
        self.collision_function = None
        self.robot_root_name = robot_root_name
        self._init_non_robot_geoms()
    
    def _init_non_robot_geoms(self):
        non_robot_bodies = get_non_robot_bodies(self.model, self.robot_root_name)
        self.non_robot_geoms = get_bodies_geoms(self.model, non_robot_bodies)
        return self.non_robot_geoms

    def estimate_bodies_in_collision_with_multiple_failures(self, failing_joints, failure_type="aggressive"):
        """
        The main method that will estimate the non-robot bodies that are estimated to be in collision upon failure_type failure at joint failing_joint.
        failing_joint: the joint that fails. (string)
        failure_type: fixed to the only failure we have, can be enum in the future. (string)
        """
        assert failure_type == "aggressive", "No other failure type is supported."

        # get attached bodies 
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, failing_joint) for failing_joint in failing_joints]
        child_bids  = self.model.jnt_bodyid[joint_ids]
        parent_bids = self.model.body_parentid[child_bids]
        all_bids = np.unique(np.concatenate((child_bids, parent_bids)))

        # get geoms in bodies
        failing_geoms = get_bodies_geoms(self.model, all_bids) 

        return self._estimate_bodies_in_collision(failing_geoms)
        


    def estimate_bodies_in_collision_with_single_failure(self, failing_joint, failure_type="aggressive"):
        """
        The main method that will estimate the non-robot bodies that are estimated to be in collision upon failure_type failure at joint failing_joint.
        failing_joint: the joint that fails. (string)
        failure_type: fixed to the only failure we have, can be enum in the future. (string)
        """
        assert failure_type == "aggressive", "No other failure type is supported."

        # get attached bodies 
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, failing_joint)
        child_bid  = self.model.jnt_bodyid[joint_id]
        parent_bid = self.model.body_parentid[child_bid]

        # get geoms in bodies
        child_geoms = get_body_geoms(self.model, child_bid) 
        parent_geoms = get_body_geoms(self.model, parent_bid) 
        failing_geoms = np.concatenate([child_geoms, parent_geoms])

        return self._estimate_bodies_in_collision(failing_geoms)

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

    def _axis_aligned_bounding_box_method(self, geom_pairs):
        raise NotImplementedError()


def test():
    robot_xml_path = "/Users/saghani/Workspace/Research/GenAISim/franka_emika_panda/scene.xml"
    model = mujoco.MjModel.from_xml_path(robot_xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)  # Update kinematics

    estimator = CollisionEstimator(model, data)
    body_ids = estimator.estimate_bodies_in_collision_with_single_failure("joint1")
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in body_ids]
    print(body_ids, body_names)

if __name__ == "__main__":
    test()
