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
        self.non_robot_geoms =  self._init_non_robot_geoms()
        self.robot_joint_names2ids, self.robot_joint_geoms, self.robot_geoms_by_ids = self._init_robot_geoms(robot_joints)
        # in larger environments, this will be not be ideal to calculate over all pairs.
        # self.all_colliding_pairs = self._get_colliding_geom_pairs(self.robot_joint_geoms, self.non_robot_geoms)
        self.all_joint_ids = self._convert_to_joint_ids(failing_joints)
        assert method_type == "AABB"
        # self._init_AABB_variables()


    def estimate_bodies_in_collision(self, failing_joints="all", failure_type="aggressive", calculate_grads=False):
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

        # get robot and non-robot geoms by id
        failing_geoms = np.unique(np.concatenate([self.robot_joint_geoms[joint_id] for joint_id in failing_joint_ids]))
        non_robot_geoms = self.non_robot_geoms

        # get robot-candidate geom pairs to check possible collision
        # preprocess this in case its slow
        colliding_id_pairs = self._get_colliding_geom_pairs(failing_geoms, non_robot_geoms)

        # internal method to check for overlapping areas
        estimated_collisions, collision_area, total_cand_area, total_robot_area, gradients_of_geoms_wrt_dof = self._axis_aligned_bounding_box_method(colliding_id_pairs, calculate_grads)
        
        # save current estimate as reference
        self.current_collision_pairs = estimated_collisions 
        self.current_collision_area = collision_area
        self.current_total_cand_area = total_cand_area
        self.current_total_robot_area = total_robot_area 

        # find the bodies the geoms belong to
        bodies_in_collision = self.model.geom_bodyid[estimated_collisions]

        # for each body pair (robot-object) calculate prob. of collison
        # prob = max(intersection/object_area, intersection/robot_area)
        sorting_idx = np.lexsort((bodies_in_collision[:,1], bodies_in_collision[:,0]))
        sorted_collision_pairs_bids = bodies_in_collision[sorting_idx]
        sorted_merged_cand_areas = np.stack((collision_area[sorting_idx], total_cand_area[sorting_idx]), axis=1)    # N, 2
        sorted_merged_robot_areas = np.stack((collision_area[sorting_idx], total_robot_area[sorting_idx]), axis=1)  # N, 2
        sorted_gradients_of_geoms_wrt_dof = gradients_of_geoms_wrt_dof[sorting_idx]

        unique_body_pairs, unique_idx = np.unique(sorted_collision_pairs_bids, return_index=True, axis=0)

        # prob_of_cand_collision_over_geoms_in_body = np.array([np.power(intersect_area.sum(axis=0), [1,-1]).prod() for intersect_area in  np.split(sorted_merged_cand_areas, unique_idx[1:])])
        # prob_of_robot_collision_over_geoms_in_body = np.array([np.power(intersect_area.sum(axis=0), [1,-1]).prod() for intersect_area in  np.split(sorted_merged_robot_areas, unique_idx[1:])])

        eps = 1e-9 # avoid 1/0 errors (0^-1)

        prob_of_robot_collision_over_geoms_in_body = np.array([
            np.power(intersect_area.sum(axis=0) + eps, [1, -1]).prod()
            for intersect_area in np.split(sorted_merged_robot_areas, unique_idx[1:])
        ])

        prob_of_cand_collision_over_geoms_in_body = np.array([
            np.power(intersect_area.sum(axis=0) + eps, [1, -1]).prod()
            for intersect_area in np.split(sorted_merged_cand_areas, unique_idx[1:])
        ])

        prob_of_collision_between_bodies = np.max((prob_of_cand_collision_over_geoms_in_body, prob_of_robot_collision_over_geoms_in_body), 0)
        
        gradients_of_bodies_wrt_dof = np.array([grads.sum(axis=0) for grads in  np.split(sorted_gradients_of_geoms_wrt_dof, unique_idx[1:])])

        gradients_of_bodies_wrt_joints = self.dof_to_jnt(gradients_of_bodies_wrt_dof, axis=1, aggregate_type="sum")

        return unique_body_pairs, prob_of_collision_between_bodies, gradients_of_bodies_wrt_joints
    
    def forward_kinematics(self, config):
        """
        Very similar to planner.collision.collision_checker.set_robot_configuration_direct, 
            except this calls mj_kinematics (only updates xpos and xmat of all bodies/geoms, ie. stage 2 of mujoco pipeline)
            while that runs forward (bunch of other things, stages 2-22 of mujoco pipeline)
        """
        if len(config) > len(self.data.qpos):
            raise ValueError(f"Config length {len(config)} > scene qpos length {len(self.data.qpos)}")
        
        self.data.qpos[:len(config)] = config
        mujoco.mj_kinematics(self.model, self.data)     # updates data xpos / xmat
        mujoco.mj_comPos(self.model, self.data)         # updated jacobian  maybe not required

    def post_mj_forward_init(self):
        self._init_AABB_variables()

    def dof_to_jnt(self, dof_mat: np.ndarray, axis: int, aggregate_type: str = "sum"):
        assert aggregate_type in ["sum", "none"], "no other aggregation exists for method."

        if axis >= dof_mat.ndim or dof_mat.shape[axis] != self.model.nv:  # size of matrix along axis != |dof|
            return dof_mat

        all_dofs = list(range(self.model.nv))
        all_dof_jntids = self.model.dof_jntid[all_dofs]
        robot_joints = self.all_joint_ids
        robot_dof_jntids_idx = np.where(np.isin(all_dof_jntids, robot_joints))[0]
        robot_dofs_mat = np.take(dof_mat, robot_dof_jntids_idx, axis)
        # aggregate
        if aggregate_type == "sum":
            robot_dof_jntids = all_dof_jntids[robot_dof_jntids_idx]
            _, unique_idx = np.unique(robot_dof_jntids, return_index=True, axis=0)
            aggregated_dof_mat = np.stack([_dof_mat.sum(axis=-1) for _dof_mat in  np.split(robot_dofs_mat, unique_idx[1:], axis=-1)], axis=-1)
            return aggregated_dof_mat
        elif aggregate_type == "none":
            return robot_dofs_mat
    
    def _init_non_robot_geoms(self, exclude_world_body=True):
        non_robot_bodies = get_non_robot_bodies(self.model, self.robot_root_name)
        non_robot_geoms = get_bodies_geoms(self.model, non_robot_bodies, keep_seperate=False)

        if exclude_world_body:
            body_names = np.array([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) for bid in self.model.geom_bodyid[non_robot_geoms]])
            non_robot_geoms = non_robot_geoms[body_names != "world"]

        return non_robot_geoms

    def _init_robot_geoms(self, all_robot_joints_by_name):
        """
        Instead of finding geoms everytime estimate_bodies_in_collision is called, pre-process it and store in a dictionary
        """
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, failing_joint) for failing_joint in all_robot_joints_by_name]
        child_bids  = self.model.jnt_bodyid[joint_ids]
        parent_bids = self.model.body_parentid[child_bids]

        # get geoms in bodies
        child_geoms = get_bodies_geoms(self.model, child_bids, keep_seperate=True) 
        parent_geoms = get_bodies_geoms(self.model, parent_bids, keep_seperate=True)
        robot_joint_name2ids = {all_robot_joints_by_name[i]:joint_ids[i] for i in range(len(joint_ids))}
        robot_joint_geoms = {joint_ids[i]: np.unique(np.concatenate((child_geoms[i], parent_geoms[i]))) for i in range(len(joint_ids)) }
        robot_geoms_by_ids = np.sort(np.unique(np.concat(list(robot_joint_geoms.values())))) 
        return robot_joint_name2ids, robot_joint_geoms, robot_geoms_by_ids

    def _convert_to_joint_ids(self, joint_names):
        if isinstance(joint_names, list):
            return [self.robot_joint_names2ids[joint_name] for joint_name in joint_names]
        else:
            assert isinstance(joint_names, str) 
            return [self.robot_joint_names2ids[joint_names]]

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
    
    def _init_AABB_variables(self):
        robot_geom_aabb = self.model.geom_aabb[self.robot_geoms_by_ids] # {joint_id: Mx6} M = |geoms_in_joint|
        corner_scale = np.array(list(itertools.product([-1,1], repeat=3)))[None]
        self.robot_geom_corners = robot_geom_aabb[:, None, :3] + (corner_scale*robot_geom_aabb[:, None, 3:])

        cand_geom_aabb = self.model.geom_aabb[self.non_robot_geoms]
        self.cand_geom_corners = cand_geom_aabb[:, None, :3] + (corner_scale*cand_geom_aabb[:, None, 3:])

        # jacobian for robot corners
        self.robot_jacp = np.zeros((len(self.robot_geoms_by_ids), 3, self.model.nv))   # Linear velocity Jacobian
        self.robot_jacr = np.zeros((len(self.robot_geoms_by_ids), 3, self.model.nv))   # Angular velocity Jacobian
        for i, geom_id in enumerate(self.robot_geoms_by_ids):
            mujoco.mj_jacGeom(self.model,self.data, self.robot_jacp[i], self.robot_jacr[i], geom_id)

        self.vec_center_to_corners = self.robot_geom_corners - robot_geom_aabb[:, None, :3]      # (num_robot_joint_geoms, 8, 3)
        
    def _axis_aligned_bounding_box_method(self, geom_pairs, calculate_grads=False):
        # robot geom corners
        indices_of_robot_geoms = np.searchsorted(self.robot_geoms_by_ids, geom_pairs[:, 0])
        robot_geom_corners = self.robot_geom_corners[indices_of_robot_geoms]
        # robot geom corner jacobians
        vec_center_to_corners_local = self.vec_center_to_corners[indices_of_robot_geoms]
        robot_jacp = self.robot_jacp[indices_of_robot_geoms]
        robot_jacr = self.robot_jacr[indices_of_robot_geoms]
        # cand. geom corners 
        indices_of_cand_geoms = np.searchsorted(self.non_robot_geoms, geom_pairs[:, 1])
        cand_geom_corners = self.cand_geom_corners[indices_of_cand_geoms]

        # transform robot geom corners from local to global frame
        robot_geom_coor = self.data.geom_xpos[geom_pairs[:,0]]
        robot_geom_rot_mat = self.data.geom_xmat[geom_pairs[:,0]].reshape(-1, 3, 3)
        robot_geom_corners_world_coor = robot_geom_corners @ robot_geom_rot_mat.transpose(0,2,1) + robot_geom_coor[:, None]

        # soft max and min over robot geom corners to get the top-right and bottom-left corners
        robot_geom_soft_maxs = log_sum_exp_beta(robot_geom_corners_world_coor, 1, True)       # (N, 3)
        robot_geom_soft_mins = log_sum_exp_beta(robot_geom_corners_world_coor, 1, False)      # (N, 3)
        
        
        # transform cand. geom corners from local to global frame
        cand_geom_coor = self.data.geom_xpos[geom_pairs[:,1]]
        cand_geom_rot_mat = self.data.geom_xmat[geom_pairs[:,1]].reshape(-1, 3, 3)
        geom_world_coor = cand_geom_corners @ cand_geom_rot_mat.transpose(0,2,1) + cand_geom_coor[:, None]

        # (hard) max and min over robot geom corners to get the top-right and bottom-left corners
        cand_geom_mins = geom_world_coor.min(axis=1)
        cand_geom_maxs = geom_world_coor.max(axis=1)

        # calculate interaction:= area of intersection in x-y axis 
        # soft mins of maxs and maxs of mins to find the top-right and bottom-left corners of the area of intersection
        geom_mins_of_maxs = log_sum_exp_beta(np.array([robot_geom_soft_maxs, cand_geom_maxs]), axis=0, max=False) # 2 N 3 -> N 3
        geom_maxs_of_mins = log_sum_exp_beta(np.array([robot_geom_soft_mins, cand_geom_mins]), axis=0, max=True)

        # calculate overlap in x and y axis
        x_overlap = np.clip(geom_mins_of_maxs[:, 0] - geom_maxs_of_mins[:, 0], a_min=0, a_max=None)
        y_overlap = np.clip(geom_mins_of_maxs[:, 1] - geom_maxs_of_mins[:, 1], a_min=0, a_max=None)

        # overlap = intersection
        intersection = x_overlap*y_overlap          # shape = (N,)
        
        # additional constraint: cand. is under the robot geom, however this makes the planner try to go under the object which is not ideal
        # cand_under_robot_geom = robot_geom_mins[:,2] + self.inflation_radius > cand_geom_maxs[:, 2] 

        # transforming into a probability, prob. of interaction = max(overlapping area / area of robot geom, overlapping area / area of cand. geom)
        # area of cand geom
        diff = (cand_geom_maxs - cand_geom_mins)[:, :2]
        cand_area = np.prod(diff, axis=1)
        # area of robot geoms
        diff = (robot_geom_soft_maxs - robot_geom_soft_mins)[:, :2]
        robot_area = np.prod(diff, axis=1)

        # return only non-zero overlaps
        possible_collision = intersection > 0

        if not calculate_grads:
            return geom_pairs[possible_collision], intersection[possible_collision], cand_area[possible_collision], robot_area[possible_collision], np.zeros((len(intersection), self.model.nv))[possible_collision]
        
        # now to calculate the derivative
        # transform vec_center_to_corners in local frame to world frame
        vec_center_to_corners_world = np.einsum('nij,nkj->nki', robot_geom_rot_mat, vec_center_to_corners_local) # (N, 8, 3)

        # construct skew matrix
        x, y, z = vec_center_to_corners_world[..., 0], vec_center_to_corners_world[..., 1], vec_center_to_corners_world[..., 2]
        skew_matrix = np.zeros((len(vec_center_to_corners_world), 8, 3, 3))
        # [[0, -r[2], r[1]],
        # [r[2], 0, -r[0]],
        # [-r[1], r[0], 0]]
        skew_matrix[..., 0, 1] = -z
        skew_matrix[..., 0, 2] = y
        skew_matrix[..., 1, 0] = z
        skew_matrix[..., 1, 2] = -x
        skew_matrix[..., 2, 0] = -y
        skew_matrix[..., 2, 1] = x

        # calculate jacobian of the robot geom corners
        robot_corner_jacobian = robot_jacp[:, None] +  np.einsum('naij,nbjv->naiv', skew_matrix, robot_jacr[:, None]) # N, 8, 3, nv

        # d(soft maxs/mins of robot corners)_dnv
        softmax_grad = softmax_beta(robot_geom_corners_world_coor, 1, True)              # N, 8, 3
        softmin_grad = softmax_beta(robot_geom_corners_world_coor, 1, False)             # N, 8, 3
        Jxy = robot_corner_jacobian[..., :2, :]   # (N,8,2,nv)

        dmax_dnv = np.einsum('nac,nacv->ncv', softmax_grad[..., :2], Jxy)  # (N,2,nv)
        dmin_dnv = np.einsum('nac,nacv->ncv', softmin_grad[..., :2], Jxy)  # (N,2,nv)

        # d(soft maxs/mins of mins/maxs over robot and cand geoms)_dnv
        dmins_of_maxs_dmax = softmax_beta(np.array([robot_geom_soft_maxs, cand_geom_maxs]), axis=0, max=False)   # 2 N 3
        dmaxs_of_mins_dmin = softmax_beta(np.array([robot_geom_soft_mins, cand_geom_mins]), axis=0, max=True)    # 2 N 3
        dmins_of_maxs_dnv = np.einsum('anc,ncv->ncv', dmins_of_maxs_dmax[..., :2], dmax_dnv)  # (2,N,2), (N,2,nv) -> (N, 2, nv)
        dmaxs_of_mins_dnv = np.einsum('anc,ncv->ncv', dmaxs_of_mins_dmin[..., :2], dmin_dnv)  # (N,2,nv)

        # d(overlap_x/y)_dnv
        dx_overlap_dnv = dmins_of_maxs_dnv[:, 0] - dmaxs_of_mins_dnv[:, 0]      # N, nv
        dy_overlap_dnv = dmins_of_maxs_dnv[:, 1] - dmaxs_of_mins_dnv[:, 1]      # N, nv

        # d(intersection)_dnv
        dintersection_dnv = dx_overlap_dnv*y_overlap[:, None] + x_overlap[:, None]*dy_overlap_dnv     # N, nv

        # return only the places where intersection > 0
        return geom_pairs[possible_collision], intersection[possible_collision], cand_area[possible_collision], robot_area[possible_collision], dintersection_dnv[possible_collision]
        

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


def softmax_beta(x, axis, max=True, beta=100):
    sgn = +1.0 if max else -1.0
    y = sgn * beta * x
    a = np.max(y, axis=axis, keepdims=True)
    w = np.exp(y - a)
    w /= np.sum(w, axis=axis, keepdims=True)
    return w

def log_sum_exp_beta(x, axis, max=True, beta=100):
    sgn = +1.0 if max else -1.0
    y = sgn * beta * x
    a = np.max(y, axis=axis, keepdims=True)
    out = sgn * (a + np.log(np.sum(np.exp(y - a), axis=axis, keepdims=True))) / beta
    return out.squeeze(axis)


if __name__ == "__main__":
    fancy_test_collision_estimator()
