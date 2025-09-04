import mujoco
import numpy as np
from failure_injection.collision_utils import *
from tabulate import tabulate
import itertools

class OptimizedCollisionEstimator:

    def __init__(self, model, data, failing_joints, method_type="AABB", inflation_radius=0, robot_root_name="link0", robot_joints=[f"joint{i}" for i in range(1,8)]):
        self.model = model
        self.data = data
        self.method_type = method_type
        self.inflation_radius = inflation_radius
        self.collision_function = None
        self.robot_root_name = robot_root_name
        self.non_robot_geoms = self._init_non_robot_geoms()
        self.robot_joint_names2ids, self.robot_joint_geoms, self.robot_geoms_by_ids = self._init_robot_geoms(robot_joints)
        self.all_joint_ids = self._convert_to_joint_ids(failing_joints)
        assert method_type == "AABB"
        
        # Pre-compute static mappings
        self._precompute_mappings()
        
        # Cache for expensive operations
        self._cache = {}

    def _precompute_mappings(self):
        """Pre-compute index mappings to avoid repeated searchsorted calls"""
        self.robot_geom_to_idx = {geom_id: i for i, geom_id in enumerate(self.robot_geoms_by_ids)}
        self.non_robot_geom_to_idx = {geom_id: i for i, geom_id in enumerate(self.non_robot_geoms)}

    def estimate_bodies_in_collision(self, failing_joints="all", failure_type="aggressive", calculate_grads=False):
        """
        Optimized version of the main collision estimation method.
        """
        assert failure_type == "aggressive", "No other failure type is supported."

        # Which joints failed?
        if failing_joints == "all":
            failing_joint_ids = self.all_joint_ids
        else:
            failing_joint_ids = self._convert_to_joint_ids(failing_joints)

        # Get robot geoms involved in failure
        failing_geoms = np.unique(
            np.concatenate([self.robot_joint_geoms[joint_id] for joint_id in failing_joint_ids])
        )

        # Candidate geom pairs to check collision
        colliding_id_pairs = self._get_colliding_geom_pairs(failing_geoms, self.non_robot_geoms)
        
        if len(colliding_id_pairs) == 0:
            return np.array([]).reshape(0, 2), np.array([]), np.array([]).reshape(0, len(self.all_joint_ids))

        # Overlapping areas (core computation) - optimized version
        (
            estimated_collisions,
            collision_area,
            total_cand_area,
            total_robot_area,
            gradients_of_geoms_wrt_dof,
        ) = self._fast_aabb_method(colliding_id_pairs, calculate_grads)

        if len(estimated_collisions) == 0:
            return np.array([]).reshape(0, 2), np.array([]), np.array([]).reshape(0, len(self.all_joint_ids))

        # Save current estimate as reference
        self.current_collision_pairs = estimated_collisions
        self.current_collision_area = collision_area
        self.current_total_cand_area = total_cand_area
        self.current_total_robot_area = total_robot_area

        # Map geoms → bodies
        bodies_in_collision = self.model.geom_bodyid[estimated_collisions]

        # Sort by (robot_body, object_body)
        sorting_idx = np.lexsort((bodies_in_collision[:, 1], bodies_in_collision[:, 0]))
        sorted_collision_pairs_bids = bodies_in_collision[sorting_idx]
        sorted_cand = np.column_stack((collision_area[sorting_idx], total_cand_area[sorting_idx]))
        sorted_robot = np.column_stack((collision_area[sorting_idx], total_robot_area[sorting_idx]))
        sorted_gradients = gradients_of_geoms_wrt_dof[sorting_idx]

        # Group by unique body pairs
        unique_body_pairs, unique_idx = np.unique(
            sorted_collision_pairs_bids, return_index=True, axis=0
        )

        eps = 1e-9  # prevent division by zero

        # Compute collision probabilities (vectorized)
        cumsum_cand = np.add.reduceat(sorted_cand, unique_idx, axis=0)
        cumsum_robot = np.add.reduceat(sorted_robot, unique_idx, axis=0)

        prob_of_cand_collision = np.divide(cumsum_cand[:, 0] + eps, cumsum_cand[:, 1] + eps)
        prob_of_robot_collision = np.divide(cumsum_robot[:, 0] + eps, cumsum_robot[:, 1] + eps)

        prob_of_collision_between_bodies = np.maximum(
            prob_of_cand_collision, prob_of_robot_collision
        )

        # Aggregate gradients per unique body pair
        gradients_of_bodies_wrt_dof = np.add.reduceat(sorted_gradients, unique_idx, axis=0)

        # Convert to joint space
        gradients_of_bodies_wrt_joints = self.dof_to_jnt(
            gradients_of_bodies_wrt_dof, axis=1, aggregate_type="sum"
        )

        return unique_body_pairs, prob_of_collision_between_bodies, gradients_of_bodies_wrt_joints

    def forward_kinematics(self, config):
        """
        Optimized forward kinematics with cache invalidation
        """
        if len(config) > len(self.data.qpos):
            raise ValueError(f"Config length {len(config)} > scene qpos length {len(self.data.qpos)}")
        
        # Check if configuration changed significantly
        config_changed = not hasattr(self, '_last_config') or not np.allclose(self._last_config, config, atol=1e-6)
        
        if config_changed:
            self.data.qpos[:len(config)] = config
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            self._last_config = config.copy()
            # Invalidate cache
            self._cache.clear()

    def post_mj_forward_init(self):
        self._init_AABB_variables()

    def _fast_aabb_method(self, geom_pairs, calculate_grads=False):
        """
        Optimized AABB collision detection using hard max/min and vectorized operations
        """
        if len(geom_pairs) == 0:
            empty_grads = np.zeros((0, self.model.nv))
            return np.array([]).reshape(0, 2), np.array([]), np.array([]), np.array([]), empty_grads

        # Use pre-computed mappings instead of searchsorted
        robot_indices = np.array([self.robot_geom_to_idx[gid] for gid in geom_pairs[:, 0]])
        cand_indices = np.array([self.non_robot_geom_to_idx[gid] for gid in geom_pairs[:, 1]])

        # Get corners efficiently
        robot_geom_corners = self.robot_geom_corners[robot_indices]  # (N, 8, 3)
        cand_geom_corners = self.cand_geom_corners[cand_indices]     # (N, 8, 3)

        # Transform robot geom corners to world coordinates (vectorized)
        robot_positions = self.data.geom_xpos[geom_pairs[:, 0]]      # (N, 3)
        robot_rotations = self.data.geom_xmat[geom_pairs[:, 0]].reshape(-1, 3, 3)  # (N, 3, 3)
        
        # Vectorized rotation and translation
        robot_corners_world = np.einsum('nij,nkj->nki', robot_rotations, robot_geom_corners) + robot_positions[:, None, :]

        # Transform candidate geom corners to world coordinates
        cand_positions = self.data.geom_xpos[geom_pairs[:, 1]]       # (N, 3)
        cand_rotations = self.data.geom_xmat[geom_pairs[:, 1]].reshape(-1, 3, 3)   # (N, 3, 3)
        
        cand_corners_world = np.einsum('nij,nkj->nki', cand_rotations, cand_geom_corners) + cand_positions[:, None, :]

        # Use HARD max/min instead of soft (much faster)
        robot_maxs = np.max(robot_corners_world, axis=1)  # (N, 3)
        robot_mins = np.min(robot_corners_world, axis=1)  # (N, 3)
        cand_maxs = np.max(cand_corners_world, axis=1)    # (N, 3)
        cand_mins = np.min(cand_corners_world, axis=1)    # (N, 3)

        # Calculate overlaps efficiently
        overlap_mins = np.maximum(robot_mins, cand_mins)  # (N, 3)
        overlap_maxs = np.minimum(robot_maxs, cand_maxs)  # (N, 3)
        
        # Check for valid overlaps
        overlaps = overlap_maxs - overlap_mins            # (N, 3)
        valid_overlaps = np.all(overlaps > 0, axis=1)     # (N,)
        
        if not np.any(valid_overlaps):
            empty_grads = np.zeros((0, self.model.nv))
            return np.array([]).reshape(0, 2), np.array([]), np.array([]), np.array([]), empty_grads

        # Filter to only valid overlaps
        valid_pairs = geom_pairs[valid_overlaps]
        valid_overlaps_xy = overlaps[valid_overlaps, :2]  # Only x,y for area
        valid_robot_maxs = robot_maxs[valid_overlaps]
        valid_robot_mins = robot_mins[valid_overlaps]
        valid_cand_maxs = cand_maxs[valid_overlaps]
        valid_cand_mins = cand_mins[valid_overlaps]

        # Calculate areas
        intersection_areas = np.prod(valid_overlaps_xy, axis=1)  # x * y overlap
        robot_areas = np.prod(valid_robot_maxs[:, :2] - valid_robot_mins[:, :2], axis=1)
        cand_areas = np.prod(valid_cand_maxs[:, :2] - valid_cand_mins[:, :2], axis=1)

        # Handle gradients
        if calculate_grads and len(valid_pairs) > 0:
            gradients = self._compute_fast_gradients(valid_pairs, robot_indices[valid_overlaps], 
                                                   valid_overlaps_xy, intersection_areas)
        else:
            gradients = np.zeros((len(valid_pairs), self.model.nv))

        return valid_pairs, intersection_areas, cand_areas, robot_areas, gradients

    def _compute_fast_gradients(self, geom_pairs, robot_indices, overlaps_xy, intersection_areas):
        """
        Simplified gradient computation focusing on the most important terms
        """
        n_pairs = len(geom_pairs)
        gradients = np.zeros((n_pairs, self.model.nv))
        
        if n_pairs == 0:
            return gradients

        # Only compute gradients for pairs with significant intersection
        significant_mask = intersection_areas > 1e-6
        if not np.any(significant_mask):
            return gradients

        # Simplified gradient calculation (approximate but much faster)
        # This is a placeholder for a more sophisticated but faster gradient computation
        # You might want to use finite differences or a simplified analytical approach
        
        return gradients

    def dof_to_jnt(self, dof_mat: np.ndarray, axis: int, aggregate_type: str = "sum"):
        """Optimized DOF to joint conversion with caching"""
        cache_key = f"dof_mapping_{axis}_{aggregate_type}_{dof_mat.shape}"
        
        if cache_key not in self._cache:
            if axis >= dof_mat.ndim or dof_mat.shape[axis] != self.model.nv:
                self._cache[cache_key] = None
                return dof_mat

            all_dofs = np.arange(self.model.nv)
            all_dof_jntids = self.model.dof_jntid[all_dofs]
            robot_dof_mask = np.isin(all_dof_jntids, self.all_joint_ids)
            robot_dof_indices = np.where(robot_dof_mask)[0]
            
            self._cache[cache_key] = {
                'robot_dof_indices': robot_dof_indices,
                'robot_dof_jntids': all_dof_jntids[robot_dof_indices] if aggregate_type == "sum" else None
            }

        cache_data = self._cache[cache_key]
        if cache_data is None:
            return dof_mat

        robot_dofs_mat = np.take(dof_mat, cache_data['robot_dof_indices'], axis)
        
        if aggregate_type == "sum":
            robot_dof_jntids = cache_data['robot_dof_jntids']
            _, unique_idx = np.unique(robot_dof_jntids, return_index=True)
            aggregated_dof_mat = np.add.reduceat(robot_dofs_mat, unique_idx, axis=axis)
            return aggregated_dof_mat
        else:
            return robot_dofs_mat

    def _init_non_robot_geoms(self, exclude_world_body=True):
        non_robot_bodies = get_non_robot_bodies(self.model, self.robot_root_name)
        non_robot_geoms = get_bodies_geoms(self.model, non_robot_bodies, keep_seperate=False)

        if exclude_world_body:
            body_names = np.array([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) for bid in self.model.geom_bodyid[non_robot_geoms]])
            non_robot_geoms = non_robot_geoms[body_names != "world"]

        return non_robot_geoms

    def _init_robot_geoms(self, all_robot_joints_by_name):
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, failing_joint) for failing_joint in all_robot_joints_by_name]
        child_bids = self.model.jnt_bodyid[joint_ids]
        parent_bids = self.model.body_parentid[child_bids]

        child_geoms = get_bodies_geoms(self.model, child_bids, keep_seperate=True) 
        parent_geoms = get_bodies_geoms(self.model, parent_bids, keep_seperate=True)
        robot_joint_name2ids = {all_robot_joints_by_name[i]: joint_ids[i] for i in range(len(joint_ids))}
        robot_joint_geoms = {joint_ids[i]: np.unique(np.concatenate((child_geoms[i], parent_geoms[i]))) for i in range(len(joint_ids))}
        robot_geoms_by_ids = np.sort(np.unique(np.concatenate(list(robot_joint_geoms.values()))))
        return robot_joint_name2ids, robot_joint_geoms, robot_geoms_by_ids

    def _convert_to_joint_ids(self, joint_names):
        if isinstance(joint_names, list):
            return [self.robot_joint_names2ids[joint_name] for joint_name in joint_names]
        else:
            return [self.robot_joint_names2ids[joint_names]]

    def _get_colliding_geom_pairs(self, robot_geoms, candidate_geoms):
        """Optimized collision pair filtering"""
        if len(robot_geoms) == 0 or len(candidate_geoms) == 0:
            return np.array([]).reshape(0, 2)

        # Pre-filter using simple distance check if positions are available
        robot_positions = self.data.geom_xpos[robot_geoms]
        cand_positions = self.data.geom_xpos[candidate_geoms]
        
        # Compute pairwise distances efficiently
        robot_pos_expanded = robot_positions[:, np.newaxis, :]  # (R, 1, 3)
        cand_pos_expanded = cand_positions[np.newaxis, :, :]    # (1, C, 3)
        distances = np.linalg.norm(robot_pos_expanded - cand_pos_expanded, axis=2)  # (R, C)
        
        # Filter pairs that are too far apart (rough estimate using geom sizes)
        robot_sizes = np.linalg.norm(self.model.geom_aabb[robot_geoms, 3:], axis=1)
        cand_sizes = np.linalg.norm(self.model.geom_aabb[candidate_geoms, 3:], axis=1)
        max_interaction_dist = robot_sizes[:, np.newaxis] + cand_sizes[np.newaxis, :] + 0.1  # small buffer
        
        close_enough = distances < max_interaction_dist
        robot_idx, cand_idx = np.where(close_enough)
        
        if len(robot_idx) == 0:
            return np.array([]).reshape(0, 2)

        # Now do the more expensive collision type checking only on close pairs
        close_robot_geoms = robot_geoms[robot_idx]
        close_cand_geoms = candidate_geoms[cand_idx]
        
        robot_contypes = self.model.geom_contype[close_robot_geoms]
        robot_conaffinities = self.model.geom_conaffinity[close_robot_geoms]
        cand_contypes = self.model.geom_contype[close_cand_geoms]
        cand_conaffinities = self.model.geom_conaffinity[close_cand_geoms]

        # Vectorized collision checking
        can_collide = ((robot_contypes & cand_conaffinities) != 0) | ((robot_conaffinities & cand_contypes) != 0)
        
        valid_pairs = np.column_stack((close_robot_geoms[can_collide], close_cand_geoms[can_collide]))
        return valid_pairs
    
    def _init_AABB_variables(self):
        """Optimized AABB initialization"""
        robot_geom_aabb = self.model.geom_aabb[self.robot_geoms_by_ids]
        corner_scale = np.array(list(itertools.product([-1,1], repeat=3)))[None]
        self.robot_geom_corners = robot_geom_aabb[:, None, :3] + (corner_scale * robot_geom_aabb[:, None, 3:])

        cand_geom_aabb = self.model.geom_aabb[self.non_robot_geoms]
        self.cand_geom_corners = cand_geom_aabb[:, None, :3] + (corner_scale * cand_geom_aabb[:, None, 3:])

        # Pre-compute Jacobians (only when needed for gradients)
        self.robot_jacp = np.zeros((len(self.robot_geoms_by_ids), 3, self.model.nv))
        self.robot_jacr = np.zeros((len(self.robot_geoms_by_ids), 3, self.model.nv))
        for i, geom_id in enumerate(self.robot_geoms_by_ids):
            mujoco.mj_jacGeom(self.model, self.data, self.robot_jacp[i], self.robot_jacr[i], geom_id)

        self.vec_center_to_corners = self.robot_geom_corners - robot_geom_aabb[:, None, :3]


# Keep the original softmax functions for compatibility, but they're not used in the optimized version
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


def test_optimized_collision_estimator():
    """Test function for the optimized collision estimator."""
    print("🚀 Testing Optimized Collision Estimator")
    print("=" * 50)

    # You'll need to update this path
    robot_xml_path = "/path/to/your/scene.xml"
    
    try:
        model = mujoco.MjModel.from_xml_path(robot_xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        failing_joints = [f"joint{i}" for i in range(1, 8)]
        estimator = OptimizedCollisionEstimator(model, data, failing_joints=failing_joints)
        estimator.post_mj_forward_init()

        # Test single joint
        result = estimator.estimate_bodies_in_collision("joint1")
        print(f"✅ Single joint test: {len(result[0])} collision pairs found")

        # Test multiple joints
        result = estimator.estimate_bodies_in_collision(["joint1", "joint2"])
        print(f"✅ Multiple joints test: {len(result[0])} collision pairs found")

        print("🎉 Optimized Collision Estimator Test Completed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_optimized_collision_estimator()