import mujoco
import numpy as np

class CollisionEstimator:

    def __init__(self, model, data, method_type="bounding_sphere", inflation_radius=0):
        self.model = model
        self.data = data
        self.method_type = method_type
        self.inflation_radius = inflation_radius
        self.collision_function = None
    
    def estimate_collision_function(self, failing_joint, failure_type="aggressive"):
        """
        The main method that will estimate the volume of collision when failure_type happens
        at failing_joint.
        failing_joint: the joint that fails (string)
        failure_type: fixed to the only failure we have. string but can be enum.
        """
        assert failure_type == "aggressive", "No other failure type is supported."

        # get attached bodies 
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, failing_joint)
        child_bid  = mujoco.jnt_bodyid[joint_id]
        parent_bid = mujoco.body_parentid[child_bid]

        # get geoms in bodies
        child_geoms = [mujoco.body_geomadr[addr] for addr in range(mujoco.body_geomadr[child_bid], mujoco.body_geomadr[child_bid] + mujoco.body_geomnum[child_bid])]
        parent_geoms = [mujoco.body_geomadr[addr] for addr in range(mujoco.body_geomadr[parent_bid], mujoco.body_geomadr[parent_bid] + mujoco.body_geomnum[parent_bid])]
        geoms_failing = child_geoms + parent_geoms

        if self.method_type == "bounding_sphere":
            self.collision_function = self._bounding_sphere_method(geoms_failing)
        elif self.method_type == "AABB":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        
    def _bounding_sphere_method(self, geom_ids):
        # get bounding sphere and world coordinates of geoms
        geom_coords = [self.data.geom_xpos[geom_id] for geom_id in geom_ids]
        geom_bounding_radius = [self.data.geom_rbound[geom_id] + self.inflation_radius for geom_id in geom_ids]
        joint_sphere_data = [[x,y,z,r] for x,y,z,r in zip(geom_coords, geom_bounding_radius)]

        # eqn of sphere: r^2 = (x-h)^2 + (y-k)^2 + (z-l)^2; (h,k,l) are coord center. if RHS < r^2, inside sphere. 
        # eqn of cylinder: constraint: z_min <= z <= z_max, equation: r^2 = (x-x_0)^2 + (y-y_0)^2

        # given the (x_i, y_i, z_i, r_i) of all the geom bounding spheres, we make the following system of linear equations:
        # [(x-x_i)^2 + (y-y_i)^2 <= r_i^2] for i in range(len(geom_ids)) with the following constraint z <= z_i
        def points_under_spheres(points):
            """
            Vectorized: given Nx3 points and M spheres, return boolean mask of length N.
            """
            points = np.asarray(points)
            mask = np.zeros(len(points), dtype=bool)
            for (xi, yi, zi, ri) in joint_sphere_data:
                dx = points[:,0] - xi
                dy = points[:,1] - yi
                cond = (dx*dx + dy*dy <= ri*ri) & (points[:,2] <= zi)
                mask |= cond
            return mask
        return points_under_spheres


    def _axis_aligned_bounding_box_method(self):
        pass

    def extract_bodies_in_collision(self):
        assert self.cocollision_function is not None

        