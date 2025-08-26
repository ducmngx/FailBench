import mujoco
import numpy as np

def get_body_descendants(model, root_bid):
    """
    Return all body IDs in the subtree rooted at root_bid.
    Vectorized using NumPy (no inner Python loop over children).
    """
    parent = np.asarray(model.body_parentid)
    descendants = set([root_bid])
    frontier = {root_bid}

    while frontier:
        # For all bodies whose parent is in frontier
        mask = np.isin(parent, list(frontier))
        children = set(np.nonzero(mask)[0])
        # Add new children to descendants
        new = children - descendants
        if not new:
            break
        descendants |= new
        frontier = new

    return descendants

def get_non_robot_bodies(model, robot_root_name):
    # Robot root ID
    robot_root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root_name)

    # All robot bodies (subtree of robot root)
    robot_bodies = get_body_descendants(model, robot_root_id)

    # All bodies in scene
    all_bodies = set(range(model.nbody))

    # Everything else is "non-robot"
    non_robot_bodies = all_bodies - robot_bodies

    return non_robot_bodies

def get_bodies_geoms(model, body_ids, keep_seperate=False):
    body_geoms = [get_body_geoms(model, body_id) for body_id in body_ids]
    if keep_seperate:
        return body_geoms
    else:
        return np.concatenate(body_geoms)


def get_body_geoms(model, body_id):
    return np.arange(model.body_geomadr[body_id], model.body_geomadr[body_id] + model.body_geomnum[body_id])