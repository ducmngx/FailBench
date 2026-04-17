"""Kinematic grasp lock — attach an object to the gripper hand.

Instead of relying on MuJoCo's contact solver to hold mesh objects (which
fails for complex collision hulls), this locks the object's free joint to
the hand body via a recorded relative transform.  Call ``update()`` every
sim step while the lock is active.

On ``release()`` the object is freed and falls under gravity — exactly what
happens on a GRIPPER_OPEN or SLIPPERY_GRIP failure.
"""

import mujoco
import numpy as np


def _pose_matrix(pos, mat3x3):
    T = np.eye(4)
    T[:3, :3] = mat3x3
    T[:3, 3] = pos
    return T


def _mat_to_quat_wxyz(R):
    """Rotation matrix → quaternion [w, x, y, z]."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


class GraspLock:
    """Kinematic object-to-hand attachment."""

    def __init__(self, model, hand_body="hand"):
        self._hand_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
        self._obj_bid = -1
        self._qpos_adr = -1
        self._qvel_adr = -1
        self._T_hand_to_obj = np.eye(4)
        self.active = False

    def attach_strict(self, model, data, obj_body_name) -> bool:
        """Attach iff at least one finger body is in contact with the object.

        Returns True if the lock engaged, False if no finger↔object contact was
        found (caller should treat this as a failed grasp and let physics run).
        """
        obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name)
        finger_bids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("left_finger", "right_finger")
        }
        finger_bids.discard(-1)
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            if (b1 == obj_bid and b2 in finger_bids) or \
               (b2 == obj_bid and b1 in finger_bids):
                self.attach(model, data, obj_body_name)
                return True
        return False

    def attach(self, model, data, obj_body_name):
        """Record hand⁻¹ @ obj and start tracking.  Call after gripper-close settle."""
        self._obj_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, obj_body_name)
        # Find the free joint for this body
        for j in range(model.njnt):
            if model.jnt_bodyid[j] == self._obj_bid and model.jnt_type[j] == 0:
                self._qpos_adr = model.jnt_qposadr[j]
                self._qvel_adr = model.jnt_dofadr[j]
                break

        T_hand = _pose_matrix(
            data.xpos[self._hand_bid],
            data.xmat[self._hand_bid].reshape(3, 3))
        T_obj = _pose_matrix(
            data.xpos[self._obj_bid],
            data.xmat[self._obj_bid].reshape(3, 3))
        self._T_hand_to_obj = np.linalg.inv(T_hand) @ T_obj
        self.active = True

    def update(self, data):
        """Set object qpos to track hand.  Call after each mj_step."""
        if not self.active:
            return
        T_hand = _pose_matrix(
            data.xpos[self._hand_bid],
            data.xmat[self._hand_bid].reshape(3, 3))
        T_obj = T_hand @ self._T_hand_to_obj
        a = self._qpos_adr
        data.qpos[a:a + 3] = T_obj[:3, 3]
        data.qpos[a + 3:a + 7] = _mat_to_quat_wxyz(T_obj[:3, :3])
        v = self._qvel_adr
        data.qvel[v:v + 6] = 0.0

    def release(self, data):
        """Stop tracking — object falls under gravity from current pose."""
        self.active = False
