"""Data capture utilities: offscreen rendering, contact extraction, state snapshots."""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContactPoint:
    """A single contact point with full physical data."""
    pos: np.ndarray            # (3,) world position
    normal: np.ndarray         # (3,) contact normal
    force: np.ndarray          # (6,) full contact force
    geom1: int                 # geom ID of first body
    geom2: int                 # geom ID of second body
    penetration: float         # contact.dist (negative = penetrating)


@dataclass
class RobotState:
    """Snapshot of robot state at a single timestep."""
    qpos: np.ndarray           # (7,) joint positions
    qvel: np.ndarray           # (7,) joint velocities
    ee_pos: np.ndarray         # (3,) end-effector world position
    gripper_ctrl: float        # ctrl[7] value
    sim_time: float


@dataclass
class SimCheckpoint:
    """Full MuJoCo simulation state for save/restore."""
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    act: np.ndarray
    time: float


# ---------------------------------------------------------------------------
# OffscreenRenderer
# ---------------------------------------------------------------------------

class OffscreenRenderer:
    """Headless RGB rendering via mujoco.Renderer."""

    def __init__(self, model: mujoco.MjModel, height: int = 480, width: int = 640,
                 camera_name: Optional[str] = "overhead_cam",
                 camera_lookat: Optional[List[float]] = None,
                 camera_distance: Optional[float] = None,
                 camera_azimuth: Optional[float] = None,
                 camera_elevation: Optional[float] = None):
        self.model = model
        self.renderer = mujoco.Renderer(model, height, width)

        if camera_name is not None:
            self.camera = camera_name
        else:
            cam = mujoco.MjvCamera()
            if camera_lookat is not None:
                cam.lookat[:] = camera_lookat
            if camera_distance is not None:
                cam.distance = camera_distance
            if camera_azimuth is not None:
                cam.azimuth = camera_azimuth
            if camera_elevation is not None:
                cam.elevation = camera_elevation
            self.camera = cam

    def render(self, data: mujoco.MjData) -> np.ndarray:
        """Render a single RGB frame. Returns (H, W, 3) uint8."""
        self.renderer.update_scene(data, self.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


# ---------------------------------------------------------------------------
# ContactExtractor
# ---------------------------------------------------------------------------

# Geom IDs for all obstacle / object geoms on the table (from scene_level2.xml).
# object3_geom = 86 is the grasped object.
_OBJECT3_GEOM_ID = 86

# Robot geom ID range (panda links + fingers). These are excluded from
# "environment contact" extraction so we only keep object-vs-obstacle contacts.
# Determined empirically from the compiled scene model; adjust if the XML changes.
_ROBOT_GEOM_PREFIX = "link"
_FINGER_GEOM_SUBSTRINGS = ("finger", "pad")


class ContactExtractor:
    """Extract 3D contact data from MuJoCo simulation state."""

    def __init__(self, model: mujoco.MjModel, target_geom_id: int = _OBJECT3_GEOM_ID):
        self.model = model
        self.target_geom_id = target_geom_id

        # Build set of robot geom IDs to exclude
        self._robot_geom_ids = set()
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if name is None:
                continue
            lower = name.lower()
            if lower.startswith(_ROBOT_GEOM_PREFIX) or any(s in lower for s in _FINGER_GEOM_SUBSTRINGS):
                self._robot_geom_ids.add(gid)

    def extract_contacts(self, model: mujoco.MjModel, data: mujoco.MjData,
                         filter_target: bool = True) -> List[ContactPoint]:
        """Extract contact points from current simulation state.

        If filter_target is True, only returns contacts involving target_geom_id
        with non-robot geoms (i.e. object3 hitting obstacles / table).
        """
        contacts: List[ContactPoint] = []
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            if filter_target:
                # Must involve the target geom
                if g1 != self.target_geom_id and g2 != self.target_geom_id:
                    continue
                # Skip robot-object contacts (finger grasping, link brushing)
                if g1 in self._robot_geom_ids or g2 in self._robot_geom_ids:
                    continue

            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)

            contacts.append(ContactPoint(
                pos=c.pos.copy(),
                normal=c.frame[:3].copy(),
                force=force,
                geom1=g1,
                geom2=g2,
                penetration=float(c.dist),
            ))
        return contacts


# ---------------------------------------------------------------------------
# RobotStateCollector
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ContactProjector — 3D world → 2D pixel projection
# ---------------------------------------------------------------------------


class ContactProjector:
    """Project 3D world contact points to 2D pixel coordinates using MuJoCo camera."""

    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480,
                 camera_name: Optional[str] = None,
                 camera_lookat: Optional[List[float]] = None,
                 camera_distance: Optional[float] = None,
                 camera_azimuth: Optional[float] = None,
                 camera_elevation: Optional[float] = None,
                 fovy: float = 45.0):
        self.width = width
        self.height = height
        self._model = model

        if camera_name is not None:
            # Named camera from XML
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id < 0:
                raise ValueError(f"Camera '{camera_name}' not found in model")
            fovy_deg = model.cam_fovy[cam_id]
            self.cam_pos = model.cam_pos[cam_id].copy()
            self.cam_rot = model.cam_mat0[cam_id].reshape(3, 3).copy()
        else:
            # Free camera — extract pose via MjvScene
            fovy_deg = fovy
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            cam = mujoco.MjvCamera()
            if camera_lookat is not None:
                cam.lookat[:] = camera_lookat
            if camera_distance is not None:
                cam.distance = camera_distance
            if camera_azimuth is not None:
                cam.azimuth = camera_azimuth
            if camera_elevation is not None:
                cam.elevation = camera_elevation
            scene = mujoco.MjvScene(model, maxgeom=1000)
            mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None,
                                   cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            self.cam_pos = np.array(scene.camera[0].pos, dtype=np.float64)
            fwd = np.array(scene.camera[0].forward, dtype=np.float64)
            up = np.array(scene.camera[0].up, dtype=np.float64)
            # MuJoCo cam frame: x=right, y=down-in-image, looks along -z
            cam_z = -fwd
            cam_y = -up
            cam_x = np.cross(cam_y, cam_z)
            cam_x /= np.linalg.norm(cam_x)
            self.cam_rot = np.stack([cam_x, cam_y, cam_z])  # (3,3) rows = axes

        # Intrinsics from vertical field-of-view
        fovy_rad = math.radians(fovy_deg)
        fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx, cy = width / 2.0, height / 2.0
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]], dtype=np.float64)

    def project(self, world_points: np.ndarray):
        """Project (N, 3) world points to pixel coordinates.

        Returns
        -------
        pixels : (N, 2) float64 — (u, v) pixel coordinates
        depths : (N,) float64 — depth in front of camera (positive = visible)
        """
        pts = np.atleast_2d(world_points).astype(np.float64)
        # Transform to camera frame: p_cam = R @ (p_world - cam_pos)
        # R rows are camera axes in world, so R @ delta gives camera-frame coords
        delta = pts - self.cam_pos
        p_cam = delta @ self.cam_rot.T  # (N, 3) in camera frame

        # MuJoCo camera convention: X=right, Y=down, looks along -Z
        # Depth is along -Z, so positive depth = in front of camera
        depth = -p_cam[:, 2]
        # Avoid division by zero
        safe_depth = np.where(depth > 1e-8, depth, 1e-8)

        u = self.K[0, 0] * p_cam[:, 0] / safe_depth + self.K[0, 2]
        v = self.K[1, 1] * p_cam[:, 1] / safe_depth + self.K[1, 2]

        return np.column_stack([u, v]), depth

    def in_frame(self, pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """Boolean mask: True for points that fall within the image and are in front of camera."""
        u, v = pixels[:, 0], pixels[:, 1]
        return (depths > 0) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)

    def geom_name(self, geom_id: int) -> str:
        """Resolve a geom ID to its name, or 'geom_<id>' if unnamed."""
        name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        return name if name else f"geom_{geom_id}"


# ---------------------------------------------------------------------------
# RobotStateCollector
# ---------------------------------------------------------------------------


class RobotStateCollector:
    """Capture robot state snapshots."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        # Resolve end-effector site ID once
        self._ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    def snapshot(self, data: mujoco.MjData) -> RobotState:
        ee_pos = data.site_xpos[self._ee_site_id].copy() if self._ee_site_id >= 0 else np.zeros(3)
        return RobotState(
            qpos=data.qpos[:7].copy(),
            qvel=data.qvel[:7].copy(),
            ee_pos=ee_pos,
            gripper_ctrl=float(data.ctrl[7]),
            sim_time=float(data.time),
        )


# ---------------------------------------------------------------------------
# SimStateCheckpoint — save / restore full MuJoCo state
# ---------------------------------------------------------------------------

class SimStateCheckpoint:
    """Save and restore full MuJoCo data state for fork-and-run pattern."""

    @staticmethod
    def save(data: mujoco.MjData) -> SimCheckpoint:
        return SimCheckpoint(
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            ctrl=data.ctrl.copy(),
            act=data.act.copy(),
            time=float(data.time),
        )

    @staticmethod
    def restore(model: mujoco.MjModel, data: mujoco.MjData, checkpoint: SimCheckpoint):
        data.qpos[:] = checkpoint.qpos
        data.qvel[:] = checkpoint.qvel
        data.ctrl[:] = checkpoint.ctrl
        data.act[:] = checkpoint.act
        data.time = checkpoint.time
        mujoco.mj_forward(model, data)
