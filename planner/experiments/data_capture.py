"""Data capture utilities: offscreen rendering, contact extraction, state snapshots."""

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
