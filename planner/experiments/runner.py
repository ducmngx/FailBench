"""Single headless experiment trial: execute trajectory, inject failures, collect data."""

import logging
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import mujoco
import numpy as np

from planner.experiments.config import ExperimentConfig, FailureConfig, FailureMode
from planner.experiments.data_capture import (
    ContactExtractor,
    ContactPoint,
    OffscreenRenderer,
    RobotState,
    RobotStateCollector,
    SimCheckpoint,
    SimStateCheckpoint,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FailureResult:
    """Contacts collected from a single failure mode."""
    failure_config: FailureConfig
    contacts: List[ContactPoint]
    impacted_geom_ids: List[int]
    had_collision: bool


@dataclass
class DataSample:
    """Full output of one experiment trial."""
    experiment_id: str
    trajectory_file: str
    seed: int
    fail_phase: int
    fail_step: int
    sim_time_at_failure: float

    # Pre-failure state (shared across all failure modes)
    pre_failure_rgb: np.ndarray
    pre_failure_depth: Optional[np.ndarray]
    pre_failure_robot: RobotState

    # Per-failure-mode results
    failure_results: List[FailureResult]

    # Aggregate contact cloud (union across all failure modes)
    aggregate_contacts: List[ContactPoint]
    all_impacted_geom_ids: List[int]

    # Optional fields
    post_failure_rgb: Optional[np.ndarray] = None
    extra_camera_views: Optional[dict] = None


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Runs a single headless pick-and-place trial with multi-failure contact collection.

    Replays pre-computed trajectories headlessly (no IK/RRT planning needed),
    then at the chosen failure point forks the sim state for each failure mode
    and collects 3D contact clouds.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(config.scene_xml_path)
        self.data = mujoco.MjData(self.model)
        _xml_stem = os.path.splitext(os.path.basename(config.scene_xml_path))[0]
        # If the XML is named generically (e.g. "scene.xml"), use the parent directory name
        if _xml_stem == "scene":
            self.scene_name = os.path.basename(os.path.dirname(config.scene_xml_path))
        else:
            self.scene_name = _xml_stem

        # Capture utilities
        self.renderer = OffscreenRenderer(
            self.model,
            height=config.image_height,
            width=config.image_width,
            camera_name=config.camera_name,
            camera_lookat=config.camera_lookat,
            camera_distance=config.camera_distance,
            camera_azimuth=config.camera_azimuth,
            camera_elevation=config.camera_elevation,
            extra_cameras=config.extra_cameras,
        )
        self.contact_extractor = ContactExtractor(self.model)
        self.state_collector = RobotStateCollector(self.model)

        # Failure injector
        from failure_injection.agressive_injector import AggressiveFailureInjector
        self.injector = AggressiveFailureInjector(self.model, self.data)

        # Trajectory manager
        from planner.utils.traj_saver import ExperimentTrajectoryManager
        self.traj_manager = ExperimentTrajectoryManager()

        # Store initial object3 position for reset
        obj_pos = self._get_body_pos("object3")
        self._object3_init_qpos = np.append(obj_pos.copy(), [0, 0, 0, 1])

        # RNG
        self._rng = random.Random(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Optional[DataSample]:
        """Execute the full trial and return collected data."""
        config = self.config
        experiment_id = config.experiment_id or f"exp_{config.seed}"

        # Load trajectory
        self.traj_manager.load_from_file(config.trajectory_file)
        scene_trajs = self.traj_manager.trajectories[self.scene_name]

        if "baseline" in config.trajectory_file:
            saved_trajectories = {"phase7": scene_trajs["baseline"]["trajectory"]}
            has_phase6 = False
        else:
            saved_trajectories = {
                "phase6": scene_trajs["phase6"]["trajectory"],
                "phase7": scene_trajs["phase7"]["trajectory"],
            }
            has_phase6 = True

        # Get the first waypoint of the first saved trajectory
        first_phase = "phase6" if has_phase6 else "phase7"
        first_config = saved_trajectories[first_phase][0]

        # Reset scene and set robot to the trajectory start with object grasped
        self._reset_object3()
        self._setup_grasped_state(first_config)

        # Decide failure phase
        if config.fail_phase is not None:
            fail_phase = config.fail_phase
        else:
            fail_phase = self._rng.choices([6, 7], weights=[0.3, 0.7], k=1)[0]

        # Collect phase trajectories in order
        phase_sequence = []
        if has_phase6:
            phase_sequence.append((6, saved_trajectories["phase6"]))
        phase_sequence.append((7, saved_trajectories["phase7"]))

        return self._run_with_failure(phase_sequence, fail_phase, experiment_id)

    def close(self):
        self.renderer.close()

    # ------------------------------------------------------------------
    # Setup: place robot at trajectory start with object grasped
    # ------------------------------------------------------------------

    def _setup_grasped_state(self, start_config: np.ndarray):
        """Set robot to the start config of the saved trajectory and grasp object3.

        The saved trajectories assume the robot is already holding the object at
        transport height. We set the joint positions directly, move object3 to be
        between the gripper fingers, close the gripper, and let physics settle.
        """
        n = min(len(start_config), 7)

        # Set arm joints to the trajectory start
        self.data.qpos[:n] = start_config[:n]
        # Finger joints slightly open for grasping
        self.data.qpos[7] = 0.02
        self.data.qpos[8] = 0.02
        self.data.ctrl[:n] = start_config[:n]
        self.data.ctrl[7] = 50.0  # partially closed
        mujoco.mj_forward(self.model, self.data)

        # Move object3 to the end-effector position
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        ee_pos = self.data.site_xpos[ee_site_id].copy()

        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object3")
        jid = self.model.body_jntadr[bid]
        qpos_adr = self.model.jnt_qposadr[jid]
        # Place object3 at EE position, slightly below
        obj_pos = ee_pos.copy()
        obj_pos[2] -= 0.02  # offset below EE
        self.data.qpos[qpos_adr:qpos_adr + 3] = obj_pos
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]  # identity quat
        # Zero object velocity
        obj_vel_adr = self.model.jnt_dofadr[jid]
        self.data.qvel[obj_vel_adr:obj_vel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Close gripper around object
        self._close_gripper_headless(target_force=5000.0)

        # Let physics settle
        self._settle(500)

    # ------------------------------------------------------------------
    # Failure phase: multi-failure fork-and-collect
    # ------------------------------------------------------------------

    def _run_with_failure(self, phase_sequence, target_fail_phase, experiment_id) -> Optional[DataSample]:
        """Replay trajectory phases, injecting failure at the chosen point."""
        config = self.config
        settle_time = 200  # physics steps per waypoint

        for phase_id, phase_traj in phase_sequence:
            # Move to start of this phase trajectory
            self.data.ctrl[:7] = phase_traj[0][:7]
            self._close_gripper_headless(target_force=5000.0)
            self._settle(settle_time)

            # Pick failure step for this phase
            if phase_id == target_fail_phase:
                if config.fail_step_offset is not None:
                    fail_at = min(config.fail_step_offset, len(phase_traj) - 1)
                else:
                    fail_at = self._rng.randint(1, max(1, len(phase_traj) - 1))

                # Execute up to failure point
                for target_config in phase_traj[:fail_at]:
                    self.data.ctrl[:7] = target_config[:7]
                    self._close_gripper_headless(target_force=5000.0)
                    for _ in range(settle_time):
                        mujoco.mj_step(self.model, self.data)

                # Capture and fork
                return self._capture_and_fork(
                    fail_phase=phase_id,
                    fail_step=fail_at,
                    experiment_id=experiment_id,
                )
            else:
                # Execute full phase without failure
                for target_config in phase_traj:
                    self.data.ctrl[:7] = target_config[:7]
                    self._close_gripper_headless(target_force=5000.0)
                    for _ in range(settle_time):
                        mujoco.mj_step(self.model, self.data)

        # If we get here, the target phase wasn't in the sequence — use last phase
        last_phase_id, last_traj = phase_sequence[-1]
        fail_at = self._rng.randint(1, max(1, len(last_traj) - 1))
        return self._capture_and_fork(
            fail_phase=last_phase_id,
            fail_step=fail_at,
            experiment_id=experiment_id,
        )

    def _capture_and_fork(self, fail_phase, fail_step, experiment_id) -> DataSample:
        """Capture pre-failure state, then fork for each failure mode."""
        config = self.config

        # --- Capture pre-failure state (all cameras) ---
        all_views = self.renderer.render_all_cameras(self.data)
        # Primary camera
        primary_name = list(all_views.keys())[0]
        pre_rgb, pre_depth = all_views[primary_name]
        # Extra cameras
        extra_views = {k: v for k, v in all_views.items() if k != primary_name}
        pre_robot = self.state_collector.snapshot(self.data)
        sim_time = float(self.data.time)

        # Save checkpoint for forking
        checkpoint = SimStateCheckpoint.save(self.data)

        # --- Run each failure mode ---
        failure_results: List[FailureResult] = []
        all_contacts: List[ContactPoint] = []
        all_geom_ids: set = set()
        post_rgb = None

        for fc in config.failure_configs:
            # Restore to pre-failure state
            SimStateCheckpoint.restore(self.model, self.data, checkpoint)

            # Inject failure
            self._inject_failure(fc)

            # Settle and collect contacts
            contacts: List[ContactPoint] = []
            for _ in range(config.post_failure_settle_steps):
                mujoco.mj_step(self.model, self.data)
                step_contacts = self.contact_extractor.extract_contacts(
                    self.model, self.data, filter_target=False, min_force=1.0)
                contacts.extend(step_contacts)

            # Unique impacted geom IDs (excluding object3 itself)
            impacted = set()
            for c in contacts:
                impacted.add(c.geom1)
                impacted.add(c.geom2)
            impacted.discard(86)  # object3_geom
            impacted_list = sorted(impacted)

            failure_results.append(FailureResult(
                failure_config=fc,
                contacts=contacts,
                impacted_geom_ids=impacted_list,
                had_collision=len(contacts) > 0,
            ))

            all_contacts.extend(contacts)
            all_geom_ids.update(impacted)

            # Capture post-failure RGB from the last mode
            post_rgb = self.renderer.render(self.data)

            # Restore model params (joint stiffness, damping, etc.)
            self.injector.restore_all_joints()

        return DataSample(
            experiment_id=experiment_id,
            trajectory_file=config.trajectory_file,
            seed=config.seed,
            fail_phase=fail_phase,
            fail_step=fail_step,
            sim_time_at_failure=sim_time,
            pre_failure_rgb=pre_rgb,
            pre_failure_depth=pre_depth,
            pre_failure_robot=pre_robot,
            extra_camera_views=extra_views if extra_views else None,
            failure_results=failure_results,
            aggregate_contacts=all_contacts,
            all_impacted_geom_ids=sorted(all_geom_ids),
            post_failure_rgb=post_rgb,
        )

    # ------------------------------------------------------------------
    # Failure injection dispatch
    # ------------------------------------------------------------------

    def _inject_failure(self, fc: FailureConfig):
        """Apply a single failure mode to the current simulation state."""
        mode = fc.mode
        if mode == FailureMode.GRIPPER_OPEN:
            self.data.ctrl[7] = 255.0
        elif mode == FailureMode.SLIPPERY_GRIP:
            self.data.ctrl[7] = fc.grip_value
        elif mode == FailureMode.SINGLE_JOINT:
            if fc.joint_names:
                self.injector.turn_off_joint_aggressive(fc.joint_names[0])
        elif mode == FailureMode.MULTI_JOINT:
            if fc.joint_names:
                self.injector.turn_off_multiple_joints(fc.joint_names)
        elif mode == FailureMode.ALL_JOINTS:
            self.injector.turn_off_all_joints()

    # ------------------------------------------------------------------
    # Headless physics helpers (no viewer, no time.sleep)
    # ------------------------------------------------------------------

    def _open_gripper_headless(self):
        self.data.ctrl[7] = 255.0
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)

    def _close_gripper_headless(self, target_force: float = 5000.0):
        """Gradual close matching close_gripper_gentle logic."""
        initial_control = self.data.ctrl[7]
        max_step = 15
        for step in range(max_step):
            progress = step / max_step
            target_control = initial_control * (1 - progress * 0.8)
            self.data.ctrl[7] = target_control
            mujoco.mj_step(self.model, self.data)

            # Check force on object3
            max_force = 0.0
            in_contact = False
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                if g1_name and g2_name:
                    pair = (g1_name.lower(), g2_name.lower())
                    finger_obj = any(
                        ("finger" in a or "pad" in a) and "object3" in b
                        for a, b in [pair, pair[::-1]]
                    )
                    if finger_obj:
                        in_contact = True
                        max_force = max(max_force, np.linalg.norm(c.f[:3]))

            if in_contact and max_force > target_force:
                break

        # Hold
        final = self.data.ctrl[7]
        for _ in range(100):
            self.data.ctrl[7] = final
            mujoco.mj_step(self.model, self.data)

    def _settle(self, steps: int):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def _get_body_pos(self, name: str) -> Optional[np.ndarray]:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            return None
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[body_id].copy()

    def _reset_object3(self):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object3")
        jid = self.model.body_jntadr[bid]
        qpos_adr = self.model.jnt_qposadr[jid]
        self.data.qpos[qpos_adr : qpos_adr + 7] = self._object3_init_qpos
        mujoco.mj_forward(self.model, self.data)

    def _reset_robot(self):
        """Set robot to a valid home position (respecting joint limits)."""
        # Panda joint limits require non-zero defaults for joint4 and joint6
        robot_model = mujoco.MjModel.from_xml_path(self.config.robot_xml_path)
        n_joints = robot_model.njnt
        home = np.zeros(n_joints)
        # Set joints to center of their limits
        for i in range(n_joints):
            lo, hi = robot_model.jnt_range[i]
            if robot_model.jnt_limited[i]:
                center = (lo + hi) / 2.0
                # Only override if 0 is outside limits
                if 0.0 < lo or 0.0 > hi:
                    home[i] = center
        self.data.qpos[:n_joints] = home
        self.data.ctrl[:7] = home[:7]
        mujoco.mj_forward(self.model, self.data)
        self._settle(200)
