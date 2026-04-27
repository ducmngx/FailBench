"""Single headless experiment trial: execute trajectory, inject failures, collect data."""

import logging
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import mujoco
import numpy as np

from planner.experiments.config import ExperimentConfig, FailureConfig, FailureMode
from planner.grasp_lock import GraspLock
from planner.experiments.data_capture import (
    ContactExtractor,
    ContactPoint,
    OffscreenRenderer,
    RobotState,
    RobotStateCollector,
    SimCheckpoint,
    SimStateCheckpoint,
)
from planner.utils.trajectory_interpolation import interpolate_trajectory

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
    task_id: str
    traj_id: int
    seed: int
    traj_progress: float       # fraction [0, 1] of trajectory at which failure was injected
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

        # Store initial grasped object position for reset
        self._grasped_obj_name = config.grasped_object_name
        obj_pos = self._get_body_pos(self._grasped_obj_name)
        self._grasped_obj_init_qpos = np.append(obj_pos.copy(), [0, 0, 0, 1])

        # Resolve grasped object geom ID dynamically (not hardcoded)
        geom_name = f"{self._grasped_obj_name}_geom"
        self._grasped_obj_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if self._grasped_obj_geom_id < 0:
            logger.warning("Grasped object geom '%s' not found in model", geom_name)

        # RNG
        self._rng = random.Random(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Optional[DataSample]:
        """Execute the full trial and return collected data."""
        import pickle
        config = self.config
        experiment_id = config.experiment_id or f"exp_{config.seed}"

        with open(config.trajectory_file, "rb") as f:
            traj_data = pickle.load(f)
        scene_traj = traj_data[self.scene_name]

        # Segmented format: {"segments": [{"name", "trajectory", "action_after"}, ...]}
        if "segments" in scene_traj:
            segments = scene_traj["segments"]
            # Reset robot to home — full mission starts from home config
            self._reset_robot()
            self._reset_grasped_object()
            # Open gripper for approach phase
            self._open_gripper_headless()
            return self._run_segmented(segments, experiment_id)

        # Legacy flat format: {"trajectory": [...]}
        trajectory = scene_traj["trajectory"]
        self._reset_grasped_object()
        self._setup_grasped_state(trajectory[0])
        return self._run_with_failure(trajectory, experiment_id)

    def close(self):
        self.renderer.close()

    # ------------------------------------------------------------------
    # Setup: place robot at trajectory start with object grasped
    # ------------------------------------------------------------------

    def _setup_grasped_state(self, start_config: np.ndarray):
        """Set robot to the start config and grasp the target object.

        The saved trajectories assume the robot is already holding the object at
        transport height. We set the joint positions directly, move the grasped
        object to be between the gripper fingers, close the gripper, and settle.
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

        # Move grasped object to the end-effector position
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        ee_pos = self.data.site_xpos[ee_site_id].copy()

        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._grasped_obj_name)
        jid = self.model.body_jntadr[bid]
        qpos_adr = self.model.jnt_qposadr[jid]
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
    # Trajectory execution with failure injection
    # ------------------------------------------------------------------

    def _run_with_failure(self, trajectory: list, experiment_id: str) -> Optional[DataSample]:
        """Interpolate trajectory, inject failure at chosen fraction, fork and collect."""
        config = self.config

        grip_ctrl = self.data.ctrl[7]

        joint_limits = np.column_stack([
            self.model.jnt_range[:7, 0],
            self.model.jnt_range[:7, 1],
        ])
        dense_traj = interpolate_trajectory(
            trajectory,
            num_points_per_segment=config.interp_points_per_segment,
            method=config.interp_method,
            joint_limits=joint_limits,
        )
        total_points = len(dense_traj)

        # Choose failure fraction
        if config.fail_fraction is not None:
            frac = config.fail_fraction
        else:
            frac = self._rng.choice(config.canonical_fail_fractions)

        fail_at_point = max(1, min(int(frac * (total_points - 1)), total_points - 1))

        # Execute trajectory up to failure point
        for pt_idx in range(fail_at_point):
            self.data.ctrl[:7] = dense_traj[pt_idx]
            self.data.ctrl[7] = grip_ctrl
            for _ in range(config.steps_per_interp_point):
                mujoco.mj_step(self.model, self.data)

        # Advance one more step at the failure target
        self.data.ctrl[:7] = dense_traj[fail_at_point]
        self.data.ctrl[7] = grip_ctrl
        for _ in range(config.steps_per_interp_point):
            mujoco.mj_step(self.model, self.data)

        qvel_norm = np.linalg.norm(self.data.qvel[:7])
        logger.info(
            "Failure at fraction=%.2f, point=%d/%d, |qvel|=%.4f rad/s",
            frac, fail_at_point, total_points, qvel_norm,
        )

        return self._capture_and_fork(traj_progress=frac, experiment_id=experiment_id)

    # ------------------------------------------------------------------
    # Segmented trajectory execution (full pick-and-place mission)
    # ------------------------------------------------------------------

    def _run_segmented(self, segments: list, experiment_id: str) -> Optional[DataSample]:
        """Replay segmented trajectory with grasp/release actions, inject failure."""
        config = self.config

        joint_limits = np.column_stack([
            self.model.jnt_range[:7, 0],
            self.model.jnt_range[:7, 1],
        ])

        # Interpolate all segments and compute total point count
        dense_segments = []
        total_points = 0
        for seg in segments:
            dense = interpolate_trajectory(
                seg["trajectory"],
                num_points_per_segment=config.interp_points_per_segment,
                method=config.interp_method,
                joint_limits=joint_limits,
            )
            dense_segments.append({
                "name": seg["name"],
                "dense": dense,
                "action_after": seg.get("action_after"),
                "start_idx": total_points,
            })
            total_points += len(dense)

        # Choose failure point — resolved to an absolute sim-step index.
        total_sim_steps = total_points * config.steps_per_interp_point
        if config.fail_sim_step is not None:
            target_sim_step = int(config.fail_sim_step)
            target_sim_step = max(1, min(target_sim_step, total_sim_steps - 1))
            frac_reported = target_sim_step / (total_sim_steps - 1)
        else:
            if config.fail_fraction is not None:
                frac = config.fail_fraction
            else:
                frac = self._rng.choice(config.canonical_fail_fractions)
            target_sim_step = int(frac * (total_sim_steps - 1))
            target_sim_step = max(1, min(target_sim_step, total_sim_steps - 1))
            frac_reported = frac  # preserve exact requested fraction in the output

        # Replay segments
        grip_ctrl = self.data.ctrl[7]  # starts open
        global_sim_step = 0
        self._grasp_lock = GraspLock(self.model)

        for seg_info in dense_segments:
            dense = seg_info["dense"]
            seg_name = seg_info["name"]

            for pt in dense:
                self.data.ctrl[:7] = pt
                self.data.ctrl[7] = grip_ctrl
                for _ in range(config.steps_per_interp_point):
                    mujoco.mj_step(self.model, self.data)
                    self._grasp_lock.update(self.data)
                    if global_sim_step == target_sim_step:
                        logger.info(
                            "Failure at sim_step=%d/%d (frac=%.3f), segment='%s'",
                            global_sim_step, total_sim_steps, frac_reported, seg_name,
                        )
                        return self._capture_and_fork(
                            traj_progress=frac_reported, experiment_id=experiment_id)
                    global_sim_step += 1

            # Segment boundary actions
            action = seg_info["action_after"]
            if action == "grasp":
                self._close_gripper_headless(target_force=5000.0)
                grip_ctrl = self.data.ctrl[7]
                self._settle(200)
                self._grasp_lock.attach(self.model, self.data, self._grasped_obj_name)
            elif action == "release":
                self._grasp_lock.release(self.data)
                self._open_gripper_headless()
                grip_ctrl = self.data.ctrl[7]
                self._settle(200)

        # Fallback: failure at end of last segment
        logger.warning("Reached end of trajectory without failure injection")
        return self._capture_and_fork(
            traj_progress=1.0, experiment_id=experiment_id)

    def _capture_and_fork(self, traj_progress: float, experiment_id: str) -> DataSample:
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

        # --- Select failure modes ---
        if config.failure_sample_mode == "sample":
            weights = [fc.probability for fc in config.failure_configs]
            selected_failures = self._rng.choices(
                config.failure_configs,
                weights=weights,
                k=config.num_failure_samples,
            )
        else:
            selected_failures = config.failure_configs

        # --- Run each failure mode ---
        failure_results: List[FailureResult] = []
        all_contacts: List[ContactPoint] = []
        all_geom_ids: set = set()
        post_rgb = None

        for fc in selected_failures:
            # Restore to pre-failure state
            SimStateCheckpoint.restore(self.model, self.data, checkpoint)

            # Inject failure
            self._inject_failure(fc)

            # Settle and collect contacts
            contacts: List[ContactPoint] = []
            for _ in range(config.post_failure_settle_steps):
                mujoco.mj_step(self.model, self.data)
                self._grasp_lock.update(self.data)
                step_contacts = self.contact_extractor.extract_contacts(
                    self.model, self.data, filter_target=False, min_force=1.0)
                contacts.extend(step_contacts)

            # Unique impacted geom IDs (excluding the grasped object itself)
            impacted = set()
            for c in contacts:
                impacted.add(c.geom1)
                impacted.add(c.geom2)
            if self._grasped_obj_geom_id >= 0:
                impacted.discard(self._grasped_obj_geom_id)
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
            task_id=config.task_id,
            traj_id=config.traj_id,
            seed=config.seed,
            traj_progress=traj_progress,
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
            self._grasp_lock.release(self.data)
            self.data.ctrl[7] = 255.0
        elif mode == FailureMode.SLIPPERY_GRIP:
            self._grasp_lock.release(self.data)
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

            # Check force on grasped object
            obj_name_lower = self._grasped_obj_name.lower()
            max_force = 0.0
            in_contact = False
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                g2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                if g1_name and g2_name:
                    pair = (g1_name.lower(), g2_name.lower())
                    finger_obj = any(
                        ("finger" in a or "pad" in a) and obj_name_lower in b
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

    def _reset_grasped_object(self):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self._grasped_obj_name)
        jid = self.model.body_jntadr[bid]
        qpos_adr = self.model.jnt_qposadr[jid]
        self.data.qpos[qpos_adr : qpos_adr + 7] = self._grasped_obj_init_qpos
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
