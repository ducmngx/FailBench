"""LIBERO demo replay with failure injection.

Mirrors :class:`planner.experiments.runner.ExperimentRunner._run_segmented` but
adapted for robosuite-named models and HDF5-sourced qpos timelines:

    1. Build ``mj_model`` from the demo's cached MJCF.
    2. Resolve robosuite ↔ FailBench names via :class:`ModelHandles`.
    3. Optionally seed full sim state from ``init_state``.
    4. Kinematic replay of (arm_qpos, finger_qpos) up to step
       ``int(fail_progress * T)``.
    5. Capture pre-failure RGB/depth/state, then fork: for each FailureConfig
       restore the checkpoint, inject the failure, settle, collect contacts.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List, Optional

import mujoco
import numpy as np

from planner.experiments.config import FailureConfig, FailureMode
from planner.experiments.data_capture import (
    ContactExtractor,
    ContactPoint,
    OffscreenRenderer,
    RobotState,
    SimStateCheckpoint,
)
from planner.experiments.libero.adapter import LiberoDemo, materialise_mjcf
from planner.experiments.libero.failure import (
    LiberoFailureInjector,
    parse_joint_spec,
)
from planner.experiments.libero.naming import resolve_model_handles
from planner.experiments.runner import DataSample, FailureResult


@dataclass
class V2CaptureSpec:
    """Enables v2-style enhanced capture inside :meth:`LiberoRunner.run_v2`.

    The defaults match the v2 schema (T=8 window, K=3 goal offsets, S=50
    settle samples over 500 steps). All knobs are exposed so the smoke test
    can shrink the trial cheaply.
    """
    window_T: int = 8
    window_stride: int = 5
    goal_offsets: tuple = (5, 15, 30)
    settle_every: int = 10  # snapshot state every N settle steps (S = settle_steps / settle_every)
    image_w: int = 320
    image_h: int = 240

logger = logging.getLogger(__name__)


@dataclass
class LiberoTrialConfig:
    """Parameters for a single LIBERO replay trial.

    Intentionally narrower than ``ExperimentConfig`` — LIBERO supplies the
    trajectory and the scene model, so we don't need scene/robot XML paths,
    interpolation knobs, or the grasped-object bookkeeping.
    """
    fail_progress: Optional[float] = None
    fail_step: Optional[int] = None
    canonical_fail_fractions: tuple = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)

    failure_configs: Optional[List[FailureConfig]] = None
    failure_sample_mode: str = "all"
    num_failure_samples: int = 1

    seed: int = 42
    image_width: int = 640
    image_height: int = 480
    post_failure_settle_steps: int = 500
    seed_from_init_state: bool = True

    # Active resistance for healthy joints during post-failure settle.
    #   "none"        : ctrl[arm] left at 0 — torque-controlled robosuite arm
    #                   sags passively along with the failed joint.
    #   "gravcomp_pd" : tau = qfrc_bias[dof] + Kp(q*-q) - Kd*qd, applied only
    #                   on healthy joints. Failed joints stay limp because
    #                   _kill_joint already zeroed their actuator gains.
    resistance_mode: str = "none"
    pd_kp: tuple = (600.0, 600.0, 600.0, 600.0, 300.0, 120.0, 120.0)
    pd_kd: Optional[tuple] = None  # None → 2*sqrt(kp) per joint


class LiberoRunner:
    """Replays one LIBERO demo and injects failures."""

    def __init__(self, demo: LiberoDemo, config: LiberoTrialConfig):
        self.demo = demo
        self.config = config

        xml_path = materialise_mjcf(demo.model_xml)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.handles = resolve_model_handles(self.model)

        if config.seed_from_init_state and demo.init_state is not None:
            self._set_full_state(demo.init_state)
        else:
            mujoco.mj_forward(self.model, self.data)

        primary = self.handles.agentview_cam
        extras = [self.handles.ee_cam] if self.handles.ee_cam else []
        self.renderer = OffscreenRenderer(
            self.model,
            height=config.image_height,
            width=config.image_width,
            camera_name=primary,
            extra_cameras=extras,
        )

        self.contact_extractor = ContactExtractor(self.model)
        # Override the robot-geom set with the robosuite-aware one.
        self.contact_extractor._robot_geom_ids = self.handles.robot_geom_ids

        self.injector = LiberoFailureInjector(self.model, self.data, self.handles)
        self._rng = random.Random(config.seed)

    def run(self, experiment_id: Optional[str] = None) -> Optional[DataSample]:
        cfg = self.config
        T = self.demo.arm_qpos.shape[0]
        if T < 2:
            logger.warning("Demo %s/%s has %d steps — skipping",
                           self.demo.hdf5_path, self.demo.demo_key, T)
            return None

        if cfg.fail_step is not None:
            fail_idx = max(1, min(int(cfg.fail_step), T - 1))
            frac = fail_idx / (T - 1)
        else:
            frac = (cfg.fail_progress
                    if cfg.fail_progress is not None
                    else self._rng.choice(cfg.canonical_fail_fractions))
            fail_idx = max(1, min(int(frac * (T - 1)), T - 1))

        # Seed pre-failure state. With per-step `full_states` from the demo we
        # can drop the sim straight into the moment of failure with the bowl
        # actually in the gripper, all object positions correct, and qvels
        # matching what robosuite's controller produced. Kinematic replay is a
        # fallback for demos that don't carry full states (older HDF5s).
        if self.demo.full_states is not None:
            self._set_full_state(self.demo.full_states[fail_idx])
        else:
            for t in range(fail_idx + 1):
                self._kinematic_step(t)

        logger.info("LIBERO demo %s/%s: failure at step %d/%d (frac=%.3f)",
                    self.demo.hdf5_path, self.demo.demo_key, fail_idx, T, frac)
        # The hold target for active resistance: the demo's pose at the moment
        # of failure. Healthy joints will be PD-controlled toward this.
        last_qpos_cmd = self.demo.arm_qpos[fail_idx].astype(np.float64).copy()
        return self._capture_and_fork(
            traj_progress=frac,
            experiment_id=experiment_id or f"libero_{self.demo.traj_id:05d}",
            last_qpos_cmd=last_qpos_cmd,
        )

    def close(self):
        self.renderer.close()

    # ---------------- v2 enhanced-capture path ----------------

    def run_v2(self, capture: V2CaptureSpec,
               failure: FailureConfig,
               experiment_id: Optional[str] = None) -> Optional[dict]:
        """Run one trial with v2 enhanced capture, returning a payload dict.

        Differs from :meth:`run` in three ways:

        1. Captures a T-frame pre-failure window (RGB+depth per camera + per-frame
           state) by seeding the sim from each demo full-state in turn.
        2. Captures per-step contacts during the settle (`contact_time`,
           `contact_force_world`) and per-N-step state snapshots (`settle_*`).
        3. Captures symmetric post-failure observations for both cameras.

        The returned dict is suitable for :meth:`planner.risk.v2_store.V2Writer.write_trial`.
        """
        from planner.experiments.libero.window import (
            sample_window_indices,
            compute_goal,
            extract_cam_calibration,
            extract_scene_metadata,
            snapshot_object_poses,
            _object_body_ids,
            object_names as _object_names_fn,
            LIBERO_CONTROL_HZ,
        )

        cfg = self.config
        demo = self.demo
        T_demo = demo.arm_qpos.shape[0]
        if T_demo < 2 or demo.full_states is None:
            logger.warning("Demo %s/%s lacks full_states or has only %d steps — v2 needs both",
                           demo.hdf5_path, demo.demo_key, T_demo)
            return None

        # --- 1. Pick the failure step exactly like run() ---
        if cfg.fail_step is not None:
            fail_idx = max(1, min(int(cfg.fail_step), T_demo - 1))
            traj_progress = fail_idx / (T_demo - 1)
        else:
            traj_progress = (cfg.fail_progress
                             if cfg.fail_progress is not None
                             else self._rng.choice(cfg.canonical_fail_fractions))
            fail_idx = max(1, min(int(traj_progress * (T_demo - 1)), T_demo - 1))

        # --- 2. Sample the window indices ---
        frame_idx = sample_window_indices(fail_idx, T_demo,
                                          T=capture.window_T,
                                          stride=capture.window_stride)

        # --- 3. Walk the window: for each frame, seed sim + render both cams
        # and capture per-frame state directly from the seeded sim (so
        # window_qpos[-1] is *identical* to pre_qpos by construction).
        H, W = capture.image_h, capture.image_w
        T = capture.window_T
        win_av_rgb = np.empty((T, H, W, 3), dtype=np.uint8)
        win_av_d = np.empty((T, H, W), dtype=np.float16)
        win_w_rgb = np.empty((T, H, W, 3), dtype=np.uint8)
        win_w_d = np.empty((T, H, W), dtype=np.float16)
        cam_wrist_pos_win = np.empty((T, 3), dtype=np.float64)
        cam_wrist_mat_win = np.empty((T, 3, 3), dtype=np.float64)
        win_qpos = np.empty((T, 7), dtype=np.float32)
        win_qpos_raw = np.empty((T, 7), dtype=np.float64)  # for finite-diff
        win_ee_pos = np.empty((T, 3), dtype=np.float32)
        win_grip = np.empty((T, 1), dtype=np.float32)

        agentview = self.handles.agentview_cam
        wrist = self.handles.ee_cam
        wrist_id = (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, wrist)
                    if wrist else -1)
        for k, idx in enumerate(frame_idx.tolist()):
            self._set_full_state(demo.full_states[idx])
            rs = self._snapshot_robot()
            win_qpos[k] = rs.qpos.astype(np.float32)
            win_qpos_raw[k] = rs.qpos
            win_ee_pos[k] = rs.ee_pos.astype(np.float32)
            win_grip[k, 0] = rs.gripper_ctrl
            views = self.renderer.render_all_cameras(self.data)
            av_rgb, av_d = views[agentview]
            win_av_rgb[k] = av_rgb
            win_av_d[k] = av_d.astype(np.float16, copy=False)
            if wrist is not None and wrist in views:
                w_rgb, w_d = views[wrist]
                win_w_rgb[k] = w_rgb
                win_w_d[k] = w_d.astype(np.float16, copy=False)
                cam_wrist_pos_win[k] = self.data.cam_xpos[wrist_id]
                cam_wrist_mat_win[k] = self.data.cam_xmat[wrist_id].reshape(3, 3)

        # Finite-diff qvel across window timestamps using the LIBERO control
        # period. ``frame_idx`` may have repeated 0s at the prefix (clipped),
        # which would give dt=0 — guard with np.maximum(diff, 1).
        dt = 1.0 / LIBERO_CONTROL_HZ
        win_qvel = np.empty((T, 7), dtype=np.float32)
        for k in range(T):
            if k == 0:
                step_dt = max(int(frame_idx[1]) - int(frame_idx[0]), 1) * dt
                win_qvel[k] = ((win_qpos_raw[1] - win_qpos_raw[0]) / step_dt).astype(np.float32)
            elif k == T - 1:
                step_dt = max(int(frame_idx[-1]) - int(frame_idx[-2]), 1) * dt
                win_qvel[k] = ((win_qpos_raw[-1] - win_qpos_raw[-2]) / step_dt).astype(np.float32)
            else:
                step_dt = max(int(frame_idx[k + 1]) - int(frame_idx[k - 1]), 1) * dt
                win_qvel[k] = ((win_qpos_raw[k + 1] - win_qpos_raw[k - 1]) / step_dt).astype(np.float32)

        # --- 4. Seed sim to fail_idx for the failure replay + capture pre fields ---
        self._set_full_state(demo.full_states[fail_idx])
        last_qpos_cmd = demo.arm_qpos[fail_idx].astype(np.float64).copy()
        pre_robot = self._snapshot_robot()

        # Read ee_states once for goal extraction (goal looks ahead in the
        # demo and we don't seed the sim there, so we read obs directly).
        import h5py
        with h5py.File(demo.hdf5_path, "r") as f:
            ee_states = np.asarray(
                f[f"data/{demo.demo_key}/obs/ee_states"]).astype(np.float64)

        window_state = {
            "window_qpos": win_qpos,
            "window_qvel": win_qvel,
            "window_ee_pos": win_ee_pos,
            "window_gripper_ctrl": win_grip,
        }
        goal = compute_goal(demo, fail_idx, ee_states, offsets=capture.goal_offsets)

        # Camera calibration (agentview is static; wrist captured above)
        av_cal = extract_cam_calibration(self.model, self.data, agentview, (W, H))
        wrist_fovy = float(self.model.cam_fovy[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, wrist)])

        # Scene metadata
        scene_meta = extract_scene_metadata(self.model, self.data)

        # Object body ids + pre poses
        obj_bids = _object_body_ids(self.model)
        obj_names = _object_names_fn(self.model, obj_bids)
        obj_pos_pre, obj_quat_pre = snapshot_object_poses(self.model, self.data, obj_bids)

        # Pre-failure renders (single frame, both cams)
        views = self.renderer.render_all_cameras(self.data)
        pre_av_rgb, pre_av_d = views[agentview]
        pre_w_rgb, pre_w_d = views[wrist]

        # is_holding flag: gripper commanded-close at fail_idx. LIBERO action
        # is positive for close; we read it from the demo if available.
        is_holding = self._is_holding_at(fail_idx)

        # --- 5. Inject failure + run settle with v2 capture ---
        self._inject_failure(failure)

        n_settle = cfg.post_failure_settle_steps
        S = n_settle // capture.settle_every
        settle_step_idx = np.empty(S, dtype=np.int32)
        settle_qpos = np.empty((S, 7), dtype=np.float32)
        settle_qvel = np.empty((S, 7), dtype=np.float32)
        settle_grip = np.empty((S, 2), dtype=np.float32) if self.handles.finger_qpos_adrs else np.zeros((S, 2), dtype=np.float32)
        settle_obj_pos = np.empty((S, len(obj_bids), 3), dtype=np.float32)
        settle_obj_quat = np.empty((S, len(obj_bids), 4), dtype=np.float32)

        contact_positions = []
        contact_forces_local = []
        contact_force_world = []
        contact_time = []
        contact_geom_pairs = []

        for step in range(n_settle):
            if last_qpos_cmd is not None:
                self._apply_resistance(last_qpos_cmd)
            mujoco.mj_step(self.model, self.data)

            # Per-step contact capture with world-frame force
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 in self.handles.robot_geom_ids and g2 in self.handles.robot_geom_ids:
                    continue
                f6 = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                if np.linalg.norm(f6[:3]) < 1.0:
                    continue
                frame = c.frame.reshape(3, 3)   # rows = (n, t1, t2) in world
                fw = frame.T @ f6[:3]
                contact_positions.append(c.pos.copy())
                contact_forces_local.append(f6.copy())
                contact_force_world.append(fw)
                contact_time.append(step)
                contact_geom_pairs.append([g1, g2])

            # State snapshot every N steps
            if (step + 1) % capture.settle_every == 0:
                k = (step + 1) // capture.settle_every - 1
                settle_step_idx[k] = step + 1
                settle_qpos[k] = [self.data.qpos[a] for a in self.handles.arm_qpos_adrs]
                settle_qvel[k] = [self.data.qvel[a] for a in self.handles.arm_dof_adrs]
                if self.handles.finger_qpos_adrs:
                    settle_grip[k] = [self.data.qpos[a] for a in self.handles.finger_qpos_adrs]
                op, oq = snapshot_object_poses(self.model, self.data, obj_bids)
                settle_obj_pos[k] = op
                settle_obj_quat[k] = oq

        # --- 6. Post-failure renders + obj poses ---
        views = self.renderer.render_all_cameras(self.data)
        post_av_rgb, post_av_d = views[agentview]
        post_w_rgb, post_w_d = views[wrist]
        obj_pos_post, obj_quat_post = snapshot_object_poses(self.model, self.data, obj_bids)

        # Restore for any subsequent call
        self.injector.restore_all()

        # --- 7. Assemble payload ---
        n_contacts = len(contact_positions)
        if n_contacts == 0:
            cp = np.zeros((0, 3), np.float32)
            cf = np.zeros((0, 6), np.float32)
            cfw = np.zeros((0, 3), np.float32)
            ct = np.zeros((0,), np.int32)
            cgp = np.zeros((0, 2), np.int32)
        else:
            cp = np.asarray(contact_positions, dtype=np.float32)
            cf = np.asarray(contact_forces_local, dtype=np.float32)
            cfw = np.asarray(contact_force_world, dtype=np.float32)
            ct = np.asarray(contact_time, dtype=np.int32)
            cgp = np.asarray(contact_geom_pairs, dtype=np.int32)
        impacted = np.unique(cgp.reshape(-1)) if cgp.size else np.zeros((0,), np.int32)

        # Failure-joint indices (1-based joint number in the failure spec)
        f_joints = np.array(
            [parse_joint_spec(n) for n in (failure.joint_names or [])],
            dtype=np.int32,
        )

        payload = {
            # window
            "window_frame_idx": frame_idx,
            **window_state,
            "window_agentview_rgb": win_av_rgb,
            "window_agentview_depth": win_av_d,
            "window_wrist_rgb": win_w_rgb,
            "window_wrist_depth": win_w_d,
            # goal
            **goal,
            # v1-compat single frame
            "pre_qpos": pre_robot.qpos,
            "pre_qvel": pre_robot.qvel,
            "pre_ee_pos": pre_robot.ee_pos,
            "pre_gripper_ctrl": np.array([pre_robot.gripper_ctrl], dtype=np.float64),
            "pre_target_qpos": last_qpos_cmd,
            "pre_rgb": pre_av_rgb,
            "pre_depth": pre_av_d.astype(np.float16, copy=False),
            "robot0_eye_in_hand_rgb": pre_w_rgb,
            "robot0_eye_in_hand_depth": pre_w_d.astype(np.float16, copy=False),
            # contacts
            "contact_positions": cp,
            "contact_forces": cf,
            "contact_force_world": cfw,
            "contact_time": ct,
            "contact_geom_pairs": cgp,
            "contact_failure_id": np.zeros((n_contacts,), dtype=np.int32),
            "impacted_geom_ids": impacted.astype(np.int32),
            # post
            "post_agentview_rgb": post_av_rgb,
            "post_agentview_depth": post_av_d.astype(np.float16, copy=False),
            "post_wrist_rgb": post_w_rgb,
            "post_wrist_depth": post_w_d.astype(np.float16, copy=False),
            # calibration
            "cam_agentview_pos": av_cal["pos"],
            "cam_agentview_mat0": av_cal["mat0"],
            "cam_agentview_fovy": av_cal["fovy"],
            "cam_agentview_size": av_cal["size"],
            "cam_wrist_pos_window": cam_wrist_pos_win,
            "cam_wrist_mat0_window": cam_wrist_mat_win,
            "cam_wrist_fovy": wrist_fovy,
            "cam_wrist_size": np.array([W, H], dtype=np.int32),
            # failure descriptor
            "failure_joints": f_joints,
            # objects
            "obj_names": obj_names,
            "obj_pos_pre": obj_pos_pre,
            "obj_quat_pre": obj_quat_pre,
            "obj_pos_post": obj_pos_post,
            "obj_quat_post": obj_quat_post,
            # settle
            "settle_step_idx": settle_step_idx,
            "settle_qpos": settle_qpos,
            "settle_qvel": settle_qvel,
            "settle_gripper_qpos": settle_grip,
            "settle_obj_pos": settle_obj_pos,
            "settle_obj_quat": settle_obj_quat,
            # attrs
            "fail_idx": int(fail_idx),
            "traj_progress": float(traj_progress),
            "failure_mode": failure.mode.name,
            "failure_prob": float(failure.probability),
            "is_holding": bool(is_holding),
            "force_frame": "contact",
            "scene_table_z": scene_meta["scene_table_z"],
            "scene_aabb_min": scene_meta["scene_aabb_min"],
            "scene_aabb_max": scene_meta["scene_aabb_max"],
            "scene_entities_json": scene_meta["scene_entities_json"],
            "robot_geom_ids": np.array(sorted(self.handles.robot_geom_ids), dtype=np.int32),
        }
        return payload

    def _is_holding_at(self, fail_idx: int) -> bool:
        """Heuristic: gripper commanded-close at fail_idx.

        LIBERO actions are 7-D; index -1 is the gripper command, positive ≈
        commanded-close. Reads directly from the demo HDF5 (the actions array
        isn't carried on :class:`LiberoDemo`).
        """
        import h5py
        try:
            with h5py.File(self.demo.hdf5_path, "r") as f:
                a = f[f"data/{self.demo.demo_key}/actions"][fail_idx, -1]
            return bool(a > 0)
        except Exception:
            return False

    def _set_full_state(self, flat_state: np.ndarray) -> None:
        # robosuite stores [time | qpos | qvel] (length 1 + nq + nv) in
        # init_state and per-step states. Reading flat_state[:nq] directly
        # treats the time field as qpos[0], shifting every object's xyz
        # by one slot — bowls/plates render at floor level instead of the
        # tabletop. Strip the time field first.
        nq = self.model.nq
        nv = self.model.nv
        expected = 1 + nq + nv
        if flat_state.shape[0] == expected:
            offset = 1
        elif flat_state.shape[0] == nq + nv:
            offset = 0
        else:
            logger.warning("init_state length %d does not match nq+nv (%d) "
                           "or 1+nq+nv (%d) — skipping",
                           flat_state.shape[0], nq + nv, expected)
            return
        self.data.qpos[:nq] = flat_state[offset:offset + nq]
        self.data.qvel[:nv] = flat_state[offset + nq:offset + nq + nv]
        mujoco.mj_forward(self.model, self.data)

    def _kinematic_step(self, t: int) -> None:
        """Drive arm + finger qpos to the demo values at step ``t``.

        We set qpos directly (kinematic replay) rather than chasing it with
        actuators — robosuite uses an OSC controller we don't want to
        reimplement. Object qpos evolves via :func:`mj_forward` so contact-
        driven object motion still happens.
        """
        for adr, q in zip(self.handles.arm_qpos_adrs, self.demo.arm_qpos[t]):
            self.data.qpos[adr] = q
        if len(self.handles.finger_qpos_adrs) == 2:
            for adr, q in zip(self.handles.finger_qpos_adrs, self.demo.finger_qpos[t]):
                self.data.qpos[adr] = q
        for adr in self.handles.arm_dof_adrs:
            self.data.qvel[adr] = 0.0
        for adr in self.handles.finger_dof_adrs:
            self.data.qvel[adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _snapshot_robot(self) -> RobotState:
        qpos = np.array([self.data.qpos[a] for a in self.handles.arm_qpos_adrs])
        qvel = np.array([self.data.qvel[a] for a in self.handles.arm_dof_adrs])
        if self.handles.ee_site_id >= 0:
            ee_pos = self.data.site_xpos[self.handles.ee_site_id].copy()
        else:
            ee_pos = np.zeros(3)
        if self.handles.finger_qpos_adrs:
            grip = float(np.mean([self.data.qpos[a]
                                  for a in self.handles.finger_qpos_adrs]))
        else:
            grip = 0.0
        return RobotState(
            qpos=qpos.astype(np.float64),
            qvel=qvel.astype(np.float64),
            ee_pos=ee_pos,
            gripper_ctrl=grip,
            sim_time=float(self.data.time),
        )

    def _apply_resistance(self, target_qpos: np.ndarray) -> None:
        """Drive healthy arm joints toward ``target_qpos`` via gravity-comp + PD.

        No-op when ``resistance_mode == "none"``. Failed joints are skipped via
        :pyattr:`LiberoFailureInjector.failed_joint_ids`; their ctrl slots stay
        at 0 (and `_kill_joint` zeroed their actuator gains anyway).
        """
        if self.config.resistance_mode == "none":
            return
        h = self.handles
        kp = np.asarray(self.config.pd_kp, dtype=np.float64)
        kd = (np.asarray(self.config.pd_kd, dtype=np.float64)
              if self.config.pd_kd is not None
              else 2.0 * np.sqrt(kp))
        # qfrc_bias is gravity + Coriolis at the current state. MuJoCo
        # populates it inside mj_step, but we want fresh values at the top of
        # this step → mj_forward keeps the cost negligible.
        mujoco.mj_forward(self.model, self.data)
        for i, jid in enumerate(h.arm_joint_ids):
            if jid in self.injector.failed_joint_ids:
                continue
            aid = h.arm_actuator_ids[i]
            if aid < 0:
                continue
            q = float(self.data.qpos[h.arm_qpos_adrs[i]])
            qd = float(self.data.qvel[h.arm_dof_adrs[i]])
            grav = float(self.data.qfrc_bias[h.arm_dof_adrs[i]])
            tau = grav + kp[i] * (target_qpos[i] - q) - kd[i] * qd
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = float(np.clip(tau, lo, hi))

    def _capture_and_fork(self, traj_progress: float,
                          experiment_id: str,
                          last_qpos_cmd: Optional[np.ndarray] = None) -> DataSample:
        cfg = self.config

        all_views = self.renderer.render_all_cameras(self.data)
        primary_name = next(iter(all_views))
        pre_rgb, pre_depth = all_views[primary_name]
        extra_views = {k: v for k, v in all_views.items() if k != primary_name}
        pre_robot = self._snapshot_robot()
        sim_time = float(self.data.time)

        checkpoint = SimStateCheckpoint.save(self.data)

        failures = (cfg.failure_configs
                    if cfg.failure_configs is not None
                    else _default_failures())
        if cfg.failure_sample_mode == "sample":
            weights = [fc.probability for fc in failures]
            failures = self._rng.choices(failures, weights=weights,
                                         k=cfg.num_failure_samples)

        failure_results: List[FailureResult] = []
        all_contacts: List[ContactPoint] = []
        all_geom_ids: set = set()
        post_rgb = None

        for fc in failures:
            SimStateCheckpoint.restore(self.model, self.data, checkpoint)
            self._inject_failure(fc)

            contacts: List[ContactPoint] = []
            for _ in range(cfg.post_failure_settle_steps):
                if last_qpos_cmd is not None:
                    self._apply_resistance(last_qpos_cmd)
                mujoco.mj_step(self.model, self.data)
                contacts.extend(self.contact_extractor.extract_contacts(
                    self.model, self.data, filter_target=False, min_force=1.0))

            impacted: set = set()
            for c in contacts:
                impacted.add(c.geom1)
                impacted.add(c.geom2)
            failure_results.append(FailureResult(
                failure_config=fc,
                contacts=contacts,
                impacted_geom_ids=sorted(impacted),
                had_collision=len(contacts) > 0,
            ))
            all_contacts.extend(contacts)
            all_geom_ids.update(impacted)
            post_rgb = self.renderer.render(self.data)

            self.injector.restore_all()

        return DataSample(
            experiment_id=experiment_id,
            trajectory_file=self.demo.hdf5_path,
            task_id=self.demo.task_id,
            traj_id=self.demo.traj_id,
            seed=cfg.seed,
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
            pre_target_qpos=last_qpos_cmd,
        )

    def _inject_failure(self, fc: FailureConfig) -> None:
        mode = fc.mode
        if mode == FailureMode.GRIPPER_OPEN:
            self._set_gripper_normalised(1.0)
        elif mode == FailureMode.SLIPPERY_GRIP:
            t = float(np.clip(fc.grip_value / 255.0, 0.0, 1.0))
            self._set_gripper_normalised(t)
        elif mode == FailureMode.SINGLE_JOINT:
            if fc.joint_names:
                self.injector.fail_single(parse_joint_spec(fc.joint_names[0]))
        elif mode == FailureMode.MULTI_JOINT:
            if fc.joint_names:
                self.injector.fail_multi([parse_joint_spec(n) for n in fc.joint_names])
        elif mode == FailureMode.ALL_JOINTS:
            self.injector.fail_all()

    def _set_gripper_normalised(self, t: float) -> None:
        """Drive the gripper to ``t`` ∈ [0, 1] where 1 = fully open.

        Handles both robosuite's two-finger actuator pair (opposite ranges)
        and FailBench's single scalar actuator with [0, 255] = [closed, open].
        """
        gid = self.handles.gripper_actuator_id
        if gid >= 0:
            lo, hi = self.model.actuator_ctrlrange[gid]
            self.data.ctrl[gid] = float(lo + t * (hi - lo))
            return
        for aid in self.handles.finger_actuator_ids:
            lo, hi = self.model.actuator_ctrlrange[aid]
            # robosuite Panda position-actuated fingers: each finger's qpos is
            # 0 when fully closed and at the |max| of its range when fully
            # open. finger1 ctrl ∈ [0, 0.04] → 0 = closed, 0.04 = open.
            # finger2 ctrl ∈ [-0.04, 0] → 0 = closed, -0.04 = open. So the
            # "open" end is the one *farther* from zero in absolute value.
            if abs(lo) > abs(hi):
                open_end, close_end = lo, hi
            else:
                open_end, close_end = hi, lo
            self.data.ctrl[aid] = float(close_end + t * (open_end - close_end))


def _default_failures() -> List[FailureConfig]:
    """LIBERO defaults — weighted toward gravity-loaded pitch joints.

    Pitch joints (j2 shoulder, j4 elbow, j6 wrist) drop the arm under gravity
    and produce the most planner-relevant contact patterns. Rotation joints
    (j1, j7) stay in the mix at low weight for dataset diversity. j3 / j5
    intentionally omitted — their failure signature is too subtle to be worth
    the trial budget.
    """
    return [
        FailureConfig(mode=FailureMode.GRIPPER_OPEN, probability=0.25),
        FailureConfig(mode=FailureMode.SLIPPERY_GRIP, grip_value=180.0, probability=0.15),
        # Pitch joints — high planner signal
        FailureConfig(mode=FailureMode.SINGLE_JOINT, joint_names=["joint2"], probability=0.18),
        FailureConfig(mode=FailureMode.SINGLE_JOINT, joint_names=["joint4"], probability=0.13),
        FailureConfig(mode=FailureMode.SINGLE_JOINT, joint_names=["joint6"], probability=0.07),
        # Rotation joints — low weight, kept for diversity
        FailureConfig(mode=FailureMode.SINGLE_JOINT, joint_names=["joint1"], probability=0.03),
        FailureConfig(mode=FailureMode.SINGLE_JOINT, joint_names=["joint7"], probability=0.02),
        # Multi-joint cascades — pitch + adjacent
        FailureConfig(mode=FailureMode.MULTI_JOINT, joint_names=["joint2", "joint4"], probability=0.05),
        FailureConfig(mode=FailureMode.MULTI_JOINT, joint_names=["joint4", "joint6"], probability=0.04),
        FailureConfig(mode=FailureMode.ALL_JOINTS, probability=0.08),
    ]
