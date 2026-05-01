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
