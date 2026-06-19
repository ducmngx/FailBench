"""Failure injection on a live LIBERO ``OffScreenRenderEnv``.

The dataset-generation path
(:class:`planner.experiments.libero.runner.LiberoRunner`) injects failures by
loading the demo's MJCF separately, stepping the sim kinematically, then
applying :class:`planner.experiments.libero.failure.LiberoFailureInjector` to
the standalone model. For the safety-rollout experiments we instead want to
step a real LIBERO ``OffScreenRenderEnv`` end-to-end (real robosuite
controllers, real reward predicate) and inject failures *into the env's own
sim* at a configurable progress point.

This module wraps a LIBERO env with that behaviour:

>>> from libero.libero.envs import OffScreenRenderEnv
>>> from planner.policy.libero_env_failure import EnvFailureScheduler
>>> from planner.experiments.config import FailureConfig, FailureMode
>>>
>>> env = OffScreenRenderEnv(bddl_file_name=..., camera_heights=240,
...                          camera_widths=320)
>>> sched = EnvFailureScheduler(
...     env, failure=FailureConfig(mode=FailureMode.SINGLE_JOINT,
...                                joints=[4], probability=1.0),
...     fail_step=120)
>>> obs = sched.reset()
>>> for action in demo_actions:
...     obs, reward, done, info = sched.step(action)
...     if done: break

Resets are clean — the injector is destroyed at the end of an episode and
rebuilt at the next ``reset()`` from the env's freshly-loaded sim.

Re-uses :class:`planner.experiments.libero.failure.LiberoFailureInjector` and
:func:`planner.experiments.libero.naming.resolve_model_handles`. The only new
logic is timing (when to inject) and the env-sim plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from planner.experiments.config import FailureConfig, FailureMode
from planner.experiments.libero.failure import (
    LiberoFailureInjector, parse_joint_spec)
from planner.experiments.libero.naming import resolve_model_handles


def _resolve_joint_indices(failure: FailureConfig) -> tuple[int, ...]:
    """Parse ``failure.joint_names`` (FailBench's canonical list-of-strings
    representation) into a tuple of 1-based joint indices.

    ``FailureConfig`` stores joints as e.g. ``["joint4", "joint6"]``;
    LiberoFailureInjector wants ``int`` indices. Empty for GRIPPER_OPEN /
    SLIPPERY_GRIP.
    """
    names = failure.joint_names or ()
    return tuple(parse_joint_spec(n) for n in names)


def unwrap_sim(sim):
    """Pull the bare ``mujoco.MjModel`` / ``mujoco.MjData`` out of a robosuite
    sim wrapper.

    Robosuite 1.4 wraps mujoco's bindings in
    ``robosuite.utils.binding_utils.MjModel`` / ``MjData``. The wrappers
    proxy attribute access, but standalone ``mujoco`` C functions reject
    them. Both wrappers expose the bare object via ``._model`` / ``._data``;
    plain mujoco objects already are the bare object — return as-is.
    """
    model = getattr(sim.model, "_model", sim.model)
    data  = getattr(sim.data,  "_data",  sim.data)
    return model, data


@dataclass
class FailureEvent:
    """Single failure-injection event log entry."""
    step: int
    mode: str
    joints: tuple
    realized: bool   # True if the injector actually ran (False if cfg disabled)


class EnvFailureScheduler:
    """Wraps a LIBERO env and injects a FailBench failure at a chosen step.

    Parameters
    ----------
    env
        A constructed LIBERO env (e.g. ``OffScreenRenderEnv``). Must expose
        ``reset()``, ``step(action)``, and ``sim`` (the robosuite-wrapped
        MjSim). We touch ``env.sim.model`` and ``env.sim.data`` to install
        the failure.
    failure
        ``FailureConfig`` describing what to inject. ``probability`` is
        ignored — this scheduler always realizes the failure (we want a
        deterministic comparison; sampling lives at the outer experiment
        sweeper).
    fail_step
        Step index (1-based as a count from the first ``step()`` after a
        ``reset()``) at which to inject. Use ``None`` for a control rollout
        with no failure.
    """

    def __init__(self, env: Any, failure: Optional[FailureConfig],
                 fail_step: Optional[int]):
        self.env = env
        self.failure = failure
        self.fail_step = fail_step

        self._injector: Optional[LiberoFailureInjector] = None
        self._step_count: int = 0
        self._injected: bool = False
        self._events: list[FailureEvent] = []

        # LIBERO's OffScreenRenderEnv reuses the same MjModel across
        # env.reset() — actuator_gainprm and other params mutated by failure
        # injection DO NOT get restored on reset.  We snapshot the clean
        # values on first reset() and re-apply them every subsequent reset.
        self._clean_snapshot: Optional[dict] = None

    # ------------------------------------------------------------------ API

    @property
    def step_count(self) -> int:
        """Steps taken since last reset."""
        return self._step_count

    @property
    def injected(self) -> bool:
        """True if the failure has fired on the current episode."""
        return self._injected

    @property
    def events(self) -> list[FailureEvent]:
        """All injection events across all resets in this object's lifetime."""
        return list(self._events)

    def reset(self, *args, **kwargs):
        """Reset the env and rebuild the failure injector from the new sim.

        On every reset we restore any mutable model parameters that an
        earlier failure may have touched, since LIBERO's underlying MjModel
        persists across resets (its env.reset() resets sim state but not
        actuator/joint metadata).  Without this, e.g. a GRIPPER_OPEN failure
        on rollout N leaves the gripper actuator gain zeroed for rollouts
        N+1, N+2, ... — every subsequent baseline runs with a dead gripper.
        """
        obs = self.env.reset(*args, **kwargs)
        self._step_count = 0
        self._injected = False
        model, data = unwrap_sim(self.env.sim)
        handles = resolve_model_handles(model)

        # First reset: snapshot the clean params; subsequent resets:
        # restore from snapshot before any new injection runs.
        if self._clean_snapshot is None:
            self._clean_snapshot = dict(
                actuator_gainprm=model.actuator_gainprm.copy(),
                actuator_biastype=model.actuator_biastype.copy(),
                actuator_gaintype=model.actuator_gaintype.copy(),
                jnt_stiffness=model.jnt_stiffness.copy(),
                dof_damping=model.dof_damping.copy(),
                jnt_range=model.jnt_range.copy(),
                dof_frictionloss=model.dof_frictionloss.copy(),
            )
        else:
            snap = self._clean_snapshot
            model.actuator_gainprm[:] = snap["actuator_gainprm"]
            model.actuator_biastype[:] = snap["actuator_biastype"]
            model.actuator_gaintype[:] = snap["actuator_gaintype"]
            model.jnt_stiffness[:] = snap["jnt_stiffness"]
            model.dof_damping[:] = snap["dof_damping"]
            model.jnt_range[:] = snap["jnt_range"]
            model.dof_frictionloss[:] = snap["dof_frictionloss"]

        # Build a fresh injector — it captures its own restore snapshot,
        # which now correctly reflects the (re)-cleaned model.
        self._injector = LiberoFailureInjector(model, data, handles)
        return obs

    def step(self, action):
        """Step the env, possibly injecting the failure first."""
        self._step_count += 1
        if (self.failure is not None
                and self.fail_step is not None
                and not self._injected
                and self._step_count >= self.fail_step):
            self._inject_now()
        return self.env.step(action)

    # ------------------------------------------------------------- internals

    def _inject_now(self) -> None:
        """Apply the failure via the LIBERO injector and log the event."""
        if self._injector is None:
            raise RuntimeError("call reset() before step() — injector is unset")
        if self.failure is None:
            return

        mode = self.failure.mode
        joints: tuple = _resolve_joint_indices(self.failure)

        model, data = unwrap_sim(self.env.sim)
        # GRIPPER_OPEN / SLIPPERY_GRIP use the LEGACY "zero / scale actuator
        # gain" semantics — this matches the v2 dataset on which the predictor
        # was trained.  Removing actuator force does NOT physically open the
        # gripper (fingers stay put, friction holds the object), but that is
        # the behavior recorded in the v2 corpus.  Changing this without
        # regenerating v2 + retraining causes a distribution shift.
        if mode == FailureMode.GRIPPER_OPEN:
            gas = []
            if self._injector.h.gripper_actuator_id >= 0:
                gas.append(self._injector.h.gripper_actuator_id)
            else:
                gas.extend(self._injector.h.finger_actuator_ids or [])
            for ga in gas:
                data.ctrl[ga] = 0.0
                model.actuator_gainprm[ga, :] = 0.0
        elif mode == FailureMode.SLIPPERY_GRIP:
            gas = []
            if self._injector.h.gripper_actuator_id >= 0:
                gas.append(self._injector.h.gripper_actuator_id)
            else:
                gas.extend(self._injector.h.finger_actuator_ids or [])
            for ga in gas:
                model.actuator_gainprm[ga, 0] *= 0.5
        elif mode == FailureMode.SINGLE_JOINT:
            if not joints:
                raise ValueError("SINGLE_JOINT failure requires joints=[i]")
            self._injector.fail_single(int(joints[0]))
        elif mode == FailureMode.MULTI_JOINT:
            if not joints:
                raise ValueError("MULTI_JOINT failure requires joints=[i,j,...]")
            self._injector.fail_multi([int(j) for j in joints])
        elif mode == FailureMode.ALL_JOINTS:
            self._injector.fail_all()
        else:
            raise ValueError(f"unknown failure mode: {mode!r}")

        self._injected = True
        self._events.append(FailureEvent(
            step=self._step_count,
            mode=mode.name,
            joints=joints,
            realized=True,
        ))

    # ----------------------------------------------------------- convenience

    @classmethod
    def from_progress(cls, env: Any, failure: Optional[FailureConfig],
                      fail_progress: Optional[float], max_steps: int
                      ) -> "EnvFailureScheduler":
        """Build a scheduler from a fractional progress in [0, 1] over a known
        max-step horizon. Mirrors the canonical FailBench convention."""
        if fail_progress is None or failure is None:
            return cls(env, failure=None, fail_step=None)
        if not 0.0 <= fail_progress <= 1.0:
            raise ValueError(
                f"fail_progress must be in [0, 1], got {fail_progress}")
        fail_step = max(1, int(round(fail_progress * max_steps)))
        return cls(env, failure=failure, fail_step=fail_step)


__all__ = ["EnvFailureScheduler", "FailureEvent", "unwrap_sim"]
