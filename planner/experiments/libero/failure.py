"""Name-aware failure injector for LIBERO/robosuite scenes.

The original ``failure_injection.agressive_injector.AggressiveFailureInjector``
hardcodes joint names (``joint1``..``joint7``) and assumes ``ctrl[7]`` is the
gripper. LIBERO models use ``robot0_joint{i}`` and a non-fixed gripper actuator
slot. This wrapper takes a :class:`ModelHandles` so every name lookup is
already resolved.
"""

from __future__ import annotations

from typing import List

import mujoco
import numpy as np

from planner.experiments.libero.naming import ModelHandles


class LiberoFailureInjector:
    """Same semantics as ``AggressiveFailureInjector`` but driven by handles."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 handles: ModelHandles):
        self.model = model
        self.data = data
        self.h = handles

        self._original_gainprm = model.actuator_gainprm.copy()
        self._original_biastype = model.actuator_biastype.copy()
        self._original_gaintype = model.actuator_gaintype.copy()
        self._original_stiffness = model.jnt_stiffness.copy()
        self._original_damping = model.dof_damping.copy()
        self._original_ranges = model.jnt_range.copy()
        self._original_frictionloss = model.dof_frictionloss.copy()
        self.failed_joint_ids: set = set()

    # ------------------------------------------------------------------

    def _kill_joint(self, joint_id: int) -> None:
        """Strip control authority + passive forces from one joint."""
        for a in range(self.model.nu):
            if int(self.model.actuator_trnid[a, 0]) == joint_id:
                self.model.actuator_gainprm[a, :] = 0.0
                self.data.ctrl[a] = 0.0
                self.model.actuator_gaintype[a] = 0
                self.model.actuator_biastype[a] = 0
        self.model.jnt_stiffness[joint_id] = 0.0
        dof_adr = int(self.model.jnt_dofadr[joint_id])
        self.model.dof_damping[dof_adr] = 0.0
        self.model.dof_frictionloss[dof_adr] = 0.0
        # Loosen the limits so the joint can move freely under gravity
        lo, hi = self._original_ranges[joint_id]
        center = 0.5 * (lo + hi)
        span = hi - lo
        self.model.jnt_range[joint_id, 0] = center - 2.0 * span
        self.model.jnt_range[joint_id, 1] = center + 2.0 * span
        self.failed_joint_ids.add(joint_id)

    # ------------------------------------------------------------------

    def fail_single(self, joint_idx: int) -> None:
        """Fail one arm joint, indexed 1..7 (matches FailBench's convention)."""
        self._kill_joint(self.h.arm_joint_ids[joint_idx - 1])

    def fail_multi(self, joint_indices: List[int]) -> None:
        for idx in joint_indices:
            self.fail_single(idx)

    def fail_all(self) -> None:
        for jid in self.h.arm_joint_ids:
            self._kill_joint(jid)

    # ------------------------------------------------------------------

    def restore_all(self) -> None:
        """Undo every parameter we touched. Cheap full-array restore."""
        if not self.failed_joint_ids:
            return
        self.model.actuator_gainprm[:] = self._original_gainprm
        self.model.actuator_biastype[:] = self._original_biastype
        self.model.actuator_gaintype[:] = self._original_gaintype
        self.model.jnt_stiffness[:] = self._original_stiffness
        self.model.dof_damping[:] = self._original_damping
        self.model.jnt_range[:] = self._original_ranges
        self.model.dof_frictionloss[:] = self._original_frictionloss
        self.failed_joint_ids.clear()


def parse_joint_spec(name: str) -> int:
    """Map FailBench failure-config joint names ('joint4') to 1-based index.

    Also accepts robosuite-style names ('robot0_joint4') so existing
    ``FailureConfig.joint_names`` lists work without modification.
    """
    if name.startswith("robot0_joint"):
        return int(name[len("robot0_joint"):])
    if name.startswith("joint"):
        return int(name[len("joint"):])
    raise ValueError(f"Unrecognised joint name '{name}' — expected joint{{i}} or robot0_joint{{i}}")
