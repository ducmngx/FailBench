"""
Shared failure cost model used by T-RRT and STOMP planners.

Encapsulates severity mapping, distance-based failure probability,
and contextual severity computation that was previously duplicated
across TRRTFailure.py and STOMP.py.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# Default interpolation table for smooth severity mapping
DEFAULT_SEVERITY_TABLE: Dict[int, float] = {
    1: 0.1, 2: 0.3, 3: 0.6, 5: 1.2, 8: 3.0,
    10: 8.0, 15: 20.0, 20: 50.0,
}


@dataclass
class SeverityConfig:
    """Configuration for the failure cost model."""

    severity_map: Dict[int, int]
    """Maps body ID → LLM severity score."""

    target_body_ids: List[int]
    """Body IDs that are pick/place targets (excluded from avoidance cost)."""

    max_failure_prob: float = 0.98
    distance_decay_rate: float = 50.0
    base_radius: float = 0.5
    failure_weight: float = 2.0

    interpolation_table: Dict[int, float] = field(
        default_factory=lambda: dict(DEFAULT_SEVERITY_TABLE)
    )


class FailureCostModel:
    """Computes failure-aware state costs for motion planners.

    This model computes the expected cost of a robot configuration
    by combining distance-based failure probabilities with LLM-derived
    severity scores for each obstacle body.
    """

    def __init__(
        self,
        scene_model: mujoco.MjModel,
        object_positions: Dict[int, np.ndarray],
        config: SeverityConfig,
    ) -> None:
        self.scene_model = scene_model
        self.object_positions = object_positions
        self.config = config

        # Pre-sort interpolation table for np.interp
        sorted_keys = sorted(config.interpolation_table.keys())
        self._interp_x = np.array(sorted_keys, dtype=float)
        self._interp_y = np.array(
            [config.interpolation_table[k] for k in sorted_keys], dtype=float
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_end_effector_position(self, config: np.ndarray) -> np.ndarray:
        """Forward-kinematics to obtain end-effector XYZ position."""
        temp_data = mujoco.MjData(self.scene_model)
        temp_data.qpos[: min(len(config), temp_data.qpos.shape[0])] = config[
            : temp_data.qpos.shape[0]
        ]
        mujoco.mj_forward(self.scene_model, temp_data)

        try:
            site_id = mujoco.mj_name2id(
                self.scene_model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
            )
            if site_id != -1:
                return temp_data.site_xpos[site_id].copy()
        except Exception:
            pass

        return np.array([0, 0, 0.5])

    def state_cost(self, ee_pos: np.ndarray) -> float:
        """Compute failure cost given an end-effector position.

        Returns the weighted sum of (failure_prob × severity) over all
        non-target objects, multiplied by ``failure_weight``.
        """
        failure_probs = self.failure_probs(ee_pos)
        total_cost = 0.0

        for obj_id, failure_prob in failure_probs.items():
            obj_pos = self.object_positions.get(obj_id)
            if obj_pos is None:
                continue
            xy_distance = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))
            severity = self._contextual_severity(obj_id, xy_distance)
            total_cost += failure_prob * severity

        return total_cost * self.config.failure_weight

    def state_cost_from_config(self, config: np.ndarray) -> float:
        """Compute failure cost from a joint configuration (does FK internally)."""
        ee_pos = self.get_end_effector_position(config)
        return self.state_cost(ee_pos)

    def failure_probs(self, ee_pos: np.ndarray) -> Dict[int, float]:
        """Distance-based failure probabilities (XY plane) for each non-target object."""
        probs: Dict[int, float] = {}
        cfg = self.config

        for obj_id, obj_pos in self.object_positions.items():
            if obj_id in cfg.target_body_ids:
                continue
            xy_distance = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))
            prob = cfg.max_failure_prob * np.exp(
                -cfg.distance_decay_rate * xy_distance
            )
            probs[obj_id] = max(0.0, min(prob, cfg.max_failure_prob))

        return probs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _smooth_severity(self, llm_severity: int) -> float:
        """Interpolate discrete LLM severity to a continuous value."""
        return float(np.interp(llm_severity, self._interp_x, self._interp_y))

    def _contextual_severity(self, obj_id: int, xy_distance: float) -> float:
        """Apply distance-based falloff to a body's severity score."""
        llm_severity = self.config.severity_map.get(obj_id, 1)
        base_severity = self._smooth_severity(llm_severity)

        base_rad = self.config.base_radius
        if llm_severity >= 15:
            danger_radius = base_rad * 8
        elif llm_severity >= 10:
            danger_radius = base_rad * 4
        elif llm_severity >= 5:
            danger_radius = base_rad * 2
        else:
            danger_radius = base_rad

        falloff = np.exp(-xy_distance / danger_radius)
        return base_severity * falloff
