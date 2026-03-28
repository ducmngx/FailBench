import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryRecord:
    """A single stored trajectory with associated metadata."""
    trajectory: List[np.ndarray]
    goal_pos: Optional[np.ndarray] = None
    me_cost: Optional[Any] = None
    safety_cost: Optional[Any] = None


class ExperimentTrajectoryManager:
    def __init__(self) -> None:
        self.trajectories: Dict[str, Dict[str, TrajectoryRecord]] = {}

    def store_trajectory(
        self,
        scenario_name: str,
        phase: str,
        trajectory: List[np.ndarray],
        start_config: Optional[np.ndarray] = None,
        goal_config: Optional[np.ndarray] = None,
        goal_pos: Optional[np.ndarray] = None,
        me_cost: Optional[Any] = None,
        safety_cost: Optional[Any] = None,
    ) -> None:
        """Store a trajectory for a given scenario and phase."""
        logger.info(f"Storing trajectory for {scenario_name} at phase '{phase}'")

        if scenario_name not in self.trajectories:
            self.trajectories[scenario_name] = {}

        self.trajectories[scenario_name][phase] = TrajectoryRecord(
            trajectory=trajectory,
            goal_pos=goal_pos,
            me_cost=me_cost,
            safety_cost=safety_cost,
        )

        logger.info(f"Stored trajectory for {scenario_name} at phase '{phase}'")

    def get_trajectory(self, scenario_name: str, phase: str) -> Optional[TrajectoryRecord]:
        """Get stored trajectory for evaluation."""
        if scenario_name not in self.trajectories:
            return None
        return self.trajectories[scenario_name].get(phase)

    def report(self) -> None:
        """Report stored trajectories."""
        logger.info("Stored Experiment Trajectories:")

        total_scenarios = len(self.trajectories)
        total_phases = sum(len(phases) for phases in self.trajectories.values())
        logger.info(f"Total scenarios: {total_scenarios}")
        logger.info(f"Total phases: {total_phases}")

        for scenario_name, phases in self.trajectories.items():
            phase_names = list(phases.keys())
            logger.info(f"- {scenario_name}: {phase_names}")

        logger.info("Detailed Trajectories:")
        for scenario_name, phases in self.trajectories.items():
            logger.info(f"Scenario: {scenario_name}")
            for phase_name, record in phases.items():
                plan = record.trajectory if record.trajectory else []
                if len(plan) == 0:
                    logger.warning(f"No traj saved in {scenario_name}/{phase_name}...")
                    continue
                logger.info(f"  Phase: {phase_name}")
                logger.info(f"    Length of traj: {len(plan)}.")

    def save_to_file(self, filename: str = "experiment_trajectories.pkl") -> None:
        """Save all trajectories to file."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.trajectories, f)
        logger.info(f"Saved trajectories to {filename}")

    def load_from_file(self, filename: str = "experiment_trajectories.pkl") -> None:
        """Load trajectories from file."""
        import pickle
        with open(filename, 'rb') as f:
            self.trajectories = pickle.load(f)
        logger.info(f"Loaded trajectories from {filename}")

    def update_from_file(self, filenames: str) -> None:
        """Load trajectories from file and adds to the dictionary of trajectories."""
        import pickle
        with open(filenames, 'rb') as f:
            trajectories = pickle.load(f)
        self.trajectories.update(trajectories)
        logger.info(f"Updated trajectories from {filenames}")
