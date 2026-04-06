"""Trajectory interpolation: convert sparse RRT waypoints into dense smooth paths."""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import List


def interpolate_trajectory(
    waypoints: List[np.ndarray],
    num_points_per_segment: int = 50,
    method: str = "cubic",
    joint_limits: np.ndarray = None,
) -> np.ndarray:
    """Interpolate sparse waypoints into a dense joint-space trajectory.

    Parameters
    ----------
    waypoints : list of (7,) arrays
        Sparse joint configurations from RRT planner.
    num_points_per_segment : int
        Number of interpolated points between each pair of consecutive waypoints.
    method : "cubic" or "linear"
        Interpolation method. Cubic gives smooth velocity/acceleration (C2).
    joint_limits : (7, 2) array, optional
        Per-joint [lower, upper] limits for clamping.

    Returns
    -------
    dense_traj : (N, 7) array
        Dense trajectory including original waypoints.
    """
    pts = np.array([wp[:7] for wp in waypoints])
    n_waypoints = len(pts)

    if n_waypoints < 2:
        return pts.copy()

    # Parameterize by cumulative joint-space arc length
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    # Avoid zero-length segments (near-duplicate waypoints)
    diffs = np.maximum(diffs, 1e-6)
    t_knots = np.concatenate([[0.0], np.cumsum(diffs)])

    n_segments = n_waypoints - 1
    total_points = n_segments * num_points_per_segment + 1
    t_dense = np.linspace(t_knots[0], t_knots[-1], total_points)

    if method == "cubic":
        cs = CubicSpline(t_knots, pts, bc_type="clamped")
        dense_traj = cs(t_dense)
    else:
        dense_traj = np.zeros((total_points, 7))
        for j in range(7):
            dense_traj[:, j] = np.interp(t_dense, t_knots, pts[:, j])

    if joint_limits is not None:
        for j in range(7):
            dense_traj[:, j] = np.clip(
                dense_traj[:, j], joint_limits[j, 0], joint_limits[j, 1]
            )

    return dense_traj
