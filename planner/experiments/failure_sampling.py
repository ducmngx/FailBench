"""Failure-point sampling strategies over the sim-step timeline.

Each strategy returns a list of absolute sim-step indices for a single
trajectory replay. Pure functions — no MuJoCo import — so they can be
used both by the dataset generator and by offline visualization.
"""

from typing import List, Optional
import numpy as np


def sample_fail_sim_step(
    total_sim_steps: int,
    strategy: str,
    rng: np.random.RandomState,
    n: int = 1,
    segment_boundaries_sim: Optional[List[int]] = None,
    boundary_margin: float = 0.05,
    n_bands: int = 6,
) -> List[int]:
    """Return ``n`` (or more, strategy-dependent) sim-step indices for one trajectory.

    Parameters
    ----------
    total_sim_steps
        Total number of sim steps in the trajectory's dense replay.
    strategy
        ``"uniform"``, ``"stratified_segments"``, or ``"stratified_bands"``.
    rng
        ``numpy.random.RandomState`` for reproducibility.
    n
        Samples per call for ``uniform`` and ``stratified_bands``. Ignored by
        ``stratified_segments`` (always returns one per segment).
    segment_boundaries_sim
        Cumulative sim-step indices at mission-segment boundaries, including
        a leading 0 and trailing ``total_sim_steps``. Required for
        ``stratified_segments``.
    boundary_margin
        Fraction of each segment to skip at the start and end (avoids
        zero-velocity boundaries from the cubic spline's clamped BCs).
    n_bands
        Number of equal-width time-progress bands for ``stratified_bands``.
    """
    lo, hi = 1, total_sim_steps - 1

    if strategy == "uniform":
        return rng.randint(lo, hi, size=n).tolist()

    if strategy == "stratified_segments":
        if segment_boundaries_sim is None:
            raise ValueError("stratified_segments requires segment_boundaries_sim")
        picks: List[int] = []
        for a, b in zip(segment_boundaries_sim[:-1], segment_boundaries_sim[1:]):
            span = b - a
            inner_lo = a + int(span * boundary_margin)
            inner_hi = b - int(span * boundary_margin)
            inner_lo = max(inner_lo, lo)
            inner_hi = min(inner_hi, hi)
            if inner_hi <= inner_lo:
                picks.append(max(lo, min((a + b) // 2, hi)))
            else:
                picks.append(int(rng.randint(inner_lo, inner_hi)))
        return picks

    if strategy == "stratified_bands":
        edges = np.linspace(lo, hi, n_bands + 1, dtype=int)
        picks = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                b = a + 1
            picks.append(int(rng.randint(a, b)))
        return picks

    raise ValueError(f"Unknown strategy: {strategy!r}")


def compute_segment_boundaries_sim(
    dense_segment_lengths: List[int], steps_per_interp_point: int
) -> List[int]:
    """Cumulative sim-step boundaries between mission segments.

    Given the per-segment dense-point counts (from ``interpolate_trajectory``)
    and the runner's ``steps_per_interp_point``, return the cumulative
    sim-step indices [0, end_of_seg_0, end_of_seg_1, ..., total].
    """
    bounds = [0]
    for n_dense in dense_segment_lengths:
        bounds.append(bounds[-1] + n_dense * steps_per_interp_point)
    return bounds
