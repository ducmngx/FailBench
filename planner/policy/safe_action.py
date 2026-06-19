"""Safety-aware action selectors driven by the contact predictor.

Three mechanisms, increasingly expressive, all share the same step() contract
and the same observation-window machinery (a small rolling buffer that keeps
the last T frames of RGB+state so the predictor's temporal-window input can
be assembled from raw env observations):

- :class:`ActionScalingPolicy` (Mechanism A): if predicted risk exceeds a
  threshold, multiplicatively scale the demo action by ``alpha`` (<1) before
  forwarding. Slows the arm down through risky configurations.
- :class:`CandidateSearchPolicy` (Mechanism B): generate ``n_candidates``
  small perturbations around the demo action, fork the env's sim for a
  1-step lookahead per candidate, score each by the predictor's marginal
  risk, and pick the lowest-risk candidate that stays within an L2 radius
  of the demo action.
- :class:`PassthroughPolicy` (baseline): just returns the demo action
  unchanged, but maintains the same observation window so we can still
  collect predictor scores for logging without affecting actions.

All three accept an optional ``masks_fn`` that takes the current env and
returns a dict of pixel masks keyed by entity name; if absent, the risk
score reduces to the heatmap sum (uniform, no per-entity weighting).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from planner.risk.inference import (
    ContactPredictor,
    marginal_heatmap,
    risk_score,
)


# Module-level defaults to mirror the dataset's window contract.
WINDOW_T = 8
STATE_DIM = 18      # qpos(7) + qvel(7) + ee(3) + grip(1)


# ---------------------------------------------------------------------------
# Observation window buffer
# ---------------------------------------------------------------------------

class ObsWindow:
    """Rolling T-frame buffer of (rgb, state) for predictor input assembly.

    Repeats the most recent frame T times on first push so the predictor
    sees a valid window before T-1 steps of warmup. This matches the
    sample_window_indices clipping behavior in the dataset path.
    """

    def __init__(self, T: int = WINDOW_T, H: int = 240, W: int = 320,
                 state_dim: int = STATE_DIM):
        self.T = T
        self.H, self.W = H, W
        self.state_dim = state_dim
        self._rgb: deque = deque(maxlen=T)
        self._state: deque = deque(maxlen=T)

    def reset(self) -> None:
        self._rgb.clear()
        self._state.clear()

    def push(self, rgb_chw: np.ndarray, state_vec: np.ndarray) -> None:
        """Append the latest observation. First push fills the buffer."""
        if rgb_chw.shape != (3, self.H, self.W):
            raise ValueError(f"rgb shape {rgb_chw.shape} != (3, {self.H}, {self.W})")
        if state_vec.shape != (self.state_dim,):
            raise ValueError(f"state shape {state_vec.shape} != ({self.state_dim},)")
        if not self._rgb:
            # Fill: replicate the first frame T times so the window is
            # immediately usable.
            for _ in range(self.T):
                self._rgb.append(rgb_chw.copy())
                self._state.append(state_vec.copy())
            return
        self._rgb.append(rgb_chw.copy())
        self._state.append(state_vec.copy())

    @property
    def ready(self) -> bool:
        return len(self._rgb) == self.T

    @property
    def rgb_window(self) -> np.ndarray:
        """(T, 3, H, W) array."""
        return np.stack(list(self._rgb))

    @property
    def state_window(self) -> np.ndarray:
        """(T, state_dim) array."""
        return np.stack(list(self._state))


# ---------------------------------------------------------------------------
# Shared scoring helper
# ---------------------------------------------------------------------------

@dataclass
class RiskQuery:
    """Result of one predictor query at the current step."""
    total_risk: float
    per_entity: dict[str, float]
    heatmap: np.ndarray
    gate_prob: float


def query_risk(predictor: ContactPredictor, obs_window: ObsWindow,
               masks: Optional[dict[str, np.ndarray]] = None,
               mode_prior: Optional[dict[str, float]] = None,
               object_values: Optional[dict[str, float]] = None,
               ) -> RiskQuery:
    """Score the current observation window with the predictor.

    If ``masks`` is None, returns the heatmap sum as ``total_risk`` and an
    empty per-entity dict — useful for "just track risk over time" logging
    without scene-aware integration.
    """
    heat, gate_prob = marginal_heatmap(predictor,
                                       obs_window.rgb_window,
                                       obs_window.state_window,
                                       mode_prior=mode_prior)
    if masks is None:
        total = float(heat.sum())
        return RiskQuery(total_risk=total, per_entity={}, heatmap=heat,
                         gate_prob=gate_prob)
    total, per = risk_score(heat, masks, object_values=object_values)
    return RiskQuery(total_risk=total, per_entity=per,
                     heatmap=heat, gate_prob=gate_prob)


# ---------------------------------------------------------------------------
# Mechanism: passthrough baseline
# ---------------------------------------------------------------------------

class PassthroughPolicy:
    """Baseline: returns the demo action verbatim.

    Still maintains the observation window and (optionally) queries the
    predictor so the rollout log can include predicted risk alongside the
    realized outcome — useful for the predicted-vs-realized correlation
    figure even on baseline rollouts.
    """

    def __init__(self, predictor: Optional[ContactPredictor] = None,
                 obs_window: Optional[ObsWindow] = None,
                 log_risk: bool = True):
        self.predictor = predictor
        self.obs_window = obs_window or ObsWindow()
        self.log_risk = log_risk and predictor is not None

    def reset(self) -> None:
        self.obs_window.reset()

    def select(self, rgb_chw: np.ndarray, state_vec: np.ndarray,
               demo_action: np.ndarray,
               masks: Optional[dict[str, np.ndarray]] = None,
               ) -> tuple[np.ndarray, dict]:
        self.obs_window.push(rgb_chw, state_vec)
        info: dict[str, Any] = {"mechanism": "passthrough", "scaled": False}
        if self.log_risk and self.predictor is not None:
            rq = query_risk(self.predictor, self.obs_window, masks=masks)
            info["risk"] = rq.total_risk
            info["gate_prob"] = rq.gate_prob
            info["per_entity"] = rq.per_entity
        return demo_action.copy(), info


# ---------------------------------------------------------------------------
# Mechanism A: action scaling
# ---------------------------------------------------------------------------

class ActionScalingPolicy:
    """Slow the arm down when predicted risk exceeds ``tau``.

    The chosen action is::

        a_safe = alpha * a_demo   if risk > tau
                 a_demo           otherwise

    ``alpha`` defaults to 0.3 (~3x slower transit through the risky region).
    ``tau`` should be calibrated against the val-set risk-score distribution
    (e.g. the 70th percentile).
    """

    def __init__(self, predictor: ContactPredictor,
                 alpha: float = 0.3,
                 tau: float = 0.0,
                 obs_window: Optional[ObsWindow] = None,
                 mode_prior: Optional[dict[str, float]] = None,
                 object_values: Optional[dict[str, float]] = None,
                 scale_gripper: bool = False):
        self.predictor = predictor
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.obs_window = obs_window or ObsWindow()
        self.mode_prior = mode_prior
        self.object_values = object_values
        self.scale_gripper = scale_gripper

    def reset(self) -> None:
        self.obs_window.reset()

    def select(self, rgb_chw: np.ndarray, state_vec: np.ndarray,
               demo_action: np.ndarray,
               masks: Optional[dict[str, np.ndarray]] = None,
               ) -> tuple[np.ndarray, dict]:
        self.obs_window.push(rgb_chw, state_vec)
        rq = query_risk(self.predictor, self.obs_window, masks=masks,
                        mode_prior=self.mode_prior,
                        object_values=self.object_values)
        scaled = rq.total_risk > self.tau
        action = demo_action.copy()
        if scaled:
            if self.scale_gripper or demo_action.size <= 6:
                action *= self.alpha
            else:
                # LIBERO 7-DoF: scale only the 6 EE-delta dims, leave gripper
                # bit alone (slowing the EE motion is the safety lever; the
                # gripper open/close is task-critical).
                action[:6] *= self.alpha
        info = {
            "mechanism": "scaling",
            "scaled": bool(scaled),
            "risk": rq.total_risk,
            "tau": self.tau,
            "alpha": self.alpha,
            "gate_prob": rq.gate_prob,
            "per_entity": rq.per_entity,
        }
        return action, info


# ---------------------------------------------------------------------------
# Mechanism B: candidate-action search with 1-step env-fork lookahead
# ---------------------------------------------------------------------------

class CandidateSearchPolicy:
    """Pick the safest among ``n_candidates`` perturbations of the demo action.

    At each step:

    1. Generate ``n_candidates`` candidates: ``demo_action + Gaussian noise``
       with stddev ``sigma`` on the 6 EE-delta dims (gripper bit clamped to
       0/1 of the demo action).
    2. For each candidate, fork the env (``env_fork_fn(env)``), apply one
       step with the candidate action, read the resulting RGB + state, query
       the predictor, score by risk.
    3. Pick the candidate whose risk is lowest **and** whose L2 distance
       from the demo action is within ``epsilon``. Fall back to the demo
       action if no candidate satisfies the constraint.

    The ``env_fork_fn`` is supplied by the caller because the right way to
    fork depends on the env shape (robosuite/MjSim vs ``mujoco.MjData``);
    we keep this layer agnostic.

    Cost: ``n_candidates`` predictor queries + ``n_candidates`` env-fork
    steps per env step. Defaults are tuned to keep per-step cost ≤ 200 ms
    on an A100 MIG slice for the GatekeeperCoordFiLMUNet predictor.
    """

    def __init__(self, predictor: ContactPredictor,
                 env: Any,
                 env_fork_fn: Callable[[Any], Any],
                 n_candidates: int = 8,
                 sigma: float = 0.02,
                 epsilon: float = 0.1,
                 include_demo: bool = True,
                 obs_extractor: Optional[
                     Callable[[Any], tuple[np.ndarray, np.ndarray]]] = None,
                 obs_window: Optional[ObsWindow] = None,
                 mode_prior: Optional[dict[str, float]] = None,
                 object_values: Optional[dict[str, float]] = None,
                 rng: Optional[np.random.Generator] = None):
        self.predictor = predictor
        self.env = env
        self.env_fork_fn = env_fork_fn
        self.n_candidates = int(n_candidates)
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        self.include_demo = bool(include_demo)
        self.obs_extractor = obs_extractor or _default_obs_extractor
        self.obs_window = obs_window or ObsWindow()
        self.mode_prior = mode_prior
        self.object_values = object_values
        self.rng = rng or np.random.default_rng()

    def reset(self) -> None:
        self.obs_window.reset()

    def select(self, rgb_chw: np.ndarray, state_vec: np.ndarray,
               demo_action: np.ndarray,
               masks: Optional[dict[str, np.ndarray]] = None,
               ) -> tuple[np.ndarray, dict]:
        # Push current state into the window so the candidate-evaluation
        # path has a populated window even before any candidate runs.
        self.obs_window.push(rgb_chw, state_vec)

        candidates = self._propose_candidates(demo_action)
        scores: list[float] = []
        for cand in candidates:
            fork = self.env_fork_fn(self.env)
            fork.step(cand)
            rgb_c, state_c = self.obs_extractor(fork)
            # Use a transient window: push the candidate's resulting obs onto
            # a copy of the current window for the predictor query.
            transient = _clone_obs_window(self.obs_window)
            transient.push(rgb_c, state_c)
            rq = query_risk(self.predictor, transient, masks=masks,
                            mode_prior=self.mode_prior,
                            object_values=self.object_values)
            scores.append(rq.total_risk)
            # NB: caller is responsible for not advancing the real env;
            # `fork` is discarded here.

        # Pick best within L2 epsilon of demo
        best_idx = -1
        best_score = float("inf")
        for i, cand in enumerate(candidates):
            l2 = float(np.linalg.norm(cand - demo_action))
            if l2 > self.epsilon:
                continue
            if scores[i] < best_score:
                best_score = scores[i]
                best_idx = i

        if best_idx < 0:
            action = demo_action.copy()
            info_picked = "demo_fallback"
        else:
            action = candidates[best_idx].copy()
            info_picked = f"candidate_{best_idx}"

        info = {
            "mechanism": "search",
            "scaled": False,
            "n_candidates": len(candidates),
            "scores": list(scores),
            "picked": info_picked,
            "picked_score": float(best_score) if best_idx >= 0 else None,
            "per_entity": {},  # could be repopulated for the picked candidate
        }
        return action, info

    def _propose_candidates(self, demo_action: np.ndarray) -> list[np.ndarray]:
        """Gaussian perturbations on the 6 EE-delta dims; gripper bit
        preserved on the demo's choice."""
        cands: list[np.ndarray] = []
        if self.include_demo:
            cands.append(demo_action.copy())
        for _ in range(self.n_candidates):
            noise = np.zeros_like(demo_action)
            noise[:6] = self.rng.normal(0.0, self.sigma, size=6)
            cands.append(demo_action + noise)
        return cands


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone_obs_window(w: ObsWindow) -> ObsWindow:
    """Shallow-copy an ObsWindow so transient candidate queries don't mutate
    the live policy buffer."""
    new = ObsWindow(T=w.T, H=w.H, W=w.W, state_dim=w.state_dim)
    new._rgb = deque(list(w._rgb), maxlen=w.T)
    new._state = deque(list(w._state), maxlen=w.T)
    return new


def _default_obs_extractor(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read (rgb_chw, state_vec) from a LIBERO env's last observation.

    Picks ``agentview_image`` (H, W, 3) → CHW and assembles a state vector
    from robosuite's standard proprio keys. Overrideable by passing a
    custom ``obs_extractor`` to :class:`CandidateSearchPolicy`.
    """
    # robosuite envs return the last observation via .render or via a
    # dedicated observable getter; LIBERO's OffScreenRenderEnv carries it on
    # `_obs`. Best-effort: try a few attributes, raise if none works.
    obs = None
    for attr in ("_last_obs", "_obs", "current_observation"):
        if hasattr(env, attr):
            obs = getattr(env, attr)
            if obs is not None:
                break
    if obs is None:
        raise RuntimeError(
            "default obs_extractor couldn't find a cached observation on the "
            "forked env — pass a custom obs_extractor to CandidateSearchPolicy")

    rgb = np.asarray(obs.get("agentview_image"))
    if rgb is None:
        raise RuntimeError("env obs missing 'agentview_image'")
    # Robosuite OpenGL framebuffer has Y up; the v2 dataset (and so the
    # predictor) stored RGB with Y down. Flip every env image before it
    # reaches the predictor.
    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb_chw = np.transpose(np.flipud(rgb).copy(),
                                (2, 0, 1)).astype(np.uint8)
    else:
        rgb_chw = np.flipud(rgb).copy().astype(np.uint8)

    # State assembly: 7 qpos + 7 qvel + 3 ee + 1 grip. Use robot0_joint_pos /
    # robot0_joint_vel / robot0_eef_pos / robot0_gripper_qpos.
    qpos = np.asarray(obs.get("robot0_joint_pos",
                              np.zeros(7)), dtype=np.float32)
    qvel = np.asarray(obs.get("robot0_joint_vel",
                              np.zeros(7)), dtype=np.float32)
    ee   = np.asarray(obs.get("robot0_eef_pos",
                              np.zeros(3)), dtype=np.float32)
    grip = np.asarray(obs.get("robot0_gripper_qpos",
                              np.zeros(1)), dtype=np.float32)
    if grip.ndim > 0 and grip.size > 1:
        grip = grip[:1]
    state_vec = np.concatenate([
        qpos[:7], qvel[:7], ee[:3], grip[:1].reshape(1)]).astype(np.float32)
    return rgb_chw, state_vec


__all__ = [
    "ObsWindow",
    "RiskQuery",
    "query_risk",
    "PassthroughPolicy",
    "ActionScalingPolicy",
    "CandidateSearchPolicy",
]
