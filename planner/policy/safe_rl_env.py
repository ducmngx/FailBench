"""Gym-compatible LIBERO env with FailBench predictor + damage reward.

Drop-in for sb3 / cleanrl / any Gym-style trainer.  Wraps LIBERO's
``OffScreenRenderEnv`` and exposes:

- standard Gym ``observation_space`` and ``action_space``
- ``reset()`` / ``step(action)`` semantics with optional automatic failure
  injection
- shaped reward
    ``r_t = r_task(s_t) − λ_pred · pred_cost(s_t) − λ_dmg · Δd_mech(s_t)``
  where individual components are also logged in ``info`` so the trainer
  can route them to TensorBoard / WandB separately
- ``info`` dict includes per-step components for logging:
    ``r_task``, ``r_pred``, ``r_damage``, ``pred_risk_total``,
    ``pred_per_body``, ``gate_prob``, ``damage_total``,
    ``per_body_damage``, ``failure_injected``

Typical use::

    from planner.policy.safe_rl_env import SafeLiberoEnv

    env = SafeLiberoEnv(
        bddl_file="external/LIBERO/libero/libero/bddl_files/.../<task>.bddl",
        demo_hdf5="datasets/libero/raw/libero_spatial/<task>_demo.hdf5",
        ckpt="notebooks/.../best_ep08_val0.0648.pt",
        lambda_pred=0.001,      # predictor cost weight (tune!)
        lambda_dmg=1.0,         # OopsieVerse damage cost weight
        failure_prob=0.1,       # probability a failure is injected per episode
        failure_modes=("GRIPPER_OPEN", "SLIPPERY_GRIP", "SINGLE_JOINT",
                        "MULTI_JOINT", "ALL_JOINTS"),
        progress_range=(0.2, 0.85),
        predictor_every_k=1,    # query predictor every K steps (1 = always)
    )

    obs, info = env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

Vectorize with sb3::

    from stable_baselines3.common.vec_env import SubprocVecEnv
    def make(): return SafeLiberoEnv(...)
    vec = SubprocVecEnv([make for _ in range(16)])

Reward gotchas:

- Predictor outputs are not normalized.  pred_risk_total typically lives in
  [0, ~5000].  Default lambda_pred=1e-3 brings it into the same magnitude as
  task reward (which is 0 or +1 in LIBERO).  Tune.
- d_mech damage values are O(0.01–1).  Default lambda_dmg=1.0 is the right
  ballpark, but the per-step DELTA is much smaller — we reward the policy
  for not increasing damage, not the cumulative amount.
- Sparse task reward — LIBERO success is binary, fired at the end.  Consider
  adding a shaped task term (proximity / staged reward) if you train from
  scratch; ours is the canonical sparse reward.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

# Gym / Gymnasium compatibility shim — sb3 wants gymnasium, but LIBERO
# brings gym 0.25.2.  We expose a gymnasium-style API (5-tuple step) and
# inherit from whatever Gym base is available.
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces
    _GYMNASIUM = False


# Module-level lazy imports keep import-time cost low when this file is
# pulled in only for type hints (e.g. by trainer code that uses a
# different env).
def _load_heavy_deps():
    from libero.libero.envs import OffScreenRenderEnv
    from planner.experiments.config import FailureConfig, FailureMode
    from planner.experiments.libero.naming import resolve_model_handles
    from planner.policy.libero_env_failure import (
        EnvFailureScheduler, unwrap_sim)
    from planner.policy.safe_action import ObsWindow, query_risk
    from planner.risk.damage import DamageAccumulator
    from planner.risk.inference import ContactPredictor, marginal_heatmap
    from scripts.safety.safety_rollout import build_entity_masks
    return dict(
        OffScreenRenderEnv=OffScreenRenderEnv,
        FailureConfig=FailureConfig,
        FailureMode=FailureMode,
        resolve_model_handles=resolve_model_handles,
        EnvFailureScheduler=EnvFailureScheduler,
        unwrap_sim=unwrap_sim,
        ObsWindow=ObsWindow,
        query_risk=query_risk,
        DamageAccumulator=DamageAccumulator,
        ContactPredictor=ContactPredictor,
        marginal_heatmap=marginal_heatmap,
        build_entity_masks=build_entity_masks,
    )


# ---------------------------------------------------------------------------
# Failure scheduling — random sampling per episode
# ---------------------------------------------------------------------------

_DEFAULT_MODE_JOINTS: Dict[str, List[int]] = {
    "GRIPPER_OPEN":  [],
    "SLIPPERY_GRIP": [],
    "SINGLE_JOINT":  [4],
    "MULTI_JOINT":   [2, 4],
    "ALL_JOINTS":    [1, 2, 3, 4, 5, 6, 7],
}


@dataclass
class FailureSchedule:
    """Per-episode failure sampling spec."""
    failure_prob: float = 0.0
    failure_modes: tuple = (
        "GRIPPER_OPEN", "SLIPPERY_GRIP",
        "SINGLE_JOINT", "MULTI_JOINT", "ALL_JOINTS")
    progress_range: tuple = (0.2, 0.85)
    # Optional per-mode joints; falls back to the canonical set above.
    mode_joints: dict = field(default_factory=dict)

    def sample(self, rng: random.Random, n_actions: int
               ) -> Optional[Tuple[str, List[int], int]]:
        """Returns (mode, joints, fail_step) or None if no failure fires."""
        if self.failure_prob <= 0 or rng.random() >= self.failure_prob:
            return None
        mode = rng.choice(self.failure_modes)
        joints = list(self.mode_joints.get(
            mode, _DEFAULT_MODE_JOINTS.get(mode, [])))
        p = rng.uniform(*self.progress_range)
        fail_step = max(1, int(p * n_actions))
        return mode, joints, fail_step


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SafeLiberoEnv(gym.Env):
    """Gym-compatible LIBERO env with predictor + d_mech reward.

    Parameters
    ----------
    bddl_file
        Path to the task's BDDL file under
        ``external/LIBERO/libero/libero/bddl_files/...``.
    demo_hdf5
        Path to the matching HDF5 demo set.  Used to seed each episode's
        initial state from a randomly-chosen demo so the open-loop
        randomization of object spawn positions stays in distribution.
        Set to ``None`` to use LIBERO's default random spawn each episode
        (less reproducible, less in-distribution).
    ckpt
        Predictor checkpoint path.  Set to ``None`` to disable the
        predictor cost (Stage A baseline — pure d_mech reward).
    lambda_pred, lambda_dmg
        Reward weights — see module docstring for tuning notes.
    failure_prob, failure_modes, progress_range
        Failure scheduling — see :class:`FailureSchedule`.
    image_h, image_w
        Camera resolution for the agentview.  Must match what the predictor
        was trained on (240x320 for our v2 checkpoints).
    predictor_every_k
        Query predictor every K steps and reuse the result.  Set to >1 to
        trade off cost-signal latency for trainer step rate.
    mode_prior
        Failure-mode prior for the marginal-heatmap query.  Defaults to
        uniform over the 5 modes.
    severity
        Per-body multiplier applied to the predictor heatmap mass before
        integrating.  Defaults to uniform 1.0.  See planner/risk/damage.py
        for LIBERO fragility values that would make sensible severity.
    rgb_in_obs
        Whether to include the agentview RGB in the policy observation
        space.  If False, the obs is proprio-only and the predictor still
        gets RGB internally (its job, not the policy's).  Default True.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        bddl_file: str,
        demo_hdf5: Optional[str] = None,
        ckpt: Optional[str] = None,
        lambda_pred: float = 1e-3,
        lambda_dmg: float = 1.0,
        failure_prob: float = 0.0,
        failure_modes: tuple = (
            "GRIPPER_OPEN", "SLIPPERY_GRIP",
            "SINGLE_JOINT", "MULTI_JOINT", "ALL_JOINTS"),
        progress_range: tuple = (0.2, 0.85),
        max_episode_steps: int = 250,
        image_h: int = 240,
        image_w: int = 320,
        predictor_every_k: int = 1,
        mode_prior: Optional[Dict[str, float]] = None,
        severity: Optional[Dict[str, float]] = None,
        rgb_in_obs: bool = True,
        pred_features_in_obs: bool = True,
        seed: Optional[int] = None,
        predictor_device: Optional[str] = None,
    ):
        deps = _load_heavy_deps()
        self._deps = deps

        self.bddl_file = str(bddl_file)
        self.demo_hdf5 = str(demo_hdf5) if demo_hdf5 else None
        self.lambda_pred = float(lambda_pred)
        self.lambda_dmg = float(lambda_dmg)
        self.max_episode_steps = int(max_episode_steps)
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.predictor_every_k = max(1, int(predictor_every_k))
        self.mode_prior = mode_prior
        self.severity = severity
        self.rgb_in_obs = bool(rgb_in_obs)
        self.pred_features_in_obs = bool(pred_features_in_obs)
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self._schedule = FailureSchedule(
            failure_prob=failure_prob,
            failure_modes=tuple(failure_modes),
            progress_range=tuple(progress_range),
        )

        # Build underlying env once
        self._env = deps["OffScreenRenderEnv"](
            bddl_file_name=self.bddl_file,
            camera_heights=self.image_h,
            camera_widths=self.image_w,
        )
        self._scheduler = deps["EnvFailureScheduler"](
            self._env, failure=None, fail_step=None)

        # Build entity masks once (for predictor mask integration)
        self._masks = deps["build_entity_masks"](
            self._env, (self.image_h, self.image_w))

        # Capture clean model snapshot before any failure mutates anything.
        # The scheduler also keeps its own; we keep one too because the env
        # is shared with the scheduler — defense in depth.
        m0, _ = deps["unwrap_sim"](self._env.sim)
        self._clean_model_state = dict(
            actuator_gainprm=m0.actuator_gainprm.copy(),
            actuator_biastype=m0.actuator_biastype.copy(),
            actuator_gaintype=m0.actuator_gaintype.copy(),
            jnt_stiffness=m0.jnt_stiffness.copy(),
            dof_damping=m0.dof_damping.copy(),
            jnt_range=m0.jnt_range.copy(),
            dof_frictionloss=m0.dof_frictionloss.copy(),
        )

        # Robot geom ids for the damage accumulator
        handles = deps["resolve_model_handles"](m0)
        self._robot_geom_ids = handles.robot_geom_ids

        # Load predictor (optional)
        self._predictor = None
        if ckpt is not None:
            self._predictor = deps["ContactPredictor"].from_checkpoint(
                str(ckpt), device=predictor_device)

        # Per-episode state — filled in reset()
        self._damage_accum = None
        self._obs_window = deps["ObsWindow"]()
        self._step_idx = 0
        self._last_pred_risk = 0.0
        self._last_pred_per_body: dict = {}
        self._last_gate_prob = float("nan")
        self._prev_damage_total = 0.0
        self._injected_event: Optional[dict] = None

        # Demo cache — load action sequences from the HDF5 once and reuse
        self._demo_cache: List[np.ndarray] = []
        self._demo_init_states: List[np.ndarray] = []
        self._load_demos()

        # Action / obs spaces
        action_dim = self._env.env.robots[0].action_dim
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        proprio_dim = 7 + 7 + 3 + 1   # qpos + qvel + ee_pos + gripper_qpos
        obs_spaces: Dict[str, "spaces.Space"] = {
            "proprio": spaces.Box(low=-np.inf, high=np.inf,
                                   shape=(proprio_dim,), dtype=np.float32),
        }
        if self.rgb_in_obs:
            obs_spaces["agentview_rgb"] = spaces.Box(
                low=0, high=255, shape=(self.image_h, self.image_w, 3),
                dtype=np.uint8)
        if self.pred_features_in_obs:
            # Fixed canonical body order for the per-body vector so the policy
            # sees the same dimensions across episodes.  Anything not present
            # in masks gets 0 by construction in _build_obs.
            self._body_names = sorted(self._masks.keys())
            K = len(self._body_names)
            obs_spaces["pred_per_body"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(K,), dtype=np.float32)
            obs_spaces["gate_prob"] = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self._body_names = []
        self.observation_space = spaces.Dict(obs_spaces)

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: Optional[int] = None,
               options: Optional[dict] = None) -> Tuple[dict, dict]:
        import mujoco
        if seed is not None:
            self._rng.seed(seed)
            self._np_rng = np.random.default_rng(seed)

        # Sample a demo for init-state seeding (if available).
        demo_idx = None
        init_state = None
        if self._demo_init_states:
            demo_idx = self._rng.randrange(len(self._demo_init_states))
            init_state = self._demo_init_states[demo_idx]

        # Sample failure event.  n_actions for progress sampling: use the
        # current demo's action length if seeded, else max_episode_steps.
        n_for_progress = (len(self._demo_cache[demo_idx])
                           if (demo_idx is not None and self._demo_cache)
                           else self.max_episode_steps)
        fail_spec = self._schedule.sample(self._rng, n_for_progress)
        if fail_spec is not None:
            mode, joints, fail_step = fail_spec
            failure = self._deps["FailureConfig"](
                mode=self._deps["FailureMode"][mode],
                probability=1.0,
                joint_names=([f"joint{j}" for j in joints]
                              if joints else None),
            )
            self._scheduler.failure = failure
            self._scheduler.fail_step = fail_step
            self._injected_event = dict(mode=mode, joints=joints,
                                         fail_step=fail_step)
        else:
            self._scheduler.failure = None
            self._scheduler.fail_step = None
            self._injected_event = None

        # Restore clean model state to defend against any leaked mutation
        m, _ = self._deps["unwrap_sim"](self._env.sim)
        for k, v in self._clean_model_state.items():
            getattr(m, k)[:] = v

        obs = self._scheduler.reset()

        # Seed sim state from demo init.  LIBERO's own evaluation harness
        # (libero/lifelong/evaluate.py) uses env.set_init_state() + 5 zero-
        # action warmup steps to settle the OSC controller.  Direct
        # mj_forward leaves the controller's goal stale, which makes the
        # very first action interpretation diverge from the demo's frame
        # and causes BC/demo-replay to fail.
        if init_state is not None:
            try:
                self._env.set_init_state(np.asarray(init_state).astype(
                    np.float64))
                # Warmup steps: 5 zero-action steps to advance the controller
                # into a stable state matching the demo's recording.
                action_dim = self._env.env.robots[0].action_dim
                zero_action = np.zeros(action_dim, dtype=np.float32)
                for _ in range(5):
                    obs, _, _, _ = self._env.step(zero_action)
            except Exception as e:
                # Fall back to direct qpos/qvel write
                model, data = self._deps["unwrap_sim"](self._env.sim)
                nq, nv = model.nq, model.nv
                flat = np.asarray(init_state, dtype=np.float64)
                off = 1 if flat.shape[0] == 1 + nq + nv else 0
                data.qpos[:nq] = flat[off:off + nq]
                data.qvel[:nv] = flat[off + nq:off + nq + nv]
                mujoco.mj_forward(model, data)

        # Rebuild damage accumulator against the fresh (post-reset) model
        model, data = self._deps["unwrap_sim"](self._env.sim)
        self._damage_accum = self._deps["DamageAccumulator"](
            model, data, self._robot_geom_ids)
        self._damage_accum.reset()
        self._obs_window = self._deps["ObsWindow"]()
        self._step_idx = 0
        self._prev_damage_total = 0.0
        self._last_pred_risk = 0.0
        self._last_pred_per_body = {}
        self._last_gate_prob = float("nan")

        return self._build_obs(obs), self._build_info(extra={})

    def step(self, action):
        obs, libero_reward, libero_done, libero_info = self._scheduler.step(action)
        self._step_idx += 1

        # Damage update
        damage_step = self._damage_accum.step()
        damage_total = self._damage_accum.total_damage
        damage_delta = damage_total - self._prev_damage_total
        self._prev_damage_total = damage_total

        # Predictor query — every K steps to keep training fast.
        if (self._predictor is not None
                and (self._step_idx - 1) % self.predictor_every_k == 0):
            self._update_predictor(obs)

        # Reward components.  LIBERO's OffScreenRenderEnv returns r=1.0
        # on the success step (BDDL predicate satisfied) and r=0 otherwise.
        # It does NOT populate info["success"] — info is typically empty.
        # So we use libero_reward directly as the success signal.
        success = bool(libero_reward > 0.5)
        r_task = float(libero_reward)
        r_pred = -self.lambda_pred * float(self._last_pred_risk)
        r_dmg = -self.lambda_dmg * float(damage_delta)
        reward = r_task + r_pred + r_dmg

        terminated = bool(libero_done) or success
        truncated = (self._step_idx >= self.max_episode_steps)

        info = self._build_info(extra=dict(
            r_task=r_task,
            r_pred=r_pred,
            r_damage=r_dmg,
            damage_step=float(damage_step),
            damage_total=float(damage_total),
            success=success,
        ))

        return self._build_obs(obs), float(reward), terminated, truncated, info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """Return the current agentview RGB in v2 orientation (Y-down)."""
        obs = self._env._get_observations() if hasattr(
            self._env, "_get_observations") else None
        if obs is None or "agentview_image" not in obs:
            return np.zeros((self.image_h, self.image_w, 3), dtype=np.uint8)
        return np.flipud(np.asarray(obs["agentview_image"])).copy()

    # --------------------------------------------------------------- internals

    def _load_demos(self) -> None:
        if not self.demo_hdf5:
            return
        import h5py
        with h5py.File(self.demo_hdf5, "r") as f:
            data = f["data"]
            for key in sorted(data.keys()):
                grp = data[key]
                if "actions" not in grp or "states" not in grp:
                    continue
                self._demo_cache.append(
                    np.asarray(grp["actions"], dtype=np.float32))
                self._demo_init_states.append(
                    np.asarray(grp["states"][0], dtype=np.float32))

    def _update_predictor(self, obs: dict) -> None:
        rgb = obs.get("agentview_image")
        if rgb is None:
            return
        rgb_v2 = np.flipud(np.asarray(rgb)).copy()
        rgb_chw = np.transpose(rgb_v2, (2, 0, 1)).astype(np.uint8)
        qpos = np.asarray(obs.get("robot0_joint_pos", np.zeros(7)))[:7]
        qvel = np.asarray(obs.get("robot0_joint_vel", np.zeros(7)))[:7]
        ee = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)))[:3]
        grip = np.asarray(obs.get("robot0_gripper_qpos",
                                     np.zeros(1))).ravel()[:1]
        state_vec = np.concatenate([qpos, qvel, ee, grip]).astype(np.float32)
        self._obs_window.push(rgb_chw, state_vec)
        rq = self._deps["query_risk"](
            self._predictor, self._obs_window,
            masks=self._masks,
            mode_prior=self.mode_prior,
            object_values=self.severity,
        )
        self._last_pred_risk = float(rq.total_risk)
        self._last_pred_per_body = dict(rq.per_entity)
        self._last_gate_prob = float(rq.gate_prob) \
            if rq.gate_prob is not None else float("nan")

    def _build_obs(self, libero_obs: dict) -> dict:
        out = {}
        # Proprio
        qpos = np.asarray(libero_obs.get("robot0_joint_pos",
                                            np.zeros(7)))[:7]
        qvel = np.asarray(libero_obs.get("robot0_joint_vel",
                                            np.zeros(7)))[:7]
        ee = np.asarray(libero_obs.get("robot0_eef_pos", np.zeros(3)))[:3]
        grip = np.asarray(libero_obs.get("robot0_gripper_qpos",
                                            np.zeros(1))).ravel()[:1]
        out["proprio"] = np.concatenate(
            [qpos, qvel, ee, grip]).astype(np.float32)
        if self.rgb_in_obs:
            rgb = libero_obs.get("agentview_image")
            if rgb is None:
                out["agentview_rgb"] = np.zeros(
                    (self.image_h, self.image_w, 3), dtype=np.uint8)
            else:
                # Y-down convention (v2-compatible) so this also matches what
                # the predictor sees internally.  If you want to keep raw
                # OpenGL orientation for the policy, drop the flipud here.
                out["agentview_rgb"] = np.flipud(
                    np.asarray(rgb)).copy().astype(np.uint8)
        if self.pred_features_in_obs:
            # Fixed-order vector over the canonical body list captured at
            # __init__ time.  Missing entries (the predictor's per_entity
            # may be sparse when masks are empty for some bodies) default to 0.
            out["pred_per_body"] = np.array(
                [self._last_pred_per_body.get(b, 0.0)
                 for b in self._body_names], dtype=np.float32)
            gate = self._last_gate_prob
            if not np.isfinite(gate):
                gate = 0.0
            out["gate_prob"] = np.array([gate], dtype=np.float32)
        return out

    def _build_info(self, *, extra: dict) -> dict:
        info = {
            "pred_risk_total": float(self._last_pred_risk),
            "pred_per_body": dict(self._last_pred_per_body),
            "gate_prob": float(self._last_gate_prob),
            "per_body_damage": (dict(self._damage_accum.per_body_damage)
                                 if self._damage_accum is not None else {}),
            "failure_injected": (dict(self._injected_event)
                                  if self._injected_event else None),
            "step_idx": int(self._step_idx),
        }
        info.update(extra)
        return info


# ---------------------------------------------------------------------------
# Convenience smoke-test entry point
# ---------------------------------------------------------------------------

def _smoke():
    """Run a random rollout to verify the env wires up.

    With ``--render window`` opens a live OpenCV window showing the
    agentview, a banner with reward / pred / damage / gate, and a small
    health-bar overlay so you can watch the sim head-on.
    With ``--render mp4 --render_path out.mp4`` writes a video instead.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl", required=True)
    ap.add_argument("--demo", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--lambda_pred", type=float, default=1e-3)
    ap.add_argument("--lambda_dmg", type=float, default=1.0)
    ap.add_argument("--failure_prob", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--render", choices=["none", "window", "mp4"],
                    default="none",
                    help="none = log to stdout only; "
                          "window = live OpenCV viewer; "
                          "mp4 = write to --render_path.")
    ap.add_argument("--render_path", default="out/safe_rl_env_smoke.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--upscale", type=int, default=2)
    args = ap.parse_args()

    env = SafeLiberoEnv(
        bddl_file=args.bddl, demo_hdf5=args.demo, ckpt=args.ckpt,
        lambda_pred=args.lambda_pred, lambda_dmg=args.lambda_dmg,
        failure_prob=args.failure_prob,
        max_episode_steps=args.steps,
    )
    obs, info = env.reset()
    print(f"observation keys: {list(obs.keys())}")
    print(f"proprio shape: {obs['proprio'].shape}")
    if "agentview_rgb" in obs:
        print(f"agentview_rgb shape: {obs['agentview_rgb'].shape}")
    print(f"action space: {env.action_space}")
    print(f"failure injected: {info['failure_injected']}")
    print()

    # Set up renderer
    writer = None
    if args.render == "window":
        import cv2
        cv2.namedWindow("SafeLiberoEnv", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SafeLiberoEnv",
                          env.image_w * args.upscale,
                          (env.image_h + 70) * args.upscale)
    elif args.render == "mp4":
        import cv2
        from pathlib import Path as _P
        _P(args.render_path).parent.mkdir(parents=True, exist_ok=True)
        H = (env.image_h + 70) * args.upscale
        W = env.image_w * args.upscale
        writer = cv2.VideoWriter(
            args.render_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps, (W, H))

    total_reward = 0.0
    for i in range(args.steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        if (args.render == "none"
                and (i < 3 or i >= args.steps - 3
                      or info.get("failure_injected"))):
            print(f"  step {i+1:3d}: r={r:+.4f}  "
                  f"(task={info['r_task']:.2f} pred={info['r_pred']:.4f} "
                  f"dmg={info['r_damage']:.4f})  "
                  f"dmg_total={info['damage_total']:.3f}  "
                  f"pred_risk={info['pred_risk_total']:.1f}  "
                  f"gate={info['gate_prob']:.2f}")

        if args.render != "none":
            import cv2
            rgb = obs.get("agentview_rgb")
            if rgb is None:
                rgb = env.render()
            # Banner above
            banner_h = 70
            H, W = rgb.shape[:2]
            banner = np.full((banner_h, W, 3), 30, dtype=np.uint8)
            fail = info.get("failure_injected")
            fail_txt = (f"failure: {fail['mode']} @ step {fail['fail_step']}"
                          if fail else "no failure")
            stage = ("PRE-FAILURE"
                     if (fail and info['step_idx'] < fail['fail_step'])
                     else ("POST-FAILURE" if fail else "NOMINAL"))
            stage_color = ((100, 200, 100) if stage == "NOMINAL"
                            else ((200, 200, 200) if stage == "PRE-FAILURE"
                                  else (240, 140, 60)))
            cv2.putText(banner, f"step {info['step_idx']:3d}/{args.steps}  "
                                  f"{fail_txt}",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(banner, stage, (6, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, stage_color,
                        2, cv2.LINE_AA)
            cv2.putText(banner,
                        f"reward {r:+.3f}  "
                        f"r_pred {info['r_pred']:+.3f}  "
                        f"r_dmg {info['r_damage']:+.3f}",
                        (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (200, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(banner,
                        f"pred_risk {info['pred_risk_total']:7.1f}  "
                        f"gate {info['gate_prob']:.2f}  "
                        f"dmg_total {info['damage_total']:.3f}",
                        (6, banner_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (170, 210, 170), 1, cv2.LINE_AA)
            composite = np.concatenate([banner, rgb], axis=0)
            if args.upscale > 1:
                composite = cv2.resize(
                    composite,
                    (composite.shape[1] * args.upscale,
                     composite.shape[0] * args.upscale),
                    interpolation=cv2.INTER_NEAREST)
            # cv2 wants BGR
            composite_bgr = composite[..., ::-1]
            if args.render == "window":
                cv2.imshow("SafeLiberoEnv", composite_bgr)
                if cv2.waitKey(max(1, 1000 // args.fps)) & 0xFF == ord("q"):
                    print("  user pressed q — exiting")
                    break
            elif writer is not None:
                writer.write(composite_bgr)

        if term or trunc:
            print(f"  episode end at step {i+1} (term={term}, trunc={trunc})")
            break

    print(f"\nTotal reward over {args.steps} steps: {total_reward:+.4f}")
    if writer is not None:
        writer.release()
        print(f"video written to {args.render_path}")
    if args.render == "window":
        import cv2
        cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    _smoke()
