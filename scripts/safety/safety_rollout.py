#!/usr/bin/env python3
"""Safety-rollout experiment runner: contact-predictor utility on LIBERO tasks.

Sweeps a matrix of (task, init_state, failure_mode, fail_progress, policy)
through LIBERO's ``OffScreenRenderEnv`` and records, per rollout:

    success_once          : env's binary success flag at episode end
    contact_mass_total    : sum of contact normal forces × dt over the episode,
                            attributed to non-robot bodies
    contact_mass_per_body : same, broken out by body name
    realized_risk         : Σᵢ value(eᵢ) · contact_mass(eᵢ)  with uniform 1.0
    pred_risk_pre_failure : predictor's marginal risk score at the failure step
    n_safety_triggers     : how many steps Mechanism A scaled the action
    failure_event         : (mode, joints, step) — the injected failure

Results stream to ``out/safety_rollouts/<task>/results.parquet`` (or csv
if pyarrow unavailable). Each row = one rollout.

Run (smoke, one task, one init, two progresses, three policies)::

    python -u -m scripts.safety.safety_rollout \\
        --ckpt $SCRATCH/failbench/runs/06172026/dualgated_state_rgb/best_ep08_val0.0648.pt \\
        --tasks KITCHEN_SCENE3_put_the_black_bowl_on_top_of_the_cabinet \\
        --n_inits 1 --progresses 0.4,0.7 --policies baseline,scaling,search \\
        --out_root $SCRATCH/failbench/safety_rollouts

Full sweep::

    --tasks KITCHEN_SCENE3_put_..., LIVING_ROOM_SCENE2_..., STUDY_SCENE2_...
    --n_inits 20 --progresses 0.1,0.25,0.4,0.55,0.7,0.85
    --policies baseline,scaling,search

Expected wall-clock: ~12 s/rollout for baseline+scaling, ~25 s for search;
full sweep ≈ 27 h on a 3g.40gb MIG.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Deferred imports — torch + LIBERO take seconds to load and we want --help
# to be fast.
def _lazy_imports():
    import torch
    from planner.experiments.config import FailureConfig, FailureMode
    from planner.policy.libero_env_failure import EnvFailureScheduler
    from planner.policy.safe_action import (
        ActionScalingPolicy,
        CandidateSearchPolicy,
        ObsWindow,
        PassthroughPolicy,
    )
    from planner.risk.inference import ContactPredictor, aabb_to_image_mask
    return dict(
        torch=torch,
        FailureConfig=FailureConfig,
        FailureMode=FailureMode,
        EnvFailureScheduler=EnvFailureScheduler,
        ActionScalingPolicy=ActionScalingPolicy,
        CandidateSearchPolicy=CandidateSearchPolicy,
        ObsWindow=ObsWindow,
        PassthroughPolicy=PassthroughPolicy,
        ContactPredictor=ContactPredictor,
        aabb_to_image_mask=aabb_to_image_mask,
    )


# ---------------------------------------------------------------------------
# Failure-mode plan (mode + joints per "category")
# ---------------------------------------------------------------------------

# Mirror the dataset's per-mode joint sampling choices.
_MODE_JOINTS = {
    "GRIPPER_OPEN":  [],
    "SLIPPERY_GRIP": [],
    "SINGLE_JOINT":  [4],
    "MULTI_JOINT":   [2, 4],
    "ALL_JOINTS":    [1, 2, 3, 4, 5, 6, 7],
}

# Policy hyperparameters. Mutated from CLI flags in main(); ``sweep_one_task``
# reads from here when building per-rollout policy instances.
_POLICY_HPARAMS = {
    "scaling_tau": 0.0,       # τ — risk threshold above which to slow down
    "scaling_alpha": 0.3,     # action multiplier when triggered
    "search_n": 4,
    "search_sigma": 0.02,
    "search_epsilon": 0.1,
}


@dataclass
class RolloutResult:
    task: str
    init_idx: int
    mode: str
    joints: tuple
    fail_progress: float
    policy: str
    success_once: bool
    n_steps: int
    contact_mass_total: float
    contact_mass_per_body: dict
    realized_risk: float
    # OopsieVerse-style mechanical damage (impulsive/sustained split + per-body
    # fragility weighting + yield threshold).  See planner/risk/damage.py.
    realized_damage_total: float
    realized_damage_per_body: dict
    final_health_per_body: dict
    damaged_objects: list           # bodies with damage > threshold
    damage_summary: dict             # per-body diagnostics (peak/timing)
    pred_risk_pre_failure: float
    pred_risk_t0: float
    pred_per_body_pre_failure: dict  # predictor mass per body via AABB masks
    n_safety_triggers: int
    rollout_seconds: float
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Contact accumulator — reads mujoco contact list each step
# ---------------------------------------------------------------------------

class ContactAccumulator:
    """Per-body accumulator of contact-force magnitude × control timestep.

    Reads ``data.ncon`` contacts each step, looks up the geom→body map
    once at reset, and bins forces by non-robot body name.

    Takes the unwrapped ``mujoco.MjModel`` / ``mujoco.MjData`` so the
    standalone ``mujoco`` C functions accept them.
    """

    def __init__(self, model, data, robot_geom_ids: set):
        self.model = model
        self.data = data
        self.robot_geom_ids = set(int(g) for g in robot_geom_ids)
        self._body_name_cache: dict[int, str] = {}
        self._geom_body: dict[int, int] = {}
        self._per_body: dict[str, float] = defaultdict(float)
        self._dt: float = float(getattr(model.opt, "timestep", 0.002))

    def reset(self) -> None:
        self._per_body.clear()
        self._geom_body.clear()
        self._body_name_cache.clear()

    def step(self) -> float:
        """Read sim contacts after a sim step; accumulate; return step total."""
        import mujoco
        model, data = self.model, self.data
        step_total = 0.0
        for i in range(int(data.ncon)):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            # Skip robot-vs-robot self-collisions
            robot1 = g1 in self.robot_geom_ids
            robot2 = g2 in self.robot_geom_ids
            if robot1 and robot2:
                continue
            # We attribute to the non-robot geom; if both are env, attribute
            # to the first (rare).
            target_geom = g2 if (robot1 and not robot2) else g1
            body_id = self._geom_body.get(target_geom)
            if body_id is None:
                body_id = int(model.geom_bodyid[target_geom])
                self._geom_body[target_geom] = body_id
            name = self._body_name_cache.get(body_id)
            if name is None:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                          body_id) or f"body_{body_id}"
                self._body_name_cache[body_id] = name
            # contact_force_world: use mj_contactForce to get the 6-vec
            cf = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, cf)
            fmag = float(np.linalg.norm(cf[:3])) * self._dt
            self._per_body[name] += fmag
            step_total += fmag
        return step_total

    @property
    def per_body(self) -> dict[str, float]:
        return dict(self._per_body)

    @property
    def total(self) -> float:
        return float(sum(self._per_body.values()))


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def _seed_sim_from_init_state(env, init_state: np.ndarray) -> dict:
    """Force the env's sim state to match the demo's recorded init_state.

    LIBERO's reset randomizes object placement.  When we then replay the
    demo's open-loop actions, the recorded actions miss the slightly-moved
    bowl/object.  Setting qpos/qvel to the saved [time|qpos|qvel] state
    pins the sim to exactly the configuration the demo was recorded under.
    """
    import mujoco
    from planner.policy.libero_env_failure import unwrap_sim
    model, data = unwrap_sim(env.sim)
    nq, nv = model.nq, model.nv
    flat = np.asarray(init_state, dtype=np.float64)
    if flat.shape[0] == 1 + nq + nv:
        off = 1
    elif flat.shape[0] == nq + nv:
        off = 0
    else:
        raise ValueError(f"init_state length {flat.shape[0]} != nq+nv ({nq+nv}) "
                          f"or 1+nq+nv ({1+nq+nv})")
    data.qpos[:nq] = flat[off:off + nq]
    data.qvel[:nv] = flat[off + nq:off + nq + nv]
    mujoco.mj_forward(model, data)
    # Re-observe so the returned obs reflects the seeded state
    return env._get_observations() if hasattr(env, "_get_observations") else None


def run_one_rollout(*, env, sim_handles, scheduler, policy, demo_actions,
                    robot_geom_ids, predictor, obs_window, masks,
                    pre_failure_step: int, init_state=None,
                    log_t0_risk: bool = True,
                    ) -> dict:
    """Run a single LIBERO rollout under one (env, scheduler, policy) combo.

    Returns dict suitable for RolloutResult.extras + per-step diagnostics.
    """
    import torch  # noqa: F401 — loaded by predictor lazily
    from planner.risk.damage import DamageAccumulator
    from planner.policy.libero_env_failure import unwrap_sim

    pred_risk_pre = float("nan")
    pred_risk_t0 = float("nan")
    pred_per_body_pre: dict = {}
    n_triggers = 0

    obs = scheduler.reset()
    if init_state is not None:
        # Pin object/joint poses to the demo's recorded init so the open-loop
        # action replay actually hits the bowl on every rollout (not just the
        # first, where the randomized init happens to align with the demo).
        seeded_obs = _seed_sim_from_init_state(env, init_state)
        if seeded_obs is not None:
            obs = seeded_obs
    # Robosuite rebuilds MjSim/MjModel/MjData on every env.reset(); accumulator
    # must be constructed against the POST-reset references, not pre-reset
    # ones (those become freed memory and silently return cached garbage).
    _model, _data = unwrap_sim(env.sim)
    accum = ContactAccumulator(_model, _data, robot_geom_ids)
    damage_accum = DamageAccumulator(_model, _data, robot_geom_ids)
    policy.reset()
    accum.reset()
    damage_accum.reset()

    success = False
    step_idx = 0
    n_actions = len(demo_actions)

    for i in range(n_actions):
        # Extract observation for the policy
        rgb_chw, state_vec = _extract_obs(obs, sim_handles)

        # Optional: log baseline t=0 risk
        if i == 0 and log_t0_risk and predictor is not None:
            from planner.policy.safe_action import query_risk
            poli_obs = obs_window
            poli_obs.push(rgb_chw, state_vec)
            rq = query_risk(predictor, poli_obs, masks=masks)
            pred_risk_t0 = rq.total_risk

        # Pre-failure risk snapshot (last step before failure fires)
        if i == max(0, pre_failure_step - 1) and predictor is not None:
            from planner.policy.safe_action import query_risk
            poli_obs = obs_window
            poli_obs.push(rgb_chw, state_vec)
            rq = query_risk(predictor, poli_obs, masks=masks)
            pred_risk_pre = rq.total_risk
            pred_per_body_pre = dict(rq.per_entity)

        demo_action = np.asarray(demo_actions[i], dtype=np.float32)
        action, info = policy.select(rgb_chw, state_vec, demo_action,
                                     masks=masks)
        if info.get("scaled"):
            n_triggers += 1

        obs, reward, done, env_info = scheduler.step(action)
        step_idx = i + 1

        accum.step()
        damage_accum.step()
        if env_info and bool(env_info.get("success", False)):
            success = True
        if done:
            break

    return dict(
        success=bool(success),
        n_steps=step_idx,
        contact_total=accum.total,
        contact_per_body=accum.per_body,
        damage_total=damage_accum.total_damage,
        damage_per_body=damage_accum.per_body_damage,
        final_health=damage_accum.per_body_health,
        damaged_set=sorted(damage_accum.damaged_set),
        damage_summary=damage_accum.damage_summary(),
        pred_risk_pre_failure=pred_risk_pre,
        pred_risk_t0=pred_risk_t0,
        pred_per_body_pre=pred_per_body_pre,
        n_safety_triggers=n_triggers,
    )


def _extract_obs(obs: dict, sim_handles) -> tuple:
    """Pull (rgb_chw, state_vec) from a LIBERO env observation dict."""
    # LIBERO key is 'agentview_image' (or similar). robosuite key for
    # joint pos: robot0_joint_pos; vel: robot0_joint_vel; ee: robot0_eef_pos;
    # gripper: robot0_gripper_qpos.
    rgb = obs.get("agentview_image")
    if rgb is None:
        for k in obs:
            if "agentview" in k and "image" in k:
                rgb = obs[k]; break
    if rgb is None:
        raise RuntimeError(f"no agentview image in obs keys: {sorted(obs)}")
    # Robosuite OpenGL framebuffer convention has Y up; the v2 dataset (which
    # the predictor was trained on) stored RGB with Y down. np.flipud aligns
    # the env image to the predictor's expected orientation.
    rgb_chw = np.transpose(
        np.flipud(np.asarray(rgb)).copy(), (2, 0, 1)).astype(np.uint8)

    qpos = np.asarray(obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float32)[:7]
    qvel = np.asarray(obs.get("robot0_joint_vel", np.zeros(7)), dtype=np.float32)[:7]
    ee   = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)[:3]
    grip = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(1)), dtype=np.float32).ravel()[:1]
    state_vec = np.concatenate([qpos, qvel, ee, grip]).astype(np.float32)
    return rgb_chw, state_vec


# ---------------------------------------------------------------------------
# Per-task setup + sweep
# ---------------------------------------------------------------------------

def build_entity_masks(env, image_hw, aabb_padding: float = 0.02) -> dict:
    """Project every non-robot body's AABB into the agentview camera.

    Uses the env's MJCF body inertials + geom bounding boxes to compute a
    rest-pose AABB per body, then projects via :func:`aabb_to_image_mask`.
    Returns a dict ``{body_name: (H, W) bool}``.
    """
    import mujoco
    from planner.experiments.libero.naming import resolve_model_handles
    from planner.policy.libero_env_failure import unwrap_sim

    # Reset first so per-task object spawn positions are realized — without
    # this geom_xpos is at the default model state (often origin) and most
    # foreground objects (bowl, ramekin, plate, cookies) project off-camera.
    env.reset()
    model, data = unwrap_sim(env.sim)
    handles = resolve_model_handles(model)

    # Per-body AABB from geom_aabb (works for mesh geoms — geom_size is
    # degenerate for meshes and would give zero-area boxes for bowls/plates).
    body_aabbs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                  body_id) or ""
        if not name or name in ("world",) or "robot0" in name:
            continue
        geom_indices = np.where(model.geom_bodyid == body_id)[0]
        if len(geom_indices) == 0:
            continue
        if any(int(g) in handles.robot_geom_ids for g in geom_indices):
            continue
        mins, maxs = [], []
        for g in geom_indices:
            # geom_aabb is (center_x,y,z, half_x,y,z) in the geom's LOCAL
            # frame.  Rotate the half-extents by the geom orientation then
            # add to world position to get a world-aligned AABB.
            aabb_local = np.asarray(model.geom_aabb[g], dtype=np.float64)
            half_local = aabb_local[3:6]
            center_local = aabb_local[0:3]
            # Geom world pose
            geom_xpos = np.asarray(data.geom_xpos[g], dtype=np.float64)
            geom_xmat = np.asarray(data.geom_xmat[g],
                                     dtype=np.float64).reshape(3, 3)
            center_world = geom_xpos + geom_xmat @ center_local
            half_world = np.abs(geom_xmat) @ half_local
            # Fallback if geom_aabb is all-zero (rare): use geom_size
            if not np.any(half_local) and not np.any(center_local):
                half_world = np.asarray(model.geom_size[g], dtype=np.float64)
                center_world = geom_xpos
            mins.append(center_world - half_world)
            maxs.append(center_world + half_world)
        if not mins:
            continue
        mn = np.min(np.stack(mins), axis=0) - aabb_padding
        mx = np.max(np.stack(maxs), axis=0) + aabb_padding
        body_aabbs[name] = (mn, mx)

    # Pull agentview cam pose + intrinsics
    cam_name = handles.agentview_cam
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    fovy = float(model.cam_fovy[cam_id])

    from planner.risk.inference import aabb_to_image_mask
    masks: dict[str, np.ndarray] = {}
    for name, (mn, mx) in body_aabbs.items():
        m = aabb_to_image_mask(mn, mx, cam_pos, cam_mat, fovy, image_hw)
        if m.any():
            masks[name] = m
    return masks


def load_demo_actions(bddl_file_name: str, init_idx: int, demo_hdf5: str):
    """Load the demo's recorded action sequence for replay."""
    import h5py
    with h5py.File(demo_hdf5, "r") as f:
        demo_key = f"demo_{init_idx}"
        if demo_key not in f["data"]:
            raise KeyError(f"{demo_key} not in {demo_hdf5}")
        actions = np.asarray(f["data"][demo_key]["actions"], dtype=np.float32)
        # Init state for env reset
        init_state = None
        if "states" in f["data"][demo_key]:
            init_state = np.asarray(f["data"][demo_key]["states"][0],
                                    dtype=np.float32)
    return actions, init_state


def sweep_one_task(*, env, deps, task_name, demo_hdf5, n_inits, modes,
                   progresses, policies, predictor, image_hw,
                   results: list, out_path: Path):
    """Run the full (init × mode × progress × policy) sub-sweep for one task."""
    from planner.policy.safe_action import ObsWindow

    masks = build_entity_masks(env, image_hw)
    print(f"  built {len(masks)} entity masks for {task_name}", flush=True)

    # Robot geom ids for the accumulator (unwrap robosuite's wrapper first).
    # Capture a one-time clean snapshot of all model params that the failure
    # injector mutates.  LIBERO's env.reset() resets sim state but does NOT
    # restore actuator/joint metadata; without this snapshot, the first
    # failure-injecting rollout permanently breaks the model and every
    # subsequent baseline runs with zero contacts.
    from planner.experiments.libero.naming import resolve_model_handles
    from planner.policy.libero_env_failure import unwrap_sim
    model, _ = unwrap_sim(env.sim)
    handles = resolve_model_handles(model)
    clean_model_state = dict(
        actuator_gainprm=model.actuator_gainprm.copy(),
        actuator_biastype=model.actuator_biastype.copy(),
        actuator_gaintype=model.actuator_gaintype.copy(),
        jnt_stiffness=model.jnt_stiffness.copy(),
        dof_damping=model.dof_damping.copy(),
        jnt_range=model.jnt_range.copy(),
        dof_frictionloss=model.dof_frictionloss.copy(),
    )

    sim_handles = handles

    def _restore_clean_model():
        m, _ = unwrap_sim(env.sim)
        for k, v in clean_model_state.items():
            getattr(m, k)[:] = v

    for init_idx in range(n_inits):
        try:
            demo_actions, init_state = load_demo_actions(
                bddl_file_name=None, init_idx=init_idx, demo_hdf5=demo_hdf5)
        except (KeyError, OSError) as e:
            print(f"  skip init {init_idx}: {e}")
            continue

        for mode in modes:
            joints = _MODE_JOINTS.get(mode, [])
            failure = deps["FailureConfig"](
                mode=deps["FailureMode"][mode],
                probability=1.0,
                joint_names=[f"joint{j}" for j in joints] if joints else None,
            )

            for progress in progresses:
                fail_step = max(1, int(progress * len(demo_actions)))

                for policy_name in policies:
                    t0 = time.time()
                    obs_window = ObsWindow()
                    policy = _build_policy(
                        policy_name, predictor, env, obs_window,
                        scaling_tau=_POLICY_HPARAMS["scaling_tau"],
                        scaling_alpha=_POLICY_HPARAMS["scaling_alpha"],
                        search_n=_POLICY_HPARAMS["search_n"],
                        search_sigma=_POLICY_HPARAMS["search_sigma"],
                        search_epsilon=_POLICY_HPARAMS["search_epsilon"],
                    )
                    # Restore the model to its clean (pre-any-failure) state
                    # before every rollout.  Without this, mutations from
                    # earlier failure injections leak across rollouts.
                    _restore_clean_model()
                    scheduler = deps["EnvFailureScheduler"](
                        env, failure, fail_step)
                    # Accumulators are built AFTER the rollout's reset() —
                    # see run_one_rollout. We pass a model_getter so the
                    # accumulator re-binds to the fresh MjModel/MjData
                    # that robosuite rebuilds on every env.reset().
                    accum = None
                    damage_accum = None

                    try:
                        out = run_one_rollout(
                            env=env, sim_handles=sim_handles,
                            scheduler=scheduler, policy=policy,
                            demo_actions=demo_actions,
                            robot_geom_ids=handles.robot_geom_ids,
                            predictor=predictor, obs_window=obs_window,
                            masks=masks, pre_failure_step=fail_step,
                            init_state=init_state,
                        )
                    except Exception as e:
                        print(f"  ERROR {task_name}/init{init_idx}/{mode}"
                              f"/{progress}/{policy_name}: {e}")
                        continue

                    risk_total = sum(out["contact_per_body"].values())
                    rr = RolloutResult(
                        task=task_name,
                        init_idx=init_idx,
                        mode=mode,
                        joints=tuple(joints),
                        fail_progress=float(progress),
                        policy=policy_name,
                        success_once=out["success"],
                        n_steps=out["n_steps"],
                        contact_mass_total=out["contact_total"],
                        contact_mass_per_body=out["contact_per_body"],
                        realized_risk=risk_total,
                        realized_damage_total=out["damage_total"],
                        realized_damage_per_body=out["damage_per_body"],
                        final_health_per_body=out["final_health"],
                        damaged_objects=list(out["damaged_set"]),
                        damage_summary=out["damage_summary"],
                        pred_risk_pre_failure=out["pred_risk_pre_failure"],
                        pred_risk_t0=out["pred_risk_t0"],
                        pred_per_body_pre_failure=out["pred_per_body_pre"],
                        n_safety_triggers=out["n_safety_triggers"],
                        rollout_seconds=time.time() - t0,
                    )
                    results.append(rr)
                    print(f"  init{init_idx} {mode:13s} p={progress:.2f} "
                          f"{policy_name:9s}  succ={int(out['success'])}  "
                          f"risk={risk_total:8.2f}  "
                          f"dmg={out['damage_total']:7.2f}  "
                          f"n={out['n_steps']:3d}  "
                          f"trig={out['n_safety_triggers']:3d}  "
                          f"({rr.rollout_seconds:.1f}s)", flush=True)

    _write_results(results, out_path)


def _build_policy(name, predictor, env, obs_window, *, scaling_tau=0.0,
                  scaling_alpha=0.3, search_n=4, search_sigma=0.02,
                  search_epsilon=0.1):
    """Factory for the three policies.

    Policy-specific tunables are exposed so the experiment runner can
    pass them from CLI flags. Defaults are intentionally conservative —
    ``scaling_tau=0.0`` fires every step (always slow down), which is the
    weakest but most reliable baseline; in production sweeps the user
    should set ``scaling_tau`` to a per-scene calibrated threshold
    (e.g. the 70th percentile of the predictor's t=0 risk on this task).
    """
    from planner.policy.safe_action import (
        ActionScalingPolicy, CandidateSearchPolicy, PassthroughPolicy)
    if name == "baseline":
        return PassthroughPolicy(predictor=predictor, obs_window=obs_window)
    if name == "scaling":
        return ActionScalingPolicy(predictor=predictor,
                                    alpha=scaling_alpha, tau=scaling_tau,
                                    obs_window=obs_window)
    if name == "search":
        return CandidateSearchPolicy(
            predictor=predictor, env=env, env_fork_fn=_fork_env,
            n_candidates=search_n, sigma=search_sigma,
            epsilon=search_epsilon,
            obs_window=obs_window)
    raise ValueError(f"unknown policy {name!r}")


def _fork_env(env):
    """Best-effort env fork for CandidateSearchPolicy. Robosuite envs don't
    natively support clone — we use deepcopy on the MjSim state and an
    in-place restore. The trade-off: state restore after the candidate is
    cheaper than building a fresh env, but corrupts the source env if the
    candidate step fails. Use with caution.
    """
    # Simplest workable fork: snapshot env state, step in-place, restore.
    # This means env_fork_fn returns the *same* env object but it's safe to
    # call .step() on it transiently — caller discards the result and we
    # rely on Mechanism B's caller never advancing the real env on a forked
    # step. The CandidateSearchPolicy in safe_action.py uses fork.step()
    # only for prediction; the real action goes through scheduler.step()
    # later. We snapshot before each fork and restore.
    return _FakeForkProxy(env)


class _FakeForkProxy:
    """Snapshot + restore wrapper used as a candidate-search env fork."""
    def __init__(self, env):
        self._env = env
        self._snapshot = _snapshot_sim(env.sim)
        self._last_obs = None

    def step(self, action):
        obs, r, done, info = self._env.step(action)
        self._last_obs = obs
        # Immediately restore so subsequent forks start from same point
        _restore_sim(self._env.sim, self._snapshot)
        return obs, r, done, info

    def __getattr__(self, item):
        # Forward attribute access — needed by the obs_extractor fallback
        return getattr(self._env, item)


def _snapshot_sim(sim):
    from planner.policy.libero_env_failure import unwrap_sim
    _, data = unwrap_sim(sim)
    return dict(
        qpos=np.asarray(data.qpos).copy(),
        qvel=np.asarray(data.qvel).copy(),
        ctrl=np.asarray(data.ctrl).copy(),
        time=float(data.time),
    )


def _restore_sim(sim, snap):
    import mujoco
    from planner.policy.libero_env_failure import unwrap_sim
    model, data = unwrap_sim(sim)
    data.qpos[:] = snap["qpos"]
    data.qvel[:] = snap["qvel"]
    data.ctrl[:] = snap["ctrl"]
    data.time = snap["time"]
    mujoco.mj_forward(model, data)


def _write_results(results: list, path: Path):
    """Persist results to parquet if pyarrow is available, else csv."""
    rows = []
    for r in results:
        d = asdict(r)
        d["joints"] = list(d["joints"])
        for k in ("contact_mass_per_body",
                  "realized_damage_per_body",
                  "final_health_per_body",
                  "damage_summary",
                  "pred_per_body_pre_failure",
                  "damaged_objects",
                  "extras"):
            if k in d:
                d[k] = json.dumps(d[k])
        rows.append(d)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            try:
                import pyarrow  # noqa: F401
                df.to_parquet(path)
            except ImportError:
                path = path.with_suffix(".csv")
                df.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)
    except ImportError:
        import csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".csv"), "w", newline="") as f:
            if not rows:
                return
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="Path to GatekeeperCoordFiLMUNet .pt checkpoint")
    ap.add_argument("--libero_root", type=Path,
                    default=Path("external/LIBERO"),
                    help="Path to LIBERO repo root (contains libero/ + datasets/)")
    ap.add_argument("--demo_root", type=Path,
                    default=Path("datasets/libero/raw"),
                    help="Path containing libero_*/<task>.hdf5 raw demo files")
    ap.add_argument("--tasks", required=True,
                    help="Comma-separated list of task names. Each name "
                          "should match a BDDL file under the LIBERO repo "
                          "and an HDF5 file under --demo_root.")
    ap.add_argument("--n_inits", type=int, default=20,
                    help="How many demo init states per task (default 20)")
    ap.add_argument("--modes", default="GRIPPER_OPEN,SLIPPERY_GRIP,SINGLE_JOINT,MULTI_JOINT,ALL_JOINTS")
    ap.add_argument("--progresses", default="0.1,0.25,0.4,0.55,0.7,0.85")
    ap.add_argument("--policies", default="baseline,scaling,search",
                    help="Comma-separated subset of {baseline,scaling,search}")
    ap.add_argument("--out_root", type=Path,
                    default=Path("out/safety_rollouts"),
                    help="Where per-task parquets land")
    ap.add_argument("--image_h", type=int, default=240)
    ap.add_argument("--image_w", type=int, default=320)
    ap.add_argument("--device", default=None,
                    help="cuda / cpu / explicit MIG UUID (set via "
                          "CUDA_VISIBLE_DEVICES before running)")
    ap.add_argument("--scaling_tau", type=float, default=0.0,
                    help="ActionScalingPolicy risk threshold. Default 0 → "
                          "always slow down during failure. Set to a per-scene "
                          "calibrated value (e.g. p70 of baseline pred_risk_t0) "
                          "for adaptive slowdown.")
    ap.add_argument("--scaling_alpha", type=float, default=0.3,
                    help="ActionScalingPolicy action multiplier when triggered.")
    ap.add_argument("--search_n", type=int, default=4,
                    help="CandidateSearchPolicy n_candidates (perturbations).")
    ap.add_argument("--search_sigma", type=float, default=0.02,
                    help="CandidateSearchPolicy perturbation stddev. Increase "
                          "if candidates produce indistinguishable predicted "
                          "risks (sweep symptom: search yields uniform low risk).")
    ap.add_argument("--search_epsilon", type=float, default=0.1)
    args = ap.parse_args()
    _POLICY_HPARAMS.update({
        "scaling_tau": args.scaling_tau,
        "scaling_alpha": args.scaling_alpha,
        "search_n": args.search_n,
        "search_sigma": args.search_sigma,
        "search_epsilon": args.search_epsilon,
    })

    deps = _lazy_imports()
    print(f"loaded torch={deps['torch'].__version__}  cuda="
          f"{deps['torch'].cuda.is_available()}")

    # Predictor
    predictor = deps["ContactPredictor"].from_checkpoint(
        args.ckpt, device=args.device)
    print(f"loaded predictor: {predictor.meta.arch}  "
          f"ep={predictor.meta.epoch}  val_heat={predictor.meta.val_heat}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    progresses = [float(p) for p in args.progresses.split(",") if p.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    image_hw = (args.image_h, args.image_w)

    for task in tasks:
        bddl_file = _resolve_bddl(args.libero_root, task)
        demo_hdf5 = _resolve_demo_hdf5(args.demo_root, task)
        print(f"\n=== {task} ===")
        print(f"  bddl: {bddl_file}")
        print(f"  demo: {demo_hdf5}")

        # Build env
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file),
            camera_heights=args.image_h,
            camera_widths=args.image_w,
        )

        results: list = []
        out_path = args.out_root / task / "results.parquet"

        try:
            sweep_one_task(
                env=env, deps=deps, task_name=task,
                demo_hdf5=str(demo_hdf5),
                n_inits=args.n_inits, modes=modes,
                progresses=progresses, policies=policies,
                predictor=predictor, image_hw=image_hw,
                results=results, out_path=out_path,
            )
        finally:
            try:
                env.close()
            except Exception:
                pass

    print("\nDONE.")
    return 0


def _resolve_bddl(libero_root: Path, task: str) -> Path:
    """Find <libero_root>/libero/libero/bddl_files/.../{task}.bddl."""
    candidates = list(libero_root.glob(f"**/bddl_files/**/{task}.bddl"))
    if not candidates:
        raise FileNotFoundError(
            f"no BDDL file matching {task}.bddl under {libero_root}")
    return candidates[0]


def _resolve_demo_hdf5(demo_root: Path, task: str) -> Path:
    """Find a demo HDF5 for ``task``. LIBERO raw demos typically use a
    ``<task>_demo.hdf5`` suffix; the canonical v2 corpus uses ``<task>.h5``.
    Try both."""
    for pattern in (f"**/{task}.hdf5", f"**/{task}_demo.hdf5",
                     f"**/{task}.h5"):
        cands = list(demo_root.glob(pattern))
        if cands:
            return cands[0]
    raise FileNotFoundError(
        f"no demo HDF5 matching {task}{{,_demo}}.hdf5 under {demo_root}")


if __name__ == "__main__":
    raise SystemExit(main())
