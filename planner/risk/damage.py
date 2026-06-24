"""OopsieVerse-style mechanical damage accumulator for FailBench rollouts.

Ports the `d_mech` formula from OopsieVerse (Balaji et al., paper §III.B):

    f_parallel,k = (f_k . a_hat) a_hat          # impulsive component
    f_perp,k     = f_k - f_parallel,k           # sustained / constrained

    F_parallel(t) = sum_k ||f_parallel,k||
    F_perp(t)     = sum_k ||f_perp,k||

    eps_mech(t)   = alpha * F_parallel + beta * F_perp
    d_mech(t)     = Lambda * max(eps_mech - E_mech, 0)

    h(t) = h(t-1) - d_mech(t)                   # per link
    h_object = min over links

Why we want this (vs raw sum ||F|| dt):
- Separating impulsive from sustained loading lets us suppress grasp-hold
  forces (sustained, brittle objects are insensitive to them) while keeping
  drop-impact forces (impulsive, brittle objects damaged easily).
- Per-link / per-body health gives unambiguous "which object got damaged"
  attribution that our 2D footprint masking can't do when AABBs overlap.

The MuJoCo plumbing is the same as our existing ContactAccumulator:
- ``data.ncon`` contacts
- ``mj_contactForce`` rotates a force into the contact frame
- ``data.contact[i].frame`` is the contact frame in world (3x3, rows are
  world-frame basis vectors), so ``frame.T @ f_contact`` gives world force
- ``data.cacc[body_id]`` is the body com acceleration in world (6-vec:
  first 3 angular, last 3 linear)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import numpy as np


# ---------------------------------------------------------------------------
# Per-body damage parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DamageParams:
    """Per-body mechanical-damage coefficients.

    Mirrors OopsieVerse ``damagesim/robosuite/params/damage_params.py``:
    ``impact_sensitivity`` (alpha), ``qs_sensitivity`` (beta), ``threshold``
    (E_mech), and a rate constant (Lambda) we choose so meaningful contacts
    produce visible health change at our sim timestep.

    Fragility intuition:
    - Brittle / impact-sensitive: alpha large, beta small, low threshold.
      Examples: porcelain (egg, ramekin, wine glass).
    - Compressible / sustained-sensitive: alpha small, beta large.
      Examples: paper cup, cake.
    - Robust structural: very high threshold, both sensitivities tiny.
      Examples: cabinet, table, world.
    """

    alpha: float = 1.0       # impact sensitivity (F_parallel weight)
    beta: float = 0.1        # sustained sensitivity (F_perp weight)
    threshold: float = 15.0  # yield threshold E_mech (N-equivalent)
    rate: float = 1.0        # damage rate Lambda; tuned so visible damage
                              # accrues over ~10s of sim steps for events
                              # well above threshold
    h_max: float = 100.0


# Pattern-matched LIBERO body fragility table.  Keys are substrings matched
# case-insensitively against body names (longest match wins, then defaults).
# Values authored by analogy to OopsieVerse RoboCasa params; not measured.
#
# Tier intuition:
#   ramekin / porcelain    -> very fragile (egg-like)
#   bowl / mug             -> fragile (wine-glass-like, sturdier than egg)
#   plate                  -> moderate (drops badly but tolerates set-down)
#   cookies / soft mesh    -> low-impact / mid-sustained (cake-like)
#   cabinet / drawer / box -> robust (large structural)
#   table / world          -> infinite (essentially indestructible)
# Rate constants below are scaled to LIBERO's sim timestep (~0.002 s) and our
# observed force magnitudes (per-body summed F_par/F_perp easily reaches
# 500–2500 N for held objects across 5–15 simultaneous contacts).  With these
# values, a clean drop-impact accrues ~10–80 damage to a porcelain object
# over its post-failure window, leaving headroom under h_max=100 instead of
# saturating immediately.  Tune per scene as needed.
# Fragile-object `rate` bumped ~10x (2026-06-23) so a single impact registers a
# meaningful health drop — the original rates were tuned for sustained loads over
# long rollouts and crushed brief impacts to ~0 even at 100+ N.
LIBERO_DAMAGE_PARAMS: Dict[str, DamageParams] = {
    "ramekin":   DamageParams(alpha=8.0, beta=0.5, threshold=10.0, rate=2.0e-4),
    "porcelain": DamageParams(alpha=8.0, beta=0.5, threshold=10.0, rate=2.0e-4),
    "bowl":      DamageParams(alpha=5.0, beta=0.5, threshold=12.0, rate=1.5e-4),
    "mug":       DamageParams(alpha=5.0, beta=0.5, threshold=12.0, rate=1.5e-4),
    "cup":       DamageParams(alpha=5.0, beta=0.5, threshold=12.0, rate=1.5e-4),
    "plate":     DamageParams(alpha=4.0, beta=0.4, threshold=15.0, rate=1.2e-4),
    "cookies":   DamageParams(alpha=2.0, beta=1.0, threshold=20.0, rate=1.0e-4),
    "cake":      DamageParams(alpha=2.0, beta=1.5, threshold=20.0, rate=1.0e-4),
    "cabinet":   DamageParams(alpha=0.05, beta=0.05, threshold=200.0, rate=1.0e-6),
    "drawer":    DamageParams(alpha=0.05, beta=0.05, threshold=200.0, rate=1.0e-6),
    "shelf":     DamageParams(alpha=0.05, beta=0.05, threshold=200.0, rate=1.0e-6),
    "table":     DamageParams(alpha=0.01, beta=0.01, threshold=500.0, rate=5.0e-7),
    "stool":     DamageParams(alpha=0.01, beta=0.01, threshold=500.0, rate=5.0e-7),
    "world":     DamageParams(alpha=0.0, beta=0.0, threshold=1e9, rate=0.0),
    # Robot self-contact: usually filtered out, but if any leak through,
    # treat the robot as moderately robust so it doesn't dominate.
    "robot0":    DamageParams(alpha=0.1, beta=0.1, threshold=100.0, rate=2.0e-6),
    "gripper":   DamageParams(alpha=0.1, beta=0.1, threshold=100.0, rate=2.0e-6),
}

# Used when no key matches.  Slightly fragile so unknown objects still
# register damage — better than silently ignoring them.
DEFAULT_DAMAGE_PARAMS = DamageParams(
    alpha=2.0, beta=0.3, threshold=20.0, rate=1.0e-5)

# Impact-energy damage: a body that loses kinetic energy in a collision (a drop
# landing, a hard hit) takes damage ~ coeff * alpha * dKE. This captures the
# impulsive damage of a *drop*, which the time-integrated sustained-force term
# (rate * F * dt) barely registers because an impact lasts only a few steps.
# Tuned so a fragile object dropped ~0.1-0.4 m takes a clearly visible health hit
# without distorting the sustained-crush channel.
IMPACT_DAMAGE_COEFF = 22.0


def lookup_damage_params(body_name: str,
                          registry: Optional[Dict[str, DamageParams]] = None,
                          default: Optional[DamageParams] = None
                          ) -> DamageParams:
    """Resolve a body name to DamageParams via case-insensitive substring."""
    reg = LIBERO_DAMAGE_PARAMS if registry is None else registry
    dflt = DEFAULT_DAMAGE_PARAMS if default is None else default
    if not body_name:
        return dflt
    name = body_name.lower()
    best_key: Optional[str] = None
    for key in reg:
        if key in name:
            if best_key is None or len(key) > len(best_key):
                best_key = key
    return reg[best_key] if best_key else dflt


# ---------------------------------------------------------------------------
# Damage accumulator
# ---------------------------------------------------------------------------

@dataclass
class BodyDamageState:
    """Per-body health bookkeeping during a rollout."""
    health: float = 100.0
    total_damage: float = 0.0
    n_damaging_steps: int = 0   # steps where d_mech > 0
    peak_step_damage: float = 0.0
    first_damage_step: Optional[int] = None


class DamageAccumulator:
    """OopsieVerse-style mechanical damage per body, integrated over rollout.

    Use alongside the existing ContactAccumulator so we can compare the
    raw sum-||F||-dt metric to the d_mech-weighted health drop on the same
    trial.

    Robot-vs-robot contacts are skipped.  Robot-vs-env contacts are
    attributed to the env body.  Env-vs-env contacts are attributed to
    the first geom's body (rare; happens with overlapping shelves etc.).
    """

    def __init__(self, model, data, robot_geom_ids: Set[int],
                 registry: Optional[Dict[str, DamageParams]] = None,
                 default_params: Optional[DamageParams] = None,
                 damage_threshold_for_set: float = 5.0,
                 held_body_ids: Optional[Set[int]] = None):
        self.model = model
        self.data = data
        self.robot_geom_ids = set(int(g) for g in robot_geom_ids)
        self.registry = registry
        self.default_params = default_params or DEFAULT_DAMAGE_PARAMS
        self.damage_threshold_for_set = damage_threshold_for_set
        # bodies the robot is simply holding: their robot-contacts (the grasp
        # grip force) must not count as damage — only impacts with the
        # environment (non-robot contacts) should.
        self.held_body_ids = set(int(b) for b in held_body_ids) \
            if held_body_ids else set()

        self._geom_body: Dict[int, int] = {}
        self._body_name: Dict[int, str] = {}
        self._params_cache: Dict[str, DamageParams] = {}
        self._state: Dict[str, BodyDamageState] = defaultdict(BodyDamageState)
        self._step_idx: int = 0
        self._prev_speed: Dict[int, float] = {}   # body_id -> last-step speed
        self._movable = None                       # lazily-computed body ids

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._geom_body.clear()
        self._body_name.clear()
        self._params_cache.clear()
        self._state.clear()
        self._step_idx = 0
        self._prev_speed.clear()

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def step(self) -> float:
        """Read contacts, accumulate d_mech per body, return total step damage."""
        import mujoco
        model, data = self.model, self.data

        # Group contacts by target body so we sum F_parallel/F_perp before
        # passing through (max, 0) thresholding (otherwise we'd threshold
        # each contact independently and lose the aggregate-load notion).
        per_body_F: Dict[str, tuple] = defaultdict(lambda: [0.0, 0.0])
        contacted: set = set()   # non-robot body ids touching something this step

        for i in range(int(data.ncon)):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            r1 = g1 in self.robot_geom_ids
            r2 = g2 in self.robot_geom_ids
            if r1 and r2:
                continue

            # World-frame force on this contact (Newton's 3rd law: equal and
            # opposite on the two bodies, same magnitude).
            cf = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, cf)
            f_contact = cf[:3]
            frame = np.asarray(con.frame, dtype=np.float64).reshape(3, 3)
            f_world = frame.T @ f_contact

            # Accumulate damage for EACH non-robot body in the contact (a bowl
            # hitting a plate damages both), decomposed against that body's own
            # acceleration. Robot bodies don't take damage.
            for gid, is_robot, other_robot in ((g1, r1, r2), (g2, r2, r1)):
                if is_robot:
                    continue
                body_id = self._geom_body.get(gid)
                if body_id is None:
                    body_id = int(model.geom_bodyid[gid])
                    self._geom_body[gid] = body_id
                name = self._body_name.get(body_id)
                if name is None:
                    name = (mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                            or f"body_{body_id}")
                    self._body_name[body_id] = name
                contacted.add(body_id)

                # A held body's grip force must not count as damage: skip the
                # robot's (gripper/arm) contact on the object it is holding.
                # Damage to a held object comes only from its impacts with the
                # environment (table, other objects) — those are env contacts,
                # not filtered here. (A static grip otherwise leaks force into
                # d_mech; decomposing it into F_par/F_perp doesn't separate it
                # cleanly because a gripped object still has noisy acceleration.)
                if other_robot and body_id in self.held_body_ids:
                    continue

                # Decompose against this body's linear acceleration (world).
                a_world = np.asarray(data.cacc[body_id][3:6], dtype=np.float64)
                a_mag = float(np.linalg.norm(a_world))
                if a_mag < 1e-6:
                    F_par = 0.0
                    F_perp = float(np.linalg.norm(f_world))
                else:
                    a_hat = a_world / a_mag
                    f_par_vec = float(f_world @ a_hat) * a_hat
                    f_perp_vec = f_world - f_par_vec
                    F_par = float(np.linalg.norm(f_par_vec))
                    F_perp = float(np.linalg.norm(f_perp_vec))

                tot = per_body_F[name]
                tot[0] += F_par
                tot[1] += F_perp

        def _params_for(name):
            p = self._params_cache.get(name)
            if p is None:
                p = lookup_damage_params(name, self.registry, self.default_params)
                self._params_cache[name] = p
            return p

        # 1. sustained-force damage (crushing): rate * max(eps - threshold, 0)
        d_by_name: Dict[str, float] = {}
        for name, (F_par, F_perp) in per_body_F.items():
            params = _params_for(name)
            eps = params.alpha * F_par + params.beta * F_perp
            d = params.rate * max(eps - params.threshold, 0.0)
            if d > 0.0:
                d_by_name[name] = d_by_name.get(name, 0.0) + d

        # 2. impact-energy damage: a moving body that loses kinetic energy in a
        # collision this step (a drop landing / a hit) takes coeff * alpha * dKE.
        if self._movable is None:
            robot_bodies = {int(model.geom_bodyid[g]) for g in self.robot_geom_ids}
            self._movable = [b for b in range(model.nbody)
                             if model.body_mass[b] > 1e-6
                             and b not in robot_bodies
                             and int(model.body_dofnum[b]) > 0]
        for bid in self._movable:
            v = float(np.linalg.norm(data.cvel[bid][3:6]))
            vprev = self._prev_speed.get(bid, v)
            self._prev_speed[bid] = v
            if bid in contacted and vprev > v:
                ke_loss = 0.5 * float(model.body_mass[bid]) * (vprev * vprev - v * v)
                if ke_loss <= 0.0:
                    continue
                name = self._body_name.get(bid)
                if name is None:
                    name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
                            or f"body_{bid}")
                    self._body_name[bid] = name
                params = _params_for(name)
                d_imp = IMPACT_DAMAGE_COEFF * params.alpha * ke_loss
                if d_imp > 0.0:
                    d_by_name[name] = d_by_name.get(name, 0.0) + d_imp

        # 3. apply accumulated damage to each body's health
        step_total_damage = 0.0
        for name, d_step in d_by_name.items():
            if d_step <= 0.0:
                continue
            params = self._params_cache[name]
            s = self._state[name]
            s.total_damage += d_step
            s.health = max(0.0, params.h_max - s.total_damage)
            s.n_damaging_steps += 1
            if d_step > s.peak_step_damage:
                s.peak_step_damage = d_step
            if s.first_damage_step is None:
                s.first_damage_step = self._step_idx
            step_total_damage += d_step

        self._step_idx += 1
        return step_total_damage

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def per_body_health(self) -> Dict[str, float]:
        """Final health per body.  Bodies never contacted are not present."""
        return {b: s.health for b, s in self._state.items()}

    @property
    def per_body_damage(self) -> Dict[str, float]:
        """Accumulated damage per body (= h_max - health)."""
        return {b: s.total_damage for b, s in self._state.items()}

    @property
    def total_damage(self) -> float:
        return float(sum(s.total_damage for s in self._state.values()))

    @property
    def damaged_set(self) -> Set[str]:
        """Bodies whose accumulated damage exceeds the configured threshold."""
        return {
            b for b, s in self._state.items()
            if s.total_damage > self.damage_threshold_for_set}

    def damage_summary(self) -> Dict[str, dict]:
        """Per-body diagnostics for the CSV / analysis path."""
        return {
            b: {
                "health": s.health,
                "damage": s.total_damage,
                "n_damaging_steps": s.n_damaging_steps,
                "peak_step_damage": s.peak_step_damage,
                "first_damage_step": s.first_damage_step,
            }
            for b, s in self._state.items()
        }
