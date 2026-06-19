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
LIBERO_DAMAGE_PARAMS: Dict[str, DamageParams] = {
    "ramekin":   DamageParams(alpha=8.0, beta=0.5, threshold=10.0, rate=2.0e-5),
    "porcelain": DamageParams(alpha=8.0, beta=0.5, threshold=10.0, rate=2.0e-5),
    "bowl":      DamageParams(alpha=5.0, beta=0.5, threshold=12.0, rate=1.5e-5),
    "mug":       DamageParams(alpha=5.0, beta=0.5, threshold=12.0, rate=1.5e-5),
    "cup":       DamageParams(alpha=5.0, beta=0.5, threshold=12.0, rate=1.5e-5),
    "plate":     DamageParams(alpha=4.0, beta=0.4, threshold=15.0, rate=1.2e-5),
    "cookies":   DamageParams(alpha=2.0, beta=1.0, threshold=20.0, rate=1.0e-5),
    "cake":      DamageParams(alpha=2.0, beta=1.5, threshold=20.0, rate=1.0e-5),
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
                 damage_threshold_for_set: float = 5.0):
        self.model = model
        self.data = data
        self.robot_geom_ids = set(int(g) for g in robot_geom_ids)
        self.registry = registry
        self.default_params = default_params or DEFAULT_DAMAGE_PARAMS
        self.damage_threshold_for_set = damage_threshold_for_set

        self._geom_body: Dict[int, int] = {}
        self._body_name: Dict[int, str] = {}
        self._params_cache: Dict[str, DamageParams] = {}
        self._state: Dict[str, BodyDamageState] = defaultdict(BodyDamageState)
        self._step_idx: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._geom_body.clear()
        self._body_name.clear()
        self._params_cache.clear()
        self._state.clear()
        self._step_idx = 0

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

        for i in range(int(data.ncon)):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            r1 = g1 in self.robot_geom_ids
            r2 = g2 in self.robot_geom_ids
            if r1 and r2:
                continue
            target_geom = g2 if (r1 and not r2) else g1
            body_id = self._geom_body.get(target_geom)
            if body_id is None:
                body_id = int(model.geom_bodyid[target_geom])
                self._geom_body[target_geom] = body_id
            name = self._body_name.get(body_id)
            if name is None:
                name = (mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                        or f"body_{body_id}")
                self._body_name[body_id] = name

            # World-frame force on this contact
            cf = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, cf)
            f_contact = cf[:3]
            frame = np.asarray(con.frame, dtype=np.float64).reshape(3, 3)
            f_world = frame.T @ f_contact

            # Decompose against body linear acceleration (world frame)
            a_world = np.asarray(data.cacc[body_id][3:6], dtype=np.float64)
            a_mag = float(np.linalg.norm(a_world))
            if a_mag < 1e-6:
                # No acceleration: all loading is "sustained/constrained".
                # This matches OopsieVerse's intuition that static-grip
                # forces should not be flagged as impulsive damage.
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

        step_total_damage = 0.0
        for name, (F_par, F_perp) in per_body_F.items():
            params = self._params_cache.get(name)
            if params is None:
                params = lookup_damage_params(
                    name, self.registry, self.default_params)
                self._params_cache[name] = params
            eps = params.alpha * F_par + params.beta * F_perp
            d_step = params.rate * max(eps - params.threshold, 0.0)
            if d_step <= 0.0:
                continue
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
