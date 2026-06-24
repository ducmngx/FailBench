"""Severity S(eᵢ, rⱼ) for the ICRA'26 risk objective (Nguyen et al., Eq. 1).

The "Failing Gracefully" formulation weights each interaction by a severity
factor S(eᵢ, rⱼ) — the potential impact of robot component rⱼ interacting with
environment entity eᵢ during a failure (hot coffee r₃ hitting a person is worse
than the empty cup r₂; a fragile ramekin is worse than the table). The planner's
safety term is

    Σₜ Σ_{rⱼ∈R} Σ_{eᵢ∈E}  Pₜ(xₜ, eᵢ, rⱼ | F) · S(eᵢ, rⱼ).

This module provides S(eᵢ, rⱼ): the robot components R = {body r₁, carried
object r₂, contents r₃}, the entity severity scale, and the failure-mode →
component mapping used to read Pₜ(·, rⱼ) off the learned predictor (object-drop
modes implicate r₂; joint-failure modes implicate the body r₁).

Entity severity is resolved from the damage fragility table in
:mod:`planner.risk.damage` (single source of truth) via the same case-insensitive
substring matching — ``akita_black_bowl_1`` → ``bowl``. Two scales are offered:
``"paper"`` reproduces the paper's discrete convention (high=10 / standard=2 /
structural=0); ``"damage"`` is the continuous impact-fragility ``alpha/threshold``
used by the earlier demo-filter (kept for backward compatibility).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from planner.risk.damage import (
    DamageParams,
    LIBERO_DAMAGE_PARAMS,
    lookup_damage_params,
)

# ---------------------------------------------------------------------------
# Robot components R = {r₁ body, r₂ carried object, r₃ contents}  (paper §III.A)
# ---------------------------------------------------------------------------

BODY = "robot_body"            # r₁ — the arm/links sweeping into a fixture
CARRIED_OBJECT = "carried_object"  # r₂ — the grasped object dropping
CONTENTS = "contents"          # r₃ — liquid/contents inside the carried object
ROBOT_COMPONENTS = (BODY, CARRIED_OBJECT, CONTENTS)

# Which robot component a failure mode implicates — lets us read the
# per-component interaction probability Pₜ(·, rⱼ) off the failure-conditioned
# predictor. Object-drop modes (gripper opens / slips) drop the carried object
# (r₂); joint-freeze modes sweep the arm body (r₁) into the scene.
FAILURE_MODE_COMPONENT: Dict[str, str] = {
    "GRIPPER_OPEN":  CARRIED_OBJECT,
    "SLIPPERY_GRIP": CARRIED_OBJECT,
    "SINGLE_JOINT":  BODY,
    "MULTI_JOINT":   BODY,
    "ALL_JOINTS":    BODY,
}


def modes_for_component(component: str) -> list:
    """Failure modes whose interaction is attributed to ``component``."""
    return [m for m, c in FAILURE_MODE_COMPONENT.items() if c == component]


# Per-component severity multiplier on top of the entity severity. Defaults to
# 1.0 for every component (severity is driven by the entity's vulnerability);
# override to encode "the heavy arm hurts more than the light bowl" or
# "spilled hot contents (r₃) are worse than the cup (r₂)".
COMPONENT_FACTOR: Dict[str, float] = {BODY: 1.0, CARRIED_OBJECT: 1.0,
                                      CONTENTS: 1.0}

# Paper severity scale (§V.B): high-severity (fragile/breakable) entities = 10,
# standard movable entities = 2, structural fixtures = 0.
PAPER_HIGH = 10.0
PAPER_STANDARD = 2.0
# Bucket boundary on the raw impact-fragility (alpha/threshold): ramekin 0.80 &
# bowl/mug/cup 0.42 → high; plate 0.27, cookies/stove 0.10 → standard; cabinet
# 2.5e-4 / table 2e-5 / world 0 → structural.
_HIGH_RAW = 0.40
_STRUCTURAL_RAW = 0.05


def _raw_value(p: DamageParams) -> float:
    """Impact-fragility scalar from damage params: ``alpha / threshold``.

    ramekin 8/10=0.80, bowl/mug 5/12=0.42, plate 4/15=0.27, cookies 2/20=0.10,
    cabinet 0.05/200≈2.5e-4, table 0.01/500≈2e-5, world 0.
    """
    if p.threshold <= 0:
        return 0.0
    return float(p.alpha) / float(p.threshold)


# Largest raw value across the registry — the normalisation constant so the
# most-fragile tier maps to ~1.0. Computed once at import.
_MAX_RAW: float = max((_raw_value(p) for p in LIBERO_DAMAGE_PARAMS.values()),
                      default=1.0) or 1.0


def object_value(body_name: str,
                 registry: Optional[Dict[str, DamageParams]] = None,
                 normalize: bool = True) -> float:
    """Value S(e) for one body name.

    Parameters
    ----------
    body_name
        MuJoCo body / entity name (e.g. ``akita_black_bowl_1``). Resolved via
        the damage registry's substring matching.
    registry
        Optional override fragility table (same shape as
        :data:`LIBERO_DAMAGE_PARAMS`). Defaults to the LIBERO table.
    normalize
        When True (default), rescale so the most-fragile tier ≈ 1.0; values
        land in ``[0, 1]``. When False, return the raw ``alpha / threshold``.
    """
    p = lookup_damage_params(body_name, registry=registry)
    raw = _raw_value(p)
    if not normalize:
        return raw
    # Re-derive the normalisation constant if a custom registry is supplied so
    # the override's own most-fragile tier maps to ~1.0.
    denom = _MAX_RAW
    if registry is not None:
        denom = max((_raw_value(q) for q in registry.values()), default=1.0) or 1.0
    return raw / denom


def severity_for_entities(names: Iterable[str],
                          registry: Optional[Dict[str, DamageParams]] = None,
                          normalize: bool = True,
                          override: Optional[Dict[str, float]] = None
                          ) -> Dict[str, float]:
    """Map a collection of entity names → value, ready for ``object_values``.

    This is exactly the dict shape :func:`planner.risk.inference.risk_score` and
    :func:`planner.policy.safe_action.query_risk` expect as ``object_values``.
    If ``override`` (a manual severity config, see :func:`load_severity_config`)
    is given, each entity's value is taken from it by substring match (falling
    back to the built-in paper severity); otherwise the continuous ``"damage"``
    scale is used.
    """
    if override is not None:
        return {n: resolve_severity(n, override=override, registry=registry)
                for n in names}
    return {n: object_value(n, registry=registry, normalize=normalize)
            for n in names}


def load_severity_config(path) -> Dict[str, float]:
    """Load a manual per-object severity config: ``{object_substring: value}``
    from a YAML (``.yaml``/``.yml``) or JSON file. Substrings are matched against
    body names (e.g. ``akita_black_bowl: 10``)."""
    import json
    from pathlib import Path as _Path
    p = _Path(path)
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return {str(k): float(v) for k, v in data.items()}


def resolve_severity(entity: str,
                     override: Optional[Dict[str, float]] = None,
                     scale: str = "paper",
                     registry: Optional[Dict[str, DamageParams]] = None) -> float:
    """Severity of ``entity``: a manual ``override`` config (longest substring
    match) takes precedence; otherwise the built-in :func:`entity_severity`."""
    if override:
        nm = entity.lower()
        best = None
        for key, val in override.items():
            if key.lower() in nm and (best is None or len(key) > best[0]):
                best = (len(key), float(val))
        if best is not None:
            return best[1]
    return entity_severity(entity, scale=scale, registry=registry)


# ---------------------------------------------------------------------------
# Paper severity S(eᵢ, rⱼ)
# ---------------------------------------------------------------------------

def entity_severity(entity: str, scale: str = "paper",
                    registry: Optional[Dict[str, DamageParams]] = None) -> float:
    """Severity of entity ``eᵢ`` independent of the component.

    ``scale="paper"`` buckets to the paper's discrete convention
    (high=10 / standard=2 / structural=0) by impact fragility; ``scale="damage"``
    returns the continuous normalised ``alpha/threshold`` value.
    """
    if scale == "damage":
        return object_value(entity, registry=registry, normalize=True)
    if scale != "paper":
        raise ValueError(f"unknown severity scale {scale!r}")
    raw = _raw_value(lookup_damage_params(entity, registry=registry))
    if raw < _STRUCTURAL_RAW:
        return 0.0
    return PAPER_HIGH if raw >= _HIGH_RAW else PAPER_STANDARD


def severity(entity: str, component: str = CARRIED_OBJECT,
             scale: str = "paper",
             component_factor: Optional[Dict[str, float]] = None,
             registry: Optional[Dict[str, DamageParams]] = None) -> float:
    """S(eᵢ, rⱼ) — severity of component ``rⱼ`` interacting with entity ``eᵢ``.

    Factors as ``entity_severity(eᵢ) · component_factor(rⱼ)``. The component
    factor defaults to :data:`COMPONENT_FACTOR` (all 1.0), so by default S
    depends only on the entity; pass ``component_factor`` to encode component
    dependence (heavy body, hot contents).
    """
    cf = component_factor if component_factor is not None else COMPONENT_FACTOR
    return entity_severity(entity, scale=scale, registry=registry) * \
        cf.get(component, 1.0)


def severity_for_entities_component(
        names: Iterable[str], component: str = CARRIED_OBJECT,
        scale: str = "paper",
        component_factor: Optional[Dict[str, float]] = None,
        registry: Optional[Dict[str, DamageParams]] = None) -> Dict[str, float]:
    """``{entity: S(entity, component)}`` for one robot component."""
    return {n: severity(n, component=component, scale=scale,
                        component_factor=component_factor, registry=registry)
            for n in names}


__all__ = [
    "object_value", "severity_for_entities",            # legacy continuous
    "load_severity_config", "resolve_severity",         # manual config override
    "entity_severity", "severity", "severity_for_entities_component",
    "BODY", "CARRIED_OBJECT", "CONTENTS", "ROBOT_COMPONENTS",
    "FAILURE_MODE_COMPONENT", "modes_for_component", "COMPONENT_FACTOR",
    "PAPER_HIGH", "PAPER_STANDARD",
]
