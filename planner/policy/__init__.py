"""Policy + safety machinery for the contact-predictor downstream experiments.

Submodules:

- :mod:`planner.policy.libero_env_failure` — wraps a LIBERO env with FailBench
  failure injection at a configurable ``traj_progress``.
- :mod:`planner.policy.safe_action` — safety-aware action selectors that
  consult a :class:`planner.risk.inference.ContactPredictor` to modulate or
  search demo actions.
"""
