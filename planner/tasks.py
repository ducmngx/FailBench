"""YAML-driven task loader and goal sampler factory.

Tasks are defined in per-scene YAML files at scenes/<scene>/tasks.yaml.
This module loads them and produces the callable goal samplers used by
generate_task_trajs.py, so new tasks can be added without modifying Python code.
"""

import os
from typing import Callable, List, Optional, Tuple

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tasks(scene: str) -> dict:
    """Load tasks.yaml for a scene. Returns the full parsed YAML dict."""
    path = os.path.join(_REPO_ROOT, "scenes", scene, "tasks.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def get_grasped_object(task_def: dict, scene_default: str = "object3") -> str:
    """Return per-task grasped_object override, or the scene-level default."""
    return task_def.get("grasped_object", scene_default)


def _is_clear(x: float, y: float,
              centers: List[Tuple[float, float]],
              min_dist: float) -> bool:
    for ox, oy in centers:
        if abs(x - ox) < min_dist and abs(y - oy) < min_dist:
            return False
    return True


def make_goal_sampler(task_def: dict,
                      obstacle_centers: List[Tuple[float, float]],
                      min_clearance: float = 0.06) -> Callable:
    """Return a callable ``rng -> (x, y)`` based on the task goal definition.

    Supported goal types
    --------------------
    zones
        Sample uniform (x, y) from a randomly chosen zone rectangle.
        If ``check_clearance: true``, retries up to 50 times to stay at
        least ``min_clearance`` away from every obstacle center.
        Zone format: ``{x: [min, max], y: [min, max]}``.

    choice
        Pick one (x, y) from the ``targets`` list at random, then add
        uniform jitter in ``[-jitter, jitter]`` to each axis.
    """
    goal = task_def["goal"]
    goal_type = goal["type"]

    if goal_type == "zones":
        zones = goal["zones"]
        check_clearance = goal.get("check_clearance", False)

        def sampler(rng):
            last_x, last_y = 0.0, 0.0
            for _ in range(50):
                zone = zones[rng.randint(len(zones))]
                x = rng.uniform(zone["x"][0], zone["x"][1])
                y = rng.uniform(zone["y"][0], zone["y"][1])
                last_x, last_y = x, y
                if not check_clearance or _is_clear(x, y, obstacle_centers, min_clearance):
                    return x, y
            return last_x, last_y  # fallback after exhausting retries

        return sampler

    if goal_type == "choice":
        targets = [tuple(t) for t in goal["targets"]]
        jitter = float(goal.get("jitter", 0.0))

        def sampler(rng):
            ox, oy = targets[rng.randint(len(targets))]
            return (ox + rng.uniform(-jitter, jitter),
                    oy + rng.uniform(-jitter, jitter))

        return sampler

    raise ValueError(f"Unknown goal type: {goal_type!r}. Expected 'zones' or 'choice'.")
