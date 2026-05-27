"""Benchmark model registry.

Every model in the contact-prediction benchmark exposes the same call
signature::

    out = model(batch)             # batch: dict of (B, ...) tensors
    out["pred"]   # (B, H, W) heatmap prediction in log1p space

so the trainer (``scripts/benchmark/train_one.py``) is model-agnostic.
"""
from __future__ import annotations

from planner.risk.benchmark_dataset import ModalityConfig
from planner.risk.models.mlp import BenchmarkMLP
from planner.risk.models.convdec import BenchmarkConvDec
from planner.risk.models.unet import BenchmarkUNet

_REGISTRY = {
    "mlp": BenchmarkMLP,
    "convdec": BenchmarkConvDec,
    "unet": BenchmarkUNet,
}


def make_model(name: str, *, modalities: ModalityConfig,
               grid_hw: tuple, T: int = 8, K: int = 3, **kwargs):
    """Build a benchmark model by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; options: {sorted(_REGISTRY)}")
    return _REGISTRY[name](modalities=modalities, grid_hw=grid_hw, T=T, K=K, **kwargs)


def available_models() -> list:
    return sorted(_REGISTRY)
