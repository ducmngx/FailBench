"""Metrics for the v2 contact-prediction benchmark.

All metrics take ``(pred, target)`` numpy arrays in *log1p space* (same as the
training loss), shape ``(B, H, W)`` or single-sample ``(H, W)``. Returns
scalars (per-batch means) so they accumulate cleanly across batches.

Convention: lower is better for {MSE, RMSE, KL, EMD}; higher is better for
{Soft-IoU, AUPRC, mass_total_ratio ideally → 1.0}. ``inference_latency_ms``
returns wall-clock per batch.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np


def _as_3d(x: np.ndarray) -> np.ndarray:
    """Ensure ``(B, H, W)`` shape — broadcast (H, W) → (1, H, W)."""
    return x[None] if x.ndim == 2 else x


# ---------------------------------------------------------------------------
# Pixel-wise losses (log1p space)
# ---------------------------------------------------------------------------

def weighted_mse_log1p(pred: np.ndarray, target: np.ndarray,
                       alpha: float = 10.0) -> float:
    """The training loss recomputed offline.

    ``w = 1 + alpha · (target > 0)``. Matches
    :func:`scripts.benchmark.train_one.weighted_mse_log1p` numerically.
    """
    pred = _as_3d(pred); target = _as_3d(target)
    fg = (target > 0).astype(np.float32)
    w = 1.0 + alpha * fg
    sq = (pred - target) ** 2
    return float((w * sq).sum() / max(w.sum(), 1e-12))


def rmse_raw(pred_log1p: np.ndarray, target_log1p: np.ndarray) -> float:
    """RMSE on the *raw* (expm1-decoded) heatmap. Interpretable units."""
    pred = np.expm1(np.clip(_as_3d(pred_log1p), 0, None))
    targ = np.expm1(_as_3d(target_log1p))
    return float(np.sqrt(((pred - targ) ** 2).mean()))


# ---------------------------------------------------------------------------
# Localisation
# ---------------------------------------------------------------------------

def soft_iou(pred: np.ndarray, target: np.ndarray,
             tau_frac: float = 0.05) -> float:
    """Mean per-trial soft IoU at threshold ``tau_frac × per-trial max``.

    Uses the binary form on per-trial threshold masks (intersection /
    union); operates in log1p space because thresholding is monotone in the
    log1p transform.
    """
    pred = _as_3d(pred); target = _as_3d(target)
    B = pred.shape[0]
    out = np.empty(B, dtype=np.float64)
    for i in range(B):
        tmax = float(target[i].max())
        if tmax <= 0:
            out[i] = float(pred[i].max() <= 0)        # both empty → IoU=1
            continue
        tau = tau_frac * tmax
        pm = pred[i] >= tau
        tm = target[i] >= tau
        inter = float((pm & tm).sum())
        uni = float((pm | tm).sum())
        out[i] = inter / max(uni, 1.0)
    return float(out.mean())


def auprc_mass(pred: np.ndarray, target: np.ndarray,
               tau_frac: float = 0.05) -> float:
    """Mean per-trial AUPRC for ``target_pixel > tau_frac · per-trial max``.

    Threshold-free localisation quality. Falls back to ``precision`` when the
    trial has no positive pixels.
    """
    from sklearn.metrics import average_precision_score

    pred = _as_3d(pred); target = _as_3d(target)
    B = pred.shape[0]
    out = []
    for i in range(B):
        tmax = float(target[i].max())
        if tmax <= 0:
            continue
        y_true = (target[i] >= tau_frac * tmax).ravel().astype(np.int32)
        if y_true.sum() == 0:
            continue
        y_score = pred[i].ravel()
        out.append(float(average_precision_score(y_true, y_score)))
    return float(np.mean(out)) if out else float("nan")


# ---------------------------------------------------------------------------
# Mass calibration & shape fidelity
# ---------------------------------------------------------------------------

def mass_total_ratio(pred_log1p: np.ndarray,
                     target_log1p: np.ndarray) -> float:
    """``sum(expm1(pred)) / sum(target)`` — averaged across trials.

    Ideal = 1.0. >1 → over-predicting mass; <1 → under-predicting.
    """
    pred = np.expm1(np.clip(_as_3d(pred_log1p), 0, None))
    targ = np.expm1(_as_3d(target_log1p))
    B = pred.shape[0]
    ratios = []
    for i in range(B):
        ts = float(targ[i].sum())
        if ts <= 1e-9:
            continue
        ratios.append(float(pred[i].sum()) / ts)
    return float(np.mean(ratios)) if ratios else float("nan")


def symmetric_kl(pred: np.ndarray, target: np.ndarray,
                 eps: float = 1e-8) -> float:
    """Symmetric KL between sum-normalised pred and target.

    Captures *shape* independent of mass scale. Operates in log1p space (a
    monotone transform; normalising afterwards yields a valid distribution).
    """
    pred = _as_3d(pred); target = _as_3d(target)
    B = pred.shape[0]
    out = []
    for i in range(B):
        p = np.clip(pred[i].ravel(), 0, None).astype(np.float64)
        t = np.clip(target[i].ravel(), 0, None).astype(np.float64)
        ps, ts = p.sum(), t.sum()
        if ps <= eps or ts <= eps:
            continue
        p /= ps; t /= ts
        p = np.clip(p, eps, None); t = np.clip(t, eps, None)
        kl_pt = float((p * np.log(p / t)).sum())
        kl_tp = float((t * np.log(t / p)).sum())
        out.append(0.5 * (kl_pt + kl_tp))
    return float(np.mean(out)) if out else float("nan")


# ---------------------------------------------------------------------------
# Inference latency
# ---------------------------------------------------------------------------

def inference_latency_ms(model, sample_batch: dict, *,
                         device: str = "cuda",
                         n_warmup: int = 3, n_iter: int = 20) -> float:
    """Mean per-batch forward-pass wall-clock in ms.

    Synchronises on CUDA between iterations. Pass a representative batch
    (already device-resident); the model is run in eval mode with no_grad.
    """
    import torch
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(sample_batch)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = model(sample_batch)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return (dt / n_iter) * 1000.0


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

ALL_METRICS = (
    "weighted_mse_log1p", "rmse_raw", "soft_iou", "auprc_mass",
    "mass_total_ratio", "symmetric_kl",
)


def compute_all(pred: np.ndarray, target: np.ndarray) -> dict:
    """Run every per-pixel metric on a single (B, H, W) batch."""
    return {
        "weighted_mse_log1p": weighted_mse_log1p(pred, target),
        "rmse_raw": rmse_raw(pred, target),
        "soft_iou": soft_iou(pred, target),
        "auprc_mass": auprc_mass(pred, target),
        "mass_total_ratio": mass_total_ratio(pred, target),
        "symmetric_kl": symmetric_kl(pred, target),
    }
