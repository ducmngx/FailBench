"""PyTorch Dataset for the FailBench heatmap regressor demo.

Wraps the pre-built `targets.npz` files (smoothed prior-weighted contact
heatmaps, one per config) joined with the per-trial npz that holds the
config inputs (`pre_qpos`, `pre_ee_pos`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class _Row:
    npz_path: Path           # full path to the trial's exp_*.npz
    target: np.ndarray       # (ny, nx) float32 — already loaded from targets.npz
    traj_id: int
    task_id: str
    experiment_id: str


class HeatmapDataset(Dataset):
    """Per-config (input_vec, target_heatmap) pairs for one scene.

    `input_vec = concat(pre_qpos (7), pre_ee_pos (3))` → shape (10,).

    Optionally restrict to a subset of `traj_ids` (used for train/val splits).
    Standardisation is applied at __getitem__ time using the supplied stats;
    if `stats` is None, returns raw values (use `fit_stats` to compute, then
    pass into a paired Dataset for the val split).
    """

    INPUT_DIM = 17  # pre_qpos (7) + pre_ee_pos (3) + pre_qvel (7)

    def __init__(self,
                 dataset_root: Path | str,
                 scene: str,
                 traj_keys: Sequence[tuple[str, int]] | None = None,
                 stats: "DatasetStats | None" = None):
        """`traj_keys` is a list of (task_id, traj_id) tuples to keep; None = all.

        traj_id is task-local (0..N per task), so the split key must include task.
        """
        self.dataset_root = Path(dataset_root)
        self.scene = scene
        self.scene_dir = self.dataset_root / scene
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(self.scene_dir)

        key_filter = None if traj_keys is None else set((str(t), int(i)) for t, i in traj_keys)

        rows: list[_Row] = []
        for task_dir in sorted(self.scene_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            tnpz = task_dir / "targets.npz"
            mcsv = task_dir / "manifest.csv"
            if not (tnpz.exists() and mcsv.exists()):
                continue
            df = pd.read_csv(mcsv)
            tdata = np.load(tnpz)
            heatmaps = tdata["target_heatmap"]
            exp_ids = list(tdata["experiment_id"])
            id_to_idx = {e: i for i, e in enumerate(exp_ids)}
            for r in df.itertuples():
                key = (str(r.task_id), int(r.traj_id))
                if key_filter is not None and key not in key_filter:
                    continue
                idx = id_to_idx.get(r.experiment_id)
                if idx is None:
                    continue
                rows.append(_Row(
                    npz_path=task_dir / r.npz_file,
                    target=heatmaps[idx].astype(np.float32, copy=False),
                    traj_id=int(r.traj_id),
                    task_id=str(r.task_id),
                    experiment_id=str(r.experiment_id),
                ))
            tdata.close()

        if not rows:
            raise RuntimeError(f"No rows found under {self.scene_dir}")
        self.rows = rows
        self.stats = stats
        # Cache grid shape for convenience.
        self.grid_shape = rows[0].target.shape

    def __len__(self) -> int:
        return len(self.rows)

    def _read_input(self, row: _Row) -> np.ndarray:
        d = np.load(row.npz_path)
        try:
            x = np.concatenate([d["pre_qpos"], d["pre_ee_pos"], d["pre_qvel"]]).astype(np.float32)
        finally:
            d.close()
        return x  # (17,)

    def __getitem__(self, i: int):
        row = self.rows[i]
        x = self._read_input(row)
        y = row.target  # (ny, nx) float32
        if self.stats is not None:
            x = (x - self.stats.x_mean) / self.stats.x_std
            y = (y - self.stats.y_mean) / self.stats.y_std
        return torch.from_numpy(x), torch.from_numpy(y), i

    @property
    def traj_keys(self) -> list[tuple[str, int]]:
        """Per-row (task_id, traj_id) tuples — the canonical split key."""
        return [(r.task_id, r.traj_id) for r in self.rows]


@dataclass
class DatasetStats:
    """Standardisation statistics computed on a training split."""
    x_mean: np.ndarray  # (10,) float32
    x_std: np.ndarray   # (10,) float32
    y_mean: np.ndarray  # (ny, nx) float32
    y_std: np.ndarray   # (ny, nx) float32

    @classmethod
    def fit(cls, ds: HeatmapDataset, max_samples: int | None = None,
            eps: float = 1e-3) -> "DatasetStats":
        """Walk the dataset once to compute per-feature input stats and per-cell target stats.

        Targets are standardised per-cell (not globally) so the loss balances
        across cells with very different baselines.
        """
        n = len(ds) if max_samples is None else min(len(ds), max_samples)
        ny, nx = ds.grid_shape
        xs = np.zeros((n, ds.INPUT_DIM), dtype=np.float64)
        ys_sum = np.zeros((ny, nx), dtype=np.float64)
        ys_sumsq = np.zeros((ny, nx), dtype=np.float64)
        for i in range(n):
            row = ds.rows[i]
            xs[i] = ds._read_input(row)
            y = row.target.astype(np.float64)
            ys_sum += y
            ys_sumsq += y * y
        x_mean = xs.mean(axis=0).astype(np.float32)
        x_std = (xs.std(axis=0) + eps).astype(np.float32)
        y_mean = (ys_sum / n).astype(np.float32)
        y_var = ys_sumsq / n - (ys_sum / n) ** 2
        y_std = (np.sqrt(np.maximum(y_var, 0)) + eps).astype(np.float32)
        return cls(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    def to_dict(self) -> dict:
        return {
            "x_mean": self.x_mean, "x_std": self.x_std,
            "y_mean": self.y_mean, "y_std": self.y_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetStats":
        return cls(x_mean=np.asarray(d["x_mean"], dtype=np.float32),
                   x_std=np.asarray(d["x_std"], dtype=np.float32),
                   y_mean=np.asarray(d["y_mean"], dtype=np.float32),
                   y_std=np.asarray(d["y_std"], dtype=np.float32))


def split_traj_keys(all_traj_keys: Iterable[tuple[str, int]],
                    val_frac: float = 0.10,
                    seed: int = 0) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Random (task_id, traj_id) split. Returns (train_keys, val_keys).

    Splitting on the (task, traj) tuple prevents leakage where the same
    traj_id (e.g., 0) exists in every task directory.
    """
    uniq = sorted({(str(t), int(i)) for t, i in all_traj_keys})
    rng = np.random.default_rng(seed)
    idx = np.arange(len(uniq))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_keys   = sorted(uniq[k] for k in idx[:n_val])
    train_keys = sorted(uniq[k] for k in idx[n_val:])
    return train_keys, val_keys
