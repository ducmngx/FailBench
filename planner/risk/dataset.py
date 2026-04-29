"""PyTorch Dataset for the FailBench heatmap regressor demo.

Wraps the pre-built `targets.npz` files (smoothed prior-weighted contact
heatmaps, one per config) joined with the per-trial npz that holds the
config inputs (`pre_qpos`, `pre_ee_pos`).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
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
    goal_pos: np.ndarray | None = None  # (3,) float32, world frame; None if not loaded
    task_idx: int | None = None         # index into HeatmapDataset.task_vocab


def _load_goal_pos_from_pkl(pkl_path: Path, scene: str) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    inner = d[scene] if scene in d else next(iter(d.values()))
    return np.asarray(inner["goal_pos"], dtype=np.float32)


class HeatmapDataset(Dataset):
    """Per-config (input_vec, target_heatmap) pairs for one scene.

    `input_vec = concat(pre_qpos (7), pre_ee_pos (3))` → shape (10,).

    Optionally restrict to a subset of `traj_ids` (used for train/val splits).
    Standardisation is applied at __getitem__ time using the supplied stats;
    if `stats` is None, returns raw values (use `fit_stats` to compute, then
    pass into a paired Dataset for the val split).
    """

    BASE_INPUT_DIM = 17   # pre_qpos (7) + pre_ee_pos (3) + pre_qvel (7)
    INPUT_DIM = 17        # legacy alias (Stage 0/1); use `instance.input_dim` for current shape

    def __init__(self,
                 dataset_root: Path | str,
                 scene: str,
                 traj_keys: Sequence[tuple[str, int]] | None = None,
                 stats: "DatasetStats | None" = None,
                 include_goal: bool = False,
                 include_task: bool = False,
                 include_rgb: bool = False,
                 include_depth: bool = False,
                 include_dinov2: bool = False,
                 dinov2_mode: str = "cls",
                 rgb_size: tuple[int, int] = (96, 128),
                 depth_clip: tuple[float, float] = (0.05, 2.0),
                 dinov2_cache_dir: Path | str | None = None,
                 trajs_dir: Path | str | None = None,
                 task_vocab: Sequence[str] | None = None):
        """`traj_keys` is a list of (task_id, traj_id) tuples to keep; None = all.

        traj_id is task-local (0..N per task), so the split key must include task.

        Stage 2 toggles:
          include_goal — append goal_pos (3,) loaded from the trajectory pkl
          include_task — append task one-hot (len(task_vocab),)

        When `task_vocab` is None it is auto-discovered from the scene dir.
        Pass it explicitly to keep train/val splits aligned on the same vocab.
        """
        self.dataset_root = Path(dataset_root)
        self.scene = scene
        self.scene_dir = self.dataset_root / scene
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(self.scene_dir)
        self.include_goal = bool(include_goal)
        self.include_task = bool(include_task)
        self.include_rgb = bool(include_rgb)
        self.include_depth = bool(include_depth)
        self.include_dinov2 = bool(include_dinov2)
        self.dinov2_mode = str(dinov2_mode)
        self.rgb_size = (int(rgb_size[0]), int(rgb_size[1]))   # (H, W)
        self.depth_clip = (float(depth_clip[0]), float(depth_clip[1]))
        if dinov2_cache_dir is not None:
            self.dinov2_cache_dir = Path(dinov2_cache_dir)
        else:
            base = "cache/dinov2" if self.dinov2_mode == "cls" else "cache/dinov2_patch4x4"
            self.dinov2_cache_dir = Path(base) / scene
        self.trajs_dir = Path(trajs_dir) if trajs_dir is not None \
            else Path("scenes") / scene / "trajs"

        # ---- Task vocabulary (used for one-hot) ----
        if task_vocab is None:
            task_vocab = sorted(
                d.name for d in self.scene_dir.iterdir()
                if d.is_dir() and (d / "manifest.csv").exists()
            )
        self.task_vocab: list[str] = list(task_vocab)
        self.task_to_idx = {t: i for i, t in enumerate(self.task_vocab)}

        key_filter = None if traj_keys is None else set((str(t), int(i)) for t, i in traj_keys)

        # ---- Goal-pos cache (loaded once per (task, traj_id) pkl) ----
        self._goal_cache: dict[tuple[str, int], np.ndarray] = {}

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
                row = _Row(
                    npz_path=task_dir / r.npz_file,
                    target=heatmaps[idx].astype(np.float32, copy=False),
                    traj_id=int(r.traj_id),
                    task_id=str(r.task_id),
                    experiment_id=str(r.experiment_id),
                    task_idx=self.task_to_idx.get(str(r.task_id)),
                )
                if self.include_goal:
                    if key not in self._goal_cache:
                        pkl = self.trajs_dir / f"{scene}_{r.task_id}_{int(r.traj_id):02d}.pkl"
                        self._goal_cache[key] = _load_goal_pos_from_pkl(pkl, scene)
                    row.goal_pos = self._goal_cache[key]
                rows.append(row)
            tdata.close()

        if not rows:
            raise RuntimeError(f"No rows found under {self.scene_dir}")
        self.rows = rows
        self.stats = stats
        # Cache grid shape for convenience.
        self.grid_shape = rows[0].target.shape

    @property
    def input_dim(self) -> int:
        d = self.BASE_INPUT_DIM
        if self.include_goal:
            d += 3
        if self.include_task:
            d += len(self.task_vocab)
        return d

    def __len__(self) -> int:
        return len(self.rows)

    def _read_input(self, row: _Row) -> np.ndarray:
        d = np.load(row.npz_path)
        try:
            parts: list[np.ndarray] = [
                np.asarray(d["pre_qpos"], dtype=np.float32),
                np.asarray(d["pre_ee_pos"], dtype=np.float32),
                np.asarray(d["pre_qvel"], dtype=np.float32),
            ]
        finally:
            d.close()
        if self.include_goal:
            assert row.goal_pos is not None, "include_goal=True but row missing goal_pos"
            parts.append(row.goal_pos.astype(np.float32, copy=False))
        if self.include_task:
            one_hot = np.zeros(len(self.task_vocab), dtype=np.float32)
            if row.task_idx is not None:
                one_hot[row.task_idx] = 1.0
            parts.append(one_hot)
        return np.concatenate(parts)

    def _read_dinov2(self, row: _Row) -> np.ndarray:
        """Returns the cached (384,) DINOv2 CLS feature for this row."""
        # row.npz_path is .../<scene>/<task>/<exp_id>.npz; cache mirrors task dir.
        task_name = row.npz_path.parent.name
        path = self.dinov2_cache_dir / task_name / (row.npz_path.stem + ".npy")
        return np.load(path).astype(np.float32, copy=False)

    def _read_rgb(self, row: _Row) -> np.ndarray:
        """Returns (C, H, W) float32. C=3 for RGB, C=4 if include_depth (RGB+D).

        Depth is clipped to `self.depth_clip` and scaled to [0, 1].
        """
        d = np.load(row.npz_path)
        try:
            rgb = np.asarray(d["pre_rgb"])           # (480, 640, 3) uint8
            depth = np.asarray(d["pre_depth"]) if self.include_depth else None  # (480, 640) float32
        finally:
            d.close()
        H, W = self.rgb_size
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))           # (3, H, W)
        if depth is None:
            return rgb
        dmin, dmax = self.depth_clip
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_AREA)
        depth = np.clip(depth, dmin, dmax)
        depth = (depth - dmin) / max(dmax - dmin, 1e-6)
        depth = depth.astype(np.float32)[None, :, :]  # (1, H, W)
        return np.concatenate([rgb, depth], axis=0)   # (4, H, W)

    def __getitem__(self, i: int):
        row = self.rows[i]
        x = self._read_input(row)
        y = row.target  # (ny, nx) float32
        if self.stats is not None:
            x = (x - self.stats.x_mean) / self.stats.x_std
            y = (y - self.stats.y_mean) / self.stats.y_std
        if self.include_dinov2:
            feat = self._read_dinov2(row)
            return torch.from_numpy(x), torch.from_numpy(feat), torch.from_numpy(y), i
        if self.include_rgb:
            rgb = self._read_rgb(row)
            return torch.from_numpy(x), torch.from_numpy(rgb), torch.from_numpy(y), i
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
        xs = np.zeros((n, ds.input_dim), dtype=np.float64)
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


class MultiSceneHeatmapDataset(Dataset):
    """Stage 5: composes per-scene `HeatmapDataset`s into one trainable stream.

    Outputs are padded to a common `max_grid_shape` with a per-row `mask` tensor
    flagging valid cells. Standardisation is per-scene/per-cell — placed in the
    padded frame to align with the model output. Loss should be masked.

    Inputs are the concatenation of:
      [base state vec (17)]
      [optional goal_pos (3)]
      [optional task one-hot — over UNION of all scene task_vocabs]
      [scene one-hot — always present]

    Returned by __getitem__:
      include_rgb=False: (x, y_padded, mask, scene_idx, idx_in_global)
      include_rgb=True:  (x, rgb, y_padded, mask, scene_idx, idx_in_global)
    """

    def __init__(self,
                 dataset_root: Path | str,
                 scenes: Sequence[str],
                 traj_keys_per_scene: dict[str, Sequence[tuple[str, int]]] | None = None,
                 stats_per_scene: dict[str, "DatasetStats"] | None = None,
                 include_goal: bool = False,
                 include_task: bool = False,
                 include_rgb: bool = False,
                 include_depth: bool = False,
                 include_dinov2: bool = False,
                 dinov2_mode: str = "cls",
                 rgb_size: tuple[int, int] = (96, 128),
                 dinov2_cache_root: Path | str | None = None,
                 max_grid_shape: tuple[int, int] | None = None,
                 task_vocab: Sequence[str] | None = None):
        self.dataset_root = Path(dataset_root)
        self.scenes = list(scenes)
        self.scene_to_idx = {s: i for i, s in enumerate(self.scenes)}
        self.include_goal = bool(include_goal)
        self.include_task = bool(include_task)
        self.include_rgb = bool(include_rgb)
        self.include_depth = bool(include_depth)
        self.include_dinov2 = bool(include_dinov2)
        self.dinov2_mode = str(dinov2_mode)
        if dinov2_cache_root is not None:
            self.dinov2_cache_root = Path(dinov2_cache_root)
        else:
            self.dinov2_cache_root = (Path("cache/dinov2") if self.dinov2_mode == "cls"
                                       else Path("cache/dinov2_patch4x4"))

        # Build a unified task vocabulary across all scenes (with scene prefix
        # to avoid name collisions like clean_nominal in level2 vs kitchen).
        if task_vocab is None:
            tv: list[str] = []
            for s in self.scenes:
                scene_dir = self.dataset_root / s
                for d in sorted(scene_dir.iterdir()) if scene_dir.is_dir() else []:
                    if d.is_dir() and (d / "manifest.csv").exists():
                        tv.append(f"{s}/{d.name}")
            self.task_vocab: list[str] = sorted(tv)
        else:
            self.task_vocab = list(task_vocab)
        self.task_to_idx = {t: i for i, t in enumerate(self.task_vocab)}

        # Build per-scene sub-datasets. Pass NO task_vocab so each sub-dataset
        # discovers its own scene-local tasks (we use the unified one ourselves).
        # Sub-datasets are constructed with include_task=False / include_goal=False
        # because we re-build the input vector here using the unified vocab.
        self.subdatasets: dict[str, HeatmapDataset] = {}
        for s in self.scenes:
            sub_keys = None if traj_keys_per_scene is None else traj_keys_per_scene.get(s)
            self.subdatasets[s] = HeatmapDataset(
                self.dataset_root, s,
                traj_keys=sub_keys,
                include_goal=include_goal,
                include_task=False,
                include_rgb=include_rgb,
                include_depth=include_depth,
                include_dinov2=include_dinov2,
                dinov2_mode=dinov2_mode,
                rgb_size=rgb_size,
                dinov2_cache_dir=(self.dinov2_cache_root / s) if include_dinov2 else None,
            )

        # Determine max grid shape across involved scenes
        if max_grid_shape is None:
            max_ny = max(sd.grid_shape[0] for sd in self.subdatasets.values())
            max_nx = max(sd.grid_shape[1] for sd in self.subdatasets.values())
            self.max_grid_shape = (max_ny, max_nx)
        else:
            self.max_grid_shape = tuple(max_grid_shape)

        # Flat row index: list of (scene, sub_index)
        self.flat_index: list[tuple[str, int]] = []
        for s in self.scenes:
            sd = self.subdatasets[s]
            for i in range(len(sd)):
                self.flat_index.append((s, i))

        # Per-scene stats applied per-cell within the padded frame
        self.stats_per_scene = stats_per_scene or {}

        # Cache per-scene mask (boolean) in the padded frame
        self._mask_per_scene: dict[str, np.ndarray] = {}
        for s, sd in self.subdatasets.items():
            ny, nx = sd.grid_shape
            m = np.zeros(self.max_grid_shape, dtype=bool)
            m[:ny, :nx] = True
            self._mask_per_scene[s] = m

    @property
    def grid_shape(self) -> tuple[int, int]:
        """For trainer compatibility — model is built around the max grid."""
        return self.max_grid_shape

    @property
    def input_dim(self) -> int:
        d = HeatmapDataset.BASE_INPUT_DIM
        if self.include_goal:
            d += 3
        if self.include_task:
            d += len(self.task_vocab)
        d += len(self.scenes)   # scene one-hot, always on for multi-scene
        return d

    @property
    def traj_keys(self) -> list[tuple[str, str, int]]:
        """(scene, task, traj_id) tuples — the multi-scene split key."""
        out: list[tuple[str, str, int]] = []
        for s in self.scenes:
            sd = self.subdatasets[s]
            for r in sd.rows:
                out.append((s, r.task_id, r.traj_id))
        return out

    def __len__(self) -> int:
        return len(self.flat_index)

    def _build_input(self, scene: str, row: _Row) -> np.ndarray:
        d = np.load(row.npz_path)
        try:
            parts: list[np.ndarray] = [
                np.asarray(d["pre_qpos"], dtype=np.float32),
                np.asarray(d["pre_ee_pos"], dtype=np.float32),
                np.asarray(d["pre_qvel"], dtype=np.float32),
            ]
        finally:
            d.close()
        if self.include_goal:
            assert row.goal_pos is not None
            parts.append(row.goal_pos.astype(np.float32, copy=False))
        if self.include_task:
            tv_key = f"{scene}/{row.task_id}"
            one_hot = np.zeros(len(self.task_vocab), dtype=np.float32)
            ti = self.task_to_idx.get(tv_key)
            if ti is not None:
                one_hot[ti] = 1.0
            parts.append(one_hot)
        # scene one-hot
        scene_one_hot = np.zeros(len(self.scenes), dtype=np.float32)
        scene_one_hot[self.scene_to_idx[scene]] = 1.0
        parts.append(scene_one_hot)
        return np.concatenate(parts)

    def __getitem__(self, i: int):
        scene, sub_i = self.flat_index[i]
        sd = self.subdatasets[scene]
        row = sd.rows[sub_i]
        x = self._build_input(scene, row)
        ny, nx = sd.grid_shape
        my, mx = self.max_grid_shape
        y_pad = np.zeros((my, mx), dtype=np.float32)
        y_pad[:ny, :nx] = row.target
        mask = self._mask_per_scene[scene]
        # standardise per-scene per-cell within mask
        st = self.stats_per_scene.get(scene)
        if st is not None:
            x = (x - st.x_mean) / st.x_std
            y_pad_std = y_pad.copy()
            y_pad_std[:ny, :nx] = (y_pad[:ny, :nx] - st.y_mean) / st.y_std
            y_pad = y_pad_std
        scene_idx = self.scene_to_idx[scene]
        if self.include_dinov2:
            feat = sd._read_dinov2(row)
            return (torch.from_numpy(x),
                    torch.from_numpy(feat),
                    torch.from_numpy(y_pad),
                    torch.from_numpy(mask),
                    scene_idx,
                    i)
        if self.include_rgb:
            rgb = sd._read_rgb(row)
            return (torch.from_numpy(x),
                    torch.from_numpy(rgb),
                    torch.from_numpy(y_pad),
                    torch.from_numpy(mask),
                    scene_idx,
                    i)
        return (torch.from_numpy(x),
                torch.from_numpy(y_pad),
                torch.from_numpy(mask),
                scene_idx,
                i)


def fit_multiscene_stats(ds: "MultiSceneHeatmapDataset",
                          y_eps: float = 0.05) -> dict[str, "DatasetStats"]:
    """Compute one DatasetStats per scene in a multi-scene dataset.

    Each scene's stats are computed from its own configs, with the input vector
    rebuilt using the multi-scene unified vocab so x_mean/x_std match what
    `__getitem__` produces.

    `y_eps` is the floor on per-cell std (5% by default) — without this, cells
    that are near-zero in training get std≈0 and any nonzero val sample
    standardises to >50, dominating the loss. The single-scene path uses
    1e-3 historically; multi-scene needs a higher floor because more configs
    leave more cells effectively constant.
    """
    out: dict[str, DatasetStats] = {}
    for scene in ds.scenes:
        sd = ds.subdatasets[scene]
        n = len(sd)
        if n == 0:
            continue
        in_dim = ds.input_dim
        ny, nx = sd.grid_shape
        xs = np.zeros((n, in_dim), dtype=np.float64)
        ys_sum = np.zeros((ny, nx), dtype=np.float64)
        ys_sumsq = np.zeros((ny, nx), dtype=np.float64)
        for i in range(n):
            row = sd.rows[i]
            xs[i] = ds._build_input(scene, row)
            y = row.target.astype(np.float64)
            ys_sum += y
            ys_sumsq += y * y
        x_mean = xs.mean(axis=0).astype(np.float32)
        x_std = (xs.std(axis=0) + 1e-3).astype(np.float32)
        y_mean = (ys_sum / n).astype(np.float32)
        y_var = ys_sumsq / n - (ys_sum / n) ** 2
        y_std = (np.sqrt(np.maximum(y_var, 0)) + y_eps).astype(np.float32)
        out[scene] = DatasetStats(x_mean=x_mean, x_std=x_std,
                                   y_mean=y_mean, y_std=y_std)
    return out


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


def split_multiscene_traj_keys(scenes_to_keys: dict[str, list[tuple[str, int]]],
                               val_frac: float = 0.10,
                               seed: int = 0) -> tuple[dict[str, list], dict[str, list]]:
    """Per-scene traj-key split. Each scene's val fraction is independent."""
    train_per_scene: dict[str, list] = {}
    val_per_scene: dict[str, list] = {}
    for s, keys in scenes_to_keys.items():
        tr, va = split_traj_keys(keys, val_frac=val_frac, seed=seed)
        train_per_scene[s] = tr
        val_per_scene[s] = va
    return train_per_scene, val_per_scene
