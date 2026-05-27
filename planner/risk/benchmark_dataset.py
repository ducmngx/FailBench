"""PyTorch dataset for the v2 contact-prediction benchmark.

Wraps :class:`LiberoV2Dataset` and assembles a fixed sample schema across all
modality configurations. Each model in the benchmark consumes a subset of the
output dict; the dataset itself reads only the keys it needs.

Output keys (only those enabled by ``modalities`` are present):

    state_window   (T, 18)  float32  per-frame qpos(7) qvel(7) ee_pos(3) grip(1)
    goal           (K, 11)  float32  goal qpos(7) ee_pos(3) grip(1)
    goal_offsets   (K,)     int32    future-step offsets [+5, +15, +30]
    rgb_window     (T, 3, H, W) float32 in [0, 1]
    depth_window   (T, 1, H, W) float32 (metres, robosuite scale)
    dino_window    (T, 384) float32  precomputed DINOv2 CLS (cache only)
    failure_mode   (5,)     float32  one-hot
    target         (H, W)   float32  agentview heatmap (raw mass-preserving)
    target_log1p   (H, W)   float32  log1p(target) — set when log1p_target=True
    trial_id       str
    task           str
    split          str

T defaults to 8, K to 3 — matches the v2 schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from planner.risk.dataset_v2 import LiberoV2Dataset, failure_mode_onehot
from planner.risk.v2_targets import Weighting, build_agentview_target


@dataclass(frozen=True)
class ModalityConfig:
    """Which input keys the dataset should load + assemble.

    Defaults match the "state-only" baseline row. Toggle flags for the other
    benchmark rows. ``dino`` requires a precomputed cache; absent files raise.
    """
    state: bool = True
    goal: bool = False
    rgb: bool = False
    depth: bool = False
    dino: bool = False
    failure_mode: bool = False   # 5-D one-hot of the sampled failure mode
    failure_joints: bool = False # 7-D multi-hot indicating which arm joints failed


@dataclass(frozen=True)
class TargetConfig:
    """How to build the agentview heatmap target."""
    sigma_px: float = 4.0
    weighting: Weighting = "force_prior"
    log1p: bool = True   # default: return both raw + log1p (model picks)


def _stack_state_window(d: dict) -> np.ndarray:
    """Per-frame state vector for the T-frame window (T, 18)."""
    qpos = np.asarray(d["window_qpos"], dtype=np.float32)              # (T, 7)
    qvel = np.asarray(d["window_qvel"], dtype=np.float32)              # (T, 7)
    ee = np.asarray(d["window_ee_pos"], dtype=np.float32)              # (T, 3)
    grip = np.asarray(d["window_gripper_ctrl"], dtype=np.float32)       # (T, 1)
    return np.concatenate([qpos, qvel, ee, grip], axis=1)               # (T, 18)


def _stack_state_single(d: dict) -> np.ndarray:
    """(1, 18) fallback from pre_* when use_window is disabled upstream."""
    qpos = np.asarray(d["pre_qpos"], dtype=np.float32).reshape(7)
    qvel = np.asarray(d["pre_qvel"], dtype=np.float32).reshape(7)
    ee = np.asarray(d["pre_ee_pos"], dtype=np.float32).reshape(3)
    grip = np.asarray(d["pre_gripper_ctrl"], dtype=np.float32).reshape(1)
    return np.concatenate([qpos, qvel, ee, grip])[None, :]              # (1, 18)


def _stack_goal(d: dict) -> np.ndarray:
    """Goal feature (K, 11) = qpos(7) ⊕ ee_pos(3) ⊕ grip(1)."""
    qpos = np.asarray(d["goal_qpos"], dtype=np.float32)                 # (K, 7)
    ee = np.asarray(d["goal_ee_pos"], dtype=np.float32)                 # (K, 3)
    grip = np.asarray(d["goal_gripper_ctrl"], dtype=np.float32)          # (K, 1)
    return np.concatenate([qpos, ee, grip], axis=1)                     # (K, 11)


def _rgb_window_chw(d: dict) -> np.ndarray:
    """(T, 3, H, W) float32 in [0, 1]."""
    rgb = np.asarray(d["window_agentview_rgb"])                         # (T, H, W, 3) u8
    return (rgb.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)


def _depth_window_chw(d: dict) -> np.ndarray:
    """(T, 1, H, W) float32 metres."""
    depth = np.asarray(d["window_agentview_depth"], dtype=np.float32)
    return depth[:, None, :, :]


class BenchmarkDataset:
    """v2 dataset adapter for the contact-heatmap benchmark.

    Parameters
    ----------
    v2_root
        Same as :class:`LiberoV2Dataset`.
    modalities
        Which input keys to assemble. See :class:`ModalityConfig`.
    target_cfg
        Heatmap target options. See :class:`TargetConfig`.
    splits
        Which v2 splits to include. Default: all three.
    dino_cache_root
        Directory holding ``<task>/<trial_id>.npy`` (T, 384) features. Required
        if ``modalities.dino=True``.

    Notes
    -----
    The wrapped :class:`LiberoV2Dataset` is configured with only the IO
    actually needed for the requested modalities — state-only runs don't pay
    for image decompression.
    """

    def __init__(self,
                 v2_root: str | Path,
                 modalities: ModalityConfig = ModalityConfig(),
                 target_cfg: TargetConfig = TargetConfig(),
                 *,
                 splits: Iterable[str] = ("libero_spatial", "libero_object", "libero_goal"),
                 dino_cache_root: Optional[str | Path] = None,
                 use_window: bool = True):
        self.modalities = modalities
        self.target_cfg = target_cfg
        self.use_window = use_window
        self.dino_cache_root = Path(dino_cache_root) if dino_cache_root else None
        if modalities.dino and self.dino_cache_root is None:
            raise ValueError("modalities.dino=True requires dino_cache_root")

        needs_rgb = modalities.rgb
        needs_depth = modalities.depth

        self._base = LiberoV2Dataset(
            v2_root,
            splits=splits,
            use_window=use_window,
            use_wrist_cam=False,
            use_depth=needs_depth,
            use_failure_mode=modalities.failure_mode,
            keep_keys=self._payload_keys(needs_rgb, needs_depth),
        )

    # --- public API -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, i: int) -> dict:
        d = self._base[i]
        out: dict = {
            "trial_id": d["trial_id"],
            "task": d["task"],
            "split": d["split"],
        }

        if self.modalities.state:
            if self.use_window and "window_qpos" in d:
                out["state_window"] = _stack_state_window(d)
            else:
                out["state_window"] = _stack_state_single(d)

        if self.modalities.goal:
            out["goal"] = _stack_goal(d)
            out["goal_offsets"] = np.asarray(d["goal_offsets"], dtype=np.int32)

        if self.modalities.rgb:
            if self.use_window and "window_agentview_rgb" in d:
                out["rgb_window"] = _rgb_window_chw(d)
            else:
                rgb = np.asarray(d["pre_rgb"], dtype=np.float32) / 255.0
                out["rgb_window"] = rgb.transpose(2, 0, 1)[None, :]      # (1, 3, H, W)

        if self.modalities.depth:
            if self.use_window and "window_agentview_depth" in d:
                out["depth_window"] = _depth_window_chw(d)
            else:
                depth = np.asarray(d["pre_depth"], dtype=np.float32)
                out["depth_window"] = depth[None, None, :, :]            # (1, 1, H, W)

        if self.modalities.dino:
            out["dino_window"] = self._load_dino(d)

        if self.modalities.failure_mode:
            out["failure_mode"] = d.get("failure_mode_onehot",
                                        failure_mode_onehot(d.get("failure_mode", "")))
        if self.modalities.failure_joints:
            # 7-D multi-hot for arm joints 1..7. Empty when GRIPPER_OPEN/SLIPPERY_GRIP.
            jh = np.zeros(7, dtype=np.float32)
            fj = d.get("failure_joints")
            if fj is not None:
                for j in np.asarray(fj).ravel():
                    j_idx = int(j) - 1  # joints are 1-based in v2
                    if 0 <= j_idx < 7:
                        jh[j_idx] = 1.0
            out["failure_joints"] = jh

        # --- target ---
        tgt = build_agentview_target(
            d,
            sigma_px=self.target_cfg.sigma_px,
            weighting=self.target_cfg.weighting,
        )
        out["target"] = tgt.heatmap
        if self.target_cfg.log1p:
            out["target_log1p"] = np.log1p(tgt.heatmap).astype(np.float32)
        out["target_mass"] = np.float32(tgt.weight_in_frame)
        return out

    # --- internals ------------------------------------------------------

    def _payload_keys(self, needs_rgb: bool, needs_depth: bool) -> set:
        """Minimal HDF5 read set for the requested modalities.

        Always includes the target inputs (contacts + agentview calibration +
        scalar failure_prob attr) and the trial identity attrs.
        """
        keys = {
            # target inputs
            "contact_positions", "contact_force_world", "contact_forces",
            "cam_agentview_pos", "cam_agentview_mat0",
            "cam_agentview_fovy", "cam_agentview_size",
            # always-on identity-ish single-frame fallbacks
            "pre_qpos", "pre_qvel", "pre_ee_pos", "pre_gripper_ctrl",
        }
        if self.modalities.state and self.use_window:
            keys |= {"window_qpos", "window_qvel",
                     "window_ee_pos", "window_gripper_ctrl"}
        if self.modalities.goal:
            keys |= {"goal_qpos", "goal_ee_pos",
                     "goal_gripper_ctrl", "goal_offsets"}
        if needs_rgb:
            if self.use_window:
                keys.add("window_agentview_rgb")
            else:
                keys.add("pre_rgb")
        if needs_depth:
            if self.use_window:
                keys.add("window_agentview_depth")
            else:
                keys.add("pre_depth")
        if self.modalities.failure_joints:
            keys.add("failure_joints")
        return keys

    def _load_dino(self, d: dict) -> np.ndarray:
        # Cache layout: <root>/<split>/<task>/<trial_id>.npy
        # Matches scripts.benchmark.precompute_dinov2_v2 output.
        path = self.dino_cache_root / d["split"] / d["task"] / f"{d['trial_id']}.npy"
        return np.load(path).astype(np.float32)


# -----------------------------------------------------------------------
# Split helpers
# -----------------------------------------------------------------------


class MarginalBenchmarkDataset:
    """Group-indexed view for the realistic benchmark.

    One sample per (split, task, demo_key, bin_idx) group. The target is the
    precomputed mode-prior-weighted marginal heatmap from
    :mod:`scripts.benchmark.build_marginal_targets`. Inputs are loaded from
    the *representative* sibling trial (siblings share pre-failure state by
    construction, so any sibling works; the precompute picks the lexically
    first).

    Modality flags work the same as :class:`BenchmarkDataset` EXCEPT
    ``failure_mode`` and ``failure_joints`` are silently disabled — the
    realistic setting by definition cannot use oracle failure descriptors.
    """

    def __init__(self,
                 v2_root: str | Path,
                 marginal_root: str | Path,
                 modalities: ModalityConfig = ModalityConfig(),
                 *,
                 splits: Iterable[str] = ("libero_spatial",),
                 use_window: bool = True,
                 dino_cache_root: Optional[str | Path] = None):
        import h5py
        import hdf5plugin  # noqa: F401
        # Strip oracle modalities from the realistic view.
        if modalities.failure_mode or modalities.failure_joints:
            modalities = ModalityConfig(
                state=modalities.state, goal=modalities.goal,
                rgb=modalities.rgb, depth=modalities.depth,
                dino=modalities.dino, failure_mode=False, failure_joints=False)
        self.modalities = modalities
        self.use_window = use_window
        self.marginal_root = Path(marginal_root)
        # Wrap a trial-indexed dataset for input loading. We only ever access
        # representative trials, so this is mostly a way to reuse all the
        # _stack_state_window / _rgb_window_chw machinery.
        self._inputs = BenchmarkDataset(
            v2_root,
            modalities=modalities,
            target_cfg=TargetConfig(log1p=True),
            splits=tuple(splits),
            dino_cache_root=dino_cache_root,
            use_window=use_window,
        )
        # Build trial_id → trial-dataset index for representative lookup.
        self._trial_to_idx = {row.trial_id: i for i, row in enumerate(self._inputs._base._index)}

        # Load all per-task marginals into one flat index.
        self._index: list = []  # list of (target, total_prob, n_sib, repr_trial_idx, split, task, demo_key)
        for split in splits:
            split_dir = self.marginal_root / split
            if not split_dir.exists():
                raise FileNotFoundError(f"missing marginal cache: {split_dir}")
            for h5_path in sorted(split_dir.glob("*__marginals.h5")):
                with h5py.File(h5_path, "r") as f:
                    keys = [k.decode() for k in f["group_keys"][()]]
                    targets = f["target"][()]                       # (N, H, W)
                    total_prob = f["total_prob"][()]
                    n_sib = f["n_siblings"][()]
                    repr_tids = [r.decode() for r in f["representative"][()]]
                    task = f.attrs["task"]
                for i, (gk, repr_tid) in enumerate(zip(keys, repr_tids)):
                    if repr_tid not in self._trial_to_idx:
                        continue
                    demo_key = gk.split("__b", 1)[0]
                    self._index.append((
                        targets[i], float(total_prob[i]), int(n_sib[i]),
                        self._trial_to_idx[repr_tid], split, task, demo_key,
                    ))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        target, total_prob, n_sib, repr_idx, split, task, demo_key = self._index[i]
        out = self._inputs[repr_idx]
        # Drop the per-trial target keys, replace with marginal.
        for k in ("target", "target_log1p", "target_mass"):
            out.pop(k, None)
        out["target"] = target.astype(np.float32)
        out["target_log1p"] = np.log1p(target).astype(np.float32)
        out["target_mass"] = np.float32(target.sum())
        out["group_total_prob"] = np.float32(total_prob)
        out["group_n_siblings"] = np.int32(n_sib)
        out["demo_key"] = demo_key
        return out


def demo_stratified_split(ds, *,
                          val_frac: float = 0.10,
                          seed: int = 0) -> tuple:
    """Group-stratified train/val split by ``(split, task, demo_key)``.

    Demo id is parsed from the v2 ``trial_id`` (format ``demo_<k>_s<seed>_b<bin>``)
    so two trials from the same demo never appear on opposite sides of the
    split. Returns ``(train_idx, val_idx)`` as numpy int64 arrays.

    Also handles :class:`MarginalBenchmarkDataset` whose index stores
    ``(target, ..., split, task, demo_key)`` tuples instead of trial rows.
    """
    indices = []
    if isinstance(ds, MarginalBenchmarkDataset):
        for i, entry in enumerate(ds._index):
            _, _, _, _, split, task, demo_key = entry
            indices.append((i, (split, task, demo_key)))
    else:
        for i, row in enumerate(ds._base._index):
            demo_key = row.trial_id.split("_s", 1)[0]    # "demo_<k>"
            group = (row.split, row.task, demo_key)
            indices.append((i, group))

    groups = {}
    for i, g in indices:
        groups.setdefault(g, []).append(i)

    rng = np.random.default_rng(seed)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)
    n_val_groups = int(round(len(group_keys) * val_frac))
    val_groups = set(group_keys[:n_val_groups])

    train_idx, val_idx = [], []
    for g, ids in groups.items():
        (val_idx if g in val_groups else train_idx).extend(ids)
    return np.asarray(sorted(train_idx), dtype=np.int64), \
           np.asarray(sorted(val_idx), dtype=np.int64)


def task_held_out_split(ds, *, n_val_tasks: int = 3, seed: int = 0):
    """Hold out ``n_val_tasks`` whole tasks for val; the rest train.

    Tests cross-task generalisation: model trained on N-k tasks must predict
    contacts on k unseen tasks. State-only models that have implicitly
    memorised the training tasks' scene layouts should generalise worse than
    vision-conditioned models that observe the new scene directly.

    Returns ``(train_idx, val_idx, val_task_names)``.
    """
    indices = []
    if isinstance(ds, MarginalBenchmarkDataset):
        for i, entry in enumerate(ds._index):
            _, _, _, _, split, task, demo_key = entry
            indices.append((i, (split, task)))
    else:
        for i, row in enumerate(ds._base._index):
            indices.append((i, (row.split, row.task)))

    groups = {}
    for i, g in indices:
        groups.setdefault(g, []).append(i)

    rng = np.random.default_rng(seed)
    task_keys = sorted(groups.keys())
    rng.shuffle(task_keys)
    val_tasks = set(task_keys[:n_val_tasks])

    train_idx, val_idx = [], []
    for g, ids in groups.items():
        (val_idx if g in val_tasks else train_idx).extend(ids)
    val_task_names = sorted(t for _, t in val_tasks)
    return (np.asarray(sorted(train_idx), dtype=np.int64),
            np.asarray(sorted(val_idx), dtype=np.int64),
            val_task_names)
