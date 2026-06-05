"""PyTorch Dataset for the LIBERO v2 contact-prediction dataset.

Built on top of :mod:`planner.risk.v2_store`. Indexes trials from the per-split
``manifest.csv`` and serves one sample dict per ``__getitem__``. Designed to
plug into ``torch.utils.data.DataLoader`` with persistent workers; each worker
opens its own :class:`V2Reader` per HDF5 file (LRU-cached by path).
"""
from __future__ import annotations

import csv
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from planner.risk.v2_store import V2Reader


_FAILURE_MODES = ("GRIPPER_OPEN", "SLIPPERY_GRIP", "SINGLE_JOINT",
                  "MULTI_JOINT", "ALL_JOINTS")


@dataclass(frozen=True)
class V2Index:
    """One row of the v2 manifest needed to fetch a trial."""
    trial_id: str
    split: str
    task: str
    h5_path: str
    fail_idx: int
    failure_mode: str
    is_holding: bool


def _load_manifest(path: Path, v2_root: Optional[Path] = None,
                   h5_layout: str = "split") -> list:
    """Load a v2 manifest, optionally rewriting h5_path under a new root.

    Parameters
    ----------
    path
        Manifest CSV path.
    v2_root
        When provided, rewrites the stored absolute ``h5_path`` under this
        new root. Layout depends on ``h5_layout``.
    h5_layout
        Either ``"split"`` (LIBERO: ``v2_root/<split>/<task>.h5``) or
        ``"flat"`` (RoboCasa: ``v2_root/<task>.h5``). Ignored when
        ``v2_root`` is None.
    """
    rows: list = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if v2_root is not None:
                if h5_layout == "split":
                    h5_path = str(v2_root / r["split"] / (r["task"] + ".h5"))
                elif h5_layout == "flat":
                    h5_path = str(v2_root / (r["task"] + ".h5"))
                else:
                    raise ValueError(f"unknown h5_layout {h5_layout!r}")
            else:
                h5_path = r["h5_path"]
            rows.append(V2Index(
                trial_id=r["trial_id"],
                split=r["split"],
                task=r["task"],
                h5_path=h5_path,
                fail_idx=int(r["fail_idx"]),
                failure_mode=r["failure_mode"],
                is_holding=(r["is_holding"].lower() == "true"),
            ))
    return rows


@functools.lru_cache(maxsize=64)
def _cached_reader(h5_path: str) -> V2Reader:
    """Per-process LRU cache of V2Reader handles.

    ``maxsize=64`` comfortably covers the 30 per-task HDF5 files in v2 even
    with all three splits open simultaneously.
    """
    return V2Reader(h5_path)


def failure_mode_onehot(mode: str) -> np.ndarray:
    """5-D one-hot in the canonical order ``_FAILURE_MODES``."""
    v = np.zeros(len(_FAILURE_MODES), dtype=np.float32)
    if mode in _FAILURE_MODES:
        v[_FAILURE_MODES.index(mode)] = 1.0
    return v


class LiberoV2Dataset:
    """A trial-level view of one or more v2 splits.

    Parameters
    ----------
    v2_root
        Directory containing ``<split>/<task>.h5`` + ``<split>/manifest.csv``.
    splits
        Which splits to include. Default: all three.
    use_window
        Include the T-frame window arrays (RGB+depth, state). When ``False``
        a sample dict still has ``pre_*`` single-frame fields.
    use_wrist_cam
        Include wrist-cam streams. Default ``True``.
    use_depth
        Include depth channels. Default ``True``.
    use_settle
        Include settle state trajectory. Default ``False`` (only needed for
        world-model training).
    use_failure_mode
        Include a 5-D one-hot failure_mode feature in the sample.
    keep_keys
        Optional explicit allow-list of payload keys. Overrides the above
        toggles. Use for minimal-IO ablations.
    """

    def __init__(self, v2_root: str | Path,
                 splits: Iterable[str] = ("libero_spatial", "libero_object", "libero_goal"),
                 *, use_window: bool = True, use_wrist_cam: bool = True,
                 use_depth: bool = True, use_settle: bool = False,
                 use_failure_mode: bool = True,
                 keep_keys: Optional[Iterable[str]] = None):
        self.v2_root = Path(v2_root)
        self.splits = tuple(splits)
        self.use_window = use_window
        self.use_wrist_cam = use_wrist_cam
        self.use_depth = use_depth
        self.use_settle = use_settle
        self.use_failure_mode = use_failure_mode
        self._keep = set(keep_keys) if keep_keys is not None else None

        self._index: list = []
        for s in self.splits:
            mp = self.v2_root / s / "manifest.csv"
            if not mp.exists():
                raise FileNotFoundError(f"Missing v2 manifest: {mp}")
            self._index.extend(_load_manifest(mp, v2_root=self.v2_root))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        row = self._index[i]
        reader = _cached_reader(row.h5_path)
        keys = self._payload_keys()
        d = reader.read_trial(row.trial_id, keys=keys)
        d["trial_id"] = row.trial_id
        d["split"] = row.split
        d["task"] = row.task
        if self.use_failure_mode:
            d["failure_mode_onehot"] = failure_mode_onehot(row.failure_mode)
        d["is_holding"] = bool(d.get("is_holding", row.is_holding))
        return d

    # -------- internals --------

    def _payload_keys(self) -> Optional[set]:
        if self._keep is not None:
            return self._keep
        keys = {
            # always-on single frame + goal + identity + contact labels
            "pre_qpos", "pre_qvel", "pre_ee_pos", "pre_gripper_ctrl",
            "pre_target_qpos", "pre_rgb",
            "goal_qpos", "goal_qvel", "goal_ee_pos", "goal_gripper_ctrl",
            "goal_offsets",
            "contact_positions", "contact_forces", "contact_force_world",
            "contact_time", "contact_geom_pairs", "contact_failure_id",
            "impacted_geom_ids",
            "post_agentview_rgb",
            "cam_agentview_pos", "cam_agentview_mat0", "cam_agentview_fovy",
            "cam_agentview_size",
            "failure_joints", "obj_names",
            "obj_pos_pre", "obj_quat_pre", "obj_pos_post", "obj_quat_post",
        }
        if self.use_depth:
            keys |= {"pre_depth", "post_agentview_depth"}
        if self.use_wrist_cam:
            keys |= {"robot0_eye_in_hand_rgb", "post_wrist_rgb",
                     "cam_wrist_pos_window", "cam_wrist_mat0_window",
                     "cam_wrist_fovy", "cam_wrist_size"}
            if self.use_depth:
                keys |= {"robot0_eye_in_hand_depth", "post_wrist_depth"}
        if self.use_window:
            keys |= {"window_frame_idx", "window_qpos", "window_qvel",
                     "window_ee_pos", "window_gripper_ctrl",
                     "window_agentview_rgb"}
            if self.use_depth:
                keys.add("window_agentview_depth")
            if self.use_wrist_cam:
                keys.add("window_wrist_rgb")
                if self.use_depth:
                    keys.add("window_wrist_depth")
        if self.use_settle:
            keys |= {"settle_step_idx", "settle_qpos", "settle_qvel",
                     "settle_gripper_qpos", "settle_obj_pos", "settle_obj_quat"}
        return keys


# --------------------------------------------------------------------------
# Pooled multi-source dataset (LIBERO + RoboCasa)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class V2Source:
    """One v2 corpus root + manifest layout.

    ``name``  — short tag stamped on each sample (``"libero"`` / ``"robocasa"``).
    ``root``  — directory containing per-task HDF5 files (and per-split
                manifests, for LIBERO).
    ``manifests``  — explicit manifest CSV paths to ingest. For LIBERO this is
                ``[root/libero_spatial/manifest.csv, root/libero_object/...]``;
                for RoboCasa it's just ``[root/manifest.csv]``.
    ``h5_layout``  — ``"split"`` (LIBERO) or ``"flat"`` (RoboCasa). Tells the
                manifest loader how to rewrite ``h5_path`` under ``root``.
    """
    name: str
    root: Path
    manifests: tuple
    h5_layout: str = "split"

    @staticmethod
    def libero(v2_root: str | Path,
               splits: Iterable[str] = ("libero_spatial", "libero_object", "libero_goal"),
               ) -> "V2Source":
        v2_root = Path(v2_root)
        return V2Source(
            name="libero",
            root=v2_root,
            manifests=tuple(v2_root / s / "manifest.csv" for s in splits),
            h5_layout="split",
        )

    @staticmethod
    def robocasa(v2_root: str | Path) -> "V2Source":
        v2_root = Path(v2_root)
        return V2Source(
            name="robocasa",
            root=v2_root,
            manifests=(v2_root / "manifest.csv",),
            h5_layout="flat",
        )


class PooledV2Dataset(LiberoV2Dataset):
    """Trial-level dataset over multiple v2 sources (LIBERO + RoboCasa).

    Subclasses :class:`LiberoV2Dataset` to inherit the ``_payload_keys`` /
    ``__getitem__`` logic; only the index construction changes.

    Sample dicts gain a ``"source"`` key (``"libero"`` / ``"robocasa"``) for
    per-source loss weighting or per-source eval splits at training time.
    """

    def __init__(self, sources: Iterable[V2Source],
                 **kwargs):
        # Skip the parent constructor's manifest loading; build _index manually.
        sources = list(sources)
        if not sources:
            raise ValueError("PooledV2Dataset needs at least one source")
        # First source's root used as a placeholder for the parent's v2_root
        # attribute (read by _payload_keys / __getitem__ only via row.h5_path).
        self.v2_root = sources[0].root
        self.splits = tuple()
        self._sources = sources
        self.use_window = kwargs.get("use_window", True)
        self.use_wrist_cam = kwargs.get("use_wrist_cam", True)
        self.use_depth = kwargs.get("use_depth", True)
        self.use_settle = kwargs.get("use_settle", False)
        self.use_failure_mode = kwargs.get("use_failure_mode", True)
        self._keep = (set(kwargs["keep_keys"])
                      if kwargs.get("keep_keys") is not None else None)

        self._index: list = []
        self._source_for_index: list = []  # parallel array of source names
        for src in sources:
            for manifest in src.manifests:
                if not manifest.exists():
                    raise FileNotFoundError(f"Missing manifest: {manifest}")
                rows = _load_manifest(manifest, v2_root=src.root,
                                      h5_layout=src.h5_layout)
                self._index.extend(rows)
                self._source_for_index.extend([src.name] * len(rows))

    def __getitem__(self, i: int) -> dict:
        d = super().__getitem__(i)
        d["source"] = self._source_for_index[i]
        return d
