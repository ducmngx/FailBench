"""HDF5 storage layer for the v2 LIBERO contact-prediction dataset.

One HDF5 file per task; per-trial groups under ``/trials/<trial_id>/``. Schema
is fixed at write time via :data:`V2_SCHEMA_VERSION`; readers can check
``file.attrs['schema_version']`` to gate behavior on future format bumps.

Compression: prefers blosc:lz4 via :mod:`hdf5plugin` (fast decode, good ratio
for u8 RGB / f16 depth). Falls back to h5py's built-in ``lzf`` codec when
hdf5plugin is unavailable. ``lzf`` is always present, comparable speed, ~70%
the ratio of blosc — good enough for development, recommend installing
hdf5plugin for production builds.
"""
from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


V2_SCHEMA_VERSION = 2
DEFAULT_WINDOW_T = 8
DEFAULT_WINDOW_STRIDE = 5
DEFAULT_GOAL_OFFSETS = (5, 15, 30)
DEFAULT_SETTLE_S = 50
DEFAULT_IMAGE_HW = (240, 320)


# --------------------------------------------------------------------------
# Compression filter discovery
# --------------------------------------------------------------------------


def _compression_kwargs() -> dict:
    """Return h5py dataset kwargs for the preferred compression filter.

    Tries blosc:lz4 via hdf5plugin; falls back to lzf. Logged once at import.
    """
    try:
        import hdf5plugin  # type: ignore
        return dict(hdf5plugin.Blosc(cname="lz4", clevel=5,
                                     shuffle=hdf5plugin.Blosc.BITSHUFFLE))
    except Exception:  # pragma: no cover — fallback path
        warnings.warn(
            "hdf5plugin not available — using h5py's built-in lzf filter. "
            "Install hdf5plugin for ~30% better compression at similar speed.",
            stacklevel=2,
        )
        return {"compression": "lzf"}


_COMP = _compression_kwargs()


# --------------------------------------------------------------------------
# Chunk shape helpers
# --------------------------------------------------------------------------


def _img_chunks(shape: tuple) -> tuple:
    """Per-frame chunk for (T, H, W[, C]) image stacks. One frame per chunk."""
    if len(shape) == 4:    # (T, H, W, C)
        return (1, shape[1], shape[2], shape[3])
    if len(shape) == 3:    # (T, H, W) depth
        return (1, shape[1], shape[2])
    return shape


def _contact_chunks(N: int) -> tuple:
    """Chunk for variable-length contact arrays."""
    return (max(1, min(N, 1024)),)


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------


# Datasets we write per trial. (key, dtype-or-None) — dtype None means
# preserve the incoming numpy dtype.
_PER_TRIAL_DATASETS = {
    # window inputs
    "window_frame_idx": np.int32,
    "window_qpos": np.float32,
    "window_qvel": np.float32,
    "window_ee_pos": np.float32,
    "window_gripper_ctrl": np.float32,
    "window_agentview_rgb": np.uint8,
    "window_agentview_depth": np.float16,
    "window_wrist_rgb": np.uint8,
    "window_wrist_depth": np.float16,
    # goal
    "goal_qpos": np.float32,
    "goal_qvel": np.float32,
    "goal_ee_pos": np.float32,
    "goal_gripper_ctrl": np.float32,
    "goal_offsets": np.int32,
    # v1-compat single-frame fields
    "pre_qpos": np.float64,
    "pre_qvel": np.float64,
    "pre_ee_pos": np.float64,
    "pre_gripper_ctrl": np.float64,
    "pre_target_qpos": np.float64,
    "pre_rgb": np.uint8,
    "pre_depth": np.float16,
    "robot0_eye_in_hand_rgb": np.uint8,
    "robot0_eye_in_hand_depth": np.float16,
    # contact arrays
    "contact_positions": np.float32,
    "contact_forces": np.float32,
    "contact_force_world": np.float32,
    "contact_time": np.int32,
    "contact_geom_pairs": np.int32,
    "contact_failure_id": np.int32,
    "impacted_geom_ids": np.int32,
    # post-failure observations (symmetric to pre)
    "post_agentview_rgb": np.uint8,
    "post_agentview_depth": np.float16,
    "post_wrist_rgb": np.uint8,
    "post_wrist_depth": np.float16,
    # camera calibration
    "cam_agentview_pos": np.float64,
    "cam_agentview_mat0": np.float64,
    "cam_agentview_fovy": np.float64,
    "cam_agentview_size": np.int32,
    "cam_wrist_pos_window": np.float64,
    "cam_wrist_mat0_window": np.float64,
    "cam_wrist_fovy": np.float64,
    "cam_wrist_size": np.int32,
    # failure descriptor
    "failure_joints": np.int32,
    # object poses
    "obj_names": None,                 # variable-length str
    "obj_pos_pre": np.float32,
    "obj_quat_pre": np.float32,
    "obj_pos_post": np.float32,
    "obj_quat_post": np.float32,
    # settle state trajectory
    "settle_step_idx": np.int32,
    "settle_qpos": np.float32,
    "settle_qvel": np.float32,
    "settle_gripper_qpos": np.float32,
    "settle_obj_pos": np.float32,
    "settle_obj_quat": np.float32,
}

# Per-trial scalar attrs.
_PER_TRIAL_SCALAR_ATTRS = (
    "trial_id", "split", "task", "demo_key", "seed", "seed_idx", "bin_idx",
    "fail_idx", "traj_progress", "failure_mode", "failure_prob", "is_holding",
    "force_frame", "scene_table_z",
)

# Array-valued attrs.
_PER_TRIAL_ARRAY_ATTRS = (
    "scene_aabb_min", "scene_aabb_max", "robot_geom_ids", "scene_entities_json",
)


class V2Writer:
    """Append-mode HDF5 writer for one task's v2 trials.

    Use as a context manager::

        with V2Writer(path, split="libero_spatial", task=task) as w:
            for trial_id, payload in trials:
                w.write_trial(trial_id, payload)
    """

    def __init__(self, h5_path: Path | str, *,
                 split: Optional[str] = None,
                 task: Optional[str] = None,
                 window_T: int = DEFAULT_WINDOW_T,
                 window_stride: int = DEFAULT_WINDOW_STRIDE,
                 goal_offsets: tuple = DEFAULT_GOAL_OFFSETS,
                 settle_S: int = DEFAULT_SETTLE_S):
        self.h5_path = Path(h5_path)
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self._split = split
        self._task = task
        self._window_T = window_T
        self._window_stride = window_stride
        self._goal_offsets = tuple(int(x) for x in goal_offsets)
        self._settle_S = settle_S
        self._f: Optional[h5py.File] = None

    def __enter__(self):
        self._f = h5py.File(self.h5_path, "a", libver="latest")
        if "schema_version" not in self._f.attrs:
            self._f.attrs["schema_version"] = V2_SCHEMA_VERSION
            if self._split is not None:
                self._f.attrs["split"] = self._split
            if self._task is not None:
                self._f.attrs["task"] = self._task
            self._f.attrs["window_T"] = self._window_T
            self._f.attrs["window_stride"] = self._window_stride
            self._f.attrs["goal_offsets"] = np.asarray(self._goal_offsets, dtype=np.int32)
            self._f.attrs["settle_S"] = self._settle_S
        if "/trials" not in self._f:
            self._f.create_group("/trials")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._f is not None:
            self._f.close()
            self._f = None

    @property
    def file(self) -> h5py.File:
        if self._f is None:
            raise RuntimeError("V2Writer used outside its context manager")
        return self._f

    def has_trial(self, trial_id: str) -> bool:
        return f"trials/{trial_id}" in self.file

    def existing_trial_ids(self) -> set:
        if "trials" not in self.file:
            return set()
        return set(self.file["trials"].keys())

    def write_trial(self, trial_id: str, payload: dict) -> None:
        """Create ``/trials/<trial_id>`` and populate it from ``payload``.

        ``payload`` is a dict mapping schema keys → numpy arrays / scalars.
        Unknown keys are silently ignored (forward-compat). Missing keys are
        skipped (the trial just doesn't have those datasets).
        """
        grp = self.file.require_group(f"trials/{trial_id}")
        # Datasets
        for name, dtype in _PER_TRIAL_DATASETS.items():
            if name not in payload:
                continue
            arr = payload[name]
            if name == "obj_names":
                self._write_str_array(grp, name, arr)
                continue
            if dtype is not None:
                arr = np.asarray(arr).astype(dtype, copy=False)
            else:
                arr = np.asarray(arr)
            kw = dict(_COMP)
            # Don't compress tiny 1-D arrays — overhead beats savings.
            if arr.ndim == 0 or arr.size <= 16:
                grp.create_dataset(name, data=arr)
                continue
            kw["chunks"] = (_img_chunks(arr.shape)
                            if name.startswith(("window_agentview_", "window_wrist_",
                                                "pre_rgb", "pre_depth",
                                                "robot0_eye_in_hand_",
                                                "post_"))
                            else None)
            if kw["chunks"] is None and arr.ndim >= 1 and arr.shape[0] > 64:
                kw["chunks"] = True
            grp.create_dataset(name, data=arr, **kw)
        # Scalar attrs
        for k in _PER_TRIAL_SCALAR_ATTRS:
            if k in payload:
                v = payload[k]
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8")
                grp.attrs[k] = v
        # Array attrs
        for k in _PER_TRIAL_ARRAY_ATTRS:
            if k in payload:
                grp.attrs[k] = payload[k]

    @staticmethod
    def _write_str_array(grp: h5py.Group, name: str, values: Iterable[str]) -> None:
        dt = h5py.string_dtype(encoding="utf-8")
        grp.create_dataset(name, data=np.asarray(list(values), dtype=object), dtype=dt)


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


class V2Reader:
    """Read-only accessor for a per-task v2 HDF5.

    Holds the file open for the reader's lifetime. Designed for use inside a
    per-worker DataLoader cache — one reader per (worker, h5 file).
    """

    def __init__(self, h5_path: Path | str):
        self.h5_path = Path(h5_path)
        self._f = h5py.File(self.h5_path, "r", libver="latest", swmr=True)

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @property
    def schema_version(self) -> int:
        return int(self._f.attrs.get("schema_version", -1))

    @property
    def split(self) -> Optional[str]:
        v = self._f.attrs.get("split")
        return v.decode("utf-8") if isinstance(v, bytes) else (str(v) if v is not None else None)

    @property
    def task(self) -> Optional[str]:
        v = self._f.attrs.get("task")
        return v.decode("utf-8") if isinstance(v, bytes) else (str(v) if v is not None else None)

    def trial_ids(self) -> list:
        return sorted(self._f["trials"].keys()) if "trials" in self._f else []

    def read_trial(self, trial_id: str,
                   keys: Optional[Iterable[str]] = None) -> dict:
        """Load all (or selected) datasets + attrs for one trial.

        Returns a dict of numpy arrays + scalar attrs. Datasets that don't
        exist in this trial group are silently omitted.
        """
        grp = self._f[f"trials/{trial_id}"]
        out: dict = {}
        ds_keys = set(_PER_TRIAL_DATASETS.keys()) if keys is None else set(keys)
        for k in ds_keys:
            if k in grp:
                ds = grp[k]
                if h5py.check_string_dtype(ds.dtype) is not None:
                    out[k] = [
                        s.decode("utf-8") if isinstance(s, bytes) else s
                        for s in ds[:]
                    ]
                else:
                    out[k] = ds[()]
        for k in _PER_TRIAL_SCALAR_ATTRS + _PER_TRIAL_ARRAY_ATTRS:
            if k in grp.attrs:
                v = grp.attrs[k]
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                out[k] = v
        return out


@contextmanager
def open_reader(h5_path: Path | str) -> Iterator[V2Reader]:
    r = V2Reader(h5_path)
    try:
        yield r
    finally:
        r.close()
