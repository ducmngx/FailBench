"""LIBERO contact-distribution training dataset.

Pairs the per-split ``labels.npz`` (produced by ``scripts/libero/build_full_labels.py``)
with the corresponding v1 trial npzs. Train/val splits are by *group*
(``(split, task, demo_key, bin_idx)`` tuple) so sibling failure trials never
cross the train/val boundary.

Schema of one item (``__getitem__`` returns a dict):

    rgb               (3, H, W) float32   ImageNet-normalized if normalize_rgb
    depth             (1, H, W) float32   raw metres (LIBERO depth is metric)
    state             (14,)     float32   pre_qpos[7] ⊕ pre_qvel[7]
    target_mass       (1, H, W) float32   log1p(aggregated mass), promoted from f16
    target_depth      (1, H, W) float32   aggregated mean depth, metres
    target_mass_total ()        float32   per-group scalar (auxiliary head)
    group_id          ()        int64     (only if return_meta)
    experiment_id     str                 (only if return_meta)
    split             str                 (only if return_meta)

Memory: all 3 splits' labels (target_mass + target_depth) total ~9 GB at
float16; that's kept resident in RAM. The target arrays cast to float32 only
inside ``__getitem__`` for the sliced group, so peak RAM stays roughly at the
float16 footprint.

Original trial npzs are read lazily and never modified.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V1_ROOT = REPO_ROOT / "datasets" / "libero" / "v1"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class _Row:
    npz_path: Path
    group_id: int       # index into self.target_mass / self.target_depth
    split: str
    experiment_id: str


def _parse_group_key(gk: str) -> Tuple[str, str, str, int]:
    """``"<split>/<task>/<demo>/binN"`` → (split, task, demo, bin_idx)."""
    parts = gk.split("/")
    if len(parts) != 4 or not parts[-1].startswith("bin"):
        raise ValueError(f"unexpected group key format: {gk!r}")
    return parts[0], parts[1], parts[2], int(parts[3][3:])


class LiberoLabelDataset(Dataset):
    """Map-style dataset for LIBERO contact-heatmap training.

    Parameters
    ----------
    v1_root : Path
        Root of ``datasets/libero/v1/``. Each split sub-dir must contain
        ``manifest.csv`` and ``labels.npz``.
    splits : tuple of str
        Subset of splits to include. Default: all three.
    image_size : (H, W) or None
        If set, RGB / depth / targets are bilinear-resized to this resolution
        on every ``__getitem__``. Default None keeps native 480×640.
    normalize_rgb : bool
        ImageNet mean/std normalization on RGB (default True).
    return_meta : bool
        Include ``group_id`` / ``experiment_id`` / ``split`` in the item dict.
    """

    def __init__(self,
                 v1_root: Path | str = DEFAULT_V1_ROOT,
                 splits: Sequence[str] = ("libero_spatial",),
                 image_size: Optional[Tuple[int, int]] = None,
                 normalize_rgb: bool = True,
                 return_meta: bool = False,
                 cache_memmap: bool = False,
                 mass_total_scale: float = 1000.0,
                 use_holding: bool = True,
                 per_trial: bool = False):
        """
        Parameters
        ----------
        cache_memmap : bool
            If True, extract ``target_mass`` and ``target_depth`` from the
            compressed ``labels.npz`` into uncompressed ``.npy`` sidecars at
            ``<split>/labels_memmap/`` on first use, and memmap them.
            This trades ~6 GB disk per split for ~zero RAM cost — required
            when loading multiple splits at once.
        mass_total_scale : float
            Divide ``target_mass_total`` by this scale so the scalar target is
            O(1). Raw values are sums of log1p mass over the H×W grid, typically
            in the 10²–10⁴ range. The default 1000.0 puts most targets in
            [0.1, 10], which keeps the auxiliary head's loss compatible with
            the primary spatial loss. Set to 1.0 to disable.
        """
        self.v1_root = Path(v1_root)
        self.splits = list(splits)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.normalize_rgb = normalize_rgb
        self.return_meta = return_meta
        self.cache_memmap = cache_memmap
        self.mass_total_scale = float(mass_total_scale)
        self.use_holding = bool(use_holding)
        self.per_trial = bool(per_trial)
        self._is_holding: dict[str, int] = {}
        if self.per_trial:
            self._init_per_trial()
            return

        # --- Load + concatenate per-split labels -------------------------------
        # When cache_memmap=False we load the whole array into RAM (default,
        # safe only for 1 split). When True we materialise per-split .npy
        # sidecars on disk and use np.memmap so peak RAM stays tiny.
        masses, depths, totals, group_keys = [], [], [], []
        split_per_group = []
        for split in self.splits:
            labels_path = self.v1_root / split / "labels.npz"
            if not labels_path.exists():
                raise FileNotFoundError(
                    f"labels.npz not found for split={split} at {labels_path}; "
                    "run scripts/libero/build_full_labels.py first.")

            if cache_memmap:
                m_arr, d_arr = _load_memmap_split(labels_path)
            else:
                with np.load(labels_path) as d:
                    m_arr = d["target_mass"][:]
                    d_arr = d["target_depth"][:]
            masses.append(m_arr)
            depths.append(d_arr)

            with np.load(labels_path) as d:
                totals.append(d["target_mass_total"][:])
                group_keys.append(d["group_keys"][:])
            split_per_group.extend([split] * len(masses[-1]))

        # Memmaps can be stacked as a list — concatenation would force a copy,
        # defeating the point. Keep per-split arrays and route through a single
        # offset table so __getitem__ can index uniformly.
        self._split_arrays_mass = masses
        self._split_arrays_depth = depths
        self._split_offsets = np.cumsum([0] + [len(m) for m in masses])  # len = n_splits+1

        self.target_mass_total = np.concatenate(totals)                  # (G,) f32
        self.group_keys = np.concatenate(group_keys).astype(np.str_)
        self.split_per_group = np.array(split_per_group)
        self.n_groups = len(self.group_keys)
        H, W = masses[0].shape[1:]
        self.native_hw = (H, W)

        # --- (split, task, demo, bin) → global group_id lookup -----------------
        self._key_to_gid: dict[Tuple[str, str, str, int], int] = {}
        for gid, gk in enumerate(self.group_keys):
            self._key_to_gid[_parse_group_key(str(gk))] = gid

        # --- Walk manifests, build per-trial rows ------------------------------
        rows: list[_Row] = []
        missing = 0
        for split in self.splits:
            df = pd.read_csv(self.v1_root / split / "manifest.csv")
            for r in df.itertuples():
                key = (split, r.task, r.demo_key, int(r.bin_idx))
                gid = self._key_to_gid.get(key)
                if gid is None:
                    missing += 1
                    continue
                rows.append(_Row(
                    npz_path=self.v1_root / split / r.npz_file,
                    group_id=gid,
                    split=split,
                    experiment_id=r.experiment_id,
                ))
        if missing:
            print(f"[LiberoLabelDataset] WARN: {missing} manifest rows had no matching group")
        self.rows = rows

        # --- Load holding sidecars (one per split, optional) -------------------
        if self.use_holding:
            n_loaded = 0
            for split in self.splits:
                holding_path = self.v1_root / split / "holding.csv"
                if not holding_path.exists():
                    print(f"[LiberoLabelDataset] WARN: no {holding_path.name} for "
                          f"split={split}; defaulting is_holding=0. Run "
                          f"scripts/libero/compute_holding_flag.py to generate.")
                    continue
                h = pd.read_csv(holding_path)
                for eid, val in zip(h["experiment_id"].values, h["is_holding"].values):
                    self._is_holding[str(eid)] = int(val)
                n_loaded += len(h)
            if n_loaded > 0:
                hits = sum(1 for r in self.rows if r.experiment_id in self._is_holding)
                rate = (sum(self._is_holding[r.experiment_id] for r in self.rows
                            if r.experiment_id in self._is_holding) / max(hits, 1))
                print(f"[LiberoLabelDataset] loaded is_holding for {hits}/{len(self.rows)} "
                      f"trials (rate={rate:.3f})")

        # --- Normalization buffers --------------------------------------------
        self._rgb_mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._rgb_std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    # ----------------------------------------------------------- per_trial init
    def _init_per_trial(self):
        """Load per-trial labels from <split>/labels_per_trial.npz."""
        masses, depths, totals = [], [], []
        exp_ids, group_keys = [], []
        failure_mode_ids, is_holdings_arr = [], []
        tasks, demo_keys, bin_idxs, splits_per_trial = [], [], [], []
        fm_names_global = None
        for split in self.splits:
            p = self.v1_root / split / "labels_per_trial.npz"
            if not p.exists():
                raise FileNotFoundError(
                    f"labels_per_trial.npz not found for split={split} at {p}; "
                    "run scripts/libero/build_per_trial_labels.py first.")
            with np.load(p) as d:
                masses.append(d["target_mass"][:])
                depths.append(d["target_depth"][:])
                totals.append(d["target_mass_total"][:])
                exp_ids.append(d["experiment_id"][:])
                group_keys.append(d["group_keys"][:])  # G entries, not N
                failure_mode_ids.append(d["failure_mode_id"][:])
                is_holdings_arr.append(d["is_holding"][:])
                tasks.append(d["task"][:])
                demo_keys.append(d["demo_key"][:])
                bin_idxs.append(d["bin_idx"][:])
                if fm_names_global is None:
                    fm_names_global = d["failure_mode_names"][:].astype(np.str_)
            splits_per_trial.extend([split] * len(masses[-1]))

        self._split_arrays_mass = masses
        self._split_arrays_depth = depths
        self._split_offsets = np.cumsum([0] + [len(m) for m in masses])
        self.target_mass_total = np.concatenate(totals)
        self.experiment_ids = np.concatenate(exp_ids).astype(np.str_)
        self.failure_mode_id = np.concatenate(failure_mode_ids).astype(np.int32)
        self.is_holding_arr = np.concatenate(is_holdings_arr).astype(np.int32)
        self.task_arr = np.concatenate(tasks).astype(np.str_)
        self.demo_key_arr = np.concatenate(demo_keys).astype(np.str_)
        self.bin_idx_arr = np.concatenate(bin_idxs).astype(np.int32)
        self.split_per_trial = np.array(splits_per_trial, dtype=object)
        self.failure_mode_names = fm_names_global
        self.n_failure_modes = len(fm_names_global)

        # Cross-split global group_id (for train/val split): hash group keys
        # uniquely across splits.
        per_split_keys = [
            np.array([f"{splits_per_trial[i + self._split_offsets[s]]}/{gk}"
                      for i, gk in enumerate(np.concatenate(group_keys[s:s+1]))], dtype=object)
            for s in range(len(self.splits))
        ]
        # Simpler: directly tag each split's group_keys with its split.
        tagged = []
        for s, split in enumerate(self.splits):
            tagged.extend(f"{split}/{g}" for g in group_keys[s])
        self.group_keys_global = np.array(tagged, dtype=np.str_)

        # Per-trial group_id: each trial's group is its (split, task, demo, bin).
        per_trial_group_keys = np.array([
            f"{self.split_per_trial[i]}/libero_{i // 9999}/dummy"
            for i in range(len(self.experiment_ids))
        ], dtype=object)  # placeholder, overwritten below
        # Real per-trial group_id: re-derive from (split, task, demo, bin).
        seen: dict[Tuple[str, str, str, int], int] = {}
        trial_group_id = np.zeros(len(self.experiment_ids), dtype=np.int32)
        for i in range(len(self.experiment_ids)):
            key = (self.split_per_trial[i], str(self.task_arr[i]),
                   str(self.demo_key_arr[i]), int(self.bin_idx_arr[i]))
            if key not in seen:
                seen[key] = len(seen)
            trial_group_id[i] = seen[key]
        self.trial_group_id = trial_group_id
        self.n_groups = len(seen)
        H, W = masses[0].shape[1:]
        self.native_hw = (H, W)

        # Build the rows list (per-trial in per_trial mode).
        rows: list[_Row] = []
        missing = 0
        manifest_lookup: dict[str, Tuple[str, str]] = {}
        for split in self.splits:
            mdf = pd.read_csv(self.v1_root / split / "manifest.csv")
            for r in mdf.itertuples():
                manifest_lookup[r.experiment_id] = (split, r.npz_file)
        for i, eid in enumerate(self.experiment_ids):
            if eid not in manifest_lookup:
                missing += 1
                continue
            split, npz_file = manifest_lookup[eid]
            rows.append(_Row(
                npz_path=self.v1_root / split / npz_file,
                group_id=int(self.trial_group_id[i]),
                split=split,
                experiment_id=str(eid),
            ))
        if missing:
            print(f"[LiberoLabelDataset] WARN: per_trial mode, {missing} trials had no manifest entry")
        self.rows = rows
        # `_trial_idx_for_row[i]` = index into the per-trial label arrays for rows[i].
        # rows[] order matches experiment_ids[] order (we built them in lockstep above).
        self._trial_idx_for_row = list(range(len(rows)))

        # Holding values come from the labels file in per_trial mode.
        for i, eid in enumerate(self.experiment_ids):
            self._is_holding[str(eid)] = int(self.is_holding_arr[i])

        # Normalization buffers
        self._rgb_mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._rgb_std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

        print(f"[LiberoLabelDataset] per_trial: {len(self.rows)} trials across "
              f"{self.n_groups} groups, {self.n_failure_modes} failure modes")

    # ----------------------------------------------------------------- __len__
    def __len__(self) -> int:
        return len(self.rows)

    # ------------------------------------------------------------- __getitem__
    def __getitem__(self, idx: int) -> dict:
        if self.per_trial:
            return self._getitem_per_trial(idx)
        row = self.rows[idx]
        with np.load(row.npz_path) as d:
            pre_rgb = d["pre_rgb"]              # (H, W, 3) uint8
            pre_depth = d["pre_depth"]           # (H, W) float32
            pre_qpos = d["pre_qpos"]             # (7,) float64
            pre_qvel = d["pre_qvel"]             # (7,) float64

        # Inputs
        rgb = torch.from_numpy(pre_rgb).permute(2, 0, 1).float() / 255.0
        if self.normalize_rgb:
            rgb = (rgb - self._rgb_mean) / self._rgb_std
        depth = torch.from_numpy(pre_depth.copy()).unsqueeze(0).float()  # (1, H, W)
        state = torch.from_numpy(
            np.concatenate([pre_qpos, pre_qvel]).astype(np.float32))     # (14,)

        # Targets — locate the correct per-split array via the offset table.
        gid = row.group_id
        split_idx = int(np.searchsorted(self._split_offsets[1:], gid, side="right"))
        local_gid = gid - int(self._split_offsets[split_idx])
        target_mass = torch.from_numpy(
            np.asarray(self._split_arrays_mass[split_idx][local_gid], dtype=np.float32)
        ).unsqueeze(0)       # (1, H, W)
        target_depth = torch.from_numpy(
            np.asarray(self._split_arrays_depth[split_idx][local_gid], dtype=np.float32)
        ).unsqueeze(0)
        target_mass_total = torch.tensor(
            float(self.target_mass_total[gid]) / self.mass_total_scale,
            dtype=torch.float32)

        # Optional resize. Note: bilinear-resizing log1p targets is an
        # approximation (mass-preserving resize would be expm1→avgpool→log1p);
        # acceptable for prototype since pred + target are resized identically.
        if self.image_size is not None and (rgb.shape[1], rgb.shape[2]) != self.image_size:
            h, w = self.image_size
            rgb = _resize(rgb, (h, w))
            depth = _resize(depth, (h, w))
            target_mass = _resize(target_mass, (h, w))
            target_depth = _resize(target_depth, (h, w))

        is_holding = torch.tensor(
            float(self._is_holding.get(row.experiment_id, 0)),
            dtype=torch.float32)

        out = {
            "rgb": rgb,
            "depth": depth,
            "state": state,
            "is_holding": is_holding,
            "target_mass": target_mass,
            "target_depth": target_depth,
            "target_mass_total": target_mass_total,
        }
        if self.return_meta:
            out["group_id"] = gid
            out["experiment_id"] = row.experiment_id
            out["split"] = row.split
        return out

    # ----------------------------------------------------- per-trial __getitem__
    def _getitem_per_trial(self, idx: int) -> dict:
        """Per-trial mode: labels indexed by trial, not group."""
        row = self.rows[idx]
        trial_idx = self._trial_idx_for_row[idx]
        with np.load(row.npz_path) as d:
            pre_rgb = d["pre_rgb"]
            pre_depth = d["pre_depth"]
            pre_qpos = d["pre_qpos"]
            pre_qvel = d["pre_qvel"]

        rgb = torch.from_numpy(pre_rgb).permute(2, 0, 1).float() / 255.0
        if self.normalize_rgb:
            rgb = (rgb - self._rgb_mean) / self._rgb_std
        depth = torch.from_numpy(pre_depth.copy()).unsqueeze(0).float()
        state = torch.from_numpy(
            np.concatenate([pre_qpos, pre_qvel]).astype(np.float32))

        # Labels: route via per-split offsets.
        split_idx = int(np.searchsorted(self._split_offsets[1:], trial_idx, side="right"))
        local_i = trial_idx - int(self._split_offsets[split_idx])
        target_mass = torch.from_numpy(
            np.asarray(self._split_arrays_mass[split_idx][local_i], dtype=np.float32)
        ).unsqueeze(0)
        target_depth = torch.from_numpy(
            np.asarray(self._split_arrays_depth[split_idx][local_i], dtype=np.float32)
        ).unsqueeze(0)
        target_mass_total = torch.tensor(
            float(self.target_mass_total[trial_idx]) / self.mass_total_scale,
            dtype=torch.float32)

        if self.image_size is not None and (rgb.shape[1], rgb.shape[2]) != self.image_size:
            h, w = self.image_size
            rgb = _resize(rgb, (h, w))
            depth = _resize(depth, (h, w))
            target_mass = _resize(target_mass, (h, w))
            target_depth = _resize(target_depth, (h, w))

        is_holding = torch.tensor(
            float(self.is_holding_arr[trial_idx]), dtype=torch.float32)
        fm_id = int(self.failure_mode_id[trial_idx])
        failure_onehot = torch.zeros(self.n_failure_modes, dtype=torch.float32)
        if 0 <= fm_id < self.n_failure_modes:
            failure_onehot[fm_id] = 1.0

        out = {
            "rgb": rgb,
            "depth": depth,
            "state": state,
            "is_holding": is_holding,
            "failure_mode_id": torch.tensor(fm_id, dtype=torch.long),
            "failure_onehot": failure_onehot,
            "target_mass": target_mass,
            "target_depth": target_depth,
            "target_mass_total": target_mass_total,
        }
        if self.return_meta:
            out["group_id"] = row.group_id
            out["experiment_id"] = row.experiment_id
            out["split"] = row.split
        return out

    # ----------------------------------------------------------- split helpers
    def train_val_split(self, val_frac: float = 0.1, seed: int = 0
                        ) -> Tuple[list[int], list[int]]:
        """Group-level train/val split.

        Returns (train_indices, val_indices) into ``self.rows`` such that no
        group has trials in both sets.
        """
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_groups)
        n_val = max(1, int(round(self.n_groups * val_frac)))
        val_groups = set(int(g) for g in perm[:n_val])
        train_groups = set(int(g) for g in perm[n_val:])
        train_idx, val_idx = [], []
        for i, r in enumerate(self.rows):
            if r.group_id in train_groups:
                train_idx.append(i)
            elif r.group_id in val_groups:
                val_idx.append(i)
        return train_idx, val_idx

    def group_id_for_row(self, idx: int) -> int:
        return self.rows[idx].group_id


def _resize(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinear-resize a (C, H, W) tensor to (C, h, w)."""
    return F.interpolate(x.unsqueeze(0), size=hw, mode="bilinear",
                         align_corners=False).squeeze(0)


def _load_memmap_split(labels_npz: Path) -> Tuple[np.memmap, np.memmap]:
    """Materialise ``target_mass`` and ``target_depth`` as uncompressed .npy
    sidecars and memmap them. One-time disk cost ~6 GB per split.

    Sidecars live at ``<split>/labels_memmap/{target_mass,target_depth}.npy``.
    Original ``labels.npz`` is never modified.
    """
    sidecar_dir = labels_npz.parent / "labels_memmap"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    mass_npy = sidecar_dir / "target_mass.npy"
    depth_npy = sidecar_dir / "target_depth.npy"
    if not (mass_npy.exists() and depth_npy.exists()):
        print(f"[LiberoLabelDataset] materialising memmap sidecars at {sidecar_dir} "
              "(one-time, ~6 GB per split)...")
        with np.load(labels_npz) as d:
            np.save(mass_npy, d["target_mass"])
            np.save(depth_npy, d["target_depth"])
    return (np.load(mass_npy, mmap_mode="r"),
            np.load(depth_npy, mmap_mode="r"))


# ---------------------------------------------------------------- smoke main
if __name__ == "__main__":
    import argparse, time

    ap = argparse.ArgumentParser()
    ap.add_argument("--v1_root", type=Path, default=DEFAULT_V1_ROOT)
    ap.add_argument("--splits", nargs="+", default=["libero_spatial"])
    ap.add_argument("--image_size", nargs=2, type=int, default=None,
                    help="If set, resize H W")
    ap.add_argument("--cache_memmap", action="store_true",
                    help="Use memmap sidecars (required for multi-split).")
    ap.add_argument("--n_iter", type=int, default=50)
    args = ap.parse_args()

    print(f"loading dataset from {args.v1_root} splits={args.splits}  "
          f"memmap={args.cache_memmap} ...")
    t0 = time.time()
    ds = LiberoLabelDataset(
        v1_root=args.v1_root,
        splits=args.splits,
        image_size=tuple(args.image_size) if args.image_size else None,
        return_meta=True,
        cache_memmap=args.cache_memmap,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")
    print(f"  groups: {ds.n_groups}  trials: {len(ds)}  native_hw: {ds.native_hw}")
    total_bytes = sum(a.nbytes for a in ds._split_arrays_mass) + \
                  sum(a.nbytes for a in ds._split_arrays_depth)
    storage = "memmap (disk)" if args.cache_memmap else "in-RAM"
    print(f"  label arrays: {total_bytes/1e9:.2f} GB ({storage})")

    print("\nshapes of first item:")
    item = ds[0]
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} "
                  f"range=[{float(v.min()):.3f}, {float(v.max()):.3f}]")
        else:
            print(f"  {k}: {v}")

    # Group-disjoint train/val split.
    train_idx, val_idx = ds.train_val_split(val_frac=0.1, seed=0)
    train_groups = {ds.rows[i].group_id for i in train_idx}
    val_groups = {ds.rows[i].group_id for i in val_idx}
    overlap = train_groups & val_groups
    print(f"\ntrain trials: {len(train_idx)}  val trials: {len(val_idx)}")
    print(f"train groups: {len(train_groups)}  val groups: {len(val_groups)}  "
          f"overlap: {len(overlap)}")
    assert not overlap, "train and val groups overlap!"

    # Throughput.
    print(f"\ntiming {args.n_iter} __getitem__ calls ...")
    t0 = time.time()
    for i in range(args.n_iter):
        _ = ds[i]
    rate = args.n_iter / (time.time() - t0)
    print(f"  {rate:.1f} items/s sequential (use DataLoader num_workers for parallelism)")
