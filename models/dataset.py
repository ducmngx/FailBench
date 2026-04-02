"""FailBench contact prediction dataset — object-centric design.

Discovers scene objects dynamically from the MuJoCo model. Each object is
represented by spatial features (position, size, type) so the model generalizes
across scenes with different numbers/types of objects.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Failure mode registry (fixed — these are the robot failure types)
# ---------------------------------------------------------------------------

FAILURE_MODE_NAMES = [
    "gripper_open", "slippery_grip",
    "single_joint_j4", "single_joint_j6",
    "multi_joint", "all_joints",
]
NUM_FAILURE_MODES = len(FAILURE_MODE_NAMES)

# Robot body names — used to exclude robot geoms from object discovery
_ROBOT_BODY_NAMES = {
    "world", "link0", "link1", "link2", "link3", "link4",
    "link5", "link6", "link7", "hand", "left_finger", "right_finger",
}

# Object type categories for one-hot encoding
OBJ_TYPES = ["soft_obstacle", "hard_obstacle", "target_object", "other_object", "table"]
NUM_OBJ_TYPES = len(OBJ_TYPES)

# Static feature vector per object: pos(3) + size(3) + type_onehot(5) = 11
OBJ_STATIC_FEAT_DIM = 3 + 3 + NUM_OBJ_TYPES

# Per-sample dynamic features appended at __getitem__ time:
# relative_to_ee(3) + distance_to_ee(1) = 4
OBJ_DYNAMIC_FEAT_DIM = 4

# Total per-object feature dimension
OBJ_FEAT_DIM = OBJ_STATIC_FEAT_DIM + OBJ_DYNAMIC_FEAT_DIM  # 15


# ---------------------------------------------------------------------------
# Scene introspection
# ---------------------------------------------------------------------------


def _classify_object(body_name: str) -> int:
    """Map body name to object type index."""
    if "table" in body_name:
        return OBJ_TYPES.index("table")
    if "soft" in body_name:
        return OBJ_TYPES.index("soft_obstacle")
    if "hard" in body_name:
        return OBJ_TYPES.index("hard_obstacle")
    if body_name.startswith("object"):
        return OBJ_TYPES.index("target_object")
    return OBJ_TYPES.index("other_object")


def discover_scene_objects(model: mujoco.MjModel) -> Tuple[List[dict], set, Dict[int, int]]:
    """Discover all non-robot objects in a MuJoCo scene.

    Returns:
        objects: list of dicts with {name, body_id, geom_ids, position, size, type_idx, features}
        robot_geom_ids: set of robot geom IDs
        geom_to_obj_idx: mapping from geom_id -> index in objects list
    """
    # Find robot bodies and geoms
    robot_body_ids = set()
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in _ROBOT_BODY_NAMES:
            robot_body_ids.add(bid)

    robot_geom_ids = {gid for gid in range(model.ngeom)
                      if model.geom_bodyid[gid] in robot_body_ids}

    # Discover environment objects (group geoms by parent body)
    body_geoms: Dict[int, List[int]] = {}
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if bid in robot_body_ids:
            continue
        # Skip floor (world body, plane geom)
        if bid == 0:
            continue
        body_geoms.setdefault(bid, []).append(gid)

    objects = []
    geom_to_obj_idx = {}

    for bid, gids in sorted(body_geoms.items()):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname is None:
            continue

        # Position: use body position from model
        pos = model.body_pos[bid].copy().astype(np.float32)

        # Size: use the first geom's size (representative)
        size = model.geom_size[gids[0]].copy().astype(np.float32)

        # Type classification
        type_idx = _classify_object(bname)
        type_onehot = np.zeros(NUM_OBJ_TYPES, dtype=np.float32)
        type_onehot[type_idx] = 1.0

        # Feature vector: [pos(3), size(3), type(5)]
        features = np.concatenate([pos, size, type_onehot])

        obj_idx = len(objects)
        for gid in gids:
            geom_to_obj_idx[gid] = obj_idx

        objects.append({
            "name": bname,
            "body_id": bid,
            "geom_ids": gids,
            "position": pos,
            "size": size,
            "type_idx": type_idx,
            "features": features,
        })

    return objects, robot_geom_ids, geom_to_obj_idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FailBenchDataset(Dataset):
    """Object-centric contact prediction dataset.

    Each sample is one (experiment, failure_mode) pair. Objects are discovered
    from the scene model, so the dataset adapts to any scene XML.

    Per sample:
        image:           (3, H, W) float32, ImageNet-normalized
        state:           (11,) float32 — [qpos(7), ee_pos(3), gripper(1)]
        failure_mode:    (6,) float32 one-hot
        obj_features:    (K, 11) float32 — per-object spatial features
        obj_mask:        (K,) bool — which slots are valid (for padding)
        target_hit:      (K,) float32 binary
        target_force:    (K,) float32 — peak force per object (0 if not hit)
        target_centroid: (K, 3) float32 — mean contact position
    """

    def __init__(self, dataset_dir: str, scene_xml_path: str,
                 image_size: Tuple[int, int] = (224, 224),
                 normalize_images: bool = True,
                 max_objects: int = 32):
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.max_objects = max_objects

        # Discover scene objects
        model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.objects, self.robot_geom_ids, self.geom_to_obj_idx = \
            discover_scene_objects(model)
        self.num_objects = len(self.objects)
        self.object_names = [o["name"] for o in self.objects]

        # Build padded static object feature matrix (shared across samples)
        self._obj_static_features = np.zeros((max_objects, OBJ_STATIC_FEAT_DIM), dtype=np.float32)
        self._obj_positions = np.zeros((max_objects, 3), dtype=np.float32)
        for i, obj in enumerate(self.objects):
            self._obj_static_features[i] = obj["features"]
            self._obj_positions[i] = obj["position"]
        self._obj_mask = np.zeros(max_objects, dtype=bool)
        self._obj_mask[:self.num_objects] = True

        # Load all NPZ files
        self.samples: List[dict] = []
        self.has_depth = False
        self.has_ee_cam = False
        npz_files = sorted(self.dataset_dir.glob("exp_*.npz"))
        for npz_path in npz_files:
            data = dict(np.load(npz_path, allow_pickle=True))
            if "pre_depth" in data:
                self.has_depth = True
            if "ee_cam_rgb" in data:
                self.has_ee_cam = True
            for fid in range(len(data["failure_modes"])):
                sample = self._build_sample(data, fid)
                sample["experiment_file"] = npz_path.name
                self.samples.append(sample)

    def _build_sample(self, data: dict, failure_id: int) -> dict:
        mask = data["contact_failure_id"] == failure_id
        positions = data["contact_positions"][mask]
        forces = data["contact_forces"][mask]
        geom_pairs = data["contact_geom_pairs"][mask]

        # Map failure mode string to canonical index
        mode_str = str(data["failure_modes"][failure_id])
        if mode_str == "single_joint":
            fm_idx = 2 if failure_id == 2 else 3
        else:
            fm_idx = {"gripper_open": 0, "slippery_grip": 1,
                      "multi_joint": 4, "all_joints": 5}[mode_str]

        # Per-object targets (sized to max_objects, padded with zeros)
        target_hit = np.zeros(self.max_objects, dtype=np.float32)
        target_force = np.zeros(self.max_objects, dtype=np.float32)
        target_centroid = np.zeros((self.max_objects, 3), dtype=np.float32)
        obj_positions: Dict[int, List[np.ndarray]] = {}
        obj_peak_force: Dict[int, float] = {}

        for i in range(len(positions)):
            g1, g2 = int(geom_pairs[i, 0]), int(geom_pairs[i, 1])
            force_mag = float(np.linalg.norm(forces[i, :3]))

            for gid in (g1, g2):
                if gid in self.robot_geom_ids:
                    continue
                oidx = self.geom_to_obj_idx.get(gid)
                if oidx is None:
                    continue
                target_hit[oidx] = 1.0
                obj_peak_force[oidx] = max(obj_peak_force.get(oidx, 0.0), force_mag)
                if oidx not in obj_positions:
                    obj_positions[oidx] = []
                obj_positions[oidx].append(positions[i])

        for oidx, peak in obj_peak_force.items():
            target_force[oidx] = peak
        for oidx, pos_list in obj_positions.items():
            target_centroid[oidx] = np.mean(pos_list, axis=0)

        state = np.concatenate([
            data["pre_qpos"].astype(np.float32),
            data["pre_ee_pos"].astype(np.float32),
            data["pre_gripper_ctrl"].astype(np.float32),
        ])

        result = {
            "image": data["pre_rgb"],
            "state": state,
            "failure_mode": fm_idx,
            "target_hit": target_hit,
            "target_force": target_force,
            "target_centroid": target_centroid,
        }
        if "pre_depth" in data:
            result["depth"] = data["pre_depth"]
        if "ee_cam_rgb" in data:
            result["ee_cam_rgb"] = data["ee_cam_rgb"]
        if "ee_cam_depth" in data:
            result["ee_cam_depth"] = data["ee_cam_depth"]
        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = self._resize(s["image"])
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.normalize_images:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std

        # Append depth as 4th channel if available
        if "depth" in s:
            depth = self._resize_depth(s["depth"])
            depth_t = torch.from_numpy(depth).unsqueeze(0).float()  # (1, H, W)
            depth_t = depth_t.clamp(0, 5.0) / 5.0
            img = torch.cat([img, depth_t], dim=0)  # (4, H, W)

        # EE camera: process same as overhead (RGBD)
        ee_img = None
        if "ee_cam_rgb" in s:
            ee_img = self._resize(s["ee_cam_rgb"])
            ee_img = torch.from_numpy(ee_img).permute(2, 0, 1).float() / 255.0
            if self.normalize_images:
                ee_img = (ee_img - mean) / std
            if "ee_cam_depth" in s:
                ee_depth = self._resize_depth(s["ee_cam_depth"])
                ee_depth_t = torch.from_numpy(ee_depth).unsqueeze(0).float().clamp(0, 5.0) / 5.0
                ee_img = torch.cat([ee_img, ee_depth_t], dim=0)

        fm_onehot = torch.zeros(NUM_FAILURE_MODES)
        fm_onehot[s["failure_mode"]] = 1.0

        # Dynamic per-object features: relative position + distance to EE
        ee_pos = s["state"][7:10]  # ee_pos is at indices 7,8,9 in state vector
        rel_pos = self._obj_positions - ee_pos  # (K, 3)
        dist = np.linalg.norm(rel_pos, axis=1, keepdims=True)  # (K, 1)

        # Concatenate static + dynamic features
        obj_features = np.concatenate([
            self._obj_static_features, rel_pos, dist
        ], axis=1)  # (K, 15)

        out = {
            "image": img,
            "state": torch.from_numpy(s["state"]),
            "failure_mode": fm_onehot,
            "obj_features": torch.from_numpy(obj_features),
            "obj_mask": torch.from_numpy(self._obj_mask.copy()),
            "target_hit": torch.from_numpy(s["target_hit"]),
            "target_force": torch.from_numpy(s["target_force"]),
            "target_centroid": torch.from_numpy(s["target_centroid"]),
        }
        if ee_img is not None:
            out["ee_image"] = ee_img
        return out

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = self.image_size
        if img.shape[0] == h and img.shape[1] == w:
            return img
        from PIL import Image
        return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))

    def _resize_depth(self, depth: np.ndarray) -> np.ndarray:
        h, w = self.image_size
        if depth.shape[0] == h and depth.shape[1] == w:
            return depth
        from PIL import Image
        pil = Image.fromarray(depth, mode='F').resize((w, h), Image.BILINEAR)
        return np.array(pil, dtype=np.float32)

    def get_experiment_ids(self) -> List[str]:
        return [s["experiment_file"] for s in self.samples]

    def split_by_experiment(self, test_experiments: List[str]):
        test_set = set(test_experiments)
        train_idx = [i for i, s in enumerate(self.samples)
                     if s["experiment_file"] not in test_set]
        test_idx = [i for i, s in enumerate(self.samples)
                    if s["experiment_file"] in test_set]
        return train_idx, test_idx
