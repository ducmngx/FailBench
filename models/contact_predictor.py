"""Object-centric contact prediction model.

Instead of a fixed-size output head tied to a specific scene, each object is
queried independently using its spatial features (position, size, type).
The model generalizes across scenes with different objects.

Architecture:
    Scene encoder:
        image -> ResNet-18 (pretrained) -> 512-d
        [qpos, ee_pos, gripper, failure_mode] -> MLP -> 64-d
        Fused scene context: 576-d

    Object encoder:
        per-object features [pos(3), size(3), type(5)] -> MLP -> 64-d

    Per-object prediction:
        concat(scene_context, obj_features) -> MLP -> (hit_logit, force, cx, cy, cz)
"""

import torch
import torch.nn as nn
import torchvision.models as models

from models.dataset import NUM_FAILURE_MODES, OBJ_FEAT_DIM


class ContactPredictor(nn.Module):
    """Object-centric contact predictor.

    Queries each object independently with shared weights, so the model
    handles any number of objects and transfers across scenes.
    """

    def __init__(self, freeze_backbone: bool = True, in_channels: int = 3,
                 use_ee_camera: bool = False, ee_in_channels: int = 3):
        super().__init__()
        self.use_ee_camera = use_ee_camera

        # --- Visual encoder (ResNet-18, pretrained) ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # If using RGBD (4 channels), extend the first conv layer
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = 0.0
            resnet.conv1 = new_conv

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        if freeze_backbone and in_channels == 3:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif freeze_backbone and in_channels != 3:
            for name, p in self.backbone.named_parameters():
                if "0.weight" not in name:
                    p.requires_grad = False
        self.visual_dim = 512

        # --- EE camera encoder (separate ResNet-18) ---
        if use_ee_camera:
            ee_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            if ee_in_channels != 3:
                old_conv = ee_resnet.conv1
                new_conv = nn.Conv2d(ee_in_channels, old_conv.out_channels,
                                     kernel_size=old_conv.kernel_size,
                                     stride=old_conv.stride,
                                     padding=old_conv.padding, bias=False)
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight
                    new_conv.weight[:, 3:] = 0.0
                ee_resnet.conv1 = new_conv
            self.ee_backbone = nn.Sequential(*list(ee_resnet.children())[:-1])
            if freeze_backbone and ee_in_channels == 3:
                for p in self.ee_backbone.parameters():
                    p.requires_grad = False
            elif freeze_backbone and ee_in_channels != 3:
                for name, p in self.ee_backbone.named_parameters():
                    if "0.weight" not in name:
                        p.requires_grad = False
            self.visual_dim = 512 + 512  # overhead + EE

        # --- State encoder ---
        state_dim = 11 + NUM_FAILURE_MODES  # qpos(7) + ee_pos(3) + gripper(1) + fm(6)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.scene_dim = self.visual_dim + 64  # 576

        # --- Object encoder ---
        # OBJ_FEAT_DIM = 15: pos(3) + size(3) + type(5) + rel_to_ee(3) + dist_to_ee(1)
        self.obj_encoder = nn.Sequential(
            nn.Linear(OBJ_FEAT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.obj_dim = 64

        # --- Per-object prediction head (shared across all objects) ---
        # Input: scene_context(576) + obj_features(64) = 640
        self.prediction_head = nn.Sequential(
            nn.Linear(self.scene_dim + self.obj_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),  # [hit_logit, peak_force, cx, cy, cz]
        )

    def encode_scene(self, image: torch.Tensor, state: torch.Tensor,
                     failure_mode: torch.Tensor,
                     ee_image: torch.Tensor = None) -> torch.Tensor:
        """Encode scene context from image(s) + robot state + failure mode.

        Returns: (B, scene_dim) scene context vector
        """
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            vis = self.backbone(image)
        vis = vis.flatten(1)  # (B, 512)

        # EE camera
        if self.use_ee_camera and ee_image is not None:
            with torch.no_grad() if not any(p.requires_grad for p in self.ee_backbone.parameters()) else torch.enable_grad():
                ee_vis = self.ee_backbone(ee_image)
            ee_vis = ee_vis.flatten(1)  # (B, 512)
            vis = torch.cat([vis, ee_vis], dim=1)  # (B, 1024)

        state_input = torch.cat([state, failure_mode], dim=1)
        state_feat = self.state_encoder(state_input)

        return torch.cat([vis, state_feat], dim=1)

    def forward(self, image, state, failure_mode, obj_features, obj_mask,
                ee_image=None):
        """Predict hit/force/centroid for each object in the scene.

        Args:
            image:        (B, C, H, W) — overhead RGBD
            state:        (B, 11)
            failure_mode: (B, 6) one-hot
            obj_features: (B, K, 15) per-object spatial features
            obj_mask:     (B, K) bool — valid object slots
            ee_image:     (B, C, H, W) optional end-effector camera RGBD

        Returns:
            hit_logits:   (B, K) — per-object hit logits
            force_preds:  (B, K, 4) — [peak_force, cx, cy, cz] per object
        """
        B, K, _ = obj_features.shape

        # Scene context
        scene_ctx = self.encode_scene(image, state, failure_mode, ee_image)

        # Object features: (B, K, 64)
        obj_feat = self.obj_encoder(obj_features)

        # Broadcast scene context to each object: (B, K, 576)
        scene_exp = scene_ctx.unsqueeze(1).expand(B, K, -1)

        # Concatenate and predict: (B, K, 640) -> (B, K, 5)
        combined = torch.cat([scene_exp, obj_feat], dim=2)
        out = self.prediction_head(combined.reshape(B * K, -1)).reshape(B, K, 5)

        hit_logits = out[..., 0]       # (B, K)
        force_preds = out[..., 1:5]    # (B, K, 4) — [force, cx, cy, cz]

        return hit_logits, force_preds
