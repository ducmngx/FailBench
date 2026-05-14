# LIBERO contact-heatmap model

A small U-Net-style RGB-D model that predicts the agentview cam-projected
contact heatmap and its companion mean-depth map from a single pre-failure
observation. Trains against the aggregated labels produced by
`scripts/libero/build_full_labels.py`; consumed via
`planner/risk/libero_dataset.py`.

Lives in `planner/risk/libero_model.py` as `LiberoHeatmapModel`. Aim is a
minimum-viable image-conditioned baseline that we can evaluate end-to-end
quickly, then iterate.

## What it predicts

For each pre-failure frame, the model outputs three things:

| Key | Shape | Meaning |
|---|---|---|
| `mass` | `(B, 1, H, W)` | `log1p(aggregated agentview-projected contact mass)` |
| `depth` | `(B, 1, H, W)` | aggregated mean depth of contacts in metres |
| `mass_total` | `(B,)` | scalar — total log1p mass per image (auxiliary head) |

`mass` is the planner-relevant signal: it's the expected force-weighted
contact density at the agentview pixel grid, taken over the dataset's failure
distribution at that pre-state. `depth` disambiguates occlusion (e.g. cabinet
vs table at the same pixel). `mass_total` is a per-image regression target
that gives the model a global "how much will go wrong" signal in addition to
the dense map.

## What it consumes

| Key | Shape | Notes |
|---|---|---|
| `rgb` | `(B, 3, H, W)` | ImageNet-normalised (the dataset class does this) |
| `depth` | `(B, 1, H, W)` | raw metres from the LIBERO renderer |
| `state` | `(B, 14)` | `pre_qpos` ⊕ `pre_qvel`; only when `use_state=True` |
| `is_holding` | `(B,)` | 0/1 — gripper is in commanded-close mode at `fail_idx`. Default on (`use_holding=True`). Derived offline from the LIBERO demo HDF5 by `scripts/libero/compute_holding_flag.py` and loaded by the dataset from `<split>/holding.csv`. |

Native LIBERO resolution is 480×640; the dataset class can resize to e.g.
240×320 to halve memory at the cost of some spatial detail.

## Architecture

ResNet-18 encoder, U-Net decoder with skip connections, optional broadcast-
tiled state feature at the bottleneck. ~14 M parameters total.

```
rgb (B, 3, H, W)
  ⊕  ─────────► conv1 (4→64, k7s2) ─► s0 (64, H/2)
depth (B,1,H,W)                            │
                                          maxpool
                                           │
                              ┌── layer1 ─►s1 (64, H/4)
                              │                │
                              │            layer2 ─►s2 (128, H/8)
                              │                            │
                              │                       layer3 ─►s3 (256, H/16)
                              │                                       │
                              │                                  layer4 ─►s4 (512, H/32)
                              │                                              │
                              │                            ┌─ state MLP ──┐  │  (optional)
                              │                            │  14 → 64     │  │
                              │                            └──── tile ────┴──┤
                              │                                              ▼
                              │                                       (512+64, H/32)
                              │                                              │
                              │                                          up4 (×2)
                              │                                       ┌──────┴
                              │                              concat s3│
                              │                                       │
                              │                                   up3 (×2)
                              │                                       │
                              │                              concat s2│
                              │                                       │
                              │                                   up2 (×2)
                              │                                       │
                              │                              concat s1│
                              │                                       │
                              │                                   up1 (×2)
                              │                                       │
                              └──────────────────────── concat s0─────┤
                                                                      │
                                                                final_up (×2)
                                                                      │
                                                                 1×1 conv (32→2)
                                                                      │
                                                          mass (1, H, W), depth (1, H, W)

bottleneck features ─► AdaptiveAvgPool2d(1) ─► Linear(512+F → 1) ─► mass_total
```

### Why these choices

- **ResNet-18 + ImageNet weights** — small enough to train on a single GPU,
  pretrained features matter at our dataset size (~15 k aggregated labels).
- **4-channel first conv** — RGB+depth fused at the very first layer; the
  depth channel weight is initialised to the mean of the RGB channels (a
  standard transfer-learning trick for adding a 4th channel).
- **U-Net decoder with skip connections** — preserves spatial detail; standard
  recipe for dense prediction at the input resolution.
- **State head off by default** — Round-4 learnability analysis showed that
  state-only inputs can't beat the constant baseline. The image stream is
  the load-bearing input; state is an opt-in for ablations.
- **Bottleneck-state fusion (not early fusion)** — when state is enabled it
  joins as a 64-D feature broadcast-tiled to the (H/32, W/32) bottleneck.
  Cheap, and lets the encoder do its own thing on the image without state
  contaminating low-level features.
- **Auxiliary scalar head** — `mass_total` regression from a global-avg-pool
  of the bottleneck. Free signal that gives the optimiser a denser loss
  surface than the spatial head alone.

## Loss

Composite training loss, called via `LiberoHeatmapModel.loss(pred, batch)`:

```
L = MSE(mass)                                          # primary
  + 0.1 · masked-MSE(depth, mask = target_mass > 1e-3) # secondary
  + 0.01 · MSE(mass_total)                             # auxiliary
```

- **Primary** — straight MSE on `log1p` mass. The log1p compression (applied
  upstream in the labels file) handles the 0–10⁴ raw dynamic range.
- **Secondary** — masked MSE on depth. The mask suppresses gradient at pixels
  where the target has no contact mass (depth is undefined there). Without
  the mask, the depth head would be punished for "wrong" zero-depth pixels
  which carry no information.
- **Auxiliary** — small-weighted regression on total log1p mass. Stops the
  bottleneck features from collapsing on samples where the spatial signal is
  faint.

Loss weights are CLI knobs (`depth_weight`, `mass_total_weight`); the defaults
match the plan and are a reasonable starting point.

## Parameter count

| Variant | Trainable | Total |
|---|---|---|
| Image-only (`use_state=False`) | 14.32 M | 14.32 M |
| With state head (`use_state=True`) | 14.54 M | 14.54 M |

Fits comfortably on a 12 GB GPU at batch 16, image size 240×320.

## How to use

```python
from planner.risk.libero_dataset import LiberoLabelDataset
from planner.risk.libero_model import LiberoHeatmapModel
from torch.utils.data import DataLoader, Subset

# 1. Build the dataset (single split by default; multi-split needs cache_memmap).
ds = LiberoLabelDataset(splits=("libero_spatial",), image_size=(240, 320))

# 2. Group-disjoint train/val split.
train_idx, val_idx = ds.train_val_split(val_frac=0.1, seed=0)
train_loader = DataLoader(Subset(ds, train_idx), batch_size=16, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(Subset(ds, val_idx), batch_size=16, num_workers=2)

# 3. Build the model.
model = LiberoHeatmapModel(use_state=False).cuda()

# 4. Training step.
for batch in train_loader:
    batch = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in batch.items()}
    pred = model(batch["rgb"], batch["depth"])
    loss, components = model.loss(pred, batch)
    loss.backward()
    # ... optimizer.step() etc.
```

A self-contained smoke test (random tensors, no dataset) is in the module's
`__main__`:

```bash
python -m planner.risk.libero_model            # image-only
python -m planner.risk.libero_model --use_state
```

It prints parameter count, forward/backward timing, output shapes, and per-
component loss values.

## Notes and gotchas

- **Bilinear resize of log1p targets is approximate.** When the dataset
  resizes the target image, it bilinear-interpolates log1p values rather than
  mass-conservative downsampling (which would be `expm1 → avgpool → log1p`).
  Predictions and targets are resized identically, so training is consistent;
  but evaluation against the native-resolution ground truth via this pipeline
  should be careful.
- **The depth head's predictions are unconstrained.** The output is linear
  (no activation), so the model can predict negative depth. The masked loss
  steers it toward physically reasonable values at high-mass pixels;
  low-mass pixels are free to drift and shouldn't be interpreted.
- **`mass_total` targets are large** (~10³ in our data). The default aux
  weight of 0.01 keeps it from dominating. If the aux head fights the
  primary, normalise `target_mass_total` outside the model or drop the
  weight further.
- **No data augmentation.** Standard image augmentations (flip, jitter, etc.)
  would need to be applied jointly to RGB+depth+target maps, since the
  target is geometrically tied to the input. Worth adding once a baseline is
  established.

## Status and next steps

- [x] Model class, loss, smoke test, parameter count.
- [ ] Training loop (`scripts/libero/train_libero_heatmap.py`) — next file.
- [ ] Single-split run on `libero_spatial` (50 epochs, single GPU).
- [ ] Evaluation notebook (`notebooks/eval_libero_model.ipynb`).
- [ ] Ablation: image-only vs `use_state=True`.
- [ ] Ablation: agentview-only vs adding wrist cam as a second decoder head.
- [ ] Ablation: aggregated target vs per-trial with failure-mode embedding.

Tracked by Round-3 of `plans/what-should-we-do-zesty-mochi.md`.
