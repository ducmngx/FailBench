# Benchmark models — architecture reference

The v2 contact-prediction benchmark trains four model families against the same
data, loss, and metric suite. This doc is the canonical reference for what
each model is, how it consumes inputs, what its parameter count is, and where
to find the code.

All models share one contract:

```python
out = model(batch)        # batch: dict of (B, ...) tensors per modality
out["pred"]               # (B, H, W) float32 — log1p-space heatmap prediction
```

Built by `planner.risk.models.make_model(name, modalities, grid_hw, T, **kwargs)`.
Registered in `planner/risk/models/__init__.py`.

All models accept the same `ModalityConfig` flags and read the same batch keys.

---

## ModalityConfig — what every model can read

Defined in `planner/risk/benchmark_dataset.py`. Toggle these to enable input
streams; the corresponding batch keys must then be present.

| Flag | Batch key | Shape (B=batch, T=window) | Source |
|---|---|---|---|
| `state` | `state_window` | (B, T, 18) | `pre_qpos⊕qvel⊕ee_pos⊕gripper_ctrl` per frame |
| `goal` | `goal`, `goal_offsets` | (B, K=3, 11), (B, K) | future commanded `qpos⊕ee⊕grip` at +5/+15/+30 steps |
| `rgb` | `rgb_window` | (B, T, 3, H, W) | agentview RGB, float32 ∈ [0, 1] |
| `depth` | `depth_window` | (B, T, 1, H, W) | agentview depth in metres |
| `dino` | `dino_window` | (B, T, 384) | precomputed DINOv2 ViT-S/14 CLS tokens |
| `failure_mode` | `failure_mode` | (B, 5) | one-hot of sampled mode (oracle) |
| `failure_joints` | `failure_joints` | (B, 7) | multi-hot of which arm joints failed (oracle) |

When `T=1` (trainer `--T 1`) the window arrays are shape `(B, 1, ...)` —
state/vision become single-frame.

---

## Model 1 — `BenchmarkMLP`

File: `planner/risk/models/mlp.py` (~110 LOC)

**Architecture:**

```
state_window (B,T,18) ──┐
goal         (B,K,11) ──┤  flatten / pool                  ┌── Linear(h, H·W)
rgb_window   (B,T,...) ──┤      │                          │      │
depth_window (B,T,...) ──┤   per-modality encoders         ▼      ▼
dino_window  (B,T,384)──┤   then concat                  MLP    reshape
failure_mode (B,5)     ──┤                                 │      │
failure_joints (B,7)   ──┘                                 ▼      ▼
                                                       hidden  (B,H,W)
```

Concrete:
- **State**: flatten `(B, T, 18) → (B, T·18)`.
- **Goal**: flatten `(B, K, 11) → (B, K·11)`.
- **RGB/Depth**: per-frame `_SmallCNN` (stride-2 × 4 → adaptive pool 4×4 → linear `(B·T, embed=128)`), then temporal mean → `(B, 128)`.
- **DINOv2**: temporal mean over T → `(B, 384)`.
- **Failure mode / joints**: concat directly `(B, 5)`, `(B, 7)`.
- **Fuse**: concat all → `(B, fused_dim)`.
- **Head**: 3-layer MLP `[512, 1024, 1024]` with SiLU + dropout → final `Linear(1024, 240·320=76,800)` → reshape `(B, 240, 320)`.

**Param counts** (T=8, grid 240×320):
- state-only: **80.3 M** (dense head dominates: 1024 × 76,800 = 78.6 M)
- state+rgb+depth: 80.6 M

**Strengths/weaknesses:**
- Zero spatial prior; the dense `Linear(h, H·W)` head treats each pixel as independent.
- Very fast inference (~0.8 ms/batch on RTX 3070).
- Param count dominated by the head; encoder is tiny.
- Tends to predict diffuse heatmaps with poor spatial concentration (IoU ≈ 0.17 in v10 evals vs ConvDec's 0.26).

**Use when**: cheapest baseline; control to verify the conv decoder is earning its keep.

---

## Model 2 — `BenchmarkConvDec`

File: `planner/risk/models/convdec.py` (~100 LOC)

**Architecture:**

```
(same per-modality encoders as MLP)
         │
         ▼
   Linear(fused, hidden, base_ch · base_hw)
         │
         ▼
  reshape (B, 128, 15, 20)         ← base feature map
         │
         ▼
   UpBlock ×3  (bilinear 2× + double 3×3 conv)
         │
         ▼
   (B, 16, 120, 160)
         │
         ▼
   Bilinear interpolate to grid_hw
         │
         ▼
   1×1 conv → (B, 1, H, W) → squeeze → (B, H, W)
```

Concrete:
- Same per-modality encoder bank as MLP.
- **Encoder MLP** projects fused features to a 128 × 15 × 20 = 38,400-element vector, reshaped to a `(B, 128, 15, 20)` base feature map.
- **UpBlock** = `Upsample(2×, bilinear) → Conv3×3 → SiLU → Conv3×3 → SiLU`. Three blocks halve channels each: 128 → 64 → 32 → 16.
- Final bilinear interpolate to grid_hw + 1×1 conv to single channel.

**Param counts** (T=8, grid 240×320):
- state-only: **20.2 M** (encoder MLP dominates: 144 → 512 → 512 → 38,400)
- state+rgb+depth: 20.5 M

**Strengths/weaknesses:**
- Spatial prior from the conv decoder makes per-pixel predictions correlated.
- 4× fewer params than MLP, better IoU (~0.26 in v10 evals), same MSE on libero_spatial.
- Inference latency similar to MLP (~0.7 ms/batch).
- Decoder is fixed-resolution; doesn't adapt to different `grid_hw` without rebuild.

**Use when**: state-only or state+frozen-features baseline. The leader for the realistic libero_spatial setting (best val MSE_log1p = 0.358 on marginal targets, 0.137 on per-trial).

---

## Model 3 — `BenchmarkUNet`

File: `planner/risk/models/unet.py` (~200 LOC). Wraps `LiberoHeatmapModel` from `planner/risk/libero_model.py` (the existing v1 model, ~12 M params ResNet-18 backbone).

**Architecture:**

```
RGB+depth window (B,T,4,H,W)
         │
   _temporal_reduce_*   ←  selected by --unet_temporal
   ┌────┴────┬─────────┬─────────────┐
   │mean     │last     │conv3d       │late_fusion
   ▼         ▼         ▼             ▼
   pixel-mean   last frame   per-channel Conv3d   per-frame encode →
   then 1 encode  then 1 encode  then 1 encode   then mean of features
        │              │              │                 │
        └──────────────┴──────────────┴─────────────────┘
                              │
                              ▼
              ResNet-18 encoder (or 8× late_fusion)
              skips s0..s4 (channels 64,64,128,256,512)
                              │
              optional state(14) → MLP(64) → tile + concat at s4
              optional failure_descriptor → FiLM-modulate every decoder block
                              │
                              ▼
           UNet decoder: up4 → up3 → up2 → up1 → final_up
           (ConvTranspose 2× + skip concat + ConvBlock)
                              │
                              ▼
                    1×1 conv → (B, 2, H, W)  [mass, depth]
                                pred = squeeze channel 0
```

**Temporal modes** (`--unet_temporal {mean, conv3d, last, late_fusion}`):
- `mean` (default): pixel-mean across T → one image → ResNet once. Loses motion.
- `conv3d`: per-channel `Conv3d(C, C, kernel=(T,1,1))` mixes time as a learned weighted pixel mean. Init'd to uniform mean.
- `last`: take frame[T-1] only. Equivalent to T=1 for vision.
- `late_fusion`: per-frame ResNet → mean-pool *features* (B, T, C, H/32, W/32) → decoder. 8× backbone compute. Strongest pre-transformer architecture.

**FiLM conditioning** (when `failure_mode` or `failure_joints` enabled): the inner model's FiLM layers modulate every decoder up-block channel-wise based on the concatenated 5+7=12-D failure descriptor.

**Param counts** (T=8, grid 240×320, state+rgb+depth):
- mean / last: **14.5 M**
- conv3d: 14.5 M (+85 params for the 3D conv)
- late_fusion: 14.5 M (same module, used 8× per forward)
- + failure_mode FiLM: +14.6 M (FiLM blocks add ~50 K)
- + failure_joints (12-D FiLM cond): same param count, larger first conv in FiLM

**Strengths/weaknesses:**
- Inherits ImageNet-pretrained ResNet-18 weights — strong low-level features.
- Native image-resolution output via UNet skips.
- Overfits hard on libero_spatial within ~5 epochs (train/val gap > 0.05); needs early stopping + WD=5e-4.
- Inference latency 3.7–7.9 ms/batch (vs 0.7 ms for ConvDec) → ~10× slower.
- Late_fusion is the most expensive (8× backbone) but the only mode that uses the window properly.

**Use when**: image-conditional benchmark. The leader for the oracle setting (best val MSE_log1p = 0.055 with all failure descriptors).

---

## Model 4 — `BenchmarkTransformer`

File: `planner/risk/models/transformer.py` (~250 LOC). **Newest model**.

**Architecture:**

```
Per-modality input projections (each enabled modality → tokens of dim d_model):
   state_window  (B,T,18)  → Linear(18→d)        → T state tokens
   goal          (B,K,11)  → Linear(11→d)        → K goal tokens
   rgb_window    (B,T,...) → per-frame CNN       → T rgb tokens
   depth_window  (B,T,...) → per-frame CNN       → T depth tokens
   dino_window   (B,T,384) → Linear(384→d)       → T dino tokens
   failure_mode  (B,5)     → Linear(5→d)         → 1 failure-mode token
   failure_joints (B,7)    → Linear(7→d)         → 1 failure-joints token

Add per-token:
   - Type embedding (learned, 8 token-types)
   - Positional embedding (sinusoidal):
       * temporal PE (length T) on per-frame tokens
       * goal PE (length K) on goal tokens
       * 2-D spatial PE (Hp × Wp) on heatmap-query tokens

Concat with learned heatmap-query tokens:
   queries: (Hp · Wp = 15 × 20 = 300) learned vectors of dim d_model

           ┌──── input tokens ────┐ ┌── heatmap queries ──┐
           ▼                       ▼ ▼                     ▼
       (B, N_kv, d_model)        (B, 300, d_model)
                       │       │
                       └──cat──┘
                           │
                           ▼
              nn.TransformerEncoder
              (6 layers, 4 heads, d=256, GELU, pre-LN, batch_first)
                           │
              extract query positions only: (B, 300, d)
                           │
                           ▼
                LayerNorm + Linear(d, 1) → (B, 300)
                           │
                  reshape (B, 1, 15, 20)
                           │
                           ▼
                bilinear interpolate to (240, 320)
                           │
                           ▼
                  squeeze → (B, H, W) prediction
```

**Concrete details:**
- `d_model=256`, `n_heads=4`, `n_layers=6`, `dropout=0.1`.
- Per-frame RGB/depth encoded with the same `_SmallCNN` as MLP/ConvDec (output dim = d_model).
- DINOv2 features just need a `Linear(384, d_model)` projection.
- Sinusoidal PE (standard `sin/cos` formula) — not learned. Heatmap queries get a 2-D PE built from row + column 1-D PEs.
- Single transformer encoder stack — no separate decoder. Queries learn to pool via self-attention.

**Param counts** (T=8, grid 240×320):
- state-only: **4.82 M** (transformer encoder dominates: 6 layers × ~790 K)
- state+rgb+depth: 5.47 M (+ 2 small CNN encoders, 150 K each)
- state+goal+rgb+failure_mode+failure_joints: 5.15 M

**Strengths/weaknesses:**
- Smallest of all four model families (5× smaller than ConvDec, 3× smaller than UNet).
- Native sequence handling — can learn non-uniform temporal weighting via attention, not just mean-pooling.
- Direct cross-frame interaction between heatmap queries and any input token.
- Self-attention over ~350 tokens is cheap (~6 ms/batch on RTX 3070).
- Output is patch-based (15×20) then bilinearly upsampled → smoother than UNet's full-resolution decoder but loses fine spatial detail.
- Untested empirically — added 2026-05-27, no full training run yet.

**Use when**: testing whether sequence structure with positional encoding helps beyond mean-pooled features. The principled "does video matter" architecture.

---

## Cross-model comparison

| Model | Params (state) | Params (state+rgb+depth) | Inference (ms/batch) | Window handling |
|---|---|---|---|---|
| MLP | 80.3 M | 80.6 M | ~0.8 | flatten T·18 + temporal mean per modality |
| ConvDec | 20.2 M | 20.5 M | ~0.7 | flatten T·18 + temporal mean per modality |
| UNet (late_fusion) | n/a (vision-required) | 14.5 M | ~7.9 | per-frame encoder + mean of features |
| Transformer | 4.8 M | 5.5 M | ~6 | per-frame tokens with PE, self-attention |

(Latencies from `runs/bench/_eval/bench_table.csv`, batch=32.)

**Param-count ordering** is the inverse of "model class capacity" — MLP has 80 M only because of its dense pixel head. ConvDec replaces that with a 1 M-param decoder. UNet pays for ResNet-18 (12 M). Transformer pays for the encoder stack (5 M). All four reach roughly similar val MSE on libero_spatial in-distribution.

---

## Reproducing the benchmark

All models share `scripts/benchmark/train_one.py`:

```bash
# State-only ConvDec (the realistic deploy leader on libero_spatial)
PYTHONPATH=. python -m scripts.benchmark.train_one \
  --model convdec --modalities state --T 1 \
  --target_form per_trial --split_by demo \
  --v2_root /home/aaron/scratch/v2_ssd --splits libero_spatial \
  --epochs 30 --patience 5 --warmup_epochs 1 --batch_size 128 --seed 0

# UNet image-conditional (oracle leader)
PYTHONPATH=. python -m scripts.benchmark.train_one \
  --model unet --modalities state,rgb,depth,failure_mode,failure_joints \
  --T 8 --unet_temporal late_fusion \
  --v2_root /home/aaron/scratch/v2_ssd --splits libero_spatial \
  --epochs 30 --patience 5 --warmup_epochs 2 --weight_decay 5e-4 \
  --batch_size 16 --seed 0

# Transformer (next-up test)
PYTHONPATH=. python -m scripts.benchmark.train_one \
  --model transformer --modalities state,rgb,depth --T 8 \
  --v2_root /home/aaron/scratch/v2_ssd --splits libero_spatial \
  --epochs 30 --patience 5 --warmup_epochs 2 --batch_size 32 --seed 0
```

---

## See also

- `docs/contact_prediction_libero_spatial.md` — full investigation log, results across all experiments.
- `docs/benchmark_experiments_log.md` — chronological table of every training run.
- `notebooks/visualize_models.ipynb` — instantiates each model and prints layer-by-layer summaries for visual exploration.
- `planner/risk/benchmark_dataset.py` — `BenchmarkDataset`, `MarginalBenchmarkDataset`, split helpers.
- `planner/risk/benchmark_metrics.py` — all eval metrics (MSE_log1p, IoU, KL, mass_ratio, AUPRC, RMSE).
- `scripts/benchmark/eval_all.py` — walks `runs/bench/*/best.pt`, produces the full metric table.
