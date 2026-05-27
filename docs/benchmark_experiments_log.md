# Benchmark experiments — chronological log

Every training run for the v2 contact-prediction benchmark on libero_spatial,
in the order it was launched. Each row links to the corresponding run dir
under `runs/bench/`. Numbers are weighted MSE on `log1p(target)` with α=10.

For *why* each experiment was run and what we concluded, see
`docs/contact_prediction_libero_spatial.md`.

For the model architectures, see `docs/benchmark_models.md`.

## Conventions

- All runs on RTX 3070, single GPU, seed 0 unless noted.
- libero_spatial split: 13,500 train / 1,500 val (demo-stratified 90/10) on per-trial target;
  3,500 train / 1,500 val on marginal target (5,000 groups, ~90/10).
- AdamW, lr 3e-4, weight_decay 1e-4 (5e-4 for UNet variants with noted regularisation).
- LinearLR warmup → CosineAnnealingLR.
- Early stop patience=5 unless noted.
- α=10 foreground reweighting in MSE_log1p loss.

## Master table (libero_spatial, seed 0)

Sorted by chronology; "best val" is `weighted_mse_log1p` on the val split.

| # | Date | Model | Modalities | T | target | split | unet_temporal | best val | best ep | run dir prefix |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 05-25 | MLP | state | 8 | per_trial | demo | — | 0.1471 | 19 | `mlp__state__seed0__20260525-200954` |
| 2 | 05-25 | ConvDec | state | 8 | per_trial | demo | — | 0.1372 | 19 | `convdec__state__seed0__20260525-203831` |
| 3 | 05-26 | UNet | state+rgb+depth | 8 | per_trial | demo | mean | 0.1386 | 5 | killed (`20260525-210741`) |
| 4 | 05-26 | UNet | state+rgb+depth | 8 | per_trial | demo | conv3d | 0.1386 | 6 | `20260526-013531` |
| 5 | 05-26 | UNet | state+rgb+depth | 8 | per_trial | demo | last | 0.1378 | 4 | `20260526-114841` |
| 6 | 05-26 | UNet | state+rgb+depth | 8 | per_trial | demo | late_fusion | 0.1372 | 7 | `20260526-123726` |
| 7 | 05-26 | ConvDec | state+dino | 8 | per_trial | demo | — | 0.1373 | 19 | `20260526-165928` |
| 8 | 05-26 | MLP | state | 1 | per_trial | demo | — | 0.1495 | 20 | `mlp__state__T1__20260526-172814` |
| 9 | 05-26 | ConvDec | state | 1 | per_trial | demo | — | **0.1366** | 18 | `convdec__state__T1__20260526-173653` |
| 10 | 05-26 | ConvDec | state (mass_total_w=0.1) | 1 | per_trial | demo | — | 0.2138 | 1 | early-stopped at ep 6 |
| 11 | 05-26 | ConvDec | state (mass_total_w=0.01) | 1 | per_trial | demo | — | 0.1651 | 16 | `20260526-195650` |
| 12 | 05-26 | ConvDec | state+failure_mode | 1 | per_trial | demo | — | 0.1036 | 20 | `convdec__state+failure_mode__T1__20260526-195107` |
| 13 | 05-26 | ConvDec | state+failure_mode+failure_joints | 1 | per_trial | demo | — | **0.0665** | 20 | `20260526-204528` |
| 14 | 05-26 | UNet | state+rgb+depth+failure_mode | 8 | per_trial | demo | late_fusion | 0.0994 | 7 | `20260526-...` |
| 15 | 05-27 | UNet | state+rgb+depth+failure_mode+failure_joints | 8 | per_trial | demo | late_fusion | **0.0551** | 12 | `unet__...__failure_mode+failure_joints__20260527-...` |
| 16 | 05-27 | ConvDec | state | 1 | **marginal** | demo | — | **0.3580** | 24 | `convdec__state__T1__marg__20260527-...` |
| 17 | 05-27 | UNet | state+rgb+depth | 8 | **marginal** | demo | late_fusion | 0.3605 | 6 | `unet__...__marg__20260527-...` |
| 18 | 05-27 | ConvDec | state | 1 | per_trial | **task-held-out** | — | 0.1620 | 14 | `convdec__state__T1__pertrial__splitT3__20260527-...` |
| 19 | 05-27 | UNet | state+rgb+depth | 8 | per_trial | **task-held-out** | late_fusion | **0.1538** | 4 | `unet__...__splitT3__20260527-...` |
| 20 | 05-27 | UNet | **rgb+depth (no state)** | 1 | per_trial | demo | last | 0.1371 | 4 | `unet__rgb+depth__T1__20260527-...` |
| 21 | 05-27 | UNet | **rgb+depth (no state)** | 8 | per_trial | demo | late_fusion | running | – | – |
| 22 | tbd | Transformer | tbd | tbd | tbd | tbd | n/a | tbd | tbd | – |

Bolded rows mark notable findings (leader within their setting).

## Findings by experiment

| Group | Experiments | Headline |
|---|---|---|
| **Initial state-only** | 1, 2, 8, 9 | ConvDec T=1 is the state-only leader; window adds 0–1 % |
| **UNet temporal modes** | 3, 4, 5, 6 | late_fusion = mean = conv3d ≈ last; vision-window adds nothing |
| **Vision features** | 7 | DINOv2 ties state-only |
| **Loss design** | 10, 11 | Mass-total auxiliary either stalls (w=0.1) or hurts MSE (w=0.01) |
| **Failure descriptors** | 12, 13 | failure_mode −24%, failure_joints −36% (free oracle signal) |
| **Vision + oracle** | 14, 15 | Vision +17% on top of full oracle info |
| **Realistic target** | 16, 17 | State-only ties vision (within 0.7%) on marginal target |
| **Cross-task held-out** | 18, 19 | Vision wins by 5% OOD — the within-distribution tie was an artefact |
| **Vision-only** | 20, 21 | Tests if pixels carry motion info that qvel doesn't |
| **Sequence model** | 22 | Transformer test of "does video help" — pending |

## Reproducible run-name decoder

Run dirs are named:
```
<model>__<modalities>__T<n>__<target_form>__<split>__seed<s>__<timestamp>
```

Older runs (pre 2026-05-27) omit `__<target_form>__<split>__` segments
(default per_trial + demo-stratified).

Example: `convdec__state+failure_mode__T1__pertrial__splitD__seed0__20260526-195107`
- model=convdec, state+failure_mode modalities, T=1 input, per_trial target,
  demo-stratified split, seed 0, timestamp.

## How to re-run an experiment from this log

```bash
# Example: re-run experiment 9 (ConvDec state-only T=1, current leader)
PYTHONPATH=. python -m scripts.benchmark.train_one \
  --model convdec --modalities state --T 1 \
  --target_form per_trial --split_by demo \
  --v2_root /home/aaron/scratch/v2_ssd --splits libero_spatial \
  --epochs 30 --patience 5 --warmup_epochs 1 \
  --batch_size 128 --num_workers 4 --seed 0
```

Read `args.json` in any run dir for the exact CLI that produced it.

## What's missing

- libero_object / libero_goal scale-up (data on USB, not yet staged)
- Multi-seed variance bars (single-seed throughout)
- Multi-camera fusion (wrist cam exists in v2 but unused)
- Two-head failure-prediction model (the principled deploy architecture)
- Diffusion model family (model E in the original plan)
