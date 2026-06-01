# FailBench contact-prediction paper plan

**Status as of 2026-06-01.** Snapshot of what's done, what's missing, the
sweep that needs to land, and what to draft in parallel.

This complements:
- `docs/benchmark_experiments_log.md` — chronological run ledger
- `docs/contact_prediction_libero_spatial.md` — investigation report (§1-18)
- `docs/benchmark_models.md` — architecture reference
- `scripts/cluster/` — Hopper SLURM scaffolding for the paper-scale sweep

---

## 1. What's done (libero_spatial, single seed)

| Component | State |
|---|---|
| v2 dataset (45k trials × 3 splits) | done |
| 2D agentview heatmap target builder | done |
| Marginal target dataset (realistic deploy setting) | done |
| Cross-task held-out split tooling | done |
| 4 model families: MLP, ConvDec, UNet (4 temporal modes), Transformer | done |
| Trainer with modality / target / split / seed flags | done |
| DINOv2 cache + dataset hooks | done |
| 24 experiments documented | done |

### Key findings to date

1. **In-distribution noise floor ≈ 0.137 weighted MSE log1p.** Five
   architectures converge there. Not an architecture problem.
2. **Oracle failure descriptors drop loss to 0.067** (−51%). Failure-mode
   uncertainty is the dominant error source.
3. **Vision adds ~0% in-dist but wins 5% OOD** under task-held-out split.
4. **Marginal target ≈ 0.36** — much harder; the realistic deploy setting.
5. **Attention does not beat mean-pool.** The 8-frame window carries no
   useful incremental temporal signal at this scale.

---

## 2. What's still needed for the paper

| # | Gap | Effort | Why it blocks publication |
|---|---|---|---|
| 1 | **Multi-seed variance bars** (3 seeds × 5 configs) | overnight on cluster | Single-seed numbers get rejected on noise grounds. Within-arch spreads (~3 %) are within plausible seed noise |
| 2 | **libero_object + libero_goal scale-up** | data already staging; ~3 h cluster sweep | The OOD vision claim (§17) is on one split. Need replication across 3 splits to call it robust |
| 3 | **Geometric baseline (Algorithm 1 / AABB)** | ~2 days | Reviewers will ask. Without it there's no point of reference |
| 4 | **Two-head failure-prediction model** | ~1 week | Oracle numbers (0.067) aren't deployable. Need a head that predicts the descriptor so §13's headline survives the realistic setting |
| 5 | **Mass calibration fix** | ~3 days | All models over-predict total mass by 28-219×. Visible in any figure a reviewer pulls up |

(1) and (2) are gated on the Hopper sweep — that's what the cluster
scaffolding is for. (3)-(5) are local-3070 work that runs in parallel.

---

## 3. The headline sweep (gated on data upload)

Defined in `scripts/cluster/sweep_configs.sh`:

> 5 model configs × 3 splits × 3 seeds = **45 runs, ~52 GPU-hours**

The 5 configs cover the conceptual axes:

| Config | Tests |
|---|---|
| ConvDec state-only T=1 | Kinematic-only leader, cheap baseline |
| UNet late_fusion state+rgb+depth T=8 | In-dist leader (locally) |
| UNet + failure_mode + failure_joints (oracle) | Full-info upper bound |
| Transformer state+rgb+depth T=8 | Sequence-model variant |
| Transformer rgb+depth T=8 (true vision-only) | Pure vision baseline |

With 8 GPUs concurrent on Hopper that's ~7 h wall-clock; with 16 GPUs
~3.5 h. Submit overnight, results in the morning.

---

## 4. Paper outline (writing in parallel)

These sections **don't depend on the sweep landing** and should be drafted
this week:

| § | Title | Source material |
|---|---|---|
| 1 | Introduction | Motivation for failure-aware planning; from §17 of the investigation doc |
| 2 | Related work | Skeleton: physics simulators (MuJoCo), LIBERO ecosystem, contact prediction (sparse), failure detection literature, world models in robotics |
| 3 | Dataset (FailBench-LIBERO) | From `docs/libero_v1_dataset.md` and v2 schema. Describe 45k trials, schema, failure modes, agentview pinhole calibration |
| 4 | Task formalisation | The "contact-at-failure" prediction task. Target form (per-trial vs marginal). Splits (demo-stratified vs task-held-out). Metrics (weighted MSE log1p, soft-IoU, mass-total) |
| 5 | Methods | From `docs/benchmark_models.md`. MLP / ConvDec / UNet / Transformer. Per-arch param count + design rationale |
| 6 | Experiments and results | **This is gated on the sweep.** Headline tables come from `runs/bench/_eval/bench_table.md` |
| 7 | Analysis | The §16 / §17 / §18 framings: in-dist noise floor, OOD reversal, attention ≠ mean. Per-failure-mode breakdown |
| 8 | Limitations | Single-camera, single-action-per-state, terminal-only target, no force prediction |
| 9 | Conclusion | "Image conditioning's value is a generalisation-regime question, not a modality question." |

What gates each section:
- §1-§5: write now while data uploads.
- §6: gated on multi-seed + 3-split sweep completion.
- §7: gated on §6 plus per-mode eval (already exists locally for libero_spatial; replicate on object/goal).
- §8: ~half a day once §6 lands.

---

## 5. Figures to make (in priority order)

| Fig | Description | Status |
|---|---|---|
| 1 | Single config: ground-truth contact overlay vs predicted, three model variants side by side | Needs §6 results |
| 2 | Per-architecture val MSE bars with 95 % CI (from 3 seeds) | Needs sweep |
| 3 | OOD reversal: in-dist vs task-held-out, per architecture, three splits | Needs OOD sweep |
| 4 | Per-failure-mode breakdown (5-mode heatmap per architecture) | Per-mode JSON already exists for libero_spatial; replicate on others |
| 5 | Marginal vs per-trial target on the same architectures | Already exists |
| 6 | Architecture diagrams (UNet, Transformer) | Take from `docs/benchmark_models.md` |
| 7 | Sample agentview pinhole projection: world contacts → image space | Already exists in `notebooks/v2_quickstart.ipynb` |

---

## 6. Writing rules of thumb

- **Don't write a "Results" section that says "0.137 weighted MSE."** It says
  nothing. Always pair with a baseline number (mean-prediction baseline,
  geometric AABB) and a per-failure-mode breakdown so the reader sees what
  drives the average.
- **Visual results in any figure with a heatmap.** The number 0.137 vs 0.072
  is incomprehensible without one side-by-side overlay.
- **Be explicit about the generalisation regime.** Every claim about
  "vision helps / doesn't help" needs the split annotated. The §17 result
  changes the story; don't bury it.
- **Don't claim the model is deploy-ready.** It uses oracle failure
  descriptors for the best numbers. Until the two-head model lands, "what
  it would take to deploy" is the honest framing.

---

## 7. Decision points still open

- **Venue.** Probably an ML/robotics conference (CoRL, ICRA, NeurIPS
  datasets-and-benchmarks). Pick venue → pick page limit → constrains
  scope of §3-§5.
- **Whether to include marginal-target results in the paper.** They're
  qualitatively different (harder, much weaker model performance). Could
  be a strong "deploy realism" section, or could split into a follow-up
  paper. Decision affects whether §3 mentions marginal targets at all.
- **Whether the two-head model has to be in the paper.** If yes, that's
  ~1-2 weeks added. If we defer to a follow-up, the paper claim narrows
  to "given oracle, contact is predictable; without oracle, it's hard."
  Honest but less ambitious.

---

## 8. This week's plan

1. **(Mon-Tue)** Data finishes uploading to Hopper. Submit one test array
   task to validate env. Submit the full 45-run sweep.
2. **(Mon-Wed, in parallel)** Draft §1, §3, §4, §5 of the paper. These
   don't need new numbers; everything is from existing docs.
3. **(Wed-Thu)** Sweep results land. Run `eval_all.py` for aggregated
   tables. Draft §6 (experiments) and §7 (analysis) from those.
4. **(Thu-Fri)** Generate Fig 1, 2, 3, 4 in notebooks.
5. **(Fri)** Decide on the §7 open questions (two-head model, marginal
   targets, venue).

If 45 runs finish in 4 hours, this all compresses; if anything in the
cluster setup goes sideways, it stretches. Build-time buffer = ~2 days.
