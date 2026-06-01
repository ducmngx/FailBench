# Session summary — 2026-06-01

End-of-day snapshot. Covers the transformer close-out, world-model design
doc, paper plan, and the move to the GMU Hopper cluster for the
paper-scale sweep.

## What landed today

### Contact-prediction (closed out)

| Run | Model | Modalities | best val | Notes |
|---|---|---|---|---|
| #22 | Transformer | state+rgb+depth (T=8) | 0.1384 | UNet late_fusion (#6) = 0.1372 → attention does not beat mean-pool |
| #23 | Transformer | rgb+depth (T=8) [contaminated] | 0.1389 | parse_modalities bug — actually included state |
| #24 | Transformer | rgb+depth (T=8) true vision-only | 0.1418 | vision alone is the weakest single modality |

Headline: five model families (MLP, ConvDec, UNet 4 temporal modes,
Transformer) converge to 0.136–0.139 within-distribution. The noise
floor is real, not an architecture problem. Architecture side of the
benchmark is **converged**.

### Bugs fixed

- `scripts/benchmark/train_one.py::parse_modalities` was inheriting
  `state=True` from `ModalityConfig`'s dataclass default, contaminating
  runs #20, #21, #23 (claimed vision-only but actually state+vision).
  Fixed in `f769a8b`.
- `planner/__init__.py` eagerly imported MuJoCo via planning algorithms,
  blocking the cluster trainer which doesn't need it. Wrapped in
  try/except in `9283a6e`.
- `planner/risk/dataset_v2.py::_load_manifest` was using absolute h5_paths
  from the manifest, which broke on the cluster where data lives at a
  different prefix. Now rewrites paths from `v2_root / split / task.h5`.
  Fixed in `0df6e19`.
- `scripts/cluster/sweep_headline.sbatch` was killing on `$PYTHONPATH`
  unset under `set -u`. Fixed in `30b1051`.

### New docs

- `docs/world_model_design.md` — design doc for the action+failure-
  conditioned world model successor project (separate repo). 12
  sections covering data requirements, architecture, loss, training,
  and image-conditioning thresholds.
- `docs/paper_plan.md` — what's done, what's missing, sweep design,
  paper outline, figure list, week-by-week timeline.
- `docs/benchmark_experiments_log.md` — updated through run #24 with
  bug-contamination note.
- `docs/contact_prediction_libero_spatial.md` §18 — Transformer
  results, parse_modalities bug, updated publishable framing.

### Cluster scaffolding (Hopper, GMU ORC)

Built and pushed `scripts/cluster/`:

- `README.md` — Hopper partition table, one-time venv setup, data
  staging via SFTP, salloc smoke test, SBATCH submit + monitor commands,
  failure modes.
- `sweep_configs.sh` — 45-row sweep table (5 model configs × 3 splits
  × 3 seeds). Standalone print mode for inspection.
- `sweep_headline.sbatch` — SLURM array job with correct `--qos=gpu`,
  typed `--gres=gpu:A100.80gb:1`, partition `gpuq`, venv activation.

### Cluster bring-up walkthrough

1. **Account + login** — `mnguy21@hopper.orc.gmu.edu`, SLURM scheduler,
   partition `gpuq` (5-day timelimit, A100 80GB primary).
2. **Data upload** — SFTP from `/media/aaron/FAILBENCH/failbench/libero/v2/`
   to `/scratch/mnguy21/data/failbench_data/libero/v2/`. All three
   splits intact (~178 GB total, 11 files per split).
3. **Env build** — pip venv (no conda needed since cluster trainer is
   pure torch+h5py, no MuJoCo). torch 2.5.1+cu121, h5py 3.14.0,
   transformers, etc. Installed in `$HOME/failbench_env`.
4. **GPU smoke test on gpu012 (A100 80GB)** — convdec state-only T=1
   for 1 epoch on 100 trials, reached `ep 1/1 train=0.20 val=0.37`.
   Full pipeline confirmed working.
5. **QOS limits discovered** — `MaxSubmitPU=40`, `MaxJobsPU=20`. Sweep
   has 45 tasks; needs chunking. Pattern: split into chunks
   ≤20 each with `--dependency=afterany:$PREV_JOB`.

## Current state at session end

- **Chunk 1** (JOB1=8017541, tasks 0-19) — running on gpuq.
  First task at `ep 1/30 train=0.30 val=0.30` after ~4 min, healthy.
- **Chunk 2** (JOB2=8017543, tasks 20-39) — queued, depends on JOB1.
- **Chunk 3 poller** (PID 1410578) — backgrounded `nohup` script that
  watches `squeue -j $JOB1` and auto-submits chunk 3 (tasks 40-44)
  when JOB1 drains. Writes the resulting JOB3 ID to `job3.txt`.
- **ETA all 45 done**: ~3 hours from end-of-session.

Outputs land in `/home/mnguy21/FailBench/runs/bench/<run_dir>/` with
`best.pt`, `metrics.json`, `args.json`, `val_curve.png` per run.

## What to do next session

1. **Check sweep completion** — `ls runs/bench/ | grep "$(date +%Y%m%d)" | wc -l` should be 45.
2. **Check for failures** — `sacct -j $JOB1,$JOB2,$JOB3 --format=JobID,State -X | grep -v COMPLETED`.
3. **Aggregate metrics** —
   ```
   PYTHONPATH=. python -m scripts.benchmark.eval_all \
     --runs_root runs/bench --v2_root /scratch/mnguy21/data/failbench_data/libero/v2
   ```
   Produces `runs/bench/_eval/bench_table.{csv,md}` and per-mode JSON.
4. **Rsync results back to local** for figure generation:
   ```
   rsync -avP mnguy21@hopper.orc.gmu.edu:FailBench/runs/bench/ ./runs/bench/
   ```
5. **Start drafting paper §1, §3, §4, §5** in parallel — see
   `docs/paper_plan.md` for the outline. These sections don't depend
   on the sweep numbers landing.
6. **Decide the §7 open questions** (venue, two-head model in or out,
   marginal targets in or out).

## Git log (this session)

```
30b1051 sweep_headline.sbatch: guard PYTHONPATH against unbound var
0df6e19 dataset_v2: rewrite manifest h5_path against v2_root
9283a6e planner/__init__.py: make planning re-exports optional
3498a0b Hopper SLURM: add required --qos=gpu and explicit GPU type in gres
8a6ca42 Hopper: switch to pip venv (no conda) — training never imports MuJoCo
6c4b2dc Cluster docs: fill in Hopper partition info from sinfo output
2e1ca60 SLURM scaffolding for paper-scale FailBench sweep on Hopper
a8eacb2 Paper plan: status, gaps, sweep design, week-by-week timeline
f769a8b Close out contact prediction: Transformer results + parse_modalities fix
8631c87 Transformer model + benchmark documentation
```

All pushed to `refactor/cleanup`.
