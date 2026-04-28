# Risk Modeling

Learning a per-config spatial contact-density field, then composing it with per-entity severity to recover the failure-impact risk used by safer-config planning.

```
trial npz                     →    targets.npz                   →    HeatmapMLP                →    integrate_per_entity
(contact_positions,                (per-config 2D heatmap                   |                          ↓
 contact_failure_id,                on per-scene grid)                      ↓                       p_interaction(eᵢ)
 failure_probs, ...)                                              predicted heatmap                     × severity S(eᵢ)
                                                                     |                                      ↓
                                                                     ↓                                  Risk(config)
                                                              eval_model.ipynb / planner cost
```

The spatial-density framing is principled: predicting a 2D field once gives you per-entity probabilities for free (integrate over each entity's footprint). It also generalises to new entities without retraining — drop in a footprint mask, integrate.

## 1. Build the target dataset

`scripts/build_density_targets.py` walks `datasets/<version>/<scene>/<task>/manifest.csv`, opens every trial npz, and writes a `targets.npz` per task directory plus a `grid.json` per scene.

```bash
python scripts/build_density_targets.py --dataset datasets/v10
```

Per-config target construction (`planner/risk/spatial.compute_target`):

1. Load `contact_positions`, `contact_failure_id`, `failure_probs` from the trial npz.
2. **Above-table filter** (`above_table_mask`) drops contacts below `SCENE_TABLE_Z[scene] - margin` (default margin = 0.01 m, which keeps object-on-table surface contacts whose z sits a few mm below the nominal table top).
3. Per-contact weight `w = failure_probs[contact_failure_id]` — marginalises the failure-mode prior.
4. Weighted `np.histogram2d` onto a fixed per-scene grid (1 cm bins, 5 cm padding around the largest table box).
5. `gaussian_filter` with σ = 2 cm. The result is a smooth, prior-weighted density.

Storage: one `targets.npz` per task dir holding `(N, ny, nx) float32` heatmaps stacked along axis 0, plus `experiment_id`, `n_contacts_above_table`, `total_weight`, `sigma_cm`, `margin`. Per-scene grid metadata in `<dataset>/<scene>/grid.json`. ~16 KB per config; ~150 MB across the full v10 dataset.

| Key flag | Default | What it does |
|---|---|---|
| `--sigma_cm` | 2.0 | Gaussian smoothing σ in cm (independent of bin size) |
| `--bin_cm` | 1.0 | Histogram bin size in cm |
| `--pad_cm` | 5.0 | Grid padding around the largest table box |
| `--margin` | -0.01 | Above-table filter offset; raise to ignore borderline edge contacts |
| `--force-weighted` | off | Also store a contact-force-weighted variant |
| `--scene` | all | Limit to one scene |
| `--workers` | half cores | Multiprocess pool over task dirs |

The script is idempotent — it skips a task dir if `targets.npz` is newer than every input npz under it.

## 2. Per-entity integration

`planner/risk/spatial.entity_footprints(model, grid)` rasterises each non-robot named body's X-Y AABB at its rest pose into a binary mask on the scene grid. `integrate_per_entity(heatmap, footprints)` returns one scalar per entity by summing heatmap mass inside the mask.

```python
from planner.risk.spatial import load_grid, entity_footprints, integrate_per_entity
import mujoco

grid = load_grid(Path("datasets/v10/scene_level2/grid.json"))
model = mujoco.MjModel.from_xml_path("scenes/scene_level2/scene.xml")
fps = entity_footprints(model, grid)

scores = integrate_per_entity(predicted_heatmap, fps)
# {'object3': 12.4, 'obstacle_soft1': 0.6, 'simple_table': 88.1, ...}

risk = sum(scores[e] * SEVERITY[e] for e in scores)   # planner cost
```

Sanity check on scene_level2 (200 sampled v10 configs):

| Entity | Spearman ρ vs raw `contact_geom_pairs` count |
|---|---|
| obstacle_hard1 | 0.97 |
| obstacle_soft1 | 0.96 |
| obstacle_soft3 | 0.94 |
| obstacle_hard3 | 0.95 |
| obstacle_hard2 | 0.93 |
| obstacle_soft2 | 0.92 |
| object3 | 0.52 (footprint overlaps the table) |
| object1 | 0.60 (footprint overlaps the table) |
| simple_table | 0.19 (footprint covers ≈ the whole grid) |

All six obstacles correlate strongly. Movable objects sit on the table so their AABB overlaps with table contacts, dragging their ρ down — expected. The table itself integrates ≈ total above-table mass and so its score is essentially a sum, not a per-entity signal — also expected, and harmless because table severity is typically low.

## 3. Train the heatmap regressor

`scripts/train_demo.py` trains a small MLP on one scene end-to-end:

```bash
python scripts/train_demo.py --scene scene_level2 --epochs 200
```

Outputs land under `runs/heatmap_<scene>_<timestamp>/`:

- `best.pt` — model weights, standardisation stats, full training history.
- `loss_curve.png` — train/val loss + de-standardised val MSE + obstacle-Spearman ρ vs epoch.
- `preds.png` — five held-out configs, each row = `[target | predicted | |error|]`.
- `history.json` — full epoch-by-epoch metrics.

### Inputs and target

```
input  : (17,)  float32     pre_qpos (7) ⊕ pre_ee_pos (3) ⊕ pre_qvel (7)
target : (ny, nx)  float32   per-cell weighted contact density (above-table, prior-weighted, σ=2 cm blurred)
```

Both are standardised — input by per-feature mean/std, target per-cell — so MSE balances cells with very different baselines.

### Model — `planner/risk/model.HeatmapMLP`

```
Linear in_dim → 256  → SiLU → Dropout 0.1
       → Linear 512  → SiLU → Dropout 0.1
       → Linear 1024 → SiLU → Dropout 0.1
       → Linear ny·nx → reshape (ny, nx)
```

~3.7 M params for scene_level2 (43 × 70 grid). Output is unbounded float in standardised space; de-standardise for visualisation and entity integration.

### Splits — `planner/risk/dataset.split_traj_keys`

90/10 random split on `(task_id, traj_id)` tuples. `traj_id` is **task-local** (each task has trajs 0..9), so a split on `traj_id` alone leaks across tasks — this matters.

### scene_level2 demo result (RTX 3070, 200 epochs)

| | qpos+ee_pos, 50 ep | + qvel, 200 ep |
|---|---|---|
| Best val MSE (original units) | 4.64 | **3.71** |
| Reduction vs mean-baseline | 63 % | **71 %** (training log) / 46 % (eval notebook) |
| Best obstacle Spearman ρ | 0.36 | **0.42** |
| Wall time | 52 s | 2 m 46 s |

The two reduction numbers differ because the training log uses `(y_std**2).mean()` as a baseline proxy and the eval notebook computes `((y_mean - y_target)**2).mean()` on the actual val split — the latter (46 %) is the honest one.

## 4. Evaluate

`notebooks/eval_model.ipynb` auto-loads the most recent `runs/heatmap_*/best.pt`, replays the exact val split, and produces:

1. **Aggregate val metrics** — MSE, R², per-sample MSE distribution, mean-baseline comparison.
2. **Per-task breakdown** — bar chart of val MSE by `task_id`.
3. **Browseable target/pred/error panels** — N random val configs.
4. **Per-entity scatter + Spearman** — one panel per entity, predicted score vs target score with the y=x line.
5. **Worst- and best-case panels** — top-K MSE outliers in both directions.
6. **Single-config inspector** — pick `i` → render heatmap triplet + per-entity bar chart.
7. **Front_cam projection overlay** — treat each grid cell as a 3D point at `(x, y, table_z)`, project through `ContactProjector`, and alpha-blend the predicted heatmap onto `pre_rgb`. Same projection the §9d cell of `inspect_dataset.ipynb` already uses for raw contacts. Run it side-by-side with the target overlay to see *where* the model thinks the contacts will land in image space.

The model output itself is **camera-agnostic** — it's a 2D field in world coordinates anchored to the table plane. The camera overlay is a rendering choice; switching cameras (front_cam → ee_cam → overhead) only changes the projection cell, not the prediction.

## 5. File map

```
planner/risk/
  spatial.py        SCENE_TABLE_Z, above_table_mask, SceneGrid, derive_scene_grid,
                    compute_target, scene_entities, entity_footprints,
                    integrate_per_entity, save_grid / load_grid
  dataset.py        HeatmapDataset, DatasetStats, split_traj_keys
  model.py          HeatmapMLP

scripts/
  build_density_targets.py    Walks manifests → writes targets.npz + grid.json
  train_demo.py               Train HeatmapMLP on one scene → runs/<ts>/

notebooks/
  inspect_dataset.ipynb       Section 10 (live density viz) and §11 (persisted-target loader + per-entity integration)
  eval_model.ipynb            Full evaluation suite + camera-overlay projection
```

## 6. Where this connects to the paper

The ICRA paper formulates failure risk as

$$\text{Risk}(x_t) = \sum_i P_\text{interaction}(e_i \mid x_t)\cdot S(e_i)$$

with $S$ hand-set per entity. Our learned $\rho(x, y \mid x_t)$ recovers per-entity interaction scores by integrating over each entity's footprint:

$$P_\text{interaction}(e_i \mid x_t) \;\propto\; \int_{\text{AABB}(e_i)} \rho(x, y \mid x_t)\, dx\, dy$$

The paper's Algorithm 1 estimates the same quantity via a geometric AABB-overlap heuristic between robot-component and entity AABBs in world frame; a one-to-one comparison against the learned predictor is a natural next experiment.

## 7. Roadmap

| Item | Why | Where it lands |
|---|---|---|
| ConvTranspose decoder | Linear head ignores grid topology | `planner/risk/model.py` |
| Add `pre_rgb` input | Workspace clutter not visible in joint state | `dataset.py` + new `HeatmapImageMLP` |
| Multi-scene training | Cross-scene transfer; shared backbone | `train_demo.py` flags + scene one-hot |
| Severity registry | Per-scene scalars for the planner cost | `planner/risk/severity.py` |
| Algorithm 1 baseline | Apples-to-apples comparison with the paper | `planner/risk/algorithm1.py` |
