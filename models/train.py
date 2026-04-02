"""Training script for the object-centric contact prediction model.

Uses leave-K-experiments-out cross-validation given the small dataset.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.dataset import FailBenchDataset
from models.contact_predictor import ContactPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


class ContactLoss(nn.Module):
    """Combined loss for object-centric hit classification + force regression.

    Only computes loss on valid (unpadded) object slots via obj_mask.
    """

    def __init__(self, w_hit=1.0, w_force=0.1, w_pos=1.0, pos_weight=None):
        super().__init__()
        self.w_hit = w_hit
        self.w_force = w_force
        self.w_pos = w_pos
        self._pos_weight = pos_weight

    def forward(self, hit_logits, force_preds, target_hit, target_force,
                target_centroid, obj_mask):
        """
        All inputs: (B, K) or (B, K, ...) with K = max_objects.
        obj_mask: (B, K) bool — only compute loss on True slots.
        """
        # Flatten to valid objects only
        valid = obj_mask.bool()
        logits_v = hit_logits[valid]         # (N_valid,)
        hits_v = target_hit[valid]           # (N_valid,)

        if self._pos_weight is not None:
            pw = self._pos_weight.to(logits_v.device)
            # Expand pos_weight to match valid entries
            # pos_weight is (K,), valid is (B, K) — gather per-object weights
            K = obj_mask.shape[1]
            pw_exp = pw.unsqueeze(0).expand_as(obj_mask)
            pw_v = pw_exp[valid]
            # Manual weighted BCE
            p = torch.sigmoid(logits_v)
            loss_hit = -(pw_v * hits_v * torch.log(p + 1e-8)
                         + (1 - hits_v) * torch.log(1 - p + 1e-8)).mean()
        else:
            loss_hit = nn.functional.binary_cross_entropy_with_logits(logits_v, hits_v)

        # Force + centroid regression — only on hit objects
        hit_mask = valid & target_hit.bool()  # (B, K)
        if hit_mask.any():
            pred_f = force_preds[..., 0][hit_mask]
            true_f = target_force[hit_mask]
            loss_force = nn.functional.mse_loss(pred_f, true_f)

            pred_pos = force_preds[..., 1:4][hit_mask]
            true_pos = target_centroid[hit_mask]
            loss_pos = nn.functional.l1_loss(pred_pos, true_pos)
        else:
            loss_force = torch.tensor(0.0, device=hit_logits.device)
            loss_pos = torch.tensor(0.0, device=hit_logits.device)

        total = self.w_hit * loss_hit + self.w_force * loss_force + self.w_pos * loss_pos

        return total, {
            "hit": loss_hit.item(),
            "force": loss_force.item(),
            "pos": loss_pos.item(),
            "total": total.item(),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    loss_parts = {"hit": 0.0, "force": 0.0, "pos": 0.0}
    n = 0

    for batch in loader:
        img = batch["image"].to(device)
        state = batch["state"].to(device)
        fm = batch["failure_mode"].to(device)
        obj_feat = batch["obj_features"].to(device)
        obj_mask = batch["obj_mask"].to(device)
        t_hit = batch["target_hit"].to(device)
        t_force = batch["target_force"].to(device)
        t_centroid = batch["target_centroid"].to(device)
        ee_img = batch["ee_image"].to(device) if "ee_image" in batch else None

        hit_logits, force_preds = model(img, state, fm, obj_feat, obj_mask, ee_img)
        loss, parts = criterion(hit_logits, force_preds, t_hit, t_force,
                                t_centroid, obj_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = img.shape[0]
        total_loss += parts["total"] * bs
        for k in loss_parts:
            loss_parts[k] += parts[k] * bs
        n += bs

    return {k: v / n for k, v in {**loss_parts, "total": total_loss}.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device, object_names, num_objects):
    model.eval()
    total_loss = 0.0
    loss_parts = {"hit": 0.0, "force": 0.0, "pos": 0.0}
    n = 0

    all_hit_preds = []
    all_hit_targets = []
    all_force_preds = []
    all_force_targets = []
    all_masks = []

    for batch in loader:
        img = batch["image"].to(device)
        state = batch["state"].to(device)
        fm = batch["failure_mode"].to(device)
        obj_feat = batch["obj_features"].to(device)
        obj_mask = batch["obj_mask"].to(device)
        t_hit = batch["target_hit"].to(device)
        t_force = batch["target_force"].to(device)
        t_centroid = batch["target_centroid"].to(device)

        ee_img = batch["ee_image"].to(device) if "ee_image" in batch else None
        hit_logits, force_preds = model(img, state, fm, obj_feat, obj_mask, ee_img)
        loss, parts = criterion(hit_logits, force_preds, t_hit, t_force,
                                t_centroid, obj_mask)

        bs = img.shape[0]
        total_loss += parts["total"] * bs
        for k in loss_parts:
            loss_parts[k] += parts[k] * bs
        n += bs

        all_hit_preds.append(torch.sigmoid(hit_logits).cpu())
        all_hit_targets.append(t_hit.cpu())
        all_force_preds.append(force_preds[..., 0].cpu())
        all_force_targets.append(t_force.cpu())
        all_masks.append(obj_mask.cpu())

    losses = {k: v / n for k, v in {**loss_parts, "total": total_loss}.items()}

    hit_preds = torch.cat(all_hit_preds)
    hit_targets = torch.cat(all_hit_targets)
    force_preds = torch.cat(all_force_preds)
    force_targets = torch.cat(all_force_targets)
    masks = torch.cat(all_masks)

    metrics = compute_metrics(hit_preds, hit_targets, force_preds, force_targets,
                              masks, object_names, num_objects)
    metrics.update(losses)
    return metrics


def compute_metrics(hit_preds, hit_targets, force_preds, force_targets,
                    obj_mask, object_names, num_objects, threshold=0.5):
    """Compute per-object and aggregate metrics, respecting the object mask."""
    pred_binary = (hit_preds >= threshold).float()

    per_obj = {}
    for k in range(num_objects):
        # Only consider samples where this object slot is valid
        valid = obj_mask[:, k]
        if not valid.any():
            continue
        preds_k = pred_binary[valid, k]
        targets_k = hit_targets[valid, k]
        tp = ((preds_k == 1) & (targets_k == 1)).sum().item()
        fp = ((preds_k == 1) & (targets_k == 0)).sum().item()
        fn = ((preds_k == 0) & (targets_k == 1)).sum().item()
        tn = ((preds_k == 0) & (targets_k == 0)).sum().item()
        total = tp + fp + fn + tn
        per_obj[object_names[k]] = {
            "accuracy": (tp + tn) / total if total > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "support": int(targets_k.sum()),
        }

    f1s = []
    for m in per_obj.values():
        p, r = m["precision"], m["recall"]
        if p + r > 0:
            f1s.append(2 * p * r / (p + r))
    macro_f1 = np.mean(f1s) if f1s else 0.0

    # Force MAE on valid hit objects
    hit_and_valid = obj_mask.bool() & hit_targets.bool()
    if hit_and_valid.any():
        force_mae = (force_preds[hit_and_valid] - force_targets[hit_and_valid]).abs().mean().item()
    else:
        force_mae = 0.0

    return {"macro_f1": macro_f1, "force_mae": force_mae, "per_object": per_obj}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def run_cross_validation(dataset, n_folds=5, epochs=50, lr=1e-3,
                         batch_size=8, device="cpu", output_dir=None):
    exp_files = sorted(set(dataset.get_experiment_ids()))
    n_exp = len(exp_files)
    fold_size = max(1, n_exp // n_folds)

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_exp)

    fold_results = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = min(start + fold_size, n_exp)
        test_exp = [exp_files[perm[i]] for i in range(start, end)]
        train_idx, test_idx = dataset.split_by_experiment(test_exp)

        if not test_idx:
            continue

        logger.info(f"Fold {fold+1}/{n_folds}: "
                    f"train={len(train_idx)}, test={len(test_idx)}, "
                    f"test_exps={test_exp}")

        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx),
                                 batch_size=batch_size)

        # Compute per-object pos_weight for class imbalance
        train_hits = torch.stack([dataset[i]["target_hit"] for i in train_idx])
        train_masks = torch.stack([dataset[i]["obj_mask"] for i in train_idx])
        # Only count valid slots
        valid_counts = train_masks.float().sum(dim=0).clamp(min=1)
        pos_counts = (train_hits * train_masks.float()).sum(dim=0)
        pos_freq = pos_counts / valid_counts
        pos_weight = ((1 - pos_freq) / pos_freq.clamp(min=0.01)).clamp(max=20)

        in_ch = 4 if dataset.has_depth else 3
        ee_ch = 4 if (dataset.has_ee_cam and dataset.has_depth) else 3
        model = ContactPredictor(
            freeze_backbone=True, in_channels=in_ch,
            use_ee_camera=dataset.has_ee_cam, ee_in_channels=ee_ch,
        ).to(device)
        criterion = ContactLoss(pos_weight=pos_weight.to(device))
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_f1 = 0.0
        best_metrics = None
        patience_counter = 0
        patience = 3  # stop after 3 eval rounds without improvement

        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()

            if epoch % 5 == 0 or epoch == epochs:
                metrics = evaluate(model, test_loader, criterion, device,
                                   dataset.object_names, dataset.num_objects)
                logger.info(f"  Epoch {epoch}: train={train_loss['total']:.4f}, "
                            f"test={metrics['total']:.4f}, "
                            f"F1={metrics['macro_f1']:.3f}, "
                            f"force_mae={metrics['force_mae']:.2f} N")
                if metrics["macro_f1"] > best_f1:
                    best_f1 = metrics["macro_f1"]
                    best_metrics = metrics
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"  Early stopping at epoch {epoch}")
                        break

        if best_metrics:
            fold_results.append(best_metrics)
            logger.info(f"  Fold {fold+1} best: F1={best_metrics['macro_f1']:.3f}, "
                        f"force_mae={best_metrics['force_mae']:.2f} N")

    # Aggregate
    if fold_results:
        mean_f1 = np.mean([r["macro_f1"] for r in fold_results])
        mean_mae = np.mean([r["force_mae"] for r in fold_results])
        logger.info(f"\nCross-validation ({n_folds} folds):")
        logger.info(f"  Macro F1: {mean_f1:.3f} +/- "
                    f"{np.std([r['macro_f1'] for r in fold_results]):.3f}")
        logger.info(f"  Force MAE: {mean_mae:.2f} +/- "
                    f"{np.std([r['force_mae'] for r in fold_results]):.2f} N")

        logger.info("\nPer-object (averaged):")
        all_obj_names = set()
        for r in fold_results:
            all_obj_names.update(r["per_object"].keys())
        for obj in sorted(all_obj_names):
            accs = [r["per_object"][obj]["accuracy"]
                    for r in fold_results if obj in r["per_object"]]
            recalls = [r["per_object"][obj]["recall"]
                       for r in fold_results if obj in r["per_object"]]
            supports = [r["per_object"][obj]["support"]
                        for r in fold_results if obj in r["per_object"]]
            if accs:
                logger.info(f"  {obj:25s}  acc={np.mean(accs):.3f}  "
                            f"recall={np.mean(recalls):.3f}  "
                            f"support={np.mean(supports):.1f}")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            results = {
                "mean_macro_f1": float(mean_f1),
                "mean_force_mae": float(mean_mae),
                "n_folds": n_folds,
                "num_objects": dataset.num_objects,
                "object_names": dataset.object_names,
                "per_fold": [{k: v for k, v in r.items() if k != "per_object"}
                             for r in fold_results],
            }
            with open(output_dir / "cv_results.json", "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_dir / 'cv_results.json'}")

    return fold_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train FailBench contact predictor")
    parser.add_argument("--dataset-dir", default="datasets/v5")
    parser.add_argument("--scene-xml", default="franka_emika_panda/scene_level2.xml")
    parser.add_argument("--output-dir", default="models/results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger.info(f"Loading dataset from {args.dataset_dir}")
    dataset = FailBenchDataset(args.dataset_dir, args.scene_xml)
    logger.info(f"Dataset: {len(dataset)} samples, {dataset.num_objects} objects discovered")
    logger.info(f"Objects: {dataset.object_names}")

    run_cross_validation(
        dataset,
        n_folds=args.folds,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
