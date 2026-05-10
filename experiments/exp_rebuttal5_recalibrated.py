"""Rebuttal Experiment 5 — Recalibrated-detector contract.

Reviewer concern: The frozen-detector contract may conflate detector
miscalibration (the adapted model's feature space has shifted, so old
class means no longer align) with genuine degraded ID/OOD separability.

This experiment tests whether the paired contamination gap persists when
detectors are recalibrated *after* adaptation using a small held-out
clean ID calibration set (1 000 labelled CIFAR-10 training samples).

Two evaluation contracts compared side-by-side:
  Frozen contract     — Mahalanobis centroids + KNN feature bank built
                        from the *pretrained* model (standard protocol).
  Recalibrated contract — After adaptation, extract features from the
                        *adapted* model on the calibration set; recompute
                        class means, shared covariance, and KNN bank.
                        Evaluate AUROC with these updated detectors.

If the paired gap (id_only > mixed) persists under the recalibrated
contract:
  → The gap is NOT an artefact of detector miscalibration; OOD gradient
    contamination genuinely degrades ID/OOD separability in feature space.
If the gap shrinks substantially:
  → Co-maintaining detectors with TTA is necessary; the gap then measures
    both separability loss and miscalibration.

Design:
  TENT × SVHN+DTD × gaussian_noise+fog × α∈{0.5,0.9}
  n=30 batches, T=10 steps, calibration set = 1 000 CIFAR-10 train samples

Output: results/rebuttal5/recalibrated_detector_results.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from experiments._common import (
    DEFAULT_BATCH_SIZE,
    bootstrap_ci,
    draw_paired_batches,
    ensure_dir,
    fresh_tent_model,
    get_device,
    get_logger,
    make_sampler,
    paired_t_test,
    save_json,
    set_seed,
    TentConfig,
    TentVariant,
    tent_step,
)
from proto_absorb.data import cifar10_train
from proto_absorb.metrics import auroc, fpr95
from proto_absorb.models import build_resnet18, load_checkpoint
from proto_absorb.scorers import (
    build_feature_bank,
    knn_score,
    mahalanobis_score,
)

OODS      = ["svhn", "dtd"]
CORRUPTS  = ["gaussian_noise", "fog"]
ALPHAS    = [0.9, 0.5]
N_BATCHES = 30
N_STEPS   = 10
KNN_K     = 50
CALIB_N   = 1_000   # held-out clean ID samples for recalibration
NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Calibration set helpers
# ---------------------------------------------------------------------------

def build_calib_dataset(data_root: str) -> Subset:
    """Return the first CALIB_N samples of CIFAR-10 train as calibration set."""
    full = cifar10_train(data_root, augment=False)
    indices = list(range(min(CALIB_N, len(full))))
    return Subset(full, indices)


@torch.no_grad()
def extract_calib_features(
    model: torch.nn.Module,
    calib_ds,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (features, labels) from calibration set using *current* model."""
    model.eval()
    loader = DataLoader(calib_ds, batch_size=256, shuffle=False, num_workers=2)
    feats_list, labels_list = [], []
    for imgs, labels in loader:
        _, feats = model(imgs.to(device), return_features=True)
        feats_list.append(feats.cpu())
        labels_list.append(labels.cpu())
    return torch.cat(feats_list), torch.cat(labels_list)


def compute_mahal_params(
    feats: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    reg: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-class means + shared inverse covariance from (feats, labels)."""
    D = feats.shape[1]
    means = torch.zeros(num_classes, D)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            means[c] = feats[mask].mean(dim=0)

    centered = feats - means[labels]
    cov = (centered.T @ centered) / max(len(feats) - 1, 1)
    cov_inv = torch.linalg.inv(cov + reg * torch.eye(D))
    return means, cov_inv


# ---------------------------------------------------------------------------
# Per-batch evaluation under both contracts
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_both_contracts(
    model: torch.nn.Module,
    images: torch.Tensor,
    is_ood: torch.Tensor,
    # frozen artefacts (from pretrained model)
    frozen_knn_bank: torch.Tensor,
    frozen_mahal_means: torch.Tensor,
    frozen_mahal_cov_inv: torch.Tensor,
    # calibration set (recalibration uses current model)
    calib_ds,
    device: torch.device,
) -> dict:
    """Evaluate KNN and Mahalanobis AUROC under frozen and recalibrated contracts."""
    model.eval()
    _, feats = model(images, return_features=True)
    feats = feats.detach()
    is_ood_np = is_ood.cpu().numpy().astype(bool)

    results = {}

    # --- Frozen contract ---
    knn_s_frozen = knn_score(feats, frozen_knn_bank.to(device), k=KNN_K).cpu().numpy()
    mhl_s_frozen = mahalanobis_score(
        feats.cpu(), frozen_mahal_means, frozen_mahal_cov_inv
    ).numpy()

    for name, scores in [("knn_frozen", knn_s_frozen), ("mahal_frozen", mhl_s_frozen)]:
        results[f"{name}_auroc"] = auroc(scores[~is_ood_np], scores[is_ood_np])
        results[f"{name}_fpr95"] = fpr95(scores[~is_ood_np], scores[is_ood_np])

    # --- Recalibrated contract ---
    cal_feats, cal_labels = extract_calib_features(model, calib_ds, device)
    recal_knn_bank = cal_feats  # adapted-model features as reference bank
    recal_means, recal_cov_inv = compute_mahal_params(cal_feats, cal_labels)

    knn_s_recal = knn_score(feats, recal_knn_bank.to(device), k=KNN_K).cpu().numpy()
    mhl_s_recal = mahalanobis_score(
        feats.cpu(), recal_means, recal_cov_inv
    ).numpy()

    for name, scores in [("knn_recal", knn_s_recal), ("mahal_recal", mhl_s_recal)]:
        results[f"{name}_auroc"] = auroc(scores[~is_ood_np], scores[is_ood_np])
        results[f"{name}_fpr95"] = fpr95(scores[~is_ood_np], scores[is_ood_np])

    return results


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "knn_frozen", "mahal_frozen",
    "knn_recal",  "mahal_recal",
]
CONDITIONS = ["no_tta", "id_only", "mixed"]


def run_cell(
    ckpt_path: str,
    data_root: str,
    corruption: str,
    alpha: float,
    ood_name: str,
    n_batches: int,
    n_steps: int,
    batch_size: int,
    master_seed: int,
    lr: float,
    device: torch.device,
    frozen_knn_bank: torch.Tensor,
    frozen_mahal_means: torch.Tensor,
    frozen_mahal_cov_inv: torch.Tensor,
    calib_ds,
    log,
) -> dict:
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity=5)
    batches = draw_paired_batches(sampler, n_batches, master_seed)

    per = {f"{m}_{c}_{cond}": []
           for m in METRIC_KEYS
           for c in ["auroc", "fpr95"]
           for cond in CONDITIONS}

    cfg = TentConfig(variant=TentVariant.VANILLA, lr=lr)

    for batch in batches:
        batch_dev = batch.to(device)
        images, is_ood = batch_dev.images, batch_dev.is_ood

        for condition in CONDITIONS:
            model, opt = fresh_tent_model(ckpt_path, device, cfg)

            if condition != "no_tta":
                for _ in range(n_steps):
                    if condition == "id_only":
                        id_mask = ~batch_dev.is_ood
                        if id_mask.any():
                            tent_step(model, opt, batch_dev.images[id_mask], cfg)
                    else:  # mixed
                        tent_step(model, opt, batch_dev.images, cfg)

            ev = evaluate_both_contracts(
                model, images, is_ood,
                frozen_knn_bank, frozen_mahal_means, frozen_mahal_cov_inv,
                calib_ds, device,
            )
            for m in METRIC_KEYS:
                for c in ["auroc", "fpr95"]:
                    per[f"{m}_{c}_{condition}"].append(ev[f"{m}_{c}"])

    # Aggregate: mean, CI, paired gap id_only - mixed
    summary: dict = {}
    for m in METRIC_KEYS:
        for c in ["auroc", "fpr95"]:
            for cond in CONDITIONS:
                key = f"{m}_{c}_{cond}"
                mean, lo, hi = bootstrap_ci(per[key])
                summary[f"{key}_mean"] = mean
                summary[f"{key}_ci_lo"] = lo
                summary[f"{key}_ci_hi"] = hi

            # paired gap
            gap_vals = [a - b for a, b in zip(per[f"{m}_{c}_id_only"],
                                               per[f"{m}_{c}_mixed"])]
            t_stat, p_val = paired_t_test(per[f"{m}_{c}_id_only"],
                                           per[f"{m}_{c}_mixed"])
            gm, glo, ghi = bootstrap_ci(gap_vals)
            summary[f"{m}_{c}_gap_mean"]   = gm
            summary[f"{m}_{c}_gap_ci_lo"]  = glo
            summary[f"{m}_{c}_gap_ci_hi"]  = ghi
            summary[f"{m}_{c}_gap_p"]      = p_val
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default="checkpoints/resnet18_cifar10.pt")
    parser.add_argument("--data-root",  default="data")
    parser.add_argument("--out",        default="results/rebuttal5")
    parser.add_argument("--batches",    type=int,   default=N_BATCHES)
    parser.add_argument("--steps",      type=int,   default=N_STEPS)
    parser.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--smoke",      action="store_true",
                        help="3 batches / 2 steps for a fast correctness check")
    args = parser.parse_args()

    log = get_logger("rebuttal5_recalibrated")
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device  = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.batches = 3
        args.steps   = 2

    # --- Build frozen artefacts from pretrained model once ---
    log.info("Building frozen detector artefacts from pretrained model...")
    model_pre = build_resnet18(num_classes=10).to(device)
    load_checkpoint(model_pre, args.ckpt, map_location=str(device))
    model_pre.eval()

    train_ds = cifar10_train(args.data_root, augment=False)
    frozen_knn_bank = build_feature_bank(
        model_pre, train_ds, device, batch_size=512, max_samples=10_000,
    )
    log.info(f"  Frozen KNN bank: {frozen_knn_bank.shape}")

    calib_ds = build_calib_dataset(args.data_root)
    cal_feats_pre, cal_labels = extract_calib_features(model_pre, calib_ds, device)
    frozen_mahal_means, frozen_mahal_cov_inv = compute_mahal_params(
        cal_feats_pre, cal_labels
    )
    log.info(f"  Frozen Mahal: means={frozen_mahal_means.shape}, "
             f"cov_inv={frozen_mahal_cov_inv.shape}")
    del model_pre  # free VRAM

    # --- Main experiment loop ---
    all_results: dict = {}
    total = len(OODS) * len(CORRUPTS) * len(ALPHAS)
    idx = 0

    for corruption in CORRUPTS:
        for ood_name in OODS:
            for alpha in ALPHAS:
                idx += 1
                key = f"{corruption}|{ood_name}|{alpha}"
                log.info(f"[{idx}/{total}] tent|{key}")

                master_seed = (
                    args.seed * 7919
                    + abs(hash(corruption)) % 997
                    + abs(hash(ood_name))   % 997
                    + int(alpha * 1000)
                )
                cell = run_cell(
                    ckpt_path=args.ckpt,
                    data_root=args.data_root,
                    corruption=corruption,
                    alpha=alpha,
                    ood_name=ood_name,
                    n_batches=args.batches,
                    n_steps=args.steps,
                    batch_size=args.batch_size,
                    master_seed=master_seed,
                    lr=args.lr,
                    device=device,
                    frozen_knn_bank=frozen_knn_bank,
                    frozen_mahal_means=frozen_mahal_means,
                    frozen_mahal_cov_inv=frozen_mahal_cov_inv,
                    calib_ds=calib_ds,
                    log=log,
                )
                all_results[key] = cell

                for m in METRIC_KEYS:
                    log.info(
                        f"  {m:14s}: "
                        f"id_only={cell[f'{m}_auroc_id_only_mean']:.4f}  "
                        f"mixed={cell[f'{m}_auroc_mixed_mean']:.4f}  "
                        f"gap={cell[f'{m}_auroc_gap_mean']:+.4f}  "
                        f"p={cell[f'{m}_auroc_gap_p']:.3g}"
                    )

    save_json({"results": all_results, "config": vars(args)},
              out_dir / "recalibrated_detector_results.json")
    log.info(f"Saved → {out_dir / 'recalibrated_detector_results.json'}")

    # --- Summary: avg over all cells at each alpha ---
    log.info("\n=== Summary by contract and alpha ===")
    log.info(f"{'contract':14s}  alpha  no_tta  id_only  mixed  paired_gap  p")
    for m in METRIC_KEYS:
        for alpha in ALPHAS:
            cells = [v for k, v in all_results.items() if f"|{alpha}" in k]
            if not cells:
                continue
            avg = lambda k: float(np.mean([c[k] for c in cells]))
            log.info(
                f"  {m:14s}  {alpha}   "
                f"{avg(f'{m}_auroc_no_tta_mean'):.4f}   "
                f"{avg(f'{m}_auroc_id_only_mean'):.4f}    "
                f"{avg(f'{m}_auroc_mixed_mean'):.4f}  "
                f"{avg(f'{m}_auroc_gap_mean'):+.4f}       "
                f"{avg(f'{m}_auroc_gap_p'):.3g}"
            )


if __name__ == "__main__":
    cli()
