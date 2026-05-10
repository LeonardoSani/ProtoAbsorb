"""Rebuttal Experiment 1 — KNN and ViM detectors (non-logit-scale).

Reviewer concern: MSP and Energy are directly affected by logit sharpening by
construction. We need stronger, non-logit-scale detectors to rule out that the
contamination effect is an artifact of using detectors that are adversarially
sensitive to entropy minimization.

This experiment reruns the paired contamination protocol with:
  - KNN (Sun et al., ICML 2022): k-NN distance in L2-normalized feature space
  - ViM (Wang et al., CVPR 2022): residual norm in feature null-space

Design:
  - Same conditions as step3 (detector breadth): SVHN + DTD, gaussian_noise + fog,
    TENT + ETA, alpha in {0.9, 0.5}, n=30 batches
  - Feature bank: CIFAR-10 training set (50K samples), extracted once at startup
  - ViM principal subspace: top-d PCA components from training features (d=256)
  - KNN k=50

Output: results/rebuttal1/knn_vim_results.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments._common import (
    DEFAULT_BATCH_SIZE,
    build_id_pool,
    build_ood_pool,
    draw_paired_batches,
    ensure_dir,
    fresh_tent_model,
    get_device,
    get_logger,
    make_sampler,
    paired_t_test,
    bootstrap_ci,
    run_condition_on_batch,
    save_json,
    set_seed,
    setup_matplotlib,
    TentConfig,
    TentVariant,
    EataConfig,
    EataState,
    eata_step,
    collect_bn_params,
    make_optimizer,
)
from proto_absorb.data import cifar10_train
from proto_absorb.models import build_resnet18, load_checkpoint
from proto_absorb.scorers import (
    build_feature_bank,
    build_vim_components,
    knn_score,
    vim_score,
)
from proto_absorb.metrics import auroc, fpr95

OODS       = ["svhn", "dtd"]
CORRUPTS   = ["gaussian_noise", "fog"]
METHODS    = ["tent", "eta"]
ALPHAS     = [0.9, 0.5]
DETECTORS  = ["knn", "vim"]
N_BATCHES  = 30
KNN_K      = 50
VIM_COMPONENTS = 256
TRAIN_MAX_SAMPLES = 10_000   # subset for speed (10K still robust)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def build_banks(ckpt_path: str, data_root: str, device: torch.device, log):
    """Build KNN feature bank and ViM components from CIFAR-10 training set."""
    log.info("Building training feature bank (this runs once)...")
    model = build_resnet18(num_classes=10).to(device)
    load_checkpoint(model, ckpt_path, map_location=str(device))
    model.eval()

    train_ds = cifar10_train(data_root, augment=False)
    train_feats = build_feature_bank(
        model, train_ds, device,
        batch_size=512,
        max_samples=TRAIN_MAX_SAMPLES,
    )
    log.info(f"  Feature bank: {train_feats.shape}")

    log.info("Computing ViM principal subspace...")
    principal_space, id_residual_scale = build_vim_components(
        train_feats,
        n_components=VIM_COMPONENTS,
    )
    log.info(f"  ViM: principal_space={principal_space.shape}, "
             f"id_residual_scale={id_residual_scale:.4f}")

    return train_feats, principal_space, id_residual_scale


# ---------------------------------------------------------------------------
# Per-batch evaluation with KNN / ViM
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_batch_extended(
    model: torch.nn.Module,
    images: torch.Tensor,
    is_ood: torch.Tensor,
    train_feats: torch.Tensor,
    principal_space: torch.Tensor,
    id_residual_scale: float,
) -> dict:
    """Evaluate KNN and ViM scores on a batch. Returns per-detector AUROC+FPR95."""
    model.eval()
    logits, feats = model(images, return_features=True)
    feats = feats.detach()
    is_ood_np = is_ood.cpu().numpy().astype(bool)

    results = {}
    for det in DETECTORS:
        if det == "knn":
            scores = knn_score(feats, train_feats.to(feats.device), k=KNN_K).cpu().numpy()
        elif det == "vim":
            scores = vim_score(feats, principal_space.to(feats.device), id_residual_scale).cpu().numpy()
        else:
            raise ValueError(det)
        id_s  = scores[~is_ood_np]
        ood_s = scores[is_ood_np]
        results[f"{det}_auroc"] = auroc(id_s, ood_s)
        results[f"{det}_fpr95"] = fpr95(id_s, ood_s)
    return results


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def run_cell(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    ood_name: str, method: str, n_batches: int, n_steps: int,
    batch_size: int, master_seed: int, lr: float, device: torch.device,
    train_feats: torch.Tensor, principal_space: torch.Tensor,
    id_residual_scale: float, log,
) -> dict:
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity=5)
    batches = draw_paired_batches(sampler, n_batches, master_seed)

    per_det = {f"{d}_{c}_{cond}": []
               for d in DETECTORS
               for c in ["auroc", "fpr95"]
               for cond in ["no_tta", "id_only", "mixed"]}

    for batch in batches:
        batch_dev = batch.to(device)
        images, is_ood = batch_dev.images, batch_dev.is_ood

        for condition in ["no_tta", "id_only", "mixed"]:
            # --- Build fresh model for this condition ---
            cfg = TentConfig(variant=TentVariant.VANILLA, lr=lr)
            model, opt = fresh_tent_model(ckpt_path, device, cfg)
            tta = "tent" if method == "tent" else "eta"

            if condition != "no_tta":
                eata_state = EataState() if tta == "eta" else None
                eata_cfg   = EataConfig(lr=lr) if tta == "eta" else None
                if tta == "eta" and eata_cfg is not None:
                    params, _ = collect_bn_params(model)
                    opt = torch.optim.SGD(params, lr=eata_cfg.lr, momentum=eata_cfg.momentum)

                for _ in range(n_steps):
                    if condition == "id_only":
                        id_mask = ~batch_dev.is_ood
                        if id_mask.any():
                            sub = batch_dev.images[id_mask]
                            if tta == "eta":
                                eata_step(model, opt, sub, eata_state, eata_cfg)
                            else:
                                from proto_absorb.tent import tent_step
                                tent_step(model, opt, sub, cfg)
                    else:  # mixed
                        if tta == "eta":
                            eata_step(model, opt, batch_dev.images, eata_state, eata_cfg)
                        else:
                            from proto_absorb.tent import tent_step
                            tent_step(model, opt, batch_dev.images, cfg)

            ev = evaluate_batch_extended(
                model, images, is_ood,
                train_feats, principal_space, id_residual_scale,
            )
            for det in DETECTORS:
                for c in ["auroc", "fpr95"]:
                    per_det[f"{det}_{c}_{condition}"].append(ev[f"{det}_{c}"])

    # Aggregate
    summary = {}
    for det in DETECTORS:
        for c in ["auroc", "fpr95"]:
            for cond in ["no_tta", "id_only", "mixed"]:
                key = f"{det}_{c}_{cond}"
                vals = per_det[key]
                mean, lo, hi = bootstrap_ci(vals)
                summary[f"{key}_mean"] = mean
                summary[f"{key}_ci_lo"] = lo
                summary[f"{key}_ci_hi"] = hi

            # paired gap id_only - mixed
            gap_vals = [a - b for a, b in
                        zip(per_det[f"{det}_{c}_id_only"], per_det[f"{det}_{c}_mixed"])]
            t_stat, p_val = paired_t_test(
                per_det[f"{det}_{c}_id_only"],
                per_det[f"{det}_{c}_mixed"],
            )
            gm, glo, ghi = bootstrap_ci(gap_vals)
            summary[f"{det}_{c}_gap_mean"] = gm
            summary[f"{det}_{c}_gap_ci_lo"] = glo
            summary[f"{det}_{c}_gap_ci_hi"] = ghi
            summary[f"{det}_{c}_gap_p"] = p_val
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/resnet18_cifar10.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/rebuttal1")
    parser.add_argument("--batches", type=int, default=N_BATCHES)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("rebuttal1_knn_vim")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device  = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.batches = 3
        args.steps   = 2

    # Build feature banks once
    train_feats, principal_space, id_residual_scale = build_banks(
        args.ckpt, args.data_root, device, log,
    )

    all_results: dict = {}
    total = len(METHODS) * len(OODS) * len(CORRUPTS) * len(ALPHAS)
    idx = 0

    for corruption in CORRUPTS:
        for ood_name in OODS:
            for method in METHODS:
                for alpha in ALPHAS:
                    idx += 1
                    key = f"{corruption}|{ood_name}|{method}|{alpha}"
                    log.info(f"[{idx}/{total}] {key}")
                    master_seed = (
                        args.seed * 7919
                        + abs(hash(corruption)) % 997
                        + abs(hash(ood_name))   % 997
                        + abs(hash(method))     % 997
                        + int(alpha * 1000)
                    )
                    cell = run_cell(
                        ckpt_path=args.ckpt, data_root=args.data_root,
                        corruption=corruption, alpha=alpha, ood_name=ood_name,
                        method=method, n_batches=args.batches,
                        n_steps=args.steps, batch_size=args.batch_size,
                        master_seed=master_seed, lr=args.lr, device=device,
                        train_feats=train_feats, principal_space=principal_space,
                        id_residual_scale=id_residual_scale, log=log,
                    )
                    all_results[key] = cell
                    for det in DETECTORS:
                        log.info(
                            f"  {det.upper()}: "
                            f"id_only_auroc={cell[f'{det}_auroc_id_only_mean']:.4f}  "
                            f"mixed_auroc={cell[f'{det}_auroc_mixed_mean']:.4f}  "
                            f"gap={cell[f'{det}_auroc_gap_mean']:+.4f}  "
                            f"p={cell[f'{det}_auroc_gap_p']:.3g}  "
                            f"FPR95_gap={cell[f'{det}_fpr95_gap_mean']:+.4f}"
                        )

    save_json({"results": all_results, "config": vars(args)},
              out_dir / "knn_vim_results.json")
    log.info(f"Saved → {out_dir / 'knn_vim_results.json'}")

    # Summary table
    log.info("\n=== Summary: alpha=0.9, avg over corruptions × OODs ===")
    log.info(f"{'method':6s} {'det':4s}  id_only_AUROC  mixed_AUROC  gap    p      FPR95_gap")
    for method in METHODS:
        for det in DETECTORS:
            cells = [
                all_results[f"{c}|{o}|{method}|0.9"]
                for c in CORRUPTS for o in OODS
                if f"{c}|{o}|{method}|0.9" in all_results
            ]
            if not cells:
                continue
            avg = lambda k: float(np.mean([v[k] for v in cells]))
            log.info(
                f"  {method:6s} {det:4s}  "
                f"{avg(f'{det}_auroc_id_only_mean'):.4f}         "
                f"{avg(f'{det}_auroc_mixed_mean'):.4f}       "
                f"{avg(f'{det}_auroc_gap_mean'):+.4f}  "
                f"{avg(f'{det}_auroc_gap_p'):.3g}  "
                f"{avg(f'{det}_fpr95_gap_mean'):+.4f}"
            )


if __name__ == "__main__":
    cli()
