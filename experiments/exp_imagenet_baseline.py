"""Step 1 (ImageNet scale) — Verify ResNet-50 clean baseline.

Expected outputs:
  - Top-1 accuracy on ImageNet val: ~76%
  - MSP-AUROC / FPR95 on NINCO and DTD: AUROC ~0.87-0.92 (NINCO), ~0.79-0.85 (DTD)

Run:
  python -m experiments.exp_imagenet_baseline --data-root data --out results/imagenet_baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    ImageNetC,
    NincoOOD,
    dtd_ood_224,
    imagenet_val_dataset,
    get_ood_dataset_224,
)
from proto_absorb.metrics import auroc, fpr95
from models import build_resnet50
from proto_absorb.scorers import energy_score, msp_score
from utils import ensure_dir, get_device, get_logger, set_seed


def evaluate_loader(model, loader, device, max_batches=None):
    """Forward pass over loader; return logits, labels."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            logits = model(imgs.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(lbls)
    return torch.cat(all_logits), torch.cat(all_labels)


def ood_metrics(id_logits, ood_logits):
    """Return dict of AUROC + FPR95 for MSP and energy scores."""
    id_msp = msp_score(id_logits).numpy()
    ood_msp = msp_score(ood_logits).numpy()
    id_eng = energy_score(id_logits).numpy()
    ood_eng = energy_score(ood_logits).numpy()
    return {
        "msp_auroc": auroc(id_msp, ood_msp),
        "msp_fpr95": fpr95(id_msp, ood_msp),
        "energy_auroc": auroc(id_eng, ood_eng),
        "energy_fpr95": fpr95(id_eng, ood_eng),
        "n_id": len(id_msp),
        "n_ood": len(ood_msp),
    }


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/imagenet_baseline")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ood", nargs="+", default=["ninco", "dtd"])
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip ImageNet val accuracy (faster; use if val set unavailable).")
    parser.add_argument("--max-val-batches", type=int, default=None,
                        help="Cap val batches for a quick smoke test.")
    parser.add_argument("--id-corruption", default="gaussian_noise",
                        help="ImageNet-C corruption for ID pool in OOD eval (sev 5).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log = get_logger("imagenet_baseline")
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    log.info("Loading ResNet-50 (torchvision pretrained)...")
    model = build_resnet50().to(device)
    model.eval()

    results: dict = {}

    # ---- Clean accuracy on ImageNet val ----
    if not args.skip_accuracy:
        log.info("Evaluating on ImageNet val...")
        try:
            val_ds = imagenet_val_dataset(args.data_root)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True)
            logits, labels = evaluate_loader(model, val_loader, device,
                                             max_batches=args.max_val_batches)
            preds = logits.argmax(dim=1)
            top1 = float((preds == labels).float().mean())
            log.info(f"  Top-1 accuracy: {top1:.4f}  (expected ~0.76)")
            results["imagenet_val_top1"] = top1
        except FileNotFoundError as e:
            log.warning(f"  Skipped (not found): {e}")

    # ---- OOD detection baselines ----
    log.info(f"Loading ID pool (ImageNet-C / {args.id_corruption} / sev=5)...")
    try:
        id_ds = ImageNetC(args.data_root, corruption=args.id_corruption, severity=5)
        id_loader = DataLoader(id_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True)
        log.info(f"  ID pool size: {len(id_ds)}")
        id_logits, _ = evaluate_loader(model, id_loader, device)
    except FileNotFoundError as e:
        log.error(f"ImageNet-C not found: {e}")
        return

    results["ood"] = {}
    for ood_name in args.ood:
        log.info(f"Evaluating OOD: {ood_name}...")
        try:
            ood_ds = get_ood_dataset_224(ood_name, args.data_root)
            ood_loader = DataLoader(ood_ds, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True)
            log.info(f"  OOD pool size: {len(ood_ds)}")
            ood_logits, _ = evaluate_loader(model, ood_loader, device)
            m = ood_metrics(id_logits, ood_logits)
            results["ood"][ood_name] = m
            log.info(
                f"  MSP  AUROC={m['msp_auroc']:.4f}  FPR95={m['msp_fpr95']:.4f}"
                f"  |  Energy AUROC={m['energy_auroc']:.4f}  FPR95={m['energy_fpr95']:.4f}"
            )
        except FileNotFoundError as e:
            log.warning(f"  Skipped (not found): {e}")

    out_path = out_dir / "baseline_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    cli()
