"""Step 1 — Baseline sanity (BLOCKING).

Goal: confirm that the checkpoint, the OOD scoring pipeline and the centroid
bank are trustworthy before running any TTA experiments. Per the reframe:

  - Clean CIFAR-10 test accuracy (target >= 93%).
  - Clean CIFAR-10 vs SVHN MSP-AUROC (expected >> 0.7).
  - Clean CIFAR-10 vs SVHN Energy-AUROC and Mahalanobis-AUROC.
  - CIFAR-10-C (severity 5) vs SVHN at t=0 for all three detectors,
    averaged over the 15 standard corruptions.

If clean MSP-AUROC <= 0.70 the script exits non-zero so downstream steps
can refuse to run on a bad backbone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments._common import (
    CIFAR10_C_CORRUPTIONS,
    CentroidBank,
    auroc_from_eval,
    energy_score,
    ensure_dir,
    get_device,
    get_logger,
    mahalanobis_score,
    msp_score,
    save_json,
    set_seed,
    setup_matplotlib,
)
from proto_absorb.data import (
    CIFAR10C,
    cifar10_test,
    eval_transform,
    get_ood_dataset,
    make_dataloader,
)
from proto_absorb.metrics import auroc as auroc_fn
from proto_absorb.models import build_resnet18, load_checkpoint


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader: DataLoader,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (logits, features, labels) over the entire loader."""
    model.eval()
    all_l, all_f, all_y = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        l, f = model(images, return_features=True)  # type: ignore[misc]
        all_l.append(l.cpu()); all_f.append(f.cpu())
        all_y.append(labels if torch.is_tensor(labels) else torch.tensor(labels))
    return (torch.cat(all_l).numpy(),
            torch.cat(all_f).numpy(),
            torch.cat(all_y).numpy())


def aurocs_for(id_logits: np.ndarray, id_feats: np.ndarray,
               ood_logits: np.ndarray, ood_feats: np.ndarray,
               bank: CentroidBank | None) -> dict[str, float]:
    out: dict[str, float] = {}
    out["msp"] = auroc_fn(
        msp_score(torch.from_numpy(id_logits)).numpy(),
        msp_score(torch.from_numpy(ood_logits)).numpy(),
    )
    out["energy"] = auroc_fn(
        energy_score(torch.from_numpy(id_logits)).numpy(),
        energy_score(torch.from_numpy(ood_logits)).numpy(),
    )
    if bank is not None and bank.cov_inv is not None:
        out["mahalanobis"] = auroc_fn(
            mahalanobis_score(torch.from_numpy(id_feats),
                              bank.centroids, bank.cov_inv).numpy(),
            mahalanobis_score(torch.from_numpy(ood_feats),
                              bank.centroids, bank.cov_inv).numpy(),
        )
    return out


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--centroids", default=None,
                        help="Centroid bank for Mahalanobis (optional).")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/step1")
    parser.add_argument("--ood", default="svhn", choices=["svhn", "cifar100"])
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--corruptions", nargs="+",
                        default=list(CIFAR10_C_CORRUPTIONS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-clean-msp-auroc", type=float, default=0.70,
                        help="Hard fail threshold (per the reframe).")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("step1")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.corruptions = args.corruptions[:2]

    model = build_resnet18(num_classes=10).to(device)
    load_checkpoint(model, args.ckpt, map_location=str(device))
    model.eval()

    bank = CentroidBank.load(args.centroids) if args.centroids else None

    log.info("==> Clean CIFAR-10 test accuracy")
    clean_test = cifar10_test(args.data_root)
    test_loader = make_dataloader(clean_test, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)
    clean_logits, clean_feats, clean_labels = collect_logits(model, test_loader, device)
    clean_acc = float((clean_logits.argmax(axis=1) == clean_labels).mean())
    log.info(f"  clean test acc: {clean_acc * 100:.2f}%")

    log.info(f"==> {args.ood.upper()} (test) features")
    ood = get_ood_dataset(args.ood, args.data_root, split="test")
    ood_loader = make_dataloader(ood, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
    ood_logits, ood_feats, _ = collect_logits(model, ood_loader, device)

    clean_aurocs = aurocs_for(clean_logits, clean_feats, ood_logits, ood_feats, bank)
    log.info(f"  clean-vs-{args.ood} AUROC: {clean_aurocs}")

    log.info(f"==> CIFAR-10-C (severity {args.severity}) vs {args.ood.upper()}, t=0")
    cifar10c_aurocs: dict[str, dict[str, float]] = {}
    for c in args.corruptions:
        try:
            ds = CIFAR10C(args.data_root, corruption=c, severity=args.severity,
                          transform=eval_transform())
        except FileNotFoundError as e:
            log.warning(f"  skipping {c}: {e}")
            continue
        loader = make_dataloader(ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
        cl, cf, cy = collect_logits(model, loader, device)
        c_acc = float((cl.argmax(axis=1) == cy).mean())
        per_corruption = aurocs_for(cl, cf, ood_logits, ood_feats, bank)
        per_corruption["id_acc"] = c_acc
        cifar10c_aurocs[c] = per_corruption
        log.info(f"  {c}: acc={c_acc*100:.2f}%  AUROC={per_corruption}")

    summary = {
        "ckpt": args.ckpt,
        "ood": args.ood,
        "clean_acc": clean_acc,
        f"clean_vs_{args.ood}_auroc": clean_aurocs,
        f"cifar10c_vs_{args.ood}_auroc_t0": cifar10c_aurocs,
    }
    save_json(summary, out_dir / "sanity.json")

    log.info(f"Saved sanity report to {out_dir / 'sanity.json'}")
    log.info(f"  clean_acc:         {clean_acc*100:.2f}%  (target >= 93%)")
    log.info(f"  clean MSP-AUROC:   {clean_aurocs['msp']:.3f}  "
             f"(threshold > {args.min_clean_msp_auroc})")

    failed = []
    if clean_aurocs["msp"] < args.min_clean_msp_auroc:
        failed.append(f"clean MSP-AUROC {clean_aurocs['msp']:.3f} < "
                      f"{args.min_clean_msp_auroc}")
    if clean_acc < 0.93:
        log.warning(f"  clean accuracy {clean_acc*100:.2f}% below 93% target "
                    "(WARN, not a hard fail)")

    if failed:
        log.error("SANITY FAIL:\n  - " + "\n  - ".join(failed))
        sys.exit(1)
    log.info("Sanity OK.")


if __name__ == "__main__":
    cli()
