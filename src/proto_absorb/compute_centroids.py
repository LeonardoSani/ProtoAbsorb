"""Compute and save class centroids on clean CIFAR-10 train set."""

from __future__ import annotations

import argparse
from pathlib import Path

from .centroids import compute_centroids
from ..data.data import cifar10_train, make_dataloader
from ..models.models import build_resnet18, load_checkpoint
from ..utils.utils import ensure_dir, get_device, get_logger, set_seed


def cli() -> None:
    p = argparse.ArgumentParser(description="Compute class centroids on clean CIFAR-10")
    p.add_argument("--ckpt", required=True, help="Path to ResNet-18 checkpoint.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="checkpoints/centroids.npz")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-cov", action="store_true", help="Skip covariance computation.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    log = get_logger("centroids")
    set_seed(args.seed)
    device = get_device()
    log.info(f"Device: {device}")

    model = build_resnet18(num_classes=10).to(device)
    load_checkpoint(model, args.ckpt, map_location=str(device))
    model.eval()

    ds = cifar10_train(args.data_root, augment=False)
    loader = make_dataloader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    log.info(f"Computing centroids on {len(ds)} samples (with_cov={not args.no_cov})")
    bank = compute_centroids(model, loader, num_classes=10, device=device,
                             with_covariance=not args.no_cov)

    ensure_dir(Path(args.out).parent)
    bank.save(args.out)
    log.info(f"Saved centroids to {args.out}: shape={tuple(bank.centroids.shape)}, "
             f"cov={bank.cov is not None}")


if __name__ == "__main__":
    cli()
