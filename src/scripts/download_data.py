"""Download CIFAR-10, CIFAR-10-C, and SVHN into ``data/``.

CIFAR-10-C is fetched from Zenodo
(https://zenodo.org/records/2535967/files/CIFAR-10-C.tar, ~2.6 GB).
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import torchvision

from ProtoAbsorb.src.data.data import CIFAR10_C_CORRUPTIONS

CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"


def _download_with_progress(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _hook(blocks: int, block_size: int, total: int) -> None:
        downloaded = blocks * block_size
        if total > 0:
            pct = min(100.0, 100.0 * downloaded / total)
            sys.stdout.write(
                f"\r  {downloaded/1e6:8.1f} / {total/1e6:8.1f} MB ({pct:5.1f}%)"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()


def fetch_cifar10c(data_root: str) -> None:
    out_dir = Path(data_root) / "CIFAR-10-C"
    if out_dir.exists() and (out_dir / "labels.npy").exists():
        missing = [
            c for c in CIFAR10_C_CORRUPTIONS if not (out_dir / f"{c}.npy").exists()
        ]
        if not missing:
            print(f"CIFAR-10-C already present at {out_dir}; skipping.")
            return
        print(f"CIFAR-10-C present but missing {len(missing)} files; redownloading.")

    tar_path = Path(data_root) / "CIFAR-10-C.tar"
    if not tar_path.exists():
        _download_with_progress(CIFAR10C_URL, tar_path)

    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(path=data_root)
    print(f"Done. CIFAR-10-C is in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--skip-cifar10c", action="store_true", help="Skip CIFAR-10-C (large download)."
    )
    parser.add_argument("--ood", choices=["svhn", "cifar100", "both"], default="svhn")
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)

    print("==> CIFAR-10")
    torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True)
    torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True)

    if not args.skip_cifar10c:
        print("==> CIFAR-10-C")
        fetch_cifar10c(args.data_root)
    else:
        print("==> Skipping CIFAR-10-C (use --skip-cifar10c=False to fetch).")

    if args.ood in ("svhn", "both"):
        print("==> SVHN (test split)")
        torchvision.datasets.SVHN(root=args.data_root, split="test", download=True)
    if args.ood in ("cifar100", "both"):
        print("==> CIFAR-100 (test split)")
        torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True)


if __name__ == "__main__":
    main()
