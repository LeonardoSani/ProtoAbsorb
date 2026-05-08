"""Datasets and mixed-batch sampling.

Provides:
 - CIFAR-10 train/test loaders (clean)
 - CIFAR-10-C dataset (per-corruption, severity-indexed)
 - OOD datasets: SVHN (default) or CIFAR-100
 - MixedBatchDataset that yields per-batch (image, label, is_ood) triples,
   where label==-1 for OOD samples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_C_CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)


def _normalize() -> T.Normalize:
    return T.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def train_transform() -> T.Compose:
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        _normalize(),
    ])


def eval_transform() -> T.Compose:
    return T.Compose([T.ToTensor(), _normalize()])


def cifar10_train(data_root: str, augment: bool = True) -> torchvision.datasets.CIFAR10:
    return torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=train_transform() if augment else eval_transform(),
        download=True,
    )


def cifar10_test(data_root: str) -> torchvision.datasets.CIFAR10:
    return torchvision.datasets.CIFAR10(
        root=data_root, train=False, transform=eval_transform(), download=True
    )


# ---------------------------------------------------------------------------
# CIFAR-10-C
# ---------------------------------------------------------------------------

class CIFAR10C(Dataset):
    """CIFAR-10-C dataset for a single corruption type at a single severity.

    Expects ``<root>/CIFAR-10-C/<corruption>.npy`` (50000 x 32 x 32 x 3, uint8)
    and ``<root>/CIFAR-10-C/labels.npy`` (50000,).
    Severity in {1..5}; each severity slice is 10000 consecutive samples.
    """

    URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"

    def __init__(self, data_root: str, corruption: str, severity: int = 5,
                 transform: Optional[T.Compose] = None):
        if corruption not in CIFAR10_C_CORRUPTIONS:
            raise ValueError(f"Unknown corruption '{corruption}'.")
        if severity < 1 or severity > 5:
            raise ValueError("severity must be in {1..5}")

        cifar_c_dir = Path(data_root) / "CIFAR-10-C"
        npy_path = cifar_c_dir / f"{corruption}.npy"
        lbl_path = cifar_c_dir / "labels.npy"
        if not npy_path.exists() or not lbl_path.exists():
            raise FileNotFoundError(
                f"CIFAR-10-C missing at {cifar_c_dir}. "
                f"Run `python -m proto_absorb.scripts.download_data` first."
            )

        all_imgs = np.load(npy_path)  # (50000, 32, 32, 3) uint8
        all_lbls = np.load(lbl_path).astype(np.int64)
        s = severity - 1
        self.images = all_imgs[s * 10000 : (s + 1) * 10000]
        self.labels = all_lbls[s * 10000 : (s + 1) * 10000]
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.images[idx]  # H,W,C uint8
        from PIL import Image
        pil = Image.fromarray(img)
        return self.transform(pil), int(self.labels[idx])


# ---------------------------------------------------------------------------
# OOD datasets
# ---------------------------------------------------------------------------

class _OODWrapper(Dataset):
    """Wraps an OOD dataset to expose (image_tensor, -1)."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.base[idx]
        img = item[0] if isinstance(item, tuple) else item
        return img, -1


def svhn_ood(data_root: str, split: str = "test") -> Dataset:
    base = torchvision.datasets.SVHN(
        root=data_root, split=split, transform=eval_transform(), download=True
    )
    return _OODWrapper(base)


def cifar100_ood(data_root: str, split: str = "test") -> Dataset:
    base = torchvision.datasets.CIFAR100(
        root=data_root, train=(split == "train"),
        transform=eval_transform(), download=True
    )
    return _OODWrapper(base)


def _resize32_eval_transform() -> T.Compose:
    """Eval transform for non-CIFAR-sized OOD datasets: resize to 32x32 then
    normalize with CIFAR-10 statistics so the input distribution matches the
    backbone's training resolution."""
    return T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        _normalize(),
    ])


def dtd_ood(data_root: str, split: str = "test") -> Dataset:
    """DTD (Describable Textures) — semantically disjoint from CIFAR-10
    (textures, no object categories). Standard far-OOD benchmark used in
    Hendrycks 2019, Liang 2018 (ODIN), Liu 2020 (Energy)."""
    base = torchvision.datasets.DTD(
        root=data_root,
        split="test" if split == "test" else "train",
        transform=_resize32_eval_transform(),
        download=True,
    )
    return _OODWrapper(base)


def places365_ood(data_root: str, split: str = "test") -> Dataset:
    """Places365 (val split, small=True ~256px) — scene categories,
    semantically disjoint from CIFAR-10 object categories."""
    base = torchvision.datasets.Places365(
        root=str(Path(data_root) / "places365"),
        split="val",
        small=True,
        transform=_resize32_eval_transform(),
        download=True,
    )
    return _OODWrapper(base)


def get_ood_dataset(name: str, data_root: str, split: str = "test") -> Dataset:
    name = name.lower()
    if name == "svhn":
        return svhn_ood(data_root, split=split)
    if name in {"cifar100", "cifar-100"}:
        return cifar100_ood(data_root, split=split)
    if name == "dtd":
        return dtd_ood(data_root, split=split)
    if name in {"places365", "places"}:
        return places365_ood(data_root, split=split)
    raise ValueError(f"Unknown OOD dataset '{name}'")


# ---------------------------------------------------------------------------
# Mixed batch sampling
# ---------------------------------------------------------------------------

@dataclass
class MixedBatch:
    """A batch of mixed ID + OOD samples."""

    images: torch.Tensor       # (B, 3, 32, 32)
    labels: torch.Tensor       # (B,) int64; -1 for OOD
    is_ood: torch.Tensor       # (B,) bool

    def to(self, device: torch.device) -> "MixedBatch":
        return MixedBatch(
            self.images.to(device, non_blocking=True),
            self.labels.to(device, non_blocking=True),
            self.is_ood.to(device, non_blocking=True),
        )


class MixedBatchSampler:
    """Sample batches of size ``batch_size`` containing roughly ``alpha`` ID samples
    and ``1-alpha`` OOD samples, drawn from ``id_dataset`` and ``ood_dataset``.

    Use ``next_batch(rng)`` to draw a batch. The number of ID/OOD samples per
    batch is ``round(alpha * batch_size)`` and ``batch_size - that``.
    """

    def __init__(self, id_dataset: Dataset, ood_dataset: Dataset,
                 alpha: float, batch_size: int = 64):
        self.id_dataset = id_dataset
        self.ood_dataset = ood_dataset
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_id = round(alpha * batch_size)
        self.n_ood = batch_size - self.n_id

    def next_batch(self, rng: np.random.Generator) -> MixedBatch:
        id_idx = rng.integers(0, len(self.id_dataset), size=self.n_id)
        ood_idx = rng.integers(0, len(self.ood_dataset), size=self.n_ood)

        imgs, lbls, is_ood = [], [], []
        for i in id_idx:
            img, lbl = self.id_dataset[int(i)]
            imgs.append(img)
            lbls.append(int(lbl))
            is_ood.append(False)
        for i in ood_idx:
            img, _ = self.ood_dataset[int(i)]
            imgs.append(img)
            lbls.append(-1)
            is_ood.append(True)

        # shuffle within batch
        order = rng.permutation(len(imgs))
        imgs = torch.stack([imgs[k] for k in order], dim=0)
        lbls_t = torch.tensor([lbls[k] for k in order], dtype=torch.long)
        ood_t = torch.tensor([is_ood[k] for k in order], dtype=torch.bool)
        return MixedBatch(imgs, lbls_t, ood_t)


def make_dataloader(dataset: Dataset, batch_size: int = 128, shuffle: bool = False,
                    num_workers: int = 2) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True
    )
