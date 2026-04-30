"""Centroid computation: frozen and dynamic regimes.

Frozen centroids are computed once on clean CIFAR-10 with the pretrained
encoder. Dynamic centroids are recomputed (in an EMA-friendly way) using the
current encoder on the ID-only samples in each adaptation batch — oracle
labels are used **only** for the dynamic update, never for TENT itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import ResNet18


@dataclass
class CentroidBank:
    """Class centroid storage with optional running covariance for Mahalanobis."""

    centroids: torch.Tensor          # (C, D)
    counts: torch.Tensor             # (C,) running sample count
    cov: Optional[torch.Tensor] = None        # (D, D) shared covariance
    cov_inv: Optional[torch.Tensor] = None    # (D, D) inverse

    @property
    def num_classes(self) -> int:
        return self.centroids.shape[0]

    @property
    def feat_dim(self) -> int:
        return self.centroids.shape[1]

    def to(self, device: torch.device) -> "CentroidBank":
        return CentroidBank(
            centroids=self.centroids.to(device),
            counts=self.counts.to(device),
            cov=self.cov.to(device) if self.cov is not None else None,
            cov_inv=self.cov_inv.to(device) if self.cov_inv is not None else None,
        )

    def clone(self) -> "CentroidBank":
        return CentroidBank(
            centroids=self.centroids.clone(),
            counts=self.counts.clone(),
            cov=None if self.cov is None else self.cov.clone(),
            cov_inv=None if self.cov_inv is None else self.cov_inv.clone(),
        )

    def save(self, path: str) -> None:
        out = {"centroids": self.centroids.cpu().numpy(),
               "counts": self.counts.cpu().numpy()}
        if self.cov is not None:
            out["cov"] = self.cov.cpu().numpy()
        if self.cov_inv is not None:
            out["cov_inv"] = self.cov_inv.cpu().numpy()
        np.savez(path, **out)

    @staticmethod
    def load(path: str) -> "CentroidBank":
        data = np.load(path)
        cov = torch.from_numpy(data["cov"]) if "cov" in data.files else None
        cov_inv = torch.from_numpy(data["cov_inv"]) if "cov_inv" in data.files else None
        return CentroidBank(
            centroids=torch.from_numpy(data["centroids"]).float(),
            counts=torch.from_numpy(data["counts"]).float(),
            cov=cov.float() if cov is not None else None,
            cov_inv=cov_inv.float() if cov_inv is not None else None,
        )


@torch.no_grad()
def compute_centroids(
    model: ResNet18,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    with_covariance: bool = True,
) -> CentroidBank:
    """Compute per-class mean features and (optionally) shared covariance."""
    model.eval()
    feat_dim = model.feat_dim
    sums = torch.zeros(num_classes, feat_dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    cov_acc = torch.zeros(feat_dim, feat_dim, device=device) if with_covariance else None
    feat_total_count = 0
    feat_global_sum = torch.zeros(feat_dim, device=device) if with_covariance else None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feats = model.features(images)  # (B, D)
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                sums[c] += feats[mask].sum(dim=0)
                counts[c] += mask.sum()

    centroids = sums / counts.clamp_min(1).unsqueeze(1)

    cov = None
    if with_covariance:
        # Tied (class-conditional) covariance, second pass.
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feats = model.features(images)
            centered = feats - centroids[labels]
            cov_acc += centered.t() @ centered
            feat_total_count += feats.shape[0]
        cov = cov_acc / max(1, feat_total_count - num_classes)

    cov_inv = None
    if cov is not None:
        # Regularize for stability.
        eps = 1e-4
        reg = cov + eps * torch.eye(feat_dim, device=device)
        cov_inv = torch.linalg.inv(reg)

    return CentroidBank(
        centroids=centroids.cpu(),
        counts=counts.cpu(),
        cov=cov.cpu() if cov is not None else None,
        cov_inv=cov_inv.cpu() if cov_inv is not None else None,
    )


@torch.no_grad()
def update_centroids_dynamic(
    bank: CentroidBank,
    feats: torch.Tensor,           # (N, D), the ID samples this step
    labels: torch.Tensor,          # (N,), oracle class labels
    momentum: float = 0.9,
) -> CentroidBank:
    """EMA update of centroids using the ID samples present in the current batch.

    For every class with samples in the batch, the centroid is moved toward the
    batch mean of that class:  mu_c <- m * mu_c + (1-m) * batch_mean_c.
    """
    new_centroids = bank.centroids.clone().to(feats.device)
    new_counts = bank.counts.clone().to(feats.device)
    for c in labels.unique().tolist():
        if c < 0:
            continue
        mask = labels == c
        if mask.any():
            mean_c = feats[mask].mean(dim=0)
            new_centroids[c] = momentum * new_centroids[c] + (1.0 - momentum) * mean_c
            new_counts[c] += mask.sum()
    return CentroidBank(centroids=new_centroids.cpu(),
                        counts=new_counts.cpu(),
                        cov=bank.cov, cov_inv=bank.cov_inv)


def nearest_centroid_distances(
    feats: torch.Tensor, centroids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row in ``feats``, return (min_distance, argmin_class).

    ``feats``: (N, D), ``centroids``: (C, D). Returns (N,) and (N,).
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    a2 = (feats ** 2).sum(dim=1, keepdim=True)
    b2 = (centroids ** 2).sum(dim=1, keepdim=True).t()
    d2 = (a2 + b2 - 2 * feats @ centroids.t()).clamp_min(0.0)
    d = d2.sqrt()
    min_d, argmin = d.min(dim=1)
    return min_d, argmin
