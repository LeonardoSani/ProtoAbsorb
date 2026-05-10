"""CoTTA: Continual Test-Time Adaptation.

Simplified faithful implementation of:
  Wang et al. "Continual Test-Time Domain Adaptation" (CVPR 2022)

Core algorithm:
  1. Augmentation-averaged predictions: compute softmax over K augmentations,
     average them as pseudo-labels.
  2. Entropy minimization on augmented images using pseudo-labels as targets.
  3. Stochastic restore: with probability p_restore, reset each BN parameter
     to its source (pre-adaptation) value to prevent error accumulation.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=1)
    return -(p * p.clamp_min(1e-12).log()).sum(dim=1)


@dataclass
class CoTTAConfig:
    lr: float = 1e-3
    momentum: float = 0.9
    n_augments: int = 32    # number of augmentations for prediction averaging
    p_restore: float = 0.01  # per-parameter restore probability each step


class CoTTAState:
    """Stores source model parameters for stochastic restore."""

    def __init__(self, model: nn.Module):
        # Save a snapshot of all BN affine params at source
        self.source_params: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.source_params[name] = p.data.clone().cpu()


# CIFAR-32 augmentation set for prediction averaging
_CIFAR_AUGS = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])


def _augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Apply random pixel-space augmentation to a batch (CPU or GPU)."""
    out = []
    for img in images:
        img_cpu = img.cpu()
        aug = _CIFAR_AUGS(img_cpu)
        out.append(aug)
    return torch.stack(out).to(images.device)


def cotta_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    state: CoTTAState,
    cfg: CoTTAConfig,
) -> None:
    """One CoTTA adaptation step."""
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()

    # ---- Step 1: augmentation-averaged pseudo-labels ----
    with torch.no_grad():
        probs_list = []
        for _ in range(cfg.n_augments):
            aug_imgs = _augment_batch(images)
            logits = model(aug_imgs)
            probs_list.append(F.softmax(logits, dim=1))
        pseudo_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # (N, C)

    # ---- Step 2: entropy minimization on augmented images ----
    aug_imgs = _augment_batch(images)
    logits = model(aug_imgs)
    log_p = F.log_softmax(logits, dim=1)
    loss = -(pseudo_probs.detach() * log_p).sum(dim=1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ---- Step 3: stochastic restore ----
    if cfg.p_restore > 0.0:
        for name, p in model.named_parameters():
            if p.requires_grad and name in state.source_params:
                if random.random() < cfg.p_restore:
                    p.data.copy_(state.source_params[name].to(p.device))
