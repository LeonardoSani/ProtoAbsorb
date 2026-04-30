"""TENT (Wang et al., 2021) and proposed fixes.

TENT minimizes the entropy of model predictions on each test batch by
updating only the affine parameters (gamma, beta) of BatchNorm layers,
while running BN in train-mode so that batch statistics adapt.

This module provides:
 - ``configure_tent_model`` / ``collect_bn_params``: the standard TENT setup.
 - ``tent_loss``                   : vanilla mean entropy loss.
 - ``weighted_entropy_loss``       : Fix A (MSP-confidence weighted entropy).
 - ``prototype_anchor_loss``       : Fix B (regularize features toward frozen
                                    nearest-class prototype, only for samples
                                    confidently classified as ID).
 - ``filter_hard_ood``             : Fix C (drop top-fraction by OOD score
                                    before TENT step).
 - ``tent_step``                   : single optimization step that supports
                                    all variants via a Strategy enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def collect_bn_params(model: nn.Module) -> tuple[list[nn.Parameter], list[str]]:
    """Return only the BatchNorm affine parameters and their names."""
    params, names = [], []
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for pname, p in module.named_parameters(recurse=False):
                if pname in {"weight", "bias"}:
                    params.append(p)
                    names.append(f"{module_name}.{pname}")
    return params, names


def configure_tent_model(model: nn.Module) -> nn.Module:
    """Prepare ``model`` for TENT-style adaptation.

    - Sets the model to eval mode globally, then BN layers to train mode so
      that batch statistics are recomputed per batch.
    - Disables ``track_running_stats`` so running averages aren't polluted.
    - Freezes all parameters except BN gamma/beta.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            for pname, p in m.named_parameters(recurse=False):
                if pname in {"weight", "bias"}:
                    p.requires_grad_(True)
    return model


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy of softmax(logits). Returns shape (N,)."""
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=1)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def tent_loss(logits: torch.Tensor) -> torch.Tensor:
    """Vanilla TENT objective: mean entropy of predictions."""
    return softmax_entropy(logits).mean()


def weighted_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Fix A: weight each sample's entropy by its MSP confidence (max softmax).

    Down-weights samples whose predictions are uncertain (likely OOD).
    """
    p = F.softmax(logits, dim=1)
    w = p.max(dim=1).values.detach()              # do not backprop through weights
    H = softmax_entropy(logits)
    denom = w.sum().clamp_min(1e-8)
    return (w * H).sum() / denom


def prototype_anchor_loss(
    feats: torch.Tensor,
    logits: torch.Tensor,
    centroids: torch.Tensor,
    confidence_threshold: float = 0.7,
) -> torch.Tensor:
    """Fix B: pull features of confidently-ID samples toward the frozen
    centroid of their predicted class.

    This anchors the encoder to the original prototypes, making it harder for
    OOD samples to be silently absorbed without distorting the ID manifold.
    """
    if feats.numel() == 0:
        return feats.new_zeros(())
    p = F.softmax(logits, dim=1)
    conf, pred = p.max(dim=1)
    mask = (conf > confidence_threshold).detach()
    if mask.sum() == 0:
        return feats.new_zeros(())
    sel_feats = feats[mask]
    sel_pred = pred[mask].detach()
    target = centroids.to(feats.device)[sel_pred]
    return ((sel_feats - target) ** 2).sum(dim=1).mean()


def filter_hard_ood(
    logits: torch.Tensor, drop_fraction: float
) -> torch.Tensor:
    """Fix C helper: return a boolean mask keeping the (1-drop_fraction)
    most-confident samples (lowest entropy) in the batch.
    """
    if drop_fraction <= 0.0:
        return torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
    H = softmax_entropy(logits).detach()
    n_keep = max(1, int(round((1.0 - drop_fraction) * logits.shape[0])))
    _, idx = torch.topk(H, k=n_keep, largest=False)
    mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
    mask[idx] = True
    return mask


# ---------------------------------------------------------------------------
# Step strategies
# ---------------------------------------------------------------------------

class TentVariant(str, Enum):
    VANILLA = "tent"
    FIX_A_WEIGHTED = "fix_a"
    FIX_B_ANCHOR = "fix_b"
    FIX_C_FILTER = "fix_c"
    FIX_AB = "fix_ab"


@dataclass
class TentConfig:
    variant: TentVariant = TentVariant.VANILLA
    lr: float = 1e-3
    momentum: float = 0.9
    # Fix B
    anchor_weight: float = 0.1
    anchor_threshold: float = 0.7
    # Fix C
    drop_fraction: float = 0.25


def make_optimizer(params: list[nn.Parameter], cfg: TentConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum)


def tent_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    cfg: TentConfig,
    centroids: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """One adaptation step. Returns a small dict of diagnostics."""
    optimizer.zero_grad(set_to_none=True)

    logits, feats = model(images, return_features=True)  # type: ignore[misc]

    if cfg.variant == TentVariant.VANILLA:
        loss = tent_loss(logits)
    elif cfg.variant == TentVariant.FIX_A_WEIGHTED:
        loss = weighted_entropy_loss(logits)
    elif cfg.variant == TentVariant.FIX_B_ANCHOR:
        if centroids is None:
            raise ValueError("Fix B requires centroids.")
        loss = (
            tent_loss(logits)
            + cfg.anchor_weight * prototype_anchor_loss(
                feats, logits, centroids, cfg.anchor_threshold
            )
        )
    elif cfg.variant == TentVariant.FIX_C_FILTER:
        mask = filter_hard_ood(logits, cfg.drop_fraction)
        loss = tent_loss(logits[mask])
    elif cfg.variant == TentVariant.FIX_AB:
        if centroids is None:
            raise ValueError("Fix A+B requires centroids.")
        loss = (
            weighted_entropy_loss(logits)
            + cfg.anchor_weight * prototype_anchor_loss(
                feats, logits, centroids, cfg.anchor_threshold
            )
        )
    else:
        raise ValueError(f"Unknown TENT variant: {cfg.variant}")

    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach().cpu())}


@torch.no_grad()
def snapshot_logits_and_features(
    model: nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass that returns (logits, features) without affecting BN stats."""
    out = model(images, return_features=True)  # type: ignore[misc]
    return out  # type: ignore[return-value]
