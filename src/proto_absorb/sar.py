"""SAR: Sharpness-Aware and Reliable Test-Time Adaptation.

Simplified faithful implementation of the key ideas from:
  Niu et al. "Towards Stable Test-Time Adaptation in Dynamic Wild World" (ICLR 2023)

Core algorithm:
  1. Reliable sample selection: keep samples with entropy < E0 (same as ETA).
  2. SAM step: compute gradient at perturbed weights (theta + rho * g/||g||),
     use that gradient to update the original theta.
  3. Model-level stability guard: if adapted model's entropy on reliable samples
     increases above a margin, revert the update.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=1)
    return -(p * p.clamp_min(1e-12).log()).sum(dim=1)


@dataclass
class SARConfig:
    lr: float = 1e-3
    momentum: float = 0.9
    e0: float = 0.4 * 0.301  # entropy threshold ≈ 0.4 * log(C) for C=10 classes
    rho: float = 0.05        # SAM perturbation radius
    margin: float = 0.2      # stability margin: revert if entropy rises > margin


@dataclass
class SARState:
    """Mutable state carried across batches."""
    step_count: int = 0
    reverts: int = 0


def _collect_bn_params(model: nn.Module):
    params, names = [], []
    for mname, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for pname, p in m.named_parameters(recurse=False):
                if pname in {"weight", "bias"}:
                    params.append(p)
                    names.append(f"{mname}.{pname}")
    return params, names


def sar_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    state: SARState,
    cfg: SARConfig,
) -> float:
    """One SAR adaptation step. Returns number of reliable samples used."""
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()

    # ---- Step 1: reliable sample selection ----
    with torch.no_grad():
        logits0 = model(images)
    H = _softmax_entropy(logits0).detach()
    mask = H < cfg.e0
    if mask.sum() == 0:
        state.step_count += 1
        return 0.0

    rel_imgs = images[mask]
    n_rel = int(mask.sum())

    # ---- Step 2: SAM first forward/backward (compute gradient at theta) ----
    optimizer.zero_grad()
    logits_rel = model(rel_imgs)
    loss = _softmax_entropy(logits_rel).mean()
    loss.backward()

    # Compute gradient norm and build perturbation
    params, _ = _collect_bn_params(model)
    grad_norms = [p.grad.norm() for p in params if p.grad is not None]
    if not grad_norms:
        state.step_count += 1
        return float(n_rel)

    total_gnorm = torch.stack(grad_norms).norm()
    if total_gnorm < 1e-8:
        optimizer.step()
        optimizer.zero_grad()
        state.step_count += 1
        return float(n_rel)

    # Perturb: theta_hat = theta + rho * grad/||grad||
    scale = cfg.rho / (total_gnorm + 1e-12)
    saved_params = []
    for p in params:
        saved_params.append(p.data.clone())
        if p.grad is not None:
            p.data.add_(p.grad, alpha=float(scale))

    # ---- Step 3: SAM second forward/backward (gradient at perturbed theta) ----
    optimizer.zero_grad()
    logits_perturb = model(rel_imgs)
    loss_perturb = _softmax_entropy(logits_perturb).mean()
    loss_perturb.backward()

    # Restore original parameters before taking the actual step
    for p, saved in zip(params, saved_params):
        p.data.copy_(saved)

    # ---- Step 4: model-level stability guard ----
    # Compute reference entropy before update
    ref_entropy = float(H[mask].mean())

    # Tentatively apply update
    saved_params2 = [p.data.clone() for p in params]
    optimizer.step()
    optimizer.zero_grad()

    # Check entropy after update
    with torch.no_grad():
        logits_after = model(rel_imgs)
    H_after = _softmax_entropy(logits_after).mean().item()

    if H_after > ref_entropy + cfg.margin:
        # Revert: entropy increased too much (sharpness alarm)
        for p, saved in zip(params, saved_params2):
            p.data.copy_(saved)
        state.reverts += 1

    state.step_count += 1
    return float(n_rel)
