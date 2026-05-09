"""ETA: Efficient Test-time Adaptation (EATA without Fisher regularizer).

Faithful but minimal implementation of the two core ideas from
Niu et al. (EATA, 2022):

1. **Reliable sample selection.** Only samples whose entropy is below a
   threshold ``E_0`` contribute to the update. This drops uncertain (often
   OOD or hard) samples from the gradient.
2. **Non-redundant sample selection.** Samples whose softmax prediction is
   too similar (cosine similarity > 1 - epsilon) to a moving-average
   prediction are also dropped, to avoid redundant updates from
   near-duplicate inputs.

Sample weighting follows the EATA recipe:
    w_i = 1 / exp(H(p_i) - E_0)   if H(p_i) < E_0 else 0,
multiplied by the redundancy mask. The loss is a weighted mean entropy.

We omit the Fisher anti-forgetting regularizer (it requires an extra forward
on a held-out source set, which is out of scope for the contamination
analysis here). EATA without Fisher is itself a published-ish baseline often
referred to as "ETA" — that is what this module implements.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    return -(p * log_p).sum(dim=1)


@dataclass
class EtaState:
    """Mutable state carried across batches by an ETA adapter."""

    moving_softmax: torch.Tensor | None = None  # running average of softmax
    momentum: float = 0.9


@dataclass
class EtaConfig:
    lr: float = 1e-3
    momentum: float = 0.9
    e0: float = 0.4 * torch.log(torch.tensor(10.0)).item()
    """Entropy threshold. Default 0.4 * ln(num_classes) following EATA paper."""
    epsilon: float = 0.05
    """Redundancy threshold; drop sample if cos(p_i, ema_p) > 1 - epsilon."""


def eta_loss_and_mask(
    logits: torch.Tensor,
    state: EtaState,
    cfg: EtaConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (scalar loss, keep_mask). Loss is mean weighted entropy on
    reliable + non-redundant samples; keep_mask is the boolean mask used.
    """
    H = softmax_entropy(logits)                           # (N,)
    p = F.softmax(logits, dim=1)                          # (N, C)

    reliable = H < cfg.e0                                 # (N,)

    if state.moving_softmax is None:
        non_redundant = torch.ones_like(reliable)
    else:
        cos = F.cosine_similarity(p, state.moving_softmax.unsqueeze(0), dim=1)
        non_redundant = cos < (1.0 - cfg.epsilon)

    keep = reliable & non_redundant
    if keep.sum() == 0:
        return logits.sum() * 0.0, keep  # zero loss that still tracks the graph

    # ETA reweighting: smaller entropy ⇒ larger weight.
    w = torch.exp(cfg.e0 - H[keep]).detach()
    H_keep = H[keep]
    loss = (w * H_keep).sum() / w.sum().clamp_min(1e-8)

    # update EMA of softmax over kept samples (paper uses all samples;
    # using kept-only avoids the EMA being dragged by dropped OOD).
    with torch.no_grad():
        batch_mean = p[keep].mean(dim=0).detach()
        if state.moving_softmax is None:
            state.moving_softmax = batch_mean
        else:
            state.moving_softmax = (
                state.momentum * state.moving_softmax
                + (1.0 - state.momentum) * batch_mean
            )

    return loss, keep


def eta_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    state: EtaState,
    cfg: EtaConfig,
) -> dict[str, float]:
    """Single ETA adaptation step on ``images``."""
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss, keep = eta_loss_and_mask(logits, state, cfg)
    if keep.sum() == 0:
        return {"loss": 0.0, "kept": 0}
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach().cpu()), "kept": int(keep.sum().cpu())}


# ---------------------------------------------------------------------------
# Backward-compat aliases (callers using old EATA names still work)
# ---------------------------------------------------------------------------
EataState = EtaState
EataConfig = EtaConfig
eata_loss_and_mask = eta_loss_and_mask
eata_step = eta_step
