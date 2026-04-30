"""OOD scoring functions: MSP, Energy, Mahalanobis.

Convention: higher score => more OOD. AUROC is then computed treating
ID samples as the negative class and OOD samples as the positive class.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def msp_score(logits: torch.Tensor) -> torch.Tensor:
    """Maximum Softmax Probability based OOD score: 1 - max_c p_c."""
    p = F.softmax(logits, dim=1)
    return 1.0 - p.max(dim=1).values


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Energy-based OOD score from Liu et al. (2020).

    E(x) = -T * logsumexp(f(x) / T). A more positive E means *lower* density;
    we keep that convention so that higher score == more OOD.
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def mahalanobis_score(
    feats: torch.Tensor,         # (N, D)
    centroids: torch.Tensor,     # (C, D)
    cov_inv: torch.Tensor,       # (D, D)
) -> torch.Tensor:
    """Min Mahalanobis distance over classes (higher == more OOD)."""
    n = feats.shape[0]
    c = centroids.shape[0]
    diffs = feats.unsqueeze(1) - centroids.unsqueeze(0)  # (N, C, D)
    # (N, C) = sum over D of (diffs @ cov_inv) * diffs
    left = diffs @ cov_inv                                # (N, C, D)
    quad = (left * diffs).sum(dim=2).clamp_min(0.0)
    d = quad.sqrt()
    return d.min(dim=1).values
