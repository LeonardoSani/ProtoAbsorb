"""OOD scoring functions: MSP, Energy, Mahalanobis, KNN, ViM.

Convention: higher score => more OOD. AUROC is then computed treating
ID samples as the negative class and OOD samples as the positive class.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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
    left = diffs @ cov_inv                                # (N, C, D)
    quad = (left * diffs).sum(dim=2).clamp_min(0.0)
    d = quad.sqrt()
    return d.min(dim=1).values


# ---------------------------------------------------------------------------
# KNN score (Sun et al., ICML 2022)
# ---------------------------------------------------------------------------

def knn_score(
    feats: torch.Tensor,        # (N, D) test features
    train_feats: torch.Tensor,  # (M, D) training feature bank
    k: int = 50,
) -> torch.Tensor:
    """k-NN distance OOD score (higher == more OOD).

    Normalizes both sets of features to the unit sphere before computing
    pairwise L2 distances, following Sun et al. (2022).
    """
    feats_n = F.normalize(feats.float(), dim=1)
    train_n = F.normalize(train_feats.float().to(feats.device), dim=1)
    # cosine sim = dot product of unit vectors; distance = sqrt(2 - 2*sim)
    # Using cdist is equivalent (after normalization, L2^2 = 2 - 2*cos)
    dists = torch.cdist(feats_n, train_n, p=2)  # (N, M)
    k_clamped = min(k, train_n.shape[0])
    knn_dists, _ = dists.topk(k_clamped, dim=1, largest=False)
    return knn_dists[:, -1]   # k-th NN distance (standard KNN-OOD)


# ---------------------------------------------------------------------------
# ViM score (Wang et al., CVPR 2022)
# ---------------------------------------------------------------------------

def build_vim_components(
    train_feats: torch.Tensor,   # (M, D) training features (CPU or GPU)
    n_components: Optional[int] = None,
    id_feats_for_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """Compute ViM principal subspace via SVD.

    Returns:
        principal_space : (D, d) matrix of top-d eigenvectors (row space of W)
        id_residual_scale : mean residual norm on training set (for normalization)
    """
    f = train_feats.float()
    f_centered = f - f.mean(dim=0, keepdim=True)
    d_feat = f.shape[1]
    if n_components is None:
        n_components = d_feat // 2  # keep half the dimensions by default

    # Economy SVD; V columns span the top-d principal directions
    try:
        _, _, Vt = torch.linalg.svd(f_centered, full_matrices=False)
        principal_space = Vt[:n_components].T.contiguous()  # (D, d)
    except Exception:
        # Fallback: numpy SVD for stability on very small batches
        U, S, Vt = np.linalg.svd(f_centered.cpu().numpy(), full_matrices=False)
        principal_space = torch.tensor(Vt[:n_components].T, dtype=torch.float32)

    # Compute average residual norm on training set for scaling
    ref = id_feats_for_scale if id_feats_for_scale is not None else f
    ref = ref.float()
    ps = principal_space.to(ref.device)
    proj = ref @ ps @ ps.T
    residuals = ref - proj
    id_scale = float(residuals.norm(dim=1).mean().clamp_min(1e-6))

    return principal_space.cpu(), id_scale


def vim_score(
    feats: torch.Tensor,           # (N, D)
    principal_space: torch.Tensor, # (D, d)  from build_vim_components
    id_residual_scale: float,
) -> torch.Tensor:
    """ViM residual-space OOD score (higher == more OOD).

    Computes the norm of the feature component outside the principal subspace,
    normalized by the mean ID residual norm so scores are comparable across
    architectures.
    """
    ps = principal_space.float().to(feats.device)
    proj = feats.float() @ ps @ ps.T      # (N, D)
    residual = feats.float() - proj        # (N, D)
    return residual.norm(dim=1) / id_residual_scale


# ---------------------------------------------------------------------------
# Feature bank extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_feature_bank(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 256,
    max_samples: Optional[int] = None,
) -> torch.Tensor:
    """Extract penultimate-layer features from model over dataset.

    Returns (N, D) float32 tensor on CPU.
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    all_feats = []
    n_seen = 0
    for imgs, *_ in loader:
        if max_samples is not None and n_seen >= max_samples:
            break
        imgs = imgs.to(device)
        out = model(imgs, return_features=True)
        feats = out[1] if isinstance(out, (tuple, list)) else model.features(imgs)
        all_feats.append(feats.cpu())
        n_seen += imgs.shape[0]
    return torch.cat(all_feats, dim=0)
