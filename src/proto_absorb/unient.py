"""UniEnt+ — Unified Entropy TTA with confidence-based split (arXiv:2404.06065).

UniEnt+ partitions each batch by entropy relative to threshold τ:
  - H(p_i) < τ  (pseudo-ID)   → minimize entropy  (standard TENT objective)
  - H(p_i) >= τ (pseudo-OOD)  → maximize entropy  (negated contribution)

Default τ = log(C) / 2, where C is the number of classes. This is the
"UniEnt+" variant: UniEnt with τ tuned per-dataset.

BN setup is identical to TENT — only BatchNorm affine parameters are updated.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from proto_absorb.tent import collect_bn_params, configure_tent_model, softmax_entropy


def configure_unient_model(model: nn.Module) -> nn.Module:
    """Prepare model for UniEnt+ adaptation — identical to configure_tent_model."""
    return configure_tent_model(model)


def unient_loss(outputs: torch.Tensor, tau: float | None = None) -> torch.Tensor:
    """UniEnt+ objective averaged over batch.

    Samples with H(p_i) < tau are entropy-minimized; samples >= tau are
    entropy-maximized (their contribution is negated). If all samples fall on
    one side the loss is still well-defined (one term is zero).

    Args:
        outputs: logits of shape (N, C).
        tau: entropy threshold. Defaults to log(C) / 2.
    """
    C = outputs.shape[1]
    if tau is None:
        tau = 0.5 * math.log(C)

    H = softmax_entropy(outputs)          # (N,)
    pseudo_id  = H < tau                  # minimize entropy
    pseudo_ood = ~pseudo_id               # maximize entropy

    loss = outputs.new_zeros(())
    if pseudo_id.any():
        loss = loss + H[pseudo_id].mean()
    if pseudo_ood.any():
        loss = loss - H[pseudo_ood].mean()
    return loss


def unient_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    tau: float | None = None,
) -> dict[str, float]:
    """Single UniEnt+ adaptation step. Returns diagnostic dict."""
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = unient_loss(logits, tau=tau)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach().cpu())}


def adapt_unient(
    model: nn.Module,
    x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    tau: float | None = None,
    n_steps: int = 1,
) -> nn.Module:
    """Multi-step UniEnt+ adaptation loop. Returns adapted model."""
    for _ in range(n_steps):
        unient_step(model, optimizer, x, tau=tau)
    return model
