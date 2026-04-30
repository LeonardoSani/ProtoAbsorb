"""Shared utilities used by all experiment scripts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from proto_absorb.centroids import (
    CentroidBank,
    nearest_centroid_distances,
    update_centroids_dynamic,
)
from proto_absorb.data import (
    CIFAR10_C_CORRUPTIONS,
    CIFAR10C,
    MixedBatch,
    MixedBatchSampler,
    cifar10_test,
    eval_transform,
    get_ood_dataset,
)
from proto_absorb.models import build_resnet18, load_checkpoint
from proto_absorb.scorers import energy_score, mahalanobis_score, msp_score
from proto_absorb.tent import (
    TentConfig,
    TentVariant,
    collect_bn_params,
    configure_tent_model,
    make_optimizer,
    snapshot_logits_and_features,
    tent_step,
)
from proto_absorb.utils import ensure_dir, get_device, get_logger, set_seed


ALPHAS = (0.9, 0.75, 0.5, 0.25)
DEFAULT_T = 20
DEFAULT_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def fresh_tent_model(ckpt_path: str, device: torch.device,
                     cfg: TentConfig) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Build a fresh model loaded from ``ckpt_path``, configured for TENT."""
    model = build_resnet18(num_classes=10).to(device)
    load_checkpoint(model, ckpt_path, map_location=str(device))
    configure_tent_model(model)
    params, _ = collect_bn_params(model)
    opt = make_optimizer(params, cfg)
    return model, opt


def build_id_pool(data_root: str, corruption: str, severity: int = 5):
    """Return the ID pool. If ``corruption == "clean"`` we fall back to the
    clean CIFAR-10 test set; otherwise we use CIFAR-10-C at the given severity.
    """
    if corruption == "clean":
        return cifar10_test(data_root)
    return CIFAR10C(data_root, corruption=corruption, severity=severity,
                    transform=eval_transform())


def build_ood_pool(data_root: str, ood_name: str = "svhn"):
    return get_ood_dataset(ood_name, data_root, split="test")


def make_sampler(data_root: str, corruption: str, alpha: float,
                 ood_name: str = "svhn", batch_size: int = DEFAULT_BATCH_SIZE,
                 severity: int = 5) -> MixedBatchSampler:
    return MixedBatchSampler(
        id_dataset=build_id_pool(data_root, corruption, severity),
        ood_dataset=build_ood_pool(data_root, ood_name),
        alpha=alpha, batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Logging primitives
# ---------------------------------------------------------------------------

def evaluate_batch(
    model: torch.nn.Module,
    batch: MixedBatch,
    centroids: Optional[CentroidBank] = None,
) -> dict:
    """Eval-only forward pass: returns scores, predictions, features for a batch."""
    was_training = {m: m.training for m in model.modules()}
    model.eval()
    try:
        with torch.no_grad():
            logits, feats = model(batch.images, return_features=True)  # type: ignore[misc]
    finally:
        for m, t in was_training.items():
            m.train(t)

    s_msp = msp_score(logits).cpu().numpy()
    s_eng = energy_score(logits).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    is_ood = batch.is_ood.cpu().numpy()
    labels = batch.labels.cpu().numpy()

    out = {
        "msp": s_msp, "energy": s_eng,
        "preds": preds, "is_ood": is_ood, "labels": labels,
        "feats": feats.cpu().numpy(),
    }
    if centroids is not None and centroids.cov_inv is not None:
        s_mhl = mahalanobis_score(
            feats.cpu(), centroids.centroids, centroids.cov_inv
        ).numpy()
        out["mahalanobis"] = s_mhl
    return out


def auroc_from_eval(eval_out: dict, score_key: str = "msp") -> float:
    from proto_absorb.metrics import auroc
    s = eval_out[score_key]
    is_ood = eval_out["is_ood"].astype(bool)
    return auroc(s[~is_ood], s[is_ood])


def id_accuracy_from_eval(eval_out: dict) -> float:
    is_ood = eval_out["is_ood"].astype(bool)
    if (~is_ood).sum() == 0:
        return 0.0
    return float((eval_out["preds"][~is_ood] == eval_out["labels"][~is_ood]).mean())


# ---------------------------------------------------------------------------
# Plot defaults
# ---------------------------------------------------------------------------

def setup_matplotlib() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Cannot serialize {type(o)}")


__all__ = [
    "ALPHAS", "DEFAULT_T", "DEFAULT_BATCH_SIZE",
    "fresh_tent_model", "make_sampler", "evaluate_batch",
    "auroc_from_eval", "id_accuracy_from_eval", "setup_matplotlib",
    "save_json",
    # re-exports
    "CIFAR10_C_CORRUPTIONS", "TentConfig", "TentVariant",
    "tent_step", "snapshot_logits_and_features",
    "CentroidBank", "nearest_centroid_distances", "update_centroids_dynamic",
    "ensure_dir", "get_device", "get_logger", "set_seed",
]
