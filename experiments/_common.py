"""Shared utilities used by all experiment scripts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

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
from proto_absorb.eata import EataConfig, EataState, eata_step
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

    p = F.softmax(logits, dim=1)
    H = -(p * p.clamp_min(1e-12).log()).sum(dim=1).cpu().numpy()
    mx = p.max(dim=1).values.cpu().numpy()
    fn = feats.norm(dim=1).cpu().numpy()

    out = {
        "msp": s_msp, "energy": s_eng,
        "preds": preds, "is_ood": is_ood, "labels": labels,
        "feats": feats.cpu().numpy(),
        "entropy": H, "max_p": mx, "feat_norm": fn,
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


# ---------------------------------------------------------------------------
# Paired-protocol primitives (used by the reframed step-N experiments)
# ---------------------------------------------------------------------------

def draw_paired_batches(
    sampler: MixedBatchSampler, n_batches: int, master_seed: int,
) -> list[MixedBatch]:
    """Draw ``n_batches`` mixed batches from a deterministic seed sequence so
    that **every condition we run** (No TTA / ID-only TENT / Mixed TENT / ...)
    sees the exact same batches when called with the same ``master_seed``.
    """
    out: list[MixedBatch] = []
    for b in range(n_batches):
        rng = np.random.default_rng(master_seed * 1_000_003 + b)
        out.append(sampler.next_batch(rng))
    return out


def run_condition_on_batch(
    ckpt_path: str, device: torch.device, batch: MixedBatch,
    condition: str, n_steps: int, lr: float,
    tta_method: str = "tent",
    centroids: Optional[torch.Tensor] = None,
    fix_a: bool = False,
    eata_state_factory: Optional[Callable[[], EataState]] = None,
) -> dict:
    """Run a single condition on a single shared batch and return per-step
    eval traces.

    ``condition`` is one of:
        - ``"no_tta"``   : evaluate at t=0 only (n_steps ignored).
        - ``"id_only"``  : at each step adapt on the ID-only slice of the
                           batch (oracle filtering); evaluate on the full batch.
        - ``"mixed"``    : adapt on the full mixed batch (the realistic open-
                           world condition).

    ``tta_method`` is ``"tent"`` or ``"eata"``.
    Returns a dict of per-step arrays of length ``n_steps + 1``.
    """
    cfg = TentConfig(
        variant=TentVariant.FIX_A_WEIGHTED if fix_a else TentVariant.VANILLA,
        lr=lr,
    )
    model, opt = fresh_tent_model(ckpt_path, device, cfg)
    batch = batch.to(device)

    eata_state: Optional[EataState] = None
    eata_cfg: Optional[EataConfig] = None
    if tta_method == "eata":
        eata_state = (eata_state_factory or EataState)()
        eata_cfg = EataConfig(lr=lr)
        # rebuild optimizer to honor EataConfig.lr (same as TentConfig.lr here)
        from proto_absorb.tent import collect_bn_params, make_optimizer
        params, _ = collect_bn_params(model)
        opt = torch.optim.SGD(params, lr=eata_cfg.lr, momentum=eata_cfg.momentum)

    n = n_steps if condition != "no_tta" else 0
    msp_curve, energy_curve, acc_curve = [], [], []
    ent_ood_curve, max_p_ood_curve, feats_norm_ood_curve = [], [], []
    msp_id_mean_curve, msp_ood_mean_curve = [], []
    for t in range(n + 1):
        ev = evaluate_batch(model, batch, centroids=None)
        is_ood = ev["is_ood"].astype(bool)
        msp_curve.append(auroc_from_eval(ev, "msp"))
        energy_curve.append(auroc_from_eval(ev, "energy"))
        acc_curve.append(id_accuracy_from_eval(ev))
        H, mx, fn = ev["entropy"], ev["max_p"], ev["feat_norm"]
        ent_ood_curve.append(float(H[is_ood].mean()) if is_ood.any() else float("nan"))
        max_p_ood_curve.append(float(mx[is_ood].mean()) if is_ood.any() else float("nan"))
        feats_norm_ood_curve.append(float(fn[is_ood].mean()) if is_ood.any() else float("nan"))
        s = ev["msp"]
        msp_id_mean_curve.append(float(s[~is_ood].mean()) if (~is_ood).any() else float("nan"))
        msp_ood_mean_curve.append(float(s[is_ood].mean()) if is_ood.any() else float("nan"))

        if t < n:
            if condition == "id_only":
                id_mask = ~batch.is_ood
                if id_mask.any():
                    sub_imgs = batch.images[id_mask]
                    if tta_method == "eata":
                        eata_step(model, opt, sub_imgs, eata_state, eata_cfg)  # type: ignore[arg-type]
                    else:
                        tent_step(model, opt, sub_imgs, cfg, centroids=centroids)
            elif condition == "mixed":
                if tta_method == "eata":
                    eata_step(model, opt, batch.images, eata_state, eata_cfg)  # type: ignore[arg-type]
                else:
                    tent_step(model, opt, batch.images, cfg, centroids=centroids)
            else:
                raise ValueError(condition)

    return {
        "msp_auroc": np.array(msp_curve),
        "energy_auroc": np.array(energy_curve),
        "id_acc": np.array(acc_curve),
        "ood_entropy": np.array(ent_ood_curve),
        "ood_max_p": np.array(max_p_ood_curve),
        "ood_feat_norm": np.array(feats_norm_ood_curve),
        "msp_id_mean": np.array(msp_id_mean_curve),
        "msp_ood_mean": np.array(msp_ood_mean_curve),
    }


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000,
                 ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, lo, hi) of a percentile bootstrap CI on the mean."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(arr.mean()), float(lo), float(hi)


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sided paired t-test. Returns (t_stat, p_value)."""
    from scipy import stats
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return float("nan"), float("nan")
    res = stats.ttest_rel(a[mask], b[mask])
    return float(res.statistic), float(res.pvalue)


__all__ = [
    "ALPHAS", "DEFAULT_T", "DEFAULT_BATCH_SIZE",
    "fresh_tent_model", "make_sampler", "evaluate_batch",
    "auroc_from_eval", "id_accuracy_from_eval", "setup_matplotlib",
    "save_json",
    "draw_paired_batches", "run_condition_on_batch",
    "bootstrap_ci", "paired_t_test",
    # re-exports
    "CIFAR10_C_CORRUPTIONS", "TentConfig", "TentVariant",
    "tent_step", "snapshot_logits_and_features",
    "EataConfig", "EataState", "eata_step",
    "CentroidBank", "nearest_centroid_distances", "update_centroids_dynamic",
    "ensure_dir", "get_device", "get_logger", "set_seed",
    "msp_score", "energy_score", "mahalanobis_score",
]
