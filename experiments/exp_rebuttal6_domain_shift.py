"""Domain-shift benchmark — Office-Home open-set split.

Reviewer-required experiment: tests whether the paired contamination gap
persists under genuine domain shift (not just corruption shift).

Setup
-----
Backbone     : ImageNet-pretrained ResNet-50 (torchvision; no task fine-tuning)
Dataset      : Office-Home (Art, Clipart, Real World; Product absent from Kaggle version)
Class split  : 25 ID (known) / 40 OOD (unknown) — alphabetically first 25
Source domain: Real World  →  KNN feature bank calibration (frozen-detector contract)
Target domains: Art, Clipart
Methods      : TENT, No-TTA
Detectors    : KNN (k=50), Energy
n_batches    : 10  (one batch per seed; reviewer spec: n=10)
T            : 10 TENT steps
alpha        : {0.9, 0.5}  →  4 cells minimum

Output: results/rebuttal6/domain_shift_results.json
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.datasets as tvd

from experiments._common import (
    bootstrap_ci,
    ensure_dir,
    get_device,
    get_logger,
    paired_t_test,
    save_json,
    set_seed,
    TentConfig,
    collect_bn_params,
    configure_tent_model,
    make_optimizer,
)
from proto_absorb.data import MixedBatch, MixedBatchSampler, imagenet_eval_transform
from proto_absorb.metrics import auroc, fpr95
from proto_absorb.models import build_resnet50
from proto_absorb.scorers import knn_score, energy_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_KNOWN    = 25        # first 25 classes alphabetically = ID
N_BATCHES  = 10        # = n seeds in reviewer's language
BATCH_SIZE = 64
N_STEPS    = 10
KNN_K      = 50
LR         = 0.00025   # same as imagenet scale experiment
SOURCE     = "Real World"
TARGETS    = ["Art", "Clipart"]
ALPHAS     = [0.9, 0.5]
KNN_MAX_SAMPLES = 5_000  # cap source bank for speed


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _OodWrapper(Dataset):
    """Wrap a Subset to always return label=-1 (signals OOD to MixedBatchSampler)."""

    def __init__(self, base: Dataset, indices: list[int]):
        self._ds = base
        self._idx = indices

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, i: int):
        img, _ = self._ds[self._idx[i]]
        return img, -1


def load_domain_split(oh_root: str, domain: str):
    """Return (id_dataset, ood_dataset, class_names) for a domain.

    Classes sorted alphabetically (matching ImageFolder ordering).
    First N_KNOWN = ID; remaining = OOD.
    id_dataset returns (img, class_idx_in_[0,N_KNOWN-1]).
    ood_dataset returns (img, -1).
    """
    ds = tvd.ImageFolder(
        root=str(Path(oh_root) / domain),
        transform=imagenet_eval_transform(),
    )
    id_indices  = [i for i, (_, c) in enumerate(ds.samples) if c < N_KNOWN]
    ood_indices = [i for i, (_, c) in enumerate(ds.samples) if c >= N_KNOWN]
    id_ds  = Subset(ds, id_indices)   # labels in [0, N_KNOWN-1]
    ood_ds = _OodWrapper(ds, ood_indices)
    return id_ds, ood_ds, ds.classes


# ---------------------------------------------------------------------------
# KNN feature bank
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_knn_bank(
    model: torch.nn.Module,
    source_id_ds: Dataset,
    device: torch.device,
    max_samples: int = KNN_MAX_SAMPLES,
) -> torch.Tensor:
    """Extract features from source-domain ID images using the PRETRAINED model."""
    model.eval()
    loader = DataLoader(source_id_ds, batch_size=128, shuffle=False,
                        num_workers=2, pin_memory=True)
    feats_list: list[torch.Tensor] = []
    n = 0
    for imgs, _ in loader:
        if n >= max_samples:
            break
        _, feats = model(imgs.to(device), return_features=True)
        feats_list.append(feats.cpu())
        n += imgs.shape[0]
    bank = torch.cat(feats_list, dim=0)[:max_samples]
    return bank


# ---------------------------------------------------------------------------
# Single-batch evaluation (KNN + Energy)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ood(
    model: torch.nn.Module,
    batch: MixedBatch,
    knn_bank: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    logits, feats = model(batch.images.to(device), return_features=True)
    logits = logits.cpu()
    feats  = feats.cpu()
    is_ood = batch.is_ood.numpy()

    eng  = energy_score(logits).numpy()
    knn_d = knn_score(feats, knn_bank, k=KNN_K).numpy()

    id_mask  = ~is_ood
    ood_mask =  is_ood
    if id_mask.sum() == 0 or ood_mask.sum() == 0:
        return {"energy_auroc": float("nan"), "knn_auroc": float("nan"),
                "energy_fpr95": float("nan"), "knn_fpr95": float("nan")}

    return {
        "energy_auroc": float(auroc(eng[id_mask],   eng[ood_mask])),
        "knn_auroc":    float(auroc(knn_d[id_mask],  knn_d[ood_mask])),
        "energy_fpr95": float(fpr95(eng[id_mask],   eng[ood_mask])),
        "knn_fpr95":    float(fpr95(knn_d[id_mask],  knn_d[ood_mask])),
    }


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def run_cell(
    model_base: torch.nn.Module,
    device: torch.device,
    id_ds: Dataset,
    ood_ds: Dataset,
    alpha: float,
    knn_bank: torch.Tensor,
    n_batches: int = N_BATCHES,
    n_steps: int   = N_STEPS,
    lr: float      = LR,
    log=None,
) -> dict:
    """Run one cell and return per-condition AUROC lists + paired stats."""
    sampler = MixedBatchSampler(id_ds, ood_ds, alpha=alpha, batch_size=BATCH_SIZE)

    # Draw all batches upfront with deterministic seeds (paired protocol)
    batches: list[MixedBatch] = []
    for b in range(n_batches):
        rng = np.random.default_rng(42 * 1_000_003 + b)
        batches.append(sampler.next_batch(rng))

    per_condition: dict[str, dict[str, list[float]]] = {
        cond: {"energy_auroc": [], "knn_auroc": [],
               "energy_fpr95": [], "knn_fpr95": []}
        for cond in ("no_tta", "id_only", "mixed")
    }

    cfg = TentConfig(lr=lr)

    for b_idx, batch in enumerate(batches):
        for condition in ("no_tta", "id_only", "mixed"):
            model = copy.deepcopy(model_base).to(device)
            configure_tent_model(model)
            params, _ = collect_bn_params(model)
            opt = make_optimizer(params, cfg)

            batch_dev = batch.to(device)
            n = n_steps if condition != "no_tta" else 0

            for _ in range(n):
                if condition == "id_only":
                    mask = ~batch_dev.is_ood
                    if not mask.any():
                        continue
                    imgs = batch_dev.images[mask]
                else:
                    imgs = batch_dev.images

                opt.zero_grad(set_to_none=True)
                logits = model(imgs)
                p = F.softmax(logits, dim=1)
                loss = -(p * p.clamp_min(1e-12).log()).sum(dim=1).mean()
                loss.backward()
                opt.step()

            ev = evaluate_ood(model, batch, knn_bank, device)
            for k, v in ev.items():
                per_condition[condition][k].append(v)

        if log:
            log.info(f"  batch {b_idx+1}/{n_batches} done")

    # Compute paired gap and t-test for each detector
    stats: dict[str, dict] = {}
    for det in ("energy", "knn"):
        key = f"{det}_auroc"
        id_vals  = per_condition["id_only"][key]
        mix_vals = per_condition["mixed"][key]
        gaps     = [a - b for a, b in zip(id_vals, mix_vals)]
        t_stat, p_val = paired_t_test(id_vals, mix_vals)
        m_gap, lo, hi = bootstrap_ci(gaps)
        stats[det] = {
            "id_only_auroc":   bootstrap_ci(id_vals),
            "mixed_auroc":     bootstrap_ci(mix_vals),
            "no_tta_auroc":    bootstrap_ci(per_condition["no_tta"][key]),
            "gap_mean":        m_gap,
            "gap_ci95":        (lo, hi),
            "t_stat":          t_stat,
            "p_value":         p_val,
            "significant":     p_val < 0.05,
        }

    return {
        "per_condition": per_condition,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--oh-root",  default="data/OfficeHome/OfficeHomeDataset_10072016")
    p.add_argument("--out",      default="results/rebuttal6/domain_shift_results.json")
    p.add_argument("--batches",  type=int, default=N_BATCHES)
    p.add_argument("--steps",    type=int, default=N_STEPS)
    p.add_argument("--lr",       type=float, default=LR)
    p.add_argument("--smoke",    action="store_true", help="2 batches, 2 steps for CI")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.batches = 2
        args.steps   = 2

    log = get_logger("domain_shift")
    device = get_device()
    log.info(f"Device: {device}")

    # Load pretrained ResNet-50 ONCE; deep-copy per cell
    log.info("Loading ImageNet-pretrained ResNet-50...")
    model_base = build_resnet50()
    model_base.eval()
    for p in model_base.parameters():
        p.requires_grad_(False)

    # Build KNN bank from source domain (frozen-detector contract)
    log.info(f"Building KNN feature bank from source domain '{SOURCE}'...")
    src_id_ds, _, src_classes = load_domain_split(args.oh_root, SOURCE)
    model_base = model_base.to(device)
    knn_bank = build_knn_bank(model_base, src_id_ds, device)
    log.info(f"KNN bank: {knn_bank.shape[0]} samples, {knn_bank.shape[1]}-dim")
    log.info(f"Known classes (first {N_KNOWN}): {src_classes[:N_KNOWN]}")

    all_results: dict = {}

    for target in TARGETS:
        log.info(f"\n=== Target domain: {target} ===")
        tgt_id_ds, tgt_ood_ds, _ = load_domain_split(args.oh_root, target)
        log.info(f"  ID samples: {len(tgt_id_ds)}, OOD samples: {len(tgt_ood_ds)}")

        for alpha in ALPHAS:
            cell_key = f"{SOURCE}→{target}_alpha{alpha}"
            log.info(f"  Running cell: {cell_key}")
            result = run_cell(
                model_base=model_base,
                device=device,
                id_ds=tgt_id_ds,
                ood_ds=tgt_ood_ds,
                alpha=alpha,
                knn_bank=knn_bank,
                n_batches=args.batches,
                n_steps=args.steps,
                lr=args.lr,
                log=log,
            )
            all_results[cell_key] = result

            # Print summary
            for det in ("energy", "knn"):
                s = result["stats"][det]
                sig = "*" if s["significant"] else " "
                log.info(
                    f"    {det.upper():6s}: gap={s['gap_mean']:+.4f} "
                    f"[{s['gap_ci95'][0]:+.4f},{s['gap_ci95'][1]:+.4f}] "
                    f"p={s['p_value']:.4f}{sig}"
                )

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    save_json(all_results, out_path)
    log.info(f"\nResults saved to {out_path}")

    # Summary table
    log.info("\n=== SUMMARY ===")
    log.info(f"{'Cell':40s} {'Det':6s} {'Gap':>8s} {'p':>8s} {'Sig':>4s}")
    log.info("-" * 70)
    for cell_key, result in all_results.items():
        for det in ("energy", "knn"):
            s = result["stats"][det]
            sig = "YES" if s["significant"] else "no"
            log.info(
                f"{cell_key:40s} {det.upper():6s} "
                f"{s['gap_mean']:+8.4f} {s['p_value']:8.4f} {sig:>4s}"
            )


if __name__ == "__main__":
    main()
