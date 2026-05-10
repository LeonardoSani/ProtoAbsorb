"""Rebuttal Experiment 2 — Extended TTA method coverage: SAR and CoTTA.

Reviewer concern: "TENT + ETA only. SAR, CoTTA, EATA needed for broad claim."

This experiment adds SAR and CoTTA to the ranking audit from step9, showing
that (1) the paired contamination gap persists for both methods, and (2) the
rank discordance between closed-set and open-world metrics holds across a
wider method family.

Methods: no_tta, tent, eta, unient_plus, sar, cotta  (6 total)
OODs:     svhn, dtd
Corrupts: gaussian_noise, fog
alpha:    0.9, 0.5
n:        30 batches

Output: results/rebuttal2/extended_baselines_results.json
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from experiments._common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_T,
    MixedBatch,
    MixedBatchSampler,
    bootstrap_ci,
    build_id_pool,
    build_ood_pool,
    draw_paired_batches,
    ensure_dir,
    fresh_tent_model,
    get_device,
    get_logger,
    make_sampler,
    paired_t_test,
    run_condition_on_batch,
    save_json,
    set_seed,
    setup_matplotlib,
    TentConfig,
    TentVariant,
    EataConfig,
    EataState,
    eata_step,
    unient_step,
    collect_bn_params,
    make_optimizer,
    auroc_from_eval,
    fpr95_from_eval,
    id_accuracy_from_eval,
    evaluate_batch,
)
from proto_absorb.models import build_resnet18, load_checkpoint
from proto_absorb.tent import configure_tent_model, tent_step
from proto_absorb.sar import SARConfig, SARState, sar_step
from proto_absorb.cotta import CoTTAConfig, CoTTAState, cotta_step

METHODS    = ["no_tta", "tent", "eta", "unient_plus", "sar", "cotta"]
OODS       = ["svhn", "dtd"]
CORRUPTS   = ["gaussian_noise", "fog"]
ALPHAS     = [0.9, 0.5]


# ---------------------------------------------------------------------------
# Method-agnostic step function
# ---------------------------------------------------------------------------

def _adapt_step(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    images: torch.Tensor,
    method: str,
    cfg_tent: TentConfig,
    cfg_eta: EataConfig,
    eta_state: EataState | None,
    sar_cfg: SARConfig,
    sar_state: SARState | None,
    cotta_cfg: CoTTAConfig,
    cotta_state: CoTTAState | None,
):
    if method == "tent":
        tent_step(model, opt, images, cfg_tent)
    elif method == "eta":
        eata_step(model, opt, images, eta_state, cfg_eta)
    elif method == "unient_plus":
        unient_step(model, opt, images)
    elif method == "sar":
        sar_step(model, opt, images, sar_state, sar_cfg)
    elif method == "cotta":
        cotta_step(model, opt, images, cotta_state, cotta_cfg)
    # no_tta: nothing


def _build_model_and_opt(ckpt_path, device, method, lr):
    """Build fresh model+optimizer for any method."""
    cfg_tent = TentConfig(variant=TentVariant.VANILLA, lr=lr)
    model, opt = fresh_tent_model(ckpt_path, device, cfg_tent)

    cfg_eta   = EataConfig(lr=lr)
    eta_state = None
    sar_cfg   = SARConfig(lr=lr)
    sar_state = None
    cotta_cfg = CoTTAConfig(lr=lr)
    cotta_state = None

    if method == "eta":
        eta_state = EataState()
        params, _ = collect_bn_params(model)
        opt = torch.optim.SGD(params, lr=lr, momentum=cfg_eta.momentum)
    elif method == "sar":
        sar_state = SARState()
        params, _ = collect_bn_params(model)
        opt = torch.optim.SGD(params, lr=lr, momentum=sar_cfg.momentum)
    elif method == "cotta":
        cotta_state = CoTTAState(model)
        params, _ = collect_bn_params(model)
        opt = torch.optim.SGD(params, lr=lr, momentum=cotta_cfg.momentum)

    return model, opt, cfg_tent, cfg_eta, eta_state, sar_cfg, sar_state, cotta_cfg, cotta_state


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def run_cell(
    ckpt_path, data_root, corruption, alpha, ood_name, method,
    n_batches, n_steps, batch_size, master_seed, lr, device,
) -> dict:
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity=5)
    batches = draw_paired_batches(sampler, n_batches, master_seed)

    mixed_auroc_list   = []
    id_only_auroc_list = []

    for batch in batches:
        batch_dev = batch.to(device)

        for condition in (["no_tta"] if method == "no_tta" else ["mixed", "id_only"]):
            (model, opt, cfg_tent, cfg_eta, eta_state, sar_cfg,
             sar_state, cotta_cfg, cotta_state) = _build_model_and_opt(
                ckpt_path, device, method, lr,
            )

            if condition != "no_tta":
                for _ in range(n_steps):
                    if condition == "id_only":
                        id_mask = ~batch_dev.is_ood
                        imgs = batch_dev.images[id_mask] if id_mask.any() else batch_dev.images
                    else:
                        imgs = batch_dev.images
                    _adapt_step(
                        model, opt, imgs, method,
                        cfg_tent, cfg_eta, eta_state,
                        sar_cfg, sar_state,
                        cotta_cfg, cotta_state,
                    )

            ev = evaluate_batch(model, batch_dev)
            auroc_val = auroc_from_eval(ev, "msp")

            if condition in ("no_tta", "mixed"):
                mixed_auroc_list.append(auroc_val)
            else:
                id_only_auroc_list.append(auroc_val)

    if method == "no_tta":
        id_only_auroc_list = list(mixed_auroc_list)

    mixed_mean   = float(np.mean(mixed_auroc_list))
    id_only_mean = float(np.mean(id_only_auroc_list))
    gap_mean     = id_only_mean - mixed_mean
    m_mix, lo_mix, hi_mix = bootstrap_ci(mixed_auroc_list)
    m_id,  lo_id,  hi_id  = bootstrap_ci(id_only_auroc_list)
    t_stat, p_val = paired_t_test(id_only_auroc_list, mixed_auroc_list)

    return {
        "mixed_auroc":      mixed_mean,
        "id_only_auroc":    id_only_mean,
        "paired_gap":       gap_mean,
        "paired_p":         p_val,
        "mixed_auroc_lo":   lo_mix,
        "mixed_auroc_hi":   hi_mix,
        "id_only_auroc_lo": lo_id,
        "id_only_auroc_hi": hi_id,
        "mixed_auroc_per_batch":   mixed_auroc_list,
        "id_only_auroc_per_batch": id_only_auroc_list,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/resnet18_cifar10.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/rebuttal2")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--oods",    nargs="+", default=OODS)
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTS)
    parser.add_argument("--alphas",  type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--steps",   type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--smoke",   action="store_true")
    args = parser.parse_args()

    log = get_logger("rebuttal2_baselines")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device  = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.batches = 3
        args.steps   = 2
        args.methods = ["no_tta", "tent", "sar"]
        args.oods    = args.oods[:1]
        args.corruptions = args.corruptions[:1]
        args.alphas  = [0.9]

    all_results: dict = {}
    total = (len(args.methods) * len(args.oods)
             * len(args.corruptions) * len(args.alphas))
    idx = 0

    for corruption in args.corruptions:
        for ood_name in args.oods:
            for method in args.methods:
                for alpha in args.alphas:
                    idx += 1
                    key = f"{corruption}|{ood_name}|{method}|{alpha}"
                    log.info(f"[{idx}/{total}] {key}")
                    master_seed = (
                        args.seed * 7919
                        + abs(hash(corruption)) % 997
                        + abs(hash(ood_name))   % 997
                        + abs(hash(method))     % 997
                        + int(alpha * 1000)
                    )
                    cell = run_cell(
                        ckpt_path=args.ckpt, data_root=args.data_root,
                        corruption=corruption, alpha=alpha, ood_name=ood_name,
                        method=method, n_batches=args.batches,
                        n_steps=args.steps, batch_size=args.batch_size,
                        master_seed=master_seed, lr=args.lr, device=device,
                    )
                    all_results[key] = cell
                    log.info(
                        f"  mixed={cell['mixed_auroc']:.4f}  "
                        f"id_only={cell['id_only_auroc']:.4f}  "
                        f"gap={cell['paired_gap']:+.4f}  "
                        f"p={cell['paired_p']:.3g}"
                    )

    save_json({"results": all_results, "config": vars(args)},
              out_dir / "extended_baselines_results.json")
    log.info(f"Saved → {out_dir / 'extended_baselines_results.json'}")

    # Ranking table at alpha=0.9, averaged over OODs × corruptions
    log.info("\n=== Ranking audit (alpha=0.9, avg over OODs × corruptions) ===")
    log.info(f"{'method':12s}  mixed_AUROC  id_only_AUROC  paired_gap  panel_B_rank")
    method_aurocs = {}
    for method in args.methods:
        cells = [
            all_results[f"{c}|{o}|{method}|0.9"]
            for c in args.corruptions for o in args.oods
            if f"{c}|{o}|{method}|0.9" in all_results
        ]
        if not cells:
            continue
        method_aurocs[method] = float(np.mean([v["mixed_auroc"] for v in cells]))

    sorted_methods = sorted(method_aurocs, key=method_aurocs.get, reverse=True)
    for rank, method in enumerate(sorted_methods, 1):
        cells = [
            all_results[f"{c}|{o}|{method}|0.9"]
            for c in args.corruptions for o in args.oods
            if f"{c}|{o}|{method}|0.9" in all_results
        ]
        avg = lambda k: float(np.mean([v[k] for v in cells]))
        log.info(
            f"  {method:12s}  {avg('mixed_auroc'):.4f}       "
            f"{avg('id_only_auroc'):.4f}         "
            f"{avg('paired_gap'):+.4f}      #{rank}"
        )

    # Spearman rank correlation (all methods, alpha=0.9)
    # Panel A = ranking by id_only_auroc (proxy for closed-set gain since no delta_acc here)
    # Panel B = ranking by mixed_auroc
    available = [m for m in args.methods if m != "no_tta"
                 and any(f"{c}|{o}|{m}|0.9" in all_results
                         for c in args.corruptions for o in args.oods)]
    if len(available) >= 3:
        panel_b = [
            float(np.mean([all_results[f"{c}|{o}|{m}|0.9"]["mixed_auroc"]
                            for c in args.corruptions for o in args.oods
                            if f"{c}|{o}|{m}|0.9" in all_results]))
            for m in available
        ]
        panel_a = [
            float(np.mean([all_results[f"{c}|{o}|{m}|0.9"]["id_only_auroc"]
                            for c in args.corruptions for o in args.oods
                            if f"{c}|{o}|{m}|0.9" in all_results]))
            for m in available
        ]
        rho, p_rho = stats.spearmanr(panel_a, panel_b)
        tau, p_tau = stats.kendalltau(panel_a, panel_b)
        log.info(f"\nRank correlation (Panel A=id_only vs Panel B=mixed): "
                 f"Spearman rho={rho:.3f} (p={p_rho:.3f}), "
                 f"Kendall tau={tau:.3f} (p={p_tau:.3f})")


if __name__ == "__main__":
    cli()
