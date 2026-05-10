"""Rebuttal Experiment 3 — Deployable "mixed-like approximation" gating baseline.

Reviewer concern: "No novel method. Need concrete actionable mitigation."
Reviewer recommendation: "Simple gating/masking heuristic — converts warning to actionable."

This experiment evaluates an adaptive entropy-gate approach (AdapTent) as a
practical, deployable approximation to the oracle ml_ condition:

  AdapTent: filter samples whose entropy exceeds the batch-median entropy
             before computing the gradient. No oracle labels needed; uses
             only the batch's own entropy distribution.

Conditions compared (same paired batches):
  no_tta    : no adaptation
  id_only   : oracle (not deployable, diagnostic only)
  mixed     : vanilla TENT on full batch
  fix_c_30  : Fix C (top-30% entropy samples removed) — previously studied
  adapTent  : adaptive median-threshold entropy gate (NEW)
  ml_like   : forward on full batch, gradient on ID slice (oracle ml_ approx)

Output: results/rebuttal3/mitigation_results.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments._common import (
    DEFAULT_BATCH_SIZE,
    bootstrap_ci,
    draw_paired_batches,
    ensure_dir,
    evaluate_batch,
    auroc_from_eval,
    fpr95_from_eval,
    id_accuracy_from_eval,
    fresh_tent_model,
    get_device,
    get_logger,
    make_sampler,
    paired_t_test,
    save_json,
    set_seed,
    setup_matplotlib,
    TentConfig,
    TentVariant,
    collect_bn_params,
    make_optimizer,
)
from proto_absorb.tent import (
    softmax_entropy as _entropy,
    tent_step,
    filter_hard_ood,
    configure_tent_model,
)
from proto_absorb.models import build_resnet18, load_checkpoint

OODS     = ["svhn", "dtd"]
CORRUPTS = ["gaussian_noise", "fog"]
ALPHAS   = [0.9, 0.5]
N_BATCHES = 30

CONDITIONS = ["no_tta", "id_only", "mixed", "fix_c_30", "adapTent"]


def _adapTent_step(model, opt, images, cfg):
    """Adaptive entropy gate: remove samples above median entropy before gradient."""
    model.train()
    with torch.no_grad():
        logits_probe = model(images)
    H = _entropy(logits_probe).detach()
    threshold = H.median()
    mask = H <= threshold
    if mask.sum() == 0:
        return
    sub = images[mask]
    tent_step(model, opt, sub, cfg)


def run_cell(
    ckpt_path, data_root, corruption, alpha, ood_name,
    n_batches, n_steps, batch_size, master_seed, lr, device,
) -> dict:
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity=5)
    batches = draw_paired_batches(sampler, n_batches, master_seed)

    per_cond_auroc = {c: [] for c in CONDITIONS}
    per_cond_fpr95 = {c: [] for c in CONDITIONS}

    for batch in batches:
        batch_dev = batch.to(device)

        for cond in CONDITIONS:
            cfg = TentConfig(variant=TentVariant.VANILLA, lr=lr)
            model, opt = fresh_tent_model(ckpt_path, device, cfg)

            if cond != "no_tta":
                for _ in range(n_steps):
                    if cond == "id_only":
                        id_mask = ~batch_dev.is_ood
                        if id_mask.any():
                            tent_step(model, opt, batch_dev.images[id_mask], cfg)
                    elif cond == "mixed":
                        tent_step(model, opt, batch_dev.images, cfg)
                    elif cond == "fix_c_30":
                        with torch.no_grad():
                            logits_probe = model(batch_dev.images)
                        keep_mask = filter_hard_ood(logits_probe, drop_fraction=0.30)
                        if keep_mask.any():
                            tent_step(model, opt, batch_dev.images[keep_mask], cfg)
                    elif cond == "adapTent":
                        _adapTent_step(model, opt, batch_dev.images, cfg)

            ev = evaluate_batch(model, batch_dev)
            per_cond_auroc[cond].append(auroc_from_eval(ev, "msp"))
            per_cond_fpr95[cond].append(fpr95_from_eval(ev, "msp"))

    summary = {}
    for cond in CONDITIONS:
        auroc_vals = per_cond_auroc[cond]
        fpr_vals   = per_cond_fpr95[cond]
        am, alo, ahi = bootstrap_ci(auroc_vals)
        fm, flo, fhi = bootstrap_ci(fpr_vals)
        summary[f"{cond}_auroc_mean"]  = am
        summary[f"{cond}_auroc_lo"]    = alo
        summary[f"{cond}_auroc_hi"]    = ahi
        summary[f"{cond}_fpr95_mean"]  = fm
        summary[f"{cond}_fpr95_lo"]    = flo
        summary[f"{cond}_fpr95_hi"]    = fhi
        summary[f"{cond}_auroc_vals"]  = auroc_vals

    # Paired gap relative to id_only
    for cond in ["mixed", "fix_c_30", "adapTent"]:
        gap_vals = [a - b for a, b in
                    zip(per_cond_auroc["id_only"], per_cond_auroc[cond])]
        t_stat, p_val = paired_t_test(per_cond_auroc["id_only"], per_cond_auroc[cond])
        gm, glo, ghi = bootstrap_ci(gap_vals)
        summary[f"gap_vs_{cond}_mean"] = gm
        summary[f"gap_vs_{cond}_lo"]   = glo
        summary[f"gap_vs_{cond}_hi"]   = ghi
        summary[f"gap_vs_{cond}_p"]    = p_val
        # gap closure fraction: how much of id_only−mixed gap does this cond close?
        base_gap = float(np.mean(per_cond_auroc["id_only"])) - float(np.mean(per_cond_auroc["mixed"]))
        if abs(base_gap) > 1e-6:
            closure = 1.0 - gm / base_gap   # fraction of gap closed
            summary[f"closure_{cond}"] = float(closure)

    return summary


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/resnet18_cifar10.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/rebuttal3")
    parser.add_argument("--oods",       nargs="+", default=OODS)
    parser.add_argument("--corruptions",nargs="+", default=CORRUPTS)
    parser.add_argument("--alphas",     type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--batches",    type=int, default=N_BATCHES)
    parser.add_argument("--steps",      type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--smoke",      action="store_true")
    args = parser.parse_args()

    log = get_logger("rebuttal3_mitigation")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device  = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.batches = 3
        args.steps   = 2
        args.oods    = args.oods[:1]
        args.corruptions = args.corruptions[:1]
        args.alphas  = [0.9]

    all_results: dict = {}
    total = len(args.oods) * len(args.corruptions) * len(args.alphas)
    idx = 0

    for corruption in args.corruptions:
        for ood_name in args.oods:
            for alpha in args.alphas:
                idx += 1
                key = f"{corruption}|{ood_name}|{alpha}"
                log.info(f"[{idx}/{total}] {key}")
                master_seed = (
                    args.seed * 7919
                    + abs(hash(corruption)) % 997
                    + abs(hash(ood_name))   % 997
                    + int(alpha * 1000)
                )
                cell = run_cell(
                    ckpt_path=args.ckpt, data_root=args.data_root,
                    corruption=corruption, alpha=alpha, ood_name=ood_name,
                    n_batches=args.batches, n_steps=args.steps,
                    batch_size=args.batch_size, master_seed=master_seed,
                    lr=args.lr, device=device,
                )
                all_results[key] = cell
                log.info(
                    f"  id_only={cell['id_only_auroc_mean']:.4f}  "
                    f"mixed={cell['mixed_auroc_mean']:.4f}  "
                    f"fix_c30={cell['fix_c_30_auroc_mean']:.4f}  "
                    f"adapTent={cell['adapTent_auroc_mean']:.4f}"
                )
                for cond in ["mixed", "fix_c_30", "adapTent"]:
                    cl = cell.get(f"closure_{cond}", float("nan"))
                    log.info(
                        f"    gap_vs_{cond}={cell[f'gap_vs_{cond}_mean']:+.4f}  "
                        f"p={cell[f'gap_vs_{cond}_p']:.3g}  "
                        f"closure={cl:.1%}"
                    )

    save_json({"results": all_results, "config": vars(args)},
              out_dir / "mitigation_results.json")
    log.info(f"Saved → {out_dir / 'mitigation_results.json'}")

    # Summary at alpha=0.9
    log.info("\n=== Mitigation summary (alpha=0.9, avg over OODs × corruptions) ===")
    log.info(f"{'condition':12s}  AUROC   gap_vs_id_only  closure")
    for cond in CONDITIONS:
        cells = [
            all_results[f"{c}|{o}|0.9"]
            for c in args.corruptions for o in args.oods
            if f"{c}|{o}|0.9" in all_results
        ]
        if not cells:
            continue
        avg_auroc = float(np.mean([v[f"{cond}_auroc_mean"] for v in cells]))
        if cond not in ("no_tta", "id_only"):
            avg_gap = float(np.mean([v.get(f"gap_vs_{cond}_mean", float("nan")) for v in cells]))
            avg_cl  = float(np.mean([v.get(f"closure_{cond}", float("nan")) for v in cells]))
            log.info(f"  {cond:12s}  {avg_auroc:.4f}  {avg_gap:+.4f}          {avg_cl:.1%}")
        else:
            log.info(f"  {cond:12s}  {avg_auroc:.4f}")


if __name__ == "__main__":
    cli()
