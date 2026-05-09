"""Step 9 — Cross-Method Ranking Audit.

Runs 4 methods × 2 OODs × 2 corruptions × 2 α = 32 cells and computes:
  - delta_id_acc   : ID accuracy change under closed-set (no-OOD) batches
  - mixed_auroc    : OOD AUROC under mixed-stream batch (MSP scorer)
  - id_only_auroc  : oracle AUROC when adapting only on ID slice
  - paired_gap     : id_only_auroc − mixed_auroc  (contamination penalty)

Methods:  no_tta, tent, eta, unient_plus
OODs:     svhn, dtd
Corrupts: gaussian_noise, fog  (sev-5)
α:        0.9, 0.5

Run:
  python -m experiments.exp_step9_ranking_audit \\
    --ckpt data/resnet18_cifar10.pt \\
    --data-root data --out results/step9 --batches 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

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
    get_device,
    get_logger,
    make_sampler,
    paired_t_test,
    run_condition_on_batch,
    save_json,
    set_seed,
    setup_matplotlib,
)

METHODS = ["no_tta", "tent", "eta", "unient_plus"]
OODS    = ["svhn", "dtd"]
CORRUPTIONS = ["gaussian_noise", "fog"]
ALPHAS  = [0.9, 0.5]


# ---------------------------------------------------------------------------
# Closed-set batch builder (alpha=1.0 → ID-only MixedBatch)
# ---------------------------------------------------------------------------

def _draw_id_only_batches(
    data_root: str, corruption: str, ood_name: str,
    batch_size: int, n_batches: int, master_seed: int, severity: int = 5,
) -> list[MixedBatch]:
    """Batches with all-ID content (no OOD).  OOD dataset still loaded but
    n_ood=0 so it is never sampled.  AUROC from these batches is 0.5 (degenerate);
    only id_acc is meaningful here.
    """
    sampler = MixedBatchSampler(
        id_dataset=build_id_pool(data_root, corruption, severity),
        ood_dataset=build_ood_pool(data_root, ood_name),
        alpha=1.0,
        batch_size=batch_size,
    )
    return draw_paired_batches(sampler, n_batches, master_seed)


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def run_cell(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    ood_name: str, n_batches: int, n_steps: int, batch_size: int,
    method: str, master_seed: int, lr: float,
    device: torch.device,
) -> dict:
    """Run mixed + closed-set protocol for one (method, corruption, ood, alpha) cell.

    Returns dict with scalar summaries and per-batch lists.
    """
    # --- Mixed-stream protocol (alpha% ID, (1-alpha)% OOD) ---
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity=5)
    batches_mixed = draw_paired_batches(sampler, n_batches, master_seed)

    mixed_auroc_list:    list[float] = []
    id_only_auroc_list:  list[float] = []

    for batch in batches_mixed:
        tta = "tent" if method in {"no_tta", "tent"} else method

        res_mixed = run_condition_on_batch(
            ckpt_path=ckpt_path, device=device, batch=batch,
            condition="no_tta" if method == "no_tta" else "mixed",
            n_steps=n_steps, lr=lr, tta_method=tta,
        )
        mixed_auroc_list.append(float(res_mixed["msp_auroc"][-1]))

        if method == "no_tta":
            # no adaptation → id_only = mixed (same eval)
            id_only_auroc_list.append(float(res_mixed["msp_auroc"][-1]))
        else:
            res_id_only = run_condition_on_batch(
                ckpt_path=ckpt_path, device=device, batch=batch,
                condition="id_only", n_steps=n_steps, lr=lr, tta_method=tta,
            )
            id_only_auroc_list.append(float(res_id_only["msp_auroc"][-1]))

    # --- Closed-set protocol (no OOD in batch) for delta_id_acc ---
    closed_seed = master_seed + 99_991   # different draw, same corruption
    batches_id = _draw_id_only_batches(
        data_root, corruption, ood_name, batch_size, n_batches,
        closed_seed, severity=5,
    )

    delta_id_acc_list: list[float] = []

    for batch in batches_id:
        # baseline: no TTA
        res_base = run_condition_on_batch(
            ckpt_path=ckpt_path, device=device, batch=batch,
            condition="no_tta", n_steps=0, lr=lr, tta_method="tent",
        )
        base_acc = float(res_base["id_acc"][-1])

        if method == "no_tta":
            delta_id_acc_list.append(0.0)
        else:
            tta = "tent" if method == "tent" else method
            res_method = run_condition_on_batch(
                ckpt_path=ckpt_path, device=device, batch=batch,
                condition="mixed", n_steps=n_steps, lr=lr, tta_method=tta,
            )
            method_acc = float(res_method["id_acc"][-1])
            delta_id_acc_list.append(method_acc - base_acc)

    # --- Aggregation ---
    mixed_auroc_mean    = float(np.mean(mixed_auroc_list))
    id_only_auroc_mean  = float(np.mean(id_only_auroc_list))
    delta_id_acc_mean   = float(np.mean(delta_id_acc_list))
    paired_gap_mean     = id_only_auroc_mean - mixed_auroc_mean

    m_mix, lo_mix, hi_mix     = bootstrap_ci(mixed_auroc_list,   seed=master_seed)
    m_id,  lo_id,  hi_id      = bootstrap_ci(id_only_auroc_list, seed=master_seed + 1)
    m_dac, lo_dac, hi_dac     = bootstrap_ci(delta_id_acc_list,  seed=master_seed + 2)

    t_stat, p_val = paired_t_test(id_only_auroc_list, mixed_auroc_list)

    return {
        "mixed_auroc":          mixed_auroc_mean,
        "mixed_auroc_lo":       lo_mix,
        "mixed_auroc_hi":       hi_mix,
        "id_only_auroc":        id_only_auroc_mean,
        "id_only_auroc_lo":     lo_id,
        "id_only_auroc_hi":     hi_id,
        "paired_gap":           paired_gap_mean,
        "delta_id_acc":         delta_id_acc_mean,
        "delta_id_acc_lo":      lo_dac,
        "delta_id_acc_hi":      hi_dac,
        "paired_t_stat":        t_stat,
        "paired_p":             p_val,
        "n_batches":            n_batches,
        "mixed_auroc_per_batch":   mixed_auroc_list,
        "id_only_auroc_per_batch": id_only_auroc_list,
        "delta_id_acc_per_batch":  delta_id_acc_list,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="Path to ResNet-18 CIFAR-10 checkpoint.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/step9")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--oods", nargs="+", default=OODS)
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTIONS)
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test: 3 batches, 2 steps, first method/OOD only.")
    args = parser.parse_args()

    log = get_logger("step9_ranking_audit")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.batches = 3
        args.steps   = 2
        args.methods = args.methods[:2]
        args.oods    = args.oods[:1]
        args.corruptions = args.corruptions[:1]
        args.alphas  = args.alphas[:1]

    total = (len(args.methods) * len(args.oods)
             * len(args.corruptions) * len(args.alphas))
    log.info(
        f"Matrix: {len(args.methods)} methods × {len(args.oods)} OODs × "
        f"{len(args.corruptions)} corruptions × {len(args.alphas)} alphas = {total} cells"
    )
    log.info(f"  batches={args.batches}  steps={args.steps}  batch_size={args.batch_size}")

    all_results: dict = {}
    cell_idx = 0

    for corruption in args.corruptions:
        for ood_name in args.oods:
            for method in args.methods:
                for alpha in args.alphas:
                    cell_idx += 1
                    key = f"{corruption}|{ood_name}|{method}|{alpha}"
                    log.info(
                        f"[{cell_idx}/{total}] "
                        f"corruption={corruption}  ood={ood_name}  "
                        f"method={method}  alpha={alpha}"
                    )
                    master_seed = (
                        args.seed * 7919
                        + abs(hash(corruption)) % 997
                        + abs(hash(ood_name))   % 997
                        + abs(hash(method))     % 997
                        + int(alpha * 1000)
                    )
                    cell = run_cell(
                        ckpt_path=args.ckpt,
                        data_root=args.data_root,
                        corruption=corruption,
                        alpha=alpha,
                        ood_name=ood_name,
                        n_batches=args.batches,
                        n_steps=args.steps,
                        batch_size=args.batch_size,
                        method=method,
                        master_seed=master_seed,
                        lr=args.lr,
                        device=device,
                    )
                    all_results[key] = cell
                    log.info(
                        f"  mixed_auroc={cell['mixed_auroc']:.4f}  "
                        f"id_only_auroc={cell['id_only_auroc']:.4f}  "
                        f"paired_gap={cell['paired_gap']:+.4f}  "
                        f"delta_id_acc={cell['delta_id_acc']:+.4f}  "
                        f"p={cell['paired_p']:.3g}"
                    )

    save_json(
        {"results": all_results, "config": vars(args)},
        out_dir / "ranking_audit_results.json",
    )
    log.info(f"Saved → {out_dir / 'ranking_audit_results.json'}")

    # Per-method summary at α=0.9, averaged over OODs × corruptions
    log.info("\n--- Method summary (α=0.9, avg over OODs × corruptions) ---")
    log.info(f"{'method':14s}  delta_id_acc  mixed_auroc  id_only_auroc  paired_gap")
    for method in args.methods:
        cells = [
            all_results[f"{c}|{o}|{method}|0.9"]
            for c in args.corruptions for o in args.oods
            if f"{c}|{o}|{method}|0.9" in all_results
        ]
        if not cells:
            continue
        avg = lambda k: float(np.mean([v[k] for v in cells]))
        log.info(
            f"  {method:12s}  {avg('delta_id_acc'):+.4f}        "
            f"{avg('mixed_auroc'):.4f}       "
            f"{avg('id_only_auroc'):.4f}         "
            f"{avg('paired_gap'):+.4f}"
        )


if __name__ == "__main__":
    cli()
