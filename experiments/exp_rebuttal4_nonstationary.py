"""Rebuttal Experiment 4 — Non-stationary alpha(t) stream.

Reviewer concern: "Non-stationary alpha(t) stream (OOD bursts + reset-frequency sweep)
— strengthens deployment realism."

We extend the streaming experiment (step10) to non-stationary contamination:

Pattern A — OOD burst:
  alpha=0.9 for t=0..29, drops to 0.3 for t=30..59 (burst), back to 0.9 for t=60..99
  Tests: does contamination cost spike during burst? Does it recover afterward?

Pattern B — Sinusoidal:
  alpha(t) = 0.5 + 0.4*cos(2*pi*t/50)  (oscillates between 0.1 and 0.9)
  Tests: does paired gap track alpha(t) dynamically?

For each pattern:
  - Run 100-batch sequential stream (no model reset)
  - Methods: TENT (main analysis) + ETA
  - OODs: SVHN + DTD
  - 5 seeds
  - Report per-batch AUROC for id_only and mixed, and the trajectory of paired gap

Output: results/rebuttal4/nonstationary_results.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from experiments._common import (
    DEFAULT_BATCH_SIZE,
    bootstrap_ci,
    build_id_pool,
    build_ood_pool,
    draw_paired_batches,
    ensure_dir,
    evaluate_batch,
    auroc_from_eval,
    fresh_tent_model,
    get_device,
    get_logger,
    save_json,
    set_seed,
    setup_matplotlib,
    TentConfig,
    TentVariant,
    EataConfig,
    EataState,
    eata_step,
    collect_bn_params,
    MixedBatch,
    MixedBatchSampler,
)
from proto_absorb.tent import tent_step, configure_tent_model
from proto_absorb.models import build_resnet18, load_checkpoint

N_BATCHES  = 100
N_SEEDS    = 5
OODS       = ["svhn", "dtd"]
METHODS    = ["tent", "eta"]
BATCH_SIZE = 64


def burst_alpha(t: int) -> float:
    """Pattern A: OOD burst at t=30..59."""
    if 30 <= t < 60:
        return 0.3
    return 0.9


def sinusoidal_alpha(t: int) -> float:
    """Pattern B: sinusoidal oscillation."""
    return float(0.5 + 0.4 * np.cos(2 * np.pi * t / 50))


PATTERNS: dict[str, Callable[[int], float]] = {
    "burst":      burst_alpha,
    "sinusoidal": sinusoidal_alpha,
}


def _draw_batch_at_alpha(
    id_pool, ood_pool, alpha: float, batch_size: int, rng: np.random.Generator,
) -> MixedBatch:
    sampler = MixedBatchSampler(
        id_dataset=id_pool, ood_dataset=ood_pool,
        alpha=alpha, batch_size=batch_size,
    )
    return sampler.next_batch(rng)


def run_stream(
    ckpt_path, data_root, corruption, ood_name, method,
    alpha_fn: Callable[[int], float], n_batches, batch_size,
    seed, lr, device,
) -> dict:
    """Run one non-stationary stream for id_only and mixed conditions.

    Both conditions share the same batch sequence and same random seed,
    so comparisons are paired.
    """
    id_pool  = build_id_pool(data_root, corruption, severity=5)
    ood_pool = build_ood_pool(data_root, ood_name)

    # Pre-draw all batches with deterministic per-batch seeds
    batches: list[MixedBatch] = []
    alpha_seq: list[float] = []
    for t in range(n_batches):
        alpha_t = float(np.clip(alpha_fn(t), 0.05, 0.95))
        alpha_seq.append(alpha_t)
        rng = np.random.default_rng(seed * 1_000_003 + t)
        batches.append(_draw_batch_at_alpha(id_pool, ood_pool, alpha_t, batch_size, rng))

    cfg = TentConfig(variant=TentVariant.VANILLA, lr=lr)
    eta_cfg = EataConfig(lr=lr)

    # Build two independent model instances (one per condition)
    def make_model():
        m, opt = fresh_tent_model(ckpt_path, device, cfg)
        if method == "eta":
            params, _ = collect_bn_params(m)
            opt = torch.optim.SGD(params, lr=lr, momentum=eta_cfg.momentum)
        return m, opt

    model_id, opt_id     = make_model()
    model_mix, opt_mix   = make_model()
    eta_state_id         = EataState() if method == "eta" else None
    eta_state_mix        = EataState() if method == "eta" else None

    auroc_id_list  = []
    auroc_mix_list = []
    alpha_list     = []

    for t, batch in enumerate(batches):
        batch_dev = batch.to(device)
        alpha_t = alpha_seq[t]
        alpha_list.append(alpha_t)

        # --- id_only condition: adapt only on ID slice ---
        id_mask = ~batch_dev.is_ood
        if id_mask.any():
            sub = batch_dev.images[id_mask]
            if method == "eta":
                eata_step(model_id, opt_id, sub, eta_state_id, eta_cfg)
            else:
                tent_step(model_id, opt_id, sub, cfg)

        # --- mixed condition: adapt on full batch ---
        if method == "eta":
            eata_step(model_mix, opt_mix, batch_dev.images, eta_state_mix, eta_cfg)
        else:
            tent_step(model_mix, opt_mix, batch_dev.images, cfg)

        ev_id  = evaluate_batch(model_id,  batch_dev)
        ev_mix = evaluate_batch(model_mix, batch_dev)
        auroc_id_list.append(auroc_from_eval(ev_id,  "msp"))
        auroc_mix_list.append(auroc_from_eval(ev_mix, "msp"))

    gap_list = [a - b for a, b in zip(auroc_id_list, auroc_mix_list)]
    return {
        "auroc_id_only":  auroc_id_list,
        "auroc_mixed":    auroc_mix_list,
        "paired_gap":     gap_list,
        "alpha_seq":      alpha_list,
        "mean_gap":       float(np.mean(gap_list)),
        "mean_auroc_id":  float(np.mean(auroc_id_list)),
        "mean_auroc_mix": float(np.mean(auroc_mix_list)),
    }


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/resnet18_cifar10.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/rebuttal4")
    parser.add_argument("--oods",       nargs="+", default=OODS)
    parser.add_argument("--methods",    nargs="+", default=METHODS)
    parser.add_argument("--patterns",   nargs="+", default=list(PATTERNS.keys()))
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--n-batches",  type=int, default=N_BATCHES)
    parser.add_argument("--n-seeds",    type=int, default=N_SEEDS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--smoke",      action="store_true")
    args = parser.parse_args()

    log = get_logger("rebuttal4_nonstationary")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device  = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.n_batches = 10
        args.n_seeds   = 2
        args.oods      = args.oods[:1]
        args.methods   = args.methods[:1]
        args.patterns  = args.patterns[:1]

    all_results: dict = {}
    total = len(args.patterns) * len(args.oods) * len(args.methods) * args.n_seeds

    idx = 0
    for pattern_name in args.patterns:
        alpha_fn = PATTERNS[pattern_name]
        for ood_name in args.oods:
            for method in args.methods:
                seed_results: list[dict] = []
                for s in range(args.n_seeds):
                    idx += 1
                    seed = args.seed * 100 + s
                    log.info(f"[{idx}/{total}] pattern={pattern_name} ood={ood_name} "
                             f"method={method} seed={seed}")
                    res = run_stream(
                        ckpt_path=args.ckpt, data_root=args.data_root,
                        corruption=args.corruption, ood_name=ood_name,
                        method=method, alpha_fn=alpha_fn,
                        n_batches=args.n_batches, batch_size=args.batch_size,
                        seed=seed, lr=args.lr, device=device,
                    )
                    seed_results.append(res)
                    log.info(f"  mean_gap={res['mean_gap']:+.4f}  "
                             f"id_only={res['mean_auroc_id']:.4f}  "
                             f"mixed={res['mean_auroc_mix']:.4f}")

                # Aggregate across seeds
                mean_gap_per_t = np.mean(
                    np.stack([np.array(r["paired_gap"]) for r in seed_results]), axis=0
                ).tolist()
                mean_id_per_t = np.mean(
                    np.stack([np.array(r["auroc_id_only"]) for r in seed_results]), axis=0
                ).tolist()
                mean_mix_per_t = np.mean(
                    np.stack([np.array(r["auroc_mixed"]) for r in seed_results]), axis=0
                ).tolist()
                overall_gap = float(np.mean([r["mean_gap"] for r in seed_results]))
                gap_std = float(np.std([r["mean_gap"] for r in seed_results]))

                key = f"{pattern_name}|{ood_name}|{method}"
                all_results[key] = {
                    "mean_gap_per_t":  mean_gap_per_t,
                    "mean_id_per_t":   mean_id_per_t,
                    "mean_mix_per_t":  mean_mix_per_t,
                    "alpha_seq":       seed_results[0]["alpha_seq"],
                    "overall_gap":     overall_gap,
                    "gap_std":         gap_std,
                    "seed_results":    seed_results,
                }
                log.info(f"  → overall gap = {overall_gap:+.4f} ± {gap_std:.4f}")

    save_json({"results": all_results, "config": vars(args)},
              out_dir / "nonstationary_results.json")
    log.info(f"Saved → {out_dir / 'nonstationary_results.json'}")

    log.info("\n=== Summary ===")
    for key, res in all_results.items():
        log.info(f"  {key}: overall_gap={res['overall_gap']:+.4f} ± {res['gap_std']:.4f}")


if __name__ == "__main__":
    cli()
