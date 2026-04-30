"""Experiment 3 — Proposed fixes for prototype absorption.

We compare:
  - No TTA           : evaluate at t=0 only.
  - TENT (vanilla)   : entropy minimization on every sample.
  - Fix A            : MSP-confidence-weighted entropy.
  - Fix B            : Vanilla TENT + prototype-anchor regularizer toward
                       frozen centroids for confidently-classified samples.
  - Fix C            : Hard OOD filtering (drop top-fraction by entropy)
                       before computing the TENT loss.
  - Fix A + B        : Combined ablation (weighted entropy + anchor).

For each method and each alpha we report:
  - AUROC at t=T
  - delta-AUROC = AUROC(T) - AUROC(0)
  - ID classification accuracy at t=T
  - Mean OOD->frozen-centroid distance at t=T
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments._common import (
    ALPHAS,
    CentroidBank,
    DEFAULT_BATCH_SIZE,
    DEFAULT_T,
    TentConfig,
    TentVariant,
    auroc_from_eval,
    ensure_dir,
    evaluate_batch,
    fresh_tent_model,
    get_device,
    get_logger,
    id_accuracy_from_eval,
    make_sampler,
    nearest_centroid_distances,
    save_json,
    set_seed,
    setup_matplotlib,
    tent_step,
)


METHODS: tuple[tuple[str, TentVariant], ...] = (
    ("No TTA", TentVariant.VANILLA),  # special-cased: 0 steps
    ("TENT", TentVariant.VANILLA),
    ("Fix A", TentVariant.FIX_A_WEIGHTED),
    ("Fix B", TentVariant.FIX_B_ANCHOR),
    ("Fix C", TentVariant.FIX_C_FILTER),
    ("Fix A+B", TentVariant.FIX_AB),
)


def run_method(
    name: str, variant: TentVariant,
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    n_steps: int, n_batches: int, batch_size: int, ood_name: str,
    severity: int, seed: int, lr: float, base_centroids: CentroidBank,
    anchor_weight: float, anchor_threshold: float, drop_fraction: float,
    device: torch.device,
) -> dict:
    rng = np.random.default_rng(seed)
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)

    final_aurocs, final_accs, final_dists, baseline_aurocs = [], [], [], []
    cents_dev = base_centroids.centroids.to(device)

    for b in range(n_batches):
        cfg = TentConfig(
            variant=variant, lr=lr,
            anchor_weight=anchor_weight, anchor_threshold=anchor_threshold,
            drop_fraction=drop_fraction,
        )
        model, opt = fresh_tent_model(ckpt_path, device, cfg)
        batch = sampler.next_batch(rng).to(device)

        ev0 = evaluate_batch(model, batch)
        baseline_aurocs.append(auroc_from_eval(ev0, "msp"))

        n_steps_this = 0 if name == "No TTA" else n_steps
        for _ in range(n_steps_this):
            tent_step(model, opt, batch.images, cfg, centroids=cents_dev)

        ev = evaluate_batch(model, batch)
        final_aurocs.append(auroc_from_eval(ev, "msp"))
        final_accs.append(id_accuracy_from_eval(ev))

        feats = torch.from_numpy(ev["feats"]).float()
        d, _ = nearest_centroid_distances(feats, base_centroids.centroids)
        d_np = d.cpu().numpy()
        is_ood = ev["is_ood"].astype(bool)
        if is_ood.any():
            final_dists.append(float(d_np[is_ood].mean()))
        else:
            final_dists.append(np.nan)

    return {
        "name": name,
        "variant": variant.value,
        "auroc_T": float(np.mean(final_aurocs)),
        "auroc_T_std": float(np.std(final_aurocs)),
        "auroc_0": float(np.mean(baseline_aurocs)),
        "delta_auroc": float(np.mean(final_aurocs) - np.mean(baseline_aurocs)),
        "id_acc_T": float(np.mean(final_accs)),
        "ood_dist_T": float(np.nanmean(final_dists)),
    }


def plot_summary(table: dict, out_dir: Path, alphas: list[float]) -> None:
    """Bar chart: methods x alphas, primary panel = AUROC@T, secondary = ID acc."""
    import matplotlib.pyplot as plt
    method_names = list(next(iter(table.values())).keys())
    n_methods = len(method_names)
    n_alphas = len(alphas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.0 * n_methods + 4, 4.4))

    x = np.arange(n_methods)
    bar_w = 0.8 / n_alphas
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_alphas))

    for i, a in enumerate(alphas):
        aurocs = [table[a][m]["auroc_T"] for m in method_names]
        accs = [table[a][m]["id_acc_T"] for m in method_names]
        ax1.bar(x + (i - n_alphas / 2) * bar_w + bar_w / 2,
                aurocs, width=bar_w, color=colors[i], label=fr"$\alpha={a}$")
        ax2.bar(x + (i - n_alphas / 2) * bar_w + bar_w / 2,
                accs, width=bar_w, color=colors[i], label=fr"$\alpha={a}$")

    for ax, ylabel, title in [
        (ax1, "AUROC at t=T", "Plot 3.1 — OOD detection AUROC by method"),
        (ax2, "ID accuracy at t=T", "Plot 3.2 — ID classification accuracy by method"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "plot_3_summary.png")
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/exp3")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--ood", default="svhn")
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--anchor-threshold", type=float, default=0.7)
    parser.add_argument("--drop-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("exp3")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    base = CentroidBank.load(args.centroids)

    if args.smoke:
        args.steps = min(args.steps, 5)
        args.batches = min(args.batches, 2)

    table: dict[float, dict[str, dict]] = {}
    for a in args.alphas:
        table[a] = {}
        for name, variant in METHODS:
            log.info(f"alpha={a}  method={name}")
            res = run_method(
                name=name, variant=variant,
                ckpt_path=args.ckpt, data_root=args.data_root,
                corruption=args.corruption, alpha=a,
                n_steps=args.steps, n_batches=args.batches,
                batch_size=args.batch_size, ood_name=args.ood,
                severity=args.severity,
                seed=args.seed + int(a * 1000) + abs(hash(name)) % 1000,
                lr=args.lr, base_centroids=base,
                anchor_weight=args.anchor_weight,
                anchor_threshold=args.anchor_threshold,
                drop_fraction=args.drop_fraction,
                device=device,
            )
            table[a][name] = res
            log.info(f"  AUROC@T={res['auroc_T']:.3f}, "
                     f"deltaAUROC={res['delta_auroc']:+.3f}, "
                     f"id_acc={res['id_acc_T']:.3f}, "
                     f"d_ood={res['ood_dist_T']:.3f}")

    plot_summary(table, out_dir, args.alphas)
    save_json({str(a): table[a] for a in args.alphas},
              out_dir / "fix_results.json")
    log.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    cli()
