"""Experiment 1 — Verify the failure mode (AUROC degradation under TENT).

Plots:
  1.1 AUROC vs adaptation steps (one curve per alpha, mean+/-std over seeds).
  1.2 AUROC degradation heatmap (corruption x alpha).
  1.3 ID classification accuracy vs adaptation steps (sanity check).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from experiments._common import (
    ALPHAS,
    CIFAR10_C_CORRUPTIONS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_T,
    TentConfig,
    auroc_from_eval,
    ensure_dir,
    evaluate_batch,
    fresh_tent_model,
    get_device,
    get_logger,
    id_accuracy_from_eval,
    make_sampler,
    save_json,
    set_seed,
    setup_matplotlib,
    tent_step,
)


def run_single_run(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    n_steps: int, n_batches: int, batch_size: int, ood_name: str,
    severity: int, seed: int, lr: float, device: torch.device,
) -> dict:
    """Run TENT for ``n_batches`` independent batches, recording AUROC + ID acc per step."""
    rng = np.random.default_rng(seed)
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)

    auroc_curves = np.zeros((n_batches, n_steps + 1), dtype=np.float64)
    acc_curves = np.zeros((n_batches, n_steps + 1), dtype=np.float64)

    for b in range(n_batches):
        cfg = TentConfig(lr=lr)
        model, opt = fresh_tent_model(ckpt_path, device, cfg)
        batch = sampler.next_batch(rng).to(device)

        for t in range(n_steps + 1):
            ev = evaluate_batch(model, batch)
            auroc_curves[b, t] = auroc_from_eval(ev, "msp")
            acc_curves[b, t] = id_accuracy_from_eval(ev)
            if t < n_steps:
                tent_step(model, opt, batch.images, cfg)

    return {
        "auroc_mean": auroc_curves.mean(0),
        "auroc_std": auroc_curves.std(0),
        "acc_mean": acc_curves.mean(0),
        "acc_std": acc_curves.std(0),
    }


def plot_auroc_vs_steps(results: dict, out_dir: Path, corruption: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ALPHAS)))
    baseline = None
    for color, (alpha, res) in zip(colors, sorted(results.items())):
        steps = np.arange(len(res["auroc_mean"]))
        m, s = res["auroc_mean"], res["auroc_std"]
        ax.plot(steps, m, color=color, label=fr"$\alpha={alpha}$", lw=2)
        ax.fill_between(steps, m - s, m + s, color=color, alpha=0.15)
        if baseline is None:
            baseline = m[0]
    if baseline is not None:
        ax.axhline(baseline, ls="--", color="gray", lw=1,
                   label=f"baseline (t=0): {baseline:.3f}")
    ax.set_xlabel("Adaptation step $t$")
    ax.set_ylabel("AUROC (MSP, OOD vs ID)")
    ax.set_title(f"Plot 1.1 — AUROC vs Adaptation Steps  ({corruption})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_1.1_auroc_vs_steps_{corruption}.png")
    plt.close(fig)


def plot_acc_vs_steps(results: dict, out_dir: Path, corruption: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ALPHAS)))
    for color, (alpha, res) in zip(colors, sorted(results.items())):
        steps = np.arange(len(res["acc_mean"]))
        m, s = res["acc_mean"], res["acc_std"]
        ax.plot(steps, m, color=color, label=fr"$\alpha={alpha}$", lw=2)
        ax.fill_between(steps, m - s, m + s, color=color, alpha=0.15)
    ax.set_xlabel("Adaptation step $t$")
    ax.set_ylabel("ID classification accuracy")
    ax.set_title(f"Plot 1.3 — ID Accuracy vs Adaptation Steps  ({corruption})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"plot_1.3_acc_vs_steps_{corruption}.png")
    plt.close(fig)


def plot_heatmap(delta_table: np.ndarray, corruptions: list[str],
                 alphas: list[float], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(corruptions) + 2), 3.5))
    vmax = np.nanmax(np.abs(delta_table))
    im = ax.imshow(delta_table, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels(corruptions, rotation=45, ha="right")
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"alpha={a}" for a in alphas])
    ax.set_title(r"Plot 1.2 — $\Delta$AUROC = AUROC(T) - AUROC(0)")
    fig.colorbar(im, ax=ax, label=r"$\Delta$ AUROC")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_1.2_auroc_degradation_heatmap.png")
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--centroids", default=None,
                        help="(unused in exp1, kept for CLI uniformity)")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/exp1")
    parser.add_argument("--corruptions", nargs="+", default=None,
                        help="If unset: all 15 standard corruptions.")
    parser.add_argument("--main-corruption", default="gaussian_noise",
                        help="Corruption used for the per-alpha curves (Plots 1.1/1.3).")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--ood", default="svhn")
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=5,
                        help="Independent batches per (alpha, corruption).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick sanity run: few corruptions, few steps.")
    args = parser.parse_args()

    log = get_logger("exp1")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.steps = min(args.steps, 5)
        args.batches = min(args.batches, 2)
        corruption_list = [args.main_corruption]
    else:
        corruption_list = args.corruptions or list(CIFAR10_C_CORRUPTIONS)

    main_results: dict[float, dict] = {}
    log.info(f"== Plots 1.1 / 1.3 (corruption={args.main_corruption}) ==")
    for a in args.alphas:
        log.info(f"  alpha={a}")
        res = run_single_run(
            args.ckpt, args.data_root, args.main_corruption, a,
            args.steps, args.batches, args.batch_size, args.ood,
            args.severity, args.seed + int(a * 1000), args.lr, device,
        )
        main_results[a] = res
        log.info(f"    AUROC: {res['auroc_mean'][0]:.3f} -> {res['auroc_mean'][-1]:.3f}, "
                 f"ID acc: {res['acc_mean'][0]:.3f} -> {res['acc_mean'][-1]:.3f}")

    plot_auroc_vs_steps(main_results, out_dir, args.main_corruption)
    plot_acc_vs_steps(main_results, out_dir, args.main_corruption)
    save_json({str(a): {k: v.tolist() for k, v in r.items()}
               for a, r in main_results.items()},
              out_dir / f"results_{args.main_corruption}.json")

    log.info("== Plot 1.2 (delta-AUROC heatmap) ==")
    delta_table = np.zeros((len(args.alphas), len(corruption_list)), dtype=np.float64)
    for i, a in enumerate(args.alphas):
        for j, c in enumerate(corruption_list):
            log.info(f"  alpha={a}, corruption={c}")
            res = run_single_run(
                args.ckpt, args.data_root, c, a,
                args.steps, max(2, args.batches // 2), args.batch_size,
                args.ood, args.severity,
                args.seed + int(a * 1000) + hash(c) % 1000,
                args.lr, device,
            )
            delta_table[i, j] = res["auroc_mean"][-1] - res["auroc_mean"][0]
    plot_heatmap(delta_table, corruption_list, args.alphas, out_dir)
    save_json({"alphas": args.alphas, "corruptions": corruption_list,
               "delta_auroc": delta_table.tolist()},
              out_dir / "heatmap_delta_auroc.json")

    log.info(f"Done. Results saved to {out_dir}")


if __name__ == "__main__":
    cli()
