"""Step 5 — Generalization across OODs and TTA methods.

Goal: show the contamination failure mode is not specific to one OOD set or
to TENT. Re-run the Step 2 paired protocol for the cross-product:

    OOD in {SVHN, CIFAR-100}  ×  TTA method in {TENT, EATA}

For each cell we report mean Delta-AUROC with 95% CI for No TTA, ID-only,
and Mixed conditions, plus a paired t-test (Mixed vs ID-only).

Adding a second backbone (ResNet-50 / ViT-S) would also fit here but is out
of scope for this script — the current implementation is single-backbone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments._common import (
    ALPHAS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_T,
    bootstrap_ci,
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


CONDITIONS = ("no_tta", "id_only", "mixed")


def run_cell(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    n_batches: int, n_steps: int, batch_size: int, ood_name: str,
    severity: int, master_seed: int, lr: float, tta_method: str,
    device: torch.device,
) -> dict:
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)
    batches = draw_paired_batches(sampler, n_batches, master_seed)
    per_cond = {c: {"delta_auroc": [], "auroc_T": []} for c in CONDITIONS}
    for batch in batches:
        for cond in CONDITIONS:
            res = run_condition_on_batch(
                ckpt_path, device, batch,
                condition=cond, n_steps=n_steps, lr=lr, tta_method=tta_method,
            )
            a0, aT = float(res["msp_auroc"][0]), float(res["msp_auroc"][-1])
            per_cond[cond]["delta_auroc"].append(aT - a0)
            per_cond[cond]["auroc_T"].append(aT)
    summary: dict = {}
    for cond in CONDITIONS:
        m, lo, hi = bootstrap_ci(per_cond[cond]["delta_auroc"], seed=master_seed)
        summary[cond] = {
            "delta_mean": m, "delta_lo": lo, "delta_hi": hi,
            "auroc_T_mean": float(np.mean(per_cond[cond]["auroc_T"])),
            "n": n_batches,
        }
    t, p = paired_t_test(per_cond["mixed"]["delta_auroc"],
                         per_cond["id_only"]["delta_auroc"])
    summary["paired_mixed_vs_id_only"] = {"t": t, "p": p}
    return summary


def grid_plot(table: dict, oods: list[str], methods: list[str], alphas: list[float],
              out_path: Path) -> None:
    import matplotlib.pyplot as plt
    rows = len(oods)
    cols = len(methods)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 2.6 * rows + 0.6),
                              sharex=True, sharey=True)
    if rows == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes.reshape(-1, 1)
    color_map = {"no_tta": "tab:gray", "id_only": "tab:blue", "mixed": "tab:red"}
    for i, ood in enumerate(oods):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            for k, a in enumerate(alphas):
                cell = table[ood][method][a]
                for off, cond in enumerate(CONDITIONS):
                    s = cell[cond]
                    y = k * (len(CONDITIONS) + 0.6) + off
                    ax.errorbar(
                        s["delta_mean"], y,
                        xerr=[[s["delta_mean"] - s["delta_lo"]],
                              [s["delta_hi"] - s["delta_mean"]]],
                        fmt="o", color=color_map[cond], capsize=3, lw=1.2, markersize=5,
                    )
            ax.axvline(0.0, ls="--", color="k", lw=1)
            ax.set_title(f"{ood} | {method}")
            if j == 0:
                ax.set_ylabel("alpha")
            if i == rows - 1:
                ax.set_xlabel(r"$\Delta$ MSP-AUROC")
            yt = []
            for k, a in enumerate(alphas):
                for off, cond in enumerate(CONDITIONS):
                    yt.append((k * (len(CONDITIONS) + 0.6) + off, f"{a}/{cond}"))
            ax.set_yticks([y for y, _ in yt])
            ax.set_yticklabels([s for _, s in yt], fontsize=6)
            ax.invert_yaxis()
    fig.suptitle("Step 5 — Generalization (rows=OOD, columns=TTA method)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--centroids", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/step5")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--oods", nargs="+", default=["svhn", "cifar100"])
    parser.add_argument("--methods", nargs="+", default=["tent", "eata"])
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("step5")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.steps = min(args.steps, 5)
        args.batches = min(args.batches, 3)

    table: dict[str, dict[str, dict[float, dict]]] = {
        ood: {m: {} for m in args.methods} for ood in args.oods
    }
    for ood in args.oods:
        for method in args.methods:
            for a in args.alphas:
                log.info(f"OOD={ood}, method={method}, alpha={a}")
                cell = run_cell(
                    ckpt_path=args.ckpt, data_root=args.data_root,
                    corruption=args.corruption, alpha=a,
                    n_batches=args.batches, n_steps=args.steps,
                    batch_size=args.batch_size, ood_name=ood,
                    severity=args.severity,
                    master_seed=args.seed * 7919 + int(a * 1000)
                                + abs(hash((ood, method))) % 997,
                    lr=args.lr, tta_method=method, device=device,
                )
                table[ood][method][a] = cell
                for cond in CONDITIONS:
                    s = cell[cond]
                    log.info(f"  {cond:8s} Delta = {s['delta_mean']:+.3f} "
                             f"[{s['delta_lo']:+.3f}, {s['delta_hi']:+.3f}]")
                tt = cell["paired_mixed_vs_id_only"]
                log.info(f"  paired t-test mixed vs id_only: t={tt['t']:.2f} p={tt['p']:.3g}")

    grid_plot(table, args.oods, args.methods, args.alphas,
              out_dir / "generalization_grid.png")

    save_json({"table": {ood: {m: {str(a): table[ood][m][a]
                                  for a in args.alphas}
                              for m in args.methods}
                         for ood in args.oods},
               "config": vars(args)},
              out_dir / "step5_results.json")
    log.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    cli()
