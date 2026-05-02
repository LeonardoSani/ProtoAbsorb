"""Step 6 — Fix A (confidence-weighted entropy) revisited under the paired protocol.

Per the reframe, Fix A is only included as a method contribution if it
*consistently* outperforms Mixed TENT in Delta-AUROC while matching or
exceeding it on ID accuracy. This script runs Fix A on the **same drawn
batches** as No TTA / ID-only TENT / Mixed TENT (Step 2 protocol) and shows
its position on the adaptation/abstention Pareto plot.
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


def run_method(
    ckpt_path: str, device: torch.device, batch,
    method: str, n_steps: int, lr: float,
) -> dict:
    """method in {no_tta, id_only_tent, mixed_tent, mixed_fix_a}."""
    if method == "no_tta":
        return run_condition_on_batch(ckpt_path, device, batch,
                                      condition="no_tta", n_steps=n_steps, lr=lr)
    if method == "id_only_tent":
        return run_condition_on_batch(ckpt_path, device, batch,
                                      condition="id_only", n_steps=n_steps, lr=lr,
                                      fix_a=False)
    if method == "mixed_tent":
        return run_condition_on_batch(ckpt_path, device, batch,
                                      condition="mixed", n_steps=n_steps, lr=lr,
                                      fix_a=False)
    if method == "mixed_fix_a":
        return run_condition_on_batch(ckpt_path, device, batch,
                                      condition="mixed", n_steps=n_steps, lr=lr,
                                      fix_a=True)
    raise ValueError(method)


METHODS = ("no_tta", "id_only_tent", "mixed_tent", "mixed_fix_a")


def run_cell(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    n_batches: int, n_steps: int, batch_size: int, ood_name: str,
    severity: int, master_seed: int, lr: float, device: torch.device,
) -> dict:
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)
    batches = draw_paired_batches(sampler, n_batches, master_seed)
    per_method: dict[str, dict[str, list]] = {
        m: {"delta_auroc": [], "delta_id_acc": [],
            "auroc_T": [], "id_acc_T": []} for m in METHODS
    }
    for batch in batches:
        for m in METHODS:
            r = run_method(ckpt_path, device, batch, m, n_steps, lr)
            au0, auT = float(r["msp_auroc"][0]), float(r["msp_auroc"][-1])
            id0, idT = float(r["id_acc"][0]), float(r["id_acc"][-1])
            per_method[m]["delta_auroc"].append(auT - au0)
            per_method[m]["delta_id_acc"].append(idT - id0)
            per_method[m]["auroc_T"].append(auT)
            per_method[m]["id_acc_T"].append(idT)
    summary: dict = {}
    for m in METHODS:
        d = per_method[m]
        au_m, au_lo, au_hi = bootstrap_ci(d["delta_auroc"], seed=master_seed)
        ia_m, ia_lo, ia_hi = bootstrap_ci(d["delta_id_acc"], seed=master_seed + 1)
        summary[m] = {
            "delta_auroc_mean": au_m, "delta_auroc_lo": au_lo, "delta_auroc_hi": au_hi,
            "delta_id_acc_mean": ia_m, "delta_id_acc_lo": ia_lo, "delta_id_acc_hi": ia_hi,
            "auroc_T_mean": float(np.mean(d["auroc_T"])),
            "id_acc_T_mean": float(np.mean(d["id_acc_T"])),
            "n": n_batches,
        }
    summary["paired_fix_a_vs_mixed_tent"] = {
        "auroc": dict(zip(("t", "p"),
                          paired_t_test(per_method["mixed_fix_a"]["delta_auroc"],
                                        per_method["mixed_tent"]["delta_auroc"]))),
        "id_acc": dict(zip(("t", "p"),
                           paired_t_test(per_method["mixed_fix_a"]["delta_id_acc"],
                                         per_method["mixed_tent"]["delta_id_acc"]))),
    }
    return summary


def pareto_plot(table: dict, alphas: list[float], out_path: Path,
                corruption: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    color_map = {
        "no_tta": "tab:gray", "id_only_tent": "tab:blue",
        "mixed_tent": "tab:red", "mixed_fix_a": "tab:green",
    }
    marker_map = {0.9: "o", 0.75: "s", 0.5: "^", 0.25: "D"}
    seen = set()
    for a in alphas:
        for m in METHODS:
            cell = table[a][m]
            label_key = m
            label = m if label_key not in seen else None
            seen.add(label_key)
            ax.errorbar(
                cell["delta_id_acc_mean"], cell["delta_auroc_mean"],
                xerr=[[cell["delta_id_acc_mean"] - cell["delta_id_acc_lo"]],
                      [cell["delta_id_acc_hi"] - cell["delta_id_acc_mean"]]],
                yerr=[[cell["delta_auroc_mean"] - cell["delta_auroc_lo"]],
                      [cell["delta_auroc_hi"] - cell["delta_auroc_mean"]]],
                fmt=marker_map.get(a, "o"), color=color_map[m],
                ecolor=color_map[m], elinewidth=1.0, capsize=3,
                markersize=8, label=label, alpha=0.9,
            )
            ax.annotate(f"a={a}",
                        (cell["delta_id_acc_mean"], cell["delta_auroc_mean"]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=color_map[m])
    ax.axvline(0.0, ls="--", color="k", lw=1)
    ax.axhline(0.0, ls="--", color="k", lw=1)
    ax.set_xlabel(r"$\Delta$ ID accuracy")
    ax.set_ylabel(r"$\Delta$ MSP-AUROC")
    ax.set_title(f"Step 6 — Adaptation/abstention Pareto, {corruption}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--centroids", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/step6")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--ood", default="svhn")
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("step6")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.steps = min(args.steps, 5)
        args.batches = min(args.batches, 3)

    table: dict[float, dict] = {}
    for a in args.alphas:
        log.info(f"alpha={a}")
        s = run_cell(
            ckpt_path=args.ckpt, data_root=args.data_root,
            corruption=args.corruption, alpha=a,
            n_batches=args.batches, n_steps=args.steps,
            batch_size=args.batch_size, ood_name=args.ood,
            severity=args.severity,
            master_seed=args.seed * 7919 + int(a * 1000),
            lr=args.lr, device=device,
        )
        table[a] = s
        for m in METHODS:
            v = s[m]
            log.info(f"  {m:14s} Delta-AUROC={v['delta_auroc_mean']:+.3f} "
                     f"[{v['delta_auroc_lo']:+.3f},{v['delta_auroc_hi']:+.3f}]  "
                     f"Delta-ID-acc={v['delta_id_acc_mean']:+.3f}")
        pa = s["paired_fix_a_vs_mixed_tent"]
        log.info(f"  paired Fix A vs Mixed TENT: AUROC t={pa['auroc']['t']:.2f} "
                 f"p={pa['auroc']['p']:.3g}, "
                 f"ID-acc t={pa['id_acc']['t']:.2f} p={pa['id_acc']['p']:.3g}")

    pareto_plot(table, args.alphas, out_dir / "pareto_fix_a.png", args.corruption)

    save_json({"table": {str(a): table[a] for a in args.alphas},
               "config": vars(args)},
              out_dir / "step6_results.json")
    log.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    cli()
