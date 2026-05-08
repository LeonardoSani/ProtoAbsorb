"""Experiment A — BatchNorm vs InstanceNorm ablation.

Tests whether shared BatchNorm statistics are a necessary component of the
contamination mechanism (beyond gradient contamination alone).

Theory:
  - BN-TENT: batch statistics computed over all samples (ID + OOD mixed) before
    the gradient step. OOD shifts BN mean/var → contaminates ID features.
  - IN-TENT: per-sample statistics → no batch-statistics contamination pathway.
    Only gradient contamination remains.

If IN-TENT paired gap < BN-TENT paired gap: BN statistics are part of mechanism.
If gaps are equal: gradient contamination alone explains the effect.

Protocol: Step 2 paired protocol (no_tta / id_only / mixed), SVHN, α=0.5,
gaussian_noise sev-5, n=30 batches.

Usage:
    python -m experiments.exp_bn_ablation \
        --ckpt checkpoints/resnet18_cifar10.pt \
        --out results/ablation_bn_vs_in
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments._common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_T,
    TentConfig,
    TentVariant,
    auroc_from_eval,
    bootstrap_ci,
    draw_paired_batches,
    ensure_dir,
    evaluate_batch,
    get_device,
    get_logger,
    id_accuracy_from_eval,
    make_sampler,
    paired_t_test,
    save_json,
    set_seed,
    setup_matplotlib,
)
from proto_absorb.models import bn_to_in, build_resnet18, load_checkpoint
from proto_absorb.tent import (
    TentConfig,
    collect_bn_params,
    collect_in_params,
    configure_in_model,
    configure_tent_model,
    make_optimizer,
    tent_step,
)

CONDITIONS = ("no_tta", "id_only", "mixed")


def fresh_bn_model(ckpt_path: str, device: torch.device, cfg: TentConfig):
    model = build_resnet18(num_classes=10).to(device)
    load_checkpoint(model, ckpt_path, map_location=str(device))
    configure_tent_model(model)
    params, _ = collect_bn_params(model)
    opt = make_optimizer(params, cfg)
    return model, opt


def fresh_in_model(ckpt_path: str, device: torch.device, cfg: TentConfig):
    base = build_resnet18(num_classes=10)
    load_checkpoint(base, ckpt_path, map_location="cpu")
    model = bn_to_in(base).to(device)
    configure_in_model(model)
    params, _ = collect_in_params(model)
    opt = make_optimizer(params, cfg)
    return model, opt


def run_one_condition(
    model_fn, ckpt_path: str, device: torch.device,
    batch, condition: str, n_steps: int, cfg: TentConfig,
) -> dict:
    model, opt = model_fn(ckpt_path, device, cfg)
    batch = batch.to(device)

    n = n_steps if condition != "no_tta" else 0

    ev0 = evaluate_batch(model, batch)
    auroc_0 = auroc_from_eval(ev0, "msp")
    id_acc_0 = id_accuracy_from_eval(ev0)

    for _ in range(n):
        if condition == "id_only":
            id_mask = ~batch.is_ood
            if id_mask.any():
                tent_step(model, opt, batch.images[id_mask], cfg)
        elif condition == "mixed":
            tent_step(model, opt, batch.images, cfg)

    evT = evaluate_batch(model, batch)
    auroc_T = auroc_from_eval(evT, "msp")
    id_acc_T = id_accuracy_from_eval(evT)

    return {
        "auroc_0": auroc_0, "auroc_T": auroc_T,
        "delta_auroc": auroc_T - auroc_0,
        "id_acc_0": id_acc_0, "id_acc_T": id_acc_T,
        "delta_id_acc": id_acc_T - id_acc_0,
    }


def run_cell(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    n_batches: int, n_steps: int, batch_size: int, ood_name: str,
    severity: int, master_seed: int, lr: float, device: torch.device,
) -> dict:
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)
    batches = draw_paired_batches(sampler, n_batches, master_seed)
    cfg = TentConfig(variant=TentVariant.VANILLA, lr=lr)

    results = {}
    for norm_type, model_fn in [("bn", fresh_bn_model), ("in", fresh_in_model)]:
        per_cond: dict[str, list] = {c: [] for c in CONDITIONS}
        for batch in batches:
            for cond in CONDITIONS:
                r = run_one_condition(model_fn, ckpt_path, device, batch, cond, n_steps, cfg)
                per_cond[cond].append(r["delta_auroc"])

        summary: dict = {}
        for cond in CONDITIONS:
            m, lo, hi = bootstrap_ci(per_cond[cond], seed=master_seed)
            summary[cond] = {"delta_mean": m, "delta_lo": lo, "delta_hi": hi,
                             "delta_per_batch": per_cond[cond]}
        t_stat, p_val = paired_t_test(per_cond["mixed"], per_cond["id_only"])
        summary["paired_mixed_vs_id_only"] = {"t": t_stat, "p": p_val}

        # paired gap = id_only delta - mixed delta per batch
        gaps = [a - b for a, b in zip(per_cond["id_only"], per_cond["mixed"])]
        gm, glo, ghi = bootstrap_ci(gaps, seed=master_seed)
        summary["paired_gap"] = {"mean": gm, "lo": glo, "hi": ghi,
                                  "per_batch": gaps}
        results[norm_type] = summary

    # t-test: is IN gap significantly smaller than BN gap?
    t_bn_vs_in, p_bn_vs_in = paired_t_test(
        results["bn"]["paired_gap"]["per_batch"],
        results["in"]["paired_gap"]["per_batch"],
    )
    results["gap_comparison"] = {
        "t_bn_vs_in": t_bn_vs_in, "p_bn_vs_in": p_bn_vs_in,
        "bn_gap_mean": results["bn"]["paired_gap"]["mean"],
        "in_gap_mean": results["in"]["paired_gap"]["mean"],
        "reduction_fraction": 1.0 - results["in"]["paired_gap"]["mean"] / results["bn"]["paired_gap"]["mean"]
            if results["bn"]["paired_gap"]["mean"] != 0 else float("nan"),
    }
    return results


def forest_plot(results: dict, out_path: Path, alpha: float, corruption: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
    })

    color_map = {"no_tta": "tab:gray", "id_only": "tab:blue", "mixed": "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=True)

    for ax, norm_type in zip(axes, ["bn", "in"]):
        r = results[norm_type]
        for i, cond in enumerate(CONDITIONS):
            s = r[cond]
            ax.errorbar(
                s["delta_mean"], i,
                xerr=[[s["delta_mean"] - s["delta_lo"]],
                      [s["delta_hi"] - s["delta_mean"]]],
                fmt="o", color=color_map[cond], capsize=4, lw=1.5, markersize=7,
                label=cond if norm_type == "bn" else None,
            )
        gap = r["paired_gap"]
        p = r["paired_mixed_vs_id_only"]["p"]
        norm_label = "BatchNorm" if norm_type == "bn" else "InstanceNorm"
        ax.axvline(0.0, ls="--", color="k", lw=1)
        ax.set_yticks(range(len(CONDITIONS)))
        ax.set_yticklabels(CONDITIONS)
        ax.invert_yaxis()
        ax.set_xlabel(r"$\Delta$AUROC (MSP)", fontsize=9)
        ax.set_title(
            f"{norm_label}\ngap={gap['mean']:+.3f} [{gap['lo']:+.3f},{gap['hi']:+.3f}]"
            f"  p={p:.2g}",
            fontsize=9,
        )

    gc = results["gap_comparison"]
    fig.suptitle(
        rf"BN vs IN ablation  —  {corruption}, $\alpha={alpha}$"
        f"\nBN gap={gc['bn_gap_mean']:+.3f}  IN gap={gc['in_gap_mean']:+.3f}"
        f"  p(BN>IN)={gc['p_bn_vs_in']:.2g}"
        f"  reduction={gc['reduction_fraction']:.0%}",
        fontsize=9,
    )
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/resnet18_cifar10.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/ablation_bn_vs_in")
    parser.add_argument("--ood", default="svhn")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("bn_ablation")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.steps = min(args.steps, 5)
        args.batches = min(args.batches, 3)

    results = run_cell(
        ckpt_path=args.ckpt, data_root=args.data_root,
        corruption=args.corruption, alpha=args.alpha,
        n_batches=args.batches, n_steps=args.steps,
        batch_size=args.batch_size, ood_name=args.ood,
        severity=args.severity,
        master_seed=args.seed * 7919 + int(args.alpha * 1000),
        lr=args.lr, device=device,
    )

    gc = results["gap_comparison"]
    log.info(f"BN gap  = {gc['bn_gap_mean']:+.4f}")
    log.info(f"IN gap  = {gc['in_gap_mean']:+.4f}")
    log.info(f"Reduction = {gc['reduction_fraction']:.1%}")
    log.info(f"p(BN gap > IN gap) = {gc['p_bn_vs_in']:.3g}")

    if gc["p_bn_vs_in"] < 0.05:
        log.info("RESULT: IN gap significantly smaller → BN statistics are part of mechanism")
    else:
        log.info("RESULT: gaps not significantly different → gradient contamination alone")

    forest_plot(
        results, out_dir / f"bn_vs_in_{args.ood}_{args.corruption}_alpha{args.alpha}.png",
        args.alpha, args.corruption,
    )
    save_json({"results": results, "config": vars(args)},
              out_dir / "results.json")
    log.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    cli()
