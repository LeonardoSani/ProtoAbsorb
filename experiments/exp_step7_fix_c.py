"""Step 7 — Fix C: threshold filtering (entropy-based OOD exclusion before TENT step).

Protocol:
  - Same paired batches as Step 2 (gaussian_noise, severity 5).
  - Conditions: no_tta, id_only, mixed_vanilla, mixed_fix_c (sweep drop_fraction τ).
  - OODs: svhn, dtd (the two with clear absolute drops).
  - Alphas: 0.9, 0.5 (high-contamination and mid-contamination).
  - Metric: MSP-AUROC delta and id_acc delta.
  - Output: Pareto plot (AUROC benefit vs id_acc cost) and summary table.

Usage:
    python -m experiments.exp_step7_fix_c \
        --ckpt checkpoints/resnet18_cifar10.pt \
        --data-root data \
        --out results/step7
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
    bootstrap_ci,
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
)

# drop_fraction = 0.0 → vanilla TENT on mixed batch
DROP_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.5]


def run_cell(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    n_batches: int, n_steps: int, batch_size: int, ood_name: str,
    severity: int, master_seed: int, lr: float, device: torch.device,
) -> dict:
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)
    batches = draw_paired_batches(sampler, n_batches, master_seed)

    conditions = ["no_tta", "id_only"] + [f"fix_c_{df}" for df in DROP_FRACTIONS]
    per_cond: dict[str, dict[str, list]] = {
        c: {"auroc_0": [], "auroc_T": [], "delta_auroc": [],
            "id_acc_0": [], "id_acc_T": [], "delta_id_acc": []}
        for c in conditions
    }

    for batch in batches:
        # no_tta and id_only via standard helper
        for cond in ["no_tta", "id_only"]:
            res = run_condition_on_batch(
                ckpt_path, device, batch, condition=cond,
                n_steps=n_steps, lr=lr, tta_method="tent",
            )
            a0, aT = float(res["msp_auroc"][0]), float(res["msp_auroc"][-1])
            i0, iT = float(res["id_acc"][0]), float(res["id_acc"][-1])
            per_cond[cond]["auroc_0"].append(a0)
            per_cond[cond]["auroc_T"].append(aT)
            per_cond[cond]["delta_auroc"].append(aT - a0)
            per_cond[cond]["id_acc_0"].append(i0)
            per_cond[cond]["id_acc_T"].append(iT)
            per_cond[cond]["delta_id_acc"].append(iT - i0)

        # Fix C sweep: run each drop_fraction on the same batch
        for df in DROP_FRACTIONS:
            cond_name = f"fix_c_{df}"
            cfg = TentConfig(variant=TentVariant.FIX_C_FILTER, lr=lr, drop_fraction=df)
            res = _run_fix_c_on_batch(ckpt_path, device, batch, n_steps, cfg)
            a0, aT = res["auroc_0"], res["auroc_T"]
            i0, iT = res["id_acc_0"], res["id_acc_T"]
            per_cond[cond_name]["auroc_0"].append(a0)
            per_cond[cond_name]["auroc_T"].append(aT)
            per_cond[cond_name]["delta_auroc"].append(aT - a0)
            per_cond[cond_name]["id_acc_0"].append(i0)
            per_cond[cond_name]["id_acc_T"].append(iT)
            per_cond[cond_name]["delta_id_acc"].append(iT - i0)

    summary: dict[str, dict] = {}
    for cond in conditions:
        d = per_cond[cond]
        m, lo, hi = bootstrap_ci(d["delta_auroc"], seed=master_seed)
        summary[cond] = {
            "delta_auroc_mean": m, "delta_auroc_lo": lo, "delta_auroc_hi": hi,
            "auroc_T_mean": float(np.mean(d["auroc_T"])),
            "id_acc_T_mean": float(np.mean(d["id_acc_T"])),
            "delta_id_acc_mean": float(np.mean(d["delta_id_acc"])),
            "n_batches": len(d["delta_auroc"]),
        }

    # paired tests: each fix_c vs id_only
    for df in DROP_FRACTIONS:
        cond_name = f"fix_c_{df}"
        t_stat, p_val = paired_t_test(
            per_cond[cond_name]["delta_auroc"],
            per_cond["id_only"]["delta_auroc"],
        )
        summary[cond_name]["paired_vs_id_only"] = {"t": t_stat, "p": p_val}
        # also vs vanilla mixed
        t_stat2, p_val2 = paired_t_test(
            per_cond[cond_name]["delta_auroc"],
            per_cond["fix_c_0.0"]["delta_auroc"],
        )
        summary[cond_name]["paired_vs_vanilla_mixed"] = {"t": t_stat2, "p": p_val2}

    return summary


def _run_fix_c_on_batch(
    ckpt_path: str, device: torch.device, batch,
    n_steps: int, cfg: TentConfig,
) -> dict:
    from experiments._common import evaluate_batch, auroc_from_eval, id_accuracy_from_eval
    from proto_absorb.tent import tent_step

    model, opt = fresh_tent_model(ckpt_path, device, cfg)
    batch = batch.to(device)

    ev0 = evaluate_batch(model, batch)
    auroc_0 = auroc_from_eval(ev0, "msp")
    id_acc_0 = id_accuracy_from_eval(ev0)

    for _ in range(n_steps):
        tent_step(model, opt, batch.images, cfg)

    evT = evaluate_batch(model, batch)
    auroc_T = auroc_from_eval(evT, "msp")
    id_acc_T = id_accuracy_from_eval(evT)

    return {"auroc_0": auroc_0, "auroc_T": auroc_T,
            "id_acc_0": id_acc_0, "id_acc_T": id_acc_T}


def pareto_plot(results: dict, ood_name: str, corruption: str,
                alphas: list[float], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
    })

    alpha_colors = {0.9: "#d73027", 0.5: "#4575b4"}
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for alpha in alphas:
        cell = results[str(alpha)]
        color = alpha_colors.get(alpha, "gray")

        # reference points
        id_ref = cell["id_only"]
        ax.scatter(id_ref["delta_id_acc_mean"], id_ref["delta_auroc_mean"],
                   c=color, marker="*", s=200, zorder=5,
                   label=f"id_only α={alpha}" if alpha == alphas[0] else None)

        xs, ys = [], []
        for df in DROP_FRACTIONS:
            c = cell[f"fix_c_{df}"]
            xs.append(c["delta_id_acc_mean"])
            ys.append(c["delta_auroc_mean"])

        ax.plot(xs, ys, "-o", color=color, lw=1.5, markersize=6,
                label=f"Fix C α={alpha}")
        for df, x, y in zip(DROP_FRACTIONS, xs, ys):
            ax.annotate(f"τ={df}", (x, y), textcoords="offset points",
                        xytext=(4, 2), fontsize=7, color=color)

    ax.axhline(0, ls="--", color="k", lw=1)
    ax.axvline(0, ls=":", color="k", lw=1)
    ax.set_xlabel("ΔAUROC (id_acc, ID accuracy change)", fontsize=10)
    ax.set_ylabel("ΔAUROC (MSP-AUROC change)", fontsize=10)
    ax.set_title(f"Fix C Pareto — {ood_name}, {corruption}", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/resnet18_cifar10.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/step7")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.9, 0.5])
    parser.add_argument("--oods", nargs="+", default=["svhn", "dtd"])
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("step7")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.steps = min(args.steps, 5)
        args.batches = min(args.batches, 3)
        args.alphas = [0.9]
        args.oods = ["svhn"]

    all_results: dict = {}
    for ood in args.oods:
        all_results[ood] = {}
        for alpha in args.alphas:
            log.info(f"OOD={ood} alpha={alpha}")
            seed = args.seed * 7919 + int(alpha * 1000)
            cell = run_cell(
                ckpt_path=args.ckpt, data_root=args.data_root,
                corruption=args.corruption, alpha=alpha,
                n_batches=args.batches, n_steps=args.steps,
                batch_size=args.batch_size, ood_name=ood,
                severity=args.severity, master_seed=seed,
                lr=args.lr, device=device,
            )
            all_results[ood][str(alpha)] = cell

            log.info(f"  id_only Δ={cell['id_only']['delta_auroc_mean']:+.3f}")
            for df in DROP_FRACTIONS:
                c = cell[f"fix_c_{df}"]
                p_id = c.get("paired_vs_id_only", {}).get("p", float("nan"))
                p_mx = c.get("paired_vs_vanilla_mixed", {}).get("p", float("nan"))
                log.info(f"  fix_c τ={df}: Δ={c['delta_auroc_mean']:+.3f}  "
                         f"p(vs id_only)={p_id:.2g}  p(vs vanilla)={p_mx:.2g}")

        pareto_plot(
            all_results[ood], ood, args.corruption,
            args.alphas, out_dir / f"pareto_{ood}_{args.corruption}.png"
        )

    save_json({"results": all_results, "config": vars(args)},
              out_dir / "step7_results.json")
    log.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    cli()
