"""Step 2 — Contamination isolation (CORE RESULT).

Goal: show that *contamination* of the adaptation batch with OOD samples,
not TTA itself, is the causal factor in OOD-detector degradation.

Protocol (per ``doc/reframe.md``):

  - Three conditions on the **same drawn batches**:
      * ``no_tta``    — evaluate at t=0, no adaptation.
      * ``id_only``   — adapt only on the ID slice of the batch (oracle).
      * ``mixed``     — adapt on the full mixed batch (realistic).
  - Same random seed and same batch draws across all three conditions.
  - n = 30..50 independent batch draws per (alpha, corruption) cell.
  - alpha in {0.9, 0.75, 0.5, 0.25}; main corruption = gaussian_noise plus
    four others.
  - Report mean Delta-AUROC with 95% CI for each condition; paired t-test
    Mixed vs ID-only.

Key figure: forest plot of Delta-AUROC with CIs across alpha values for the
main corruption, plus a generalization panel across corruptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments._common import (
    ALPHAS,
    CIFAR10_C_CORRUPTIONS,
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
    severity: int, master_seed: int, lr: float, device: torch.device,
) -> dict:
    """Run all three conditions on the same drawn batches, return per-batch traces."""
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)
    batches = draw_paired_batches(sampler, n_batches, master_seed)

    per_cond: dict[str, dict[str, list]] = {
        c: {"auroc_0": [], "auroc_T": [], "delta_auroc": [],
            "id_acc_0": [], "id_acc_T": [], "delta_id_acc": []}
        for c in CONDITIONS
    }
    for b, batch in enumerate(batches):
        for cond in CONDITIONS:
            res = run_condition_on_batch(
                ckpt_path, device, batch,
                condition=cond, n_steps=n_steps, lr=lr, tta_method="tent",
            )
            a0, aT = float(res["msp_auroc"][0]), float(res["msp_auroc"][-1])
            i0, iT = float(res["id_acc"][0]), float(res["id_acc"][-1])
            per_cond[cond]["auroc_0"].append(a0)
            per_cond[cond]["auroc_T"].append(aT)
            per_cond[cond]["delta_auroc"].append(aT - a0)
            per_cond[cond]["id_acc_0"].append(i0)
            per_cond[cond]["id_acc_T"].append(iT)
            per_cond[cond]["delta_id_acc"].append(iT - i0)

    summary: dict[str, dict] = {}
    for cond in CONDITIONS:
        d = per_cond[cond]
        m, lo, hi = bootstrap_ci(d["delta_auroc"], seed=master_seed)
        summary[cond] = {
            "delta_auroc_mean": m, "delta_auroc_lo": lo, "delta_auroc_hi": hi,
            "auroc_T_mean": float(np.mean(d["auroc_T"])),
            "id_acc_T_mean": float(np.mean(d["id_acc_T"])),
            "delta_id_acc_mean": float(np.mean(d["delta_id_acc"])),
            "n_batches": len(d["delta_auroc"]),
            "delta_auroc_per_batch": d["delta_auroc"],
        }
    t_stat, p_val = paired_t_test(
        per_cond["mixed"]["delta_auroc"], per_cond["id_only"]["delta_auroc"]
    )
    summary["paired_test_mixed_vs_id_only"] = {"t": t_stat, "p": p_val}
    return summary


_COND_LABEL = {
    "no_tta":  "No-Adapt",
    "id_only": "ID-Oracle",
    "mixed":   "Mixed-Adapt",
}
_COND_COLOR = {
    "no_tta":  "tab:gray",
    "id_only": "tab:blue",
    "mixed":   "tab:red",
}


def forest_plot(results: dict, alphas: list[float], out_path: Path,
                title: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(alphas) * 3 + 1.4))
    y_positions = []
    yticks_labels = []
    legend_done: set = set()
    y = 0
    for a in alphas:
        for cond in CONDITIONS:
            s = results[a][cond]
            lbl = _COND_LABEL[cond] if cond not in legend_done else None
            if lbl:
                legend_done.add(cond)
            ax.errorbar(
                s["delta_auroc_mean"], y,
                xerr=[[s["delta_auroc_mean"] - s["delta_auroc_lo"]],
                      [s["delta_auroc_hi"] - s["delta_auroc_mean"]]],
                fmt="o", color=_COND_COLOR[cond], capsize=4, lw=1.5, markersize=6,
                label=lbl,
            )
            y_positions.append(y)
            yticks_labels.append(rf"$\alpha={a}$  {_COND_LABEL[cond]}")
            y += 1
        y += 0.5
    ax.axvline(0.0, ls="--", color="k", lw=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(yticks_labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$AUROC = AUROC(T) $-$ AUROC(0)  (95% CI)")
    ax.set_title(title)
    handles, labels_leg = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_leg, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--centroids", default=None,
                        help="Unused in step 2 (kept for CLI uniformity).")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/step2")
    parser.add_argument("--main-corruption", default="gaussian_noise")
    parser.add_argument("--extra-corruptions", nargs="+",
                        default=["fog", "impulse_noise", "elastic_transform", "pixelate"])
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--ood", default="svhn",
                        choices=["svhn", "cifar100", "dtd", "places365"])
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=30,
                        help="Independent batch draws per cell (target 30-50).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("step2")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.steps = min(args.steps, 5)
        args.batches = min(args.batches, 3)
        args.extra_corruptions = []

    log.info(f"== Main corruption: {args.main_corruption} ==")
    main_results: dict[float, dict] = {}
    for a in args.alphas:
        log.info(f"  alpha={a}, n_batches={args.batches}")
        s = run_cell(
            ckpt_path=args.ckpt, data_root=args.data_root,
            corruption=args.main_corruption, alpha=a,
            n_batches=args.batches, n_steps=args.steps,
            batch_size=args.batch_size, ood_name=args.ood,
            severity=args.severity,
            master_seed=args.seed * 7919 + int(a * 1000), lr=args.lr,
            device=device,
        )
        main_results[a] = s
        for cond in CONDITIONS:
            d = s[cond]
            log.info(f"    {cond:8s}  Delta-AUROC = {d['delta_auroc_mean']:+.3f} "
                     f"[{d['delta_auroc_lo']:+.3f}, {d['delta_auroc_hi']:+.3f}]  "
                     f"AUROC(T)={d['auroc_T_mean']:.3f}")
        log.info(f"    paired t-test (mixed vs id_only): "
                 f"t={s['paired_test_mixed_vs_id_only']['t']:.2f}  "
                 f"p={s['paired_test_mixed_vs_id_only']['p']:.3g}")

    forest_plot(main_results, args.alphas,
                out_dir / f"forest_{args.main_corruption}.png",
                title=f"Paired contamination gap — {args.main_corruption} (95% CI)")

    log.info("== Other corruptions (sanity / generalization) ==")
    other_results: dict[str, dict[float, dict]] = {}
    for c in args.extra_corruptions:
        other_results[c] = {}
        for a in args.alphas:
            log.info(f"  corruption={c}, alpha={a}")
            s = run_cell(
                ckpt_path=args.ckpt, data_root=args.data_root,
                corruption=c, alpha=a,
                n_batches=max(args.batches // 2, 5), n_steps=args.steps,
                batch_size=args.batch_size, ood_name=args.ood,
                severity=args.severity,
                master_seed=args.seed * 7919 + int(a * 1000) + abs(hash(c)) % 997,
                lr=args.lr, device=device,
            )
            other_results[c][a] = s
        forest_plot(other_results[c], args.alphas,
                    out_dir / f"forest_{c}.png",
                    title=f"Paired contamination gap — {c} (95% CI)")

    save_json({"main": {str(a): main_results[a] for a in args.alphas},
               "other": {c: {str(a): other_results[c][a] for a in args.alphas}
                         for c in other_results},
               "config": vars(args)},
              out_dir / "step2_results.json")

    log.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    cli()
