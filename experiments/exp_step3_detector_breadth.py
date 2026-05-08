"""Step 3 — Detector breadth.

Goal: show the contamination-induced AUROC degradation is not an artifact of
MSP — it reproduces under Energy and Mahalanobis as well.

Same paired protocol as Step 2 (No TTA / ID-only TENT / Mixed TENT on shared
batches). At each step we compute MSP, Energy, and Mahalanobis on the full
mixed batch using the *current* model's features against frozen centroids.

Key figure: 3-panel plot (one per detector) at a representative alpha
(default 0.5), showing Delta-AUROC with 95% CIs across the three conditions.
All three should show same-sign degradation for Mixed TENT vs ID-only TENT.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments._common import (
    ALPHAS,
    CIFAR10_C_CORRUPTIONS,
    CentroidBank,
    DEFAULT_BATCH_SIZE,
    DEFAULT_T,
    EataConfig,
    EataState,
    TentConfig,
    TentVariant,
    bootstrap_ci,
    draw_paired_batches,
    energy_score,
    ensure_dir,
    evaluate_batch,
    fresh_tent_model,
    get_device,
    get_logger,
    id_accuracy_from_eval,
    mahalanobis_score,
    make_sampler,
    msp_score,
    paired_t_test,
    save_json,
    set_seed,
    setup_matplotlib,
    tent_step,
)
from proto_absorb.eata import eata_step


CONDITIONS = ("no_tta", "id_only", "mixed")


def _scores(ev: dict, bank: CentroidBank | None) -> dict[str, np.ndarray]:
    out = {"msp": ev["msp"], "energy": ev["energy"]}
    if bank is not None and bank.cov_inv is not None:
        feats = torch.from_numpy(ev["feats"]).float()
        out["mahalanobis"] = mahalanobis_score(
            feats, bank.centroids, bank.cov_inv
        ).numpy()
    return out


def _auroc(score: np.ndarray, is_ood: np.ndarray) -> float:
    from proto_absorb.metrics import auroc
    return auroc(score[~is_ood], score[is_ood])


def run_condition_full_detectors(
    ckpt_path: str, device: torch.device, batch,
    condition: str, n_steps: int, lr: float,
    bank: CentroidBank | None,
) -> dict:
    """Like run_condition_on_batch but reports MSP/Energy/Mahalanobis at t=0,T."""
    cfg = TentConfig(variant=TentVariant.VANILLA, lr=lr)
    model, opt = fresh_tent_model(ckpt_path, device, cfg)
    batch = batch.to(device)

    n = n_steps if condition != "no_tta" else 0

    ev0 = evaluate_batch(model, batch, centroids=bank)
    is_ood = ev0["is_ood"].astype(bool)
    s0 = _scores(ev0, bank)
    auroc_0 = {k: _auroc(v, is_ood) for k, v in s0.items()}
    id_acc_0 = id_accuracy_from_eval(ev0)

    for _ in range(n):
        if condition == "id_only":
            id_mask = ~batch.is_ood
            if id_mask.any():
                tent_step(model, opt, batch.images[id_mask], cfg)
        elif condition == "mixed":
            tent_step(model, opt, batch.images, cfg)
        else:
            raise ValueError(condition)

    evT = evaluate_batch(model, batch, centroids=bank)
    sT = _scores(evT, bank)
    auroc_T = {k: _auroc(v, is_ood) for k, v in sT.items()}
    id_acc_T = id_accuracy_from_eval(evT)

    return {"auroc_0": auroc_0, "auroc_T": auroc_T,
            "id_acc_0": id_acc_0, "id_acc_T": id_acc_T}


def run_cell(
    ckpt_path: str, data_root: str, corruption: str, alpha: float,
    n_batches: int, n_steps: int, batch_size: int, ood_name: str,
    severity: int, master_seed: int, lr: float, bank: CentroidBank | None,
    device: torch.device,
) -> dict:
    sampler = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)
    batches = draw_paired_batches(sampler, n_batches, master_seed)

    detectors = ["msp", "energy"] + (["mahalanobis"] if bank is not None and bank.cov_inv is not None else [])
    per_cond = {c: {d: {"d0": [], "dT": [], "delta": []} for d in detectors}
                for c in CONDITIONS}
    id_acc = {c: {"acc_T": []} for c in CONDITIONS}
    for batch in batches:
        for cond in CONDITIONS:
            r = run_condition_full_detectors(
                ckpt_path, device, batch, cond, n_steps, lr, bank
            )
            for d in detectors:
                per_cond[cond][d]["d0"].append(r["auroc_0"][d])
                per_cond[cond][d]["dT"].append(r["auroc_T"][d])
                per_cond[cond][d]["delta"].append(r["auroc_T"][d] - r["auroc_0"][d])
            id_acc[cond]["acc_T"].append(r["id_acc_T"])

    summary: dict = {"detectors": detectors, "n_batches": n_batches}
    for d in detectors:
        summary[d] = {}
        for cond in CONDITIONS:
            arr = per_cond[cond][d]["delta"]
            m, lo, hi = bootstrap_ci(arr, seed=master_seed)
            summary[d][cond] = {
                "delta_mean": m, "delta_lo": lo, "delta_hi": hi,
                "auroc_T": float(np.mean(per_cond[cond][d]["dT"])),
                "auroc_0": float(np.mean(per_cond[cond][d]["d0"])),
                "delta_per_batch": arr,
            }
        t, p = paired_t_test(per_cond["mixed"][d]["delta"],
                             per_cond["id_only"][d]["delta"])
        summary[d]["paired_mixed_vs_id_only"] = {"t": t, "p": p}
    summary["id_acc_T"] = {c: float(np.mean(id_acc[c]["acc_T"])) for c in CONDITIONS}
    return summary


def three_panel_plot(summary: dict, out_path: Path, alpha: float,
                     corruption: str) -> None:
    import matplotlib.pyplot as plt
    detectors = summary["detectors"]
    fig, axes = plt.subplots(1, len(detectors), figsize=(4.0 * len(detectors), 3.6),
                              sharey=True)
    if len(detectors) == 1:
        axes = [axes]
    color_map = {"no_tta": "tab:gray", "id_only": "tab:blue", "mixed": "tab:red"}
    for ax, d in zip(axes, detectors):
        for i, cond in enumerate(CONDITIONS):
            s = summary[d][cond]
            ax.errorbar(
                s["delta_mean"], i,
                xerr=[[s["delta_mean"] - s["delta_lo"]],
                      [s["delta_hi"] - s["delta_mean"]]],
                fmt="o", color=color_map[cond], capsize=4, lw=1.5, markersize=7,
            )
        p = summary[d]["paired_mixed_vs_id_only"]["p"]
        ax.axvline(0.0, ls="--", color="k", lw=1)
        ax.set_yticks(range(len(CONDITIONS)))
        ax.set_yticklabels(CONDITIONS)
        ax.invert_yaxis()
        ax.set_xlabel(r"$\Delta$AUROC")
        ax.set_title(f"{d}\np(mixed vs id_only)={p:.2g}")
    fig.suptitle(rf"Step 3 — Detector breadth, {corruption}, $\alpha={alpha}$")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--centroids", required=True,
                        help="Frozen centroid bank with covariance for Mahalanobis.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/step3")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Representative alpha (per the reframe, default 0.5).")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="If set, run multi-alpha and produce one figure per alpha.")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--ood", default="svhn",
                        choices=["svhn", "cifar100", "dtd", "places365"])
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("step3")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    bank = CentroidBank.load(args.centroids)
    if bank.cov_inv is None:
        log.warning("Centroid bank has no covariance; Mahalanobis will be skipped.")

    if args.smoke:
        args.steps = min(args.steps, 5)
        args.batches = min(args.batches, 3)

    alphas = args.alphas if args.alphas else [args.alpha]
    out_summary: dict[str, dict] = {}
    for a in alphas:
        log.info(f"alpha={a}")
        s = run_cell(
            ckpt_path=args.ckpt, data_root=args.data_root,
            corruption=args.corruption, alpha=a,
            n_batches=args.batches, n_steps=args.steps,
            batch_size=args.batch_size, ood_name=args.ood,
            severity=args.severity,
            master_seed=args.seed * 7919 + int(a * 1000),
            lr=args.lr, bank=bank, device=device,
        )
        out_summary[str(a)] = s
        for d in s["detectors"]:
            log.info(f"  detector={d}")
            for cond in CONDITIONS:
                v = s[d][cond]
                log.info(f"    {cond:8s} Delta = {v['delta_mean']:+.3f} "
                         f"[{v['delta_lo']:+.3f}, {v['delta_hi']:+.3f}] "
                         f"AUROC(T)={v['auroc_T']:.3f}")
            tt = s[d]["paired_mixed_vs_id_only"]
            log.info(f"    paired t-test mixed vs id_only: t={tt['t']:.2f} p={tt['p']:.3g}")

        three_panel_plot(s, out_dir / f"detector_breadth_{args.corruption}_alpha{a}.png",
                          a, args.corruption)

    save_json({"results": out_summary, "config": vars(args)},
              out_dir / "step3_results.json")
    log.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    cli()
