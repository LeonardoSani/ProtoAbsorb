"""ViT-S/16 backbone check — one-cell paired protocol.

Goal: verify the adaptation-abstention conflict holds for a second backbone
(ViT-S/16) with LayerNorm-based TENT. ViT has no BatchNorm; LN computes
per-token statistics so the shared-batch-statistics contamination pathway is
absent. If the paired gap survives, it implicates gradient contamination alone.
If it collapses, it confirms BN as the primary mechanism (consistent with
Ablation A).

Protocol:
  OOD = SVHN, alpha = 0.5, corruption = gaussian_noise sev-5, n = 30 batches.
  Three conditions on the same drawn batches: no_tta / id_only / mixed.
  TENT via LayerNorm affine params.

Outputs: results/vit_backbone/vit_step2_results.json
         results/vit_backbone/forest_gaussian_noise.png
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments._common import (
    auroc_from_eval,
    bootstrap_ci,
    draw_paired_batches,
    ensure_dir,
    get_device,
    get_logger,
    id_accuracy_from_eval,
    paired_t_test,
    save_json,
    set_seed,
    setup_matplotlib,
    tent_step,
    TentConfig,
    TentVariant,
)
from proto_absorb.data import (
    CIFAR10C,
    MixedBatch,
    MixedBatchSampler,
    eval_transform_224,
    svhn_ood_224,
)
from proto_absorb.models import build_vit_small, load_checkpoint
from proto_absorb.tent import collect_ln_params, configure_vit_tent_model
from proto_absorb.scorers import energy_score, msp_score


CONDITIONS = ("no_tta", "id_only", "mixed")


def fresh_vit_tent_model(
    ckpt_path: str, device: torch.device, lr: float
) -> tuple[nn.Module, torch.optim.Optimizer]:
    model = build_vit_small(num_classes=10).to(device)
    load_checkpoint(model, ckpt_path, map_location=str(device))
    configure_vit_tent_model(model)
    params, _ = collect_ln_params(model)
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    return model, opt


def evaluate_batch_vit(model: nn.Module, batch: MixedBatch) -> dict:
    """Eval-only forward pass for ViT (no centroids)."""
    import torch.nn.functional as F
    was_training = {m: m.training for m in model.modules()}
    model.eval()
    try:
        with torch.no_grad():
            logits, feats = model(batch.images, return_features=True)
    finally:
        for m, t in was_training.items():
            m.train(t)

    s_msp = msp_score(logits).cpu().numpy()
    s_eng = energy_score(logits).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    is_ood = batch.is_ood.cpu().numpy()
    labels = batch.labels.cpu().numpy()
    p = F.softmax(logits, dim=1)
    H = -(p * p.clamp_min(1e-12).log()).sum(dim=1).cpu().numpy()
    mx = p.max(dim=1).values.cpu().numpy()
    return {
        "msp": s_msp, "energy": s_eng,
        "preds": preds, "is_ood": is_ood, "labels": labels,
        "entropy": H, "max_p": mx,
    }


def run_condition_vit(
    ckpt_path: str, device: torch.device, batch: MixedBatch,
    condition: str, n_steps: int, lr: float,
) -> dict:
    model, opt = fresh_vit_tent_model(ckpt_path, device, lr)
    batch = batch.to(device)
    cfg = TentConfig(variant=TentVariant.VANILLA, lr=lr)

    n = n_steps if condition != "no_tta" else 0
    msp_curve, energy_curve, acc_curve = [], [], []
    for t in range(n + 1):
        ev = evaluate_batch_vit(model, batch)
        is_ood = ev["is_ood"].astype(bool)
        msp_curve.append(auroc_from_eval(ev, "msp"))
        energy_curve.append(auroc_from_eval(ev, "energy"))
        acc = float((ev["preds"][~is_ood] == ev["labels"][~is_ood]).mean()) if (~is_ood).any() else float("nan")
        acc_curve.append(acc)

        if t < n:
            if condition == "id_only":
                id_mask = ~batch.is_ood
                if id_mask.any():
                    tent_step(model, opt, batch.images[id_mask], cfg)
            elif condition == "mixed":
                tent_step(model, opt, batch.images, cfg)

    return {
        "msp_auroc": np.array(msp_curve),
        "energy_auroc": np.array(energy_curve),
        "id_acc": np.array(acc_curve),
    }


def run_cell(
    ckpt_path: str, data_root: str, alpha: float,
    n_batches: int, n_steps: int, batch_size: int,
    severity: int, master_seed: int, lr: float, device: torch.device,
) -> dict:
    id_pool = CIFAR10C(data_root, corruption="gaussian_noise", severity=severity,
                       transform=eval_transform_224())
    ood_pool = svhn_ood_224(data_root)
    sampler = MixedBatchSampler(id_pool, ood_pool, alpha=alpha, batch_size=batch_size)
    batches = draw_paired_batches(sampler, n_batches, master_seed)

    per_cond: dict[str, dict[str, list]] = {
        c: {"auroc_0": [], "auroc_T": [], "delta_auroc": [],
            "id_acc_T": [], "delta_id_acc": []}
        for c in CONDITIONS
    }
    for b, batch in enumerate(batches):
        for cond in CONDITIONS:
            res = run_condition_vit(ckpt_path, device, batch, cond, n_steps, lr)
            a0, aT = float(res["msp_auroc"][0]), float(res["msp_auroc"][-1])
            iT = float(res["id_acc"][-1])
            i0 = float(res["id_acc"][0])
            per_cond[cond]["auroc_0"].append(a0)
            per_cond[cond]["auroc_T"].append(aT)
            per_cond[cond]["delta_auroc"].append(aT - a0)
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


def forest_plot(summary: dict, alpha: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    color_map = {"no_tta": "tab:gray", "id_only": "tab:blue", "mixed": "tab:red"}
    for y, cond in enumerate(CONDITIONS):
        s = summary[cond]
        ax.errorbar(
            s["delta_auroc_mean"], y,
            xerr=[[s["delta_auroc_mean"] - s["delta_auroc_lo"]],
                  [s["delta_auroc_hi"] - s["delta_auroc_mean"]]],
            fmt="o", color=color_map[cond], capsize=4, lw=1.5, markersize=7,
            label=cond,
        )
    ax.axvline(0, ls="--", color="k", lw=1)
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels(CONDITIONS)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$AUROC = AUROC(T) - AUROC(0)  (95% CI)")
    paired_p = summary["paired_test_mixed_vs_id_only"]["p"]
    ax.set_title(
        f"ViT-S/16 — SVHN OOD, α={alpha}, gaussian_noise sev-5\n"
        f"paired gap p={paired_p:.2e}"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/vit_backbone")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("vit_backbone")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    if args.smoke:
        args.steps = min(args.steps, 3)
        args.batches = min(args.batches, 3)

    log.info(f"ViT backbone check: SVHN, alpha={args.alpha}, n={args.batches} batches")
    summary = run_cell(
        ckpt_path=args.ckpt, data_root=args.data_root,
        alpha=args.alpha, n_batches=args.batches, n_steps=args.steps,
        batch_size=args.batch_size, severity=args.severity,
        master_seed=args.seed, lr=args.lr, device=device,
    )

    for cond in CONDITIONS:
        d = summary[cond]
        log.info(f"  {cond:8s}  Delta-AUROC = {d['delta_auroc_mean']:+.3f} "
                 f"[{d['delta_auroc_lo']:+.3f}, {d['delta_auroc_hi']:+.3f}]  "
                 f"AUROC(T)={d['auroc_T_mean']:.3f}")
    pt = summary["paired_test_mixed_vs_id_only"]
    log.info(f"  paired gap (id_only - mixed): t={pt['t']:.2f}  p={pt['p']:.3g}")

    # compare to ResNet-18 BN result (Ablation A: paired gap = +0.130 at alpha=0.5)
    gap = summary["id_only"]["delta_auroc_mean"] - summary["mixed"]["delta_auroc_mean"]
    log.info(f"  paired gap = {gap:+.3f}  (ResNet-18/BN baseline: +0.130)")

    forest_plot(summary, args.alpha,
                out_dir / f"forest_gaussian_noise_alpha{args.alpha}.png")
    save_json({"summary": summary, "config": vars(args)},
              out_dir / "vit_step2_results.json")
    log.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    cli()
