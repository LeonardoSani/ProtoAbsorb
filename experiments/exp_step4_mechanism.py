"""Step 4 — Mechanism (no derivative claim) + Pareto plot.

Goal: support the confidence-sharpening + separability-gap explanation
without making any per-sample derivative claims. Per the reframe, we measure
per-step (t = 0..T) for Mixed TENT vs ID-only TENT:

  - Mean max-softmax on OOD samples       (should INCREASE under Mixed)
  - Mean entropy on OOD samples           (should DECREASE under Mixed)
  - OOD score gap = mean OOD score - mean ID score  (should SHRINK)
  - Feature norm of OOD representations   (controls for global BN rescaling)
  - Normalized centroid margin
        ( ||phi(OOD) - mu_c*||  -  ||phi(ID) - mu_cID|| )  /  ||phi||

Key figure: 4-panel mechanism plot (entropy / max-softmax / margin / feature
norm), one curve per condition (Mixed vs ID-only TENT).

Pareto figure (bonus): Delta-ID-accuracy on x, Delta-OOD-AUROC on y. One
point per (method, alpha, step) over the trace.
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
    draw_paired_batches,
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


CONDITIONS = ("id_only", "mixed")


def run_one_batch(
    ckpt_path: str, device: torch.device, batch,
    condition: str, n_steps: int, lr: float, bank: CentroidBank,
) -> dict[str, np.ndarray]:
    """Run TENT under a single condition and capture per-step diagnostics."""
    cfg = TentConfig(variant=TentVariant.VANILLA, lr=lr)
    model, opt = fresh_tent_model(ckpt_path, device, cfg)
    batch = batch.to(device)
    is_ood = batch.is_ood.cpu().numpy().astype(bool)

    arrs: dict[str, list[float]] = {
        "mean_max_p_ood": [], "mean_entropy_ood": [],
        "msp_gap": [], "energy_gap": [],
        "feat_norm_ood": [], "feat_norm_id": [],
        "centroid_margin": [],
        "auroc_msp": [], "id_acc": [],
    }
    for t in range(n_steps + 1):
        ev = evaluate_batch(model, batch, centroids=bank)
        H, mx, fn = ev["entropy"], ev["max_p"], ev["feat_norm"]
        s_msp, s_eng = ev["msp"], ev["energy"]
        arrs["mean_max_p_ood"].append(float(mx[is_ood].mean()) if is_ood.any() else np.nan)
        arrs["mean_entropy_ood"].append(float(H[is_ood].mean()) if is_ood.any() else np.nan)
        if is_ood.any() and (~is_ood).any():
            arrs["msp_gap"].append(float(s_msp[is_ood].mean() - s_msp[~is_ood].mean()))
            arrs["energy_gap"].append(float(s_eng[is_ood].mean() - s_eng[~is_ood].mean()))
        else:
            arrs["msp_gap"].append(np.nan); arrs["energy_gap"].append(np.nan)
        arrs["feat_norm_ood"].append(float(fn[is_ood].mean()) if is_ood.any() else np.nan)
        arrs["feat_norm_id"].append(float(fn[~is_ood].mean()) if (~is_ood).any() else np.nan)

        feats_t = torch.from_numpy(ev["feats"]).float()
        d, _ = nearest_centroid_distances(feats_t, bank.centroids)
        d_np = d.numpy()
        nrm = float(np.linalg.norm(ev["feats"], axis=1).mean())
        if is_ood.any() and (~is_ood).any():
            margin = (d_np[is_ood].mean() - d_np[~is_ood].mean()) / max(nrm, 1e-8)
            arrs["centroid_margin"].append(float(margin))
        else:
            arrs["centroid_margin"].append(np.nan)

        arrs["auroc_msp"].append(auroc_from_eval(ev, "msp"))
        arrs["id_acc"].append(id_accuracy_from_eval(ev))

        if t < n_steps:
            if condition == "id_only":
                id_mask = ~batch.is_ood
                if id_mask.any():
                    tent_step(model, opt, batch.images[id_mask], cfg)
            elif condition == "mixed":
                tent_step(model, opt, batch.images, cfg)
            else:
                raise ValueError(condition)
    return {k: np.array(v) for k, v in arrs.items()}


def aggregate(traces: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = traces[0].keys()
    out: dict[str, np.ndarray] = {}
    for k in keys:
        stack = np.stack([t[k] for t in traces], axis=0)
        out[f"{k}_mean"] = np.nanmean(stack, axis=0)
        out[f"{k}_std"] = np.nanstd(stack, axis=0)
    return out


def four_panel_plot(by_cond: dict[str, dict[str, np.ndarray]], out_path: Path,
                    alpha: float, corruption: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    panels = [
        ("mean_entropy_ood", "Mean entropy on OOD", axes[0, 0]),
        ("mean_max_p_ood", "Mean max-softmax on OOD", axes[0, 1]),
        ("centroid_margin", "Normalized centroid margin", axes[1, 0]),
        ("feat_norm_ood", "OOD feature norm", axes[1, 1]),
    ]
    color_map = {"id_only": "tab:blue", "mixed": "tab:red"}
    for key, title, ax in panels:
        for cond, agg in by_cond.items():
            m = agg[f"{key}_mean"]; s = agg[f"{key}_std"]
            steps = np.arange(len(m))
            ax.plot(steps, m, color=color_map[cond], lw=2, label=cond)
            ax.fill_between(steps, m - s, m + s, color=color_map[cond], alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("step t")
        ax.legend(frameon=False)
    fig.suptitle(rf"Step 4 — Mechanism, {corruption}, $\alpha={alpha}$")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def pareto_plot(records: list[dict], out_path: Path,
                corruption: str) -> None:
    """records: each item has alpha, condition, step, delta_id_acc, delta_auroc."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    color_map = {"id_only": "tab:blue", "mixed": "tab:red"}
    marker_map = {0.9: "o", 0.75: "s", 0.5: "^", 0.25: "D"}
    seen = set()
    for r in records:
        c = r["condition"]; a = r["alpha"]
        label = None
        key = (c, a)
        if key not in seen:
            seen.add(key)
            label = f"{c}, alpha={a}"
        ax.scatter(r["delta_id_acc"], r["delta_auroc"],
                   c=color_map[c], marker=marker_map.get(a, "o"),
                   alpha=0.55, s=28, label=label,
                   edgecolor="black" if r["step"] == 0 else "none",
                   linewidth=0.7)
    ax.axhline(0.0, ls="--", color="k", lw=1)
    ax.axvline(0.0, ls="--", color="k", lw=1)
    ax.set_xlabel(r"$\Delta$ ID accuracy (vs t=0)")
    ax.set_ylabel(r"$\Delta$ OOD AUROC (vs t=0)")
    ax.set_title(rf"Step 4 — Adaptation/abstention Pareto, {corruption}")
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="results/step4")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Representative alpha for the 4-panel plot.")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS),
                        help="Alphas to include in the Pareto plot.")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--ood", default="svhn")
    parser.add_argument("--steps", type=int, default=DEFAULT_T)
    parser.add_argument("--batches", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    log = get_logger("step4")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    device = get_device()
    log.info(f"Device: {device}")

    bank = CentroidBank.load(args.centroids)

    if args.smoke:
        args.steps = min(args.steps, 5)
        args.batches = min(args.batches, 3)

    log.info(f"== 4-panel mechanism, alpha={args.alpha} ==")
    sampler = make_sampler(args.data_root, args.corruption, args.alpha,
                           args.ood, args.batch_size, args.severity)
    batches = draw_paired_batches(sampler, args.batches,
                                  args.seed * 7919 + int(args.alpha * 1000))
    by_cond_traces: dict[str, list[dict[str, np.ndarray]]] = {c: [] for c in CONDITIONS}
    for batch in batches:
        for cond in CONDITIONS:
            by_cond_traces[cond].append(
                run_one_batch(args.ckpt, device, batch, cond, args.steps, args.lr, bank)
            )
    by_cond = {c: aggregate(by_cond_traces[c]) for c in CONDITIONS}
    four_panel_plot(by_cond, out_dir / f"mechanism_{args.corruption}_alpha{args.alpha}.png",
                    args.alpha, args.corruption)

    log.info("== Pareto: deltaID-acc vs deltaAUROC over (method, alpha, step) ==")
    records: list[dict] = []
    for a in args.alphas:
        sampler_a = make_sampler(args.data_root, args.corruption, a,
                                 args.ood, args.batch_size, args.severity)
        batches_a = draw_paired_batches(sampler_a, args.batches,
                                        args.seed * 7919 + int(a * 1000))
        for cond in CONDITIONS:
            traces = [
                run_one_batch(args.ckpt, device, b, cond, args.steps, args.lr, bank)
                for b in batches_a
            ]
            agg = aggregate(traces)
            id_acc0 = float(agg["id_acc_mean"][0])
            au0 = float(agg["auroc_msp_mean"][0])
            for t in range(len(agg["id_acc_mean"])):
                records.append({
                    "alpha": a, "condition": cond, "step": t,
                    "delta_id_acc": float(agg["id_acc_mean"][t] - id_acc0),
                    "delta_auroc": float(agg["auroc_msp_mean"][t] - au0),
                })
    pareto_plot(records, out_dir / f"pareto_{args.corruption}.png", args.corruption)

    save_json({"by_condition_alpha_main": {
                  c: {k: v.tolist() for k, v in by_cond[c].items()} for c in CONDITIONS},
               "pareto_records": records,
               "config": vars(args)},
              out_dir / "step4_results.json")
    log.info(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    cli()
