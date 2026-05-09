"""Step 8 — Four-Condition Confound Control.

Decomposes the contamination cost into three additive components to settle
the batch-size confound objection and strengthen the BN-mechanism claim.

Conditions (each gets a fresh model; all conditions see the same batch):
  1. id_subbatch      — adapt on ID slice only (αB samples)          [oracle]
  2. id_fullmatch     — pad OOD slots with extra corrupted ID → B     [count control]
  3. mixed_maskedloss — full mixed forward (OOD in BN stats), loss on ID only
  4. mixed            — full mixed forward + full loss                 [realistic]

Decomposition:
  id_subbatch − id_fullmatch      = sample-count artifact   (expected ≈ 0)
  id_fullmatch − mixed_maskedloss = BN-statistics contamination  (expected dominant)
  mixed_maskedloss − mixed        = direct OOD-gradient contamination

Cells (n=30 paired batches, TENT vanilla, T=20 steps):
  Main:     SVHN α=0.5 + Places365 α=0.9
  Appendix: SVHN α=0.9 + Places365 α=0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments._common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_T,
    auroc_from_eval,
    bootstrap_ci,
    build_id_pool,
    draw_paired_batches,
    ensure_dir,
    evaluate_batch,
    fresh_tent_model,
    fpr95_from_eval,
    get_device,
    get_logger,
    id_accuracy_from_eval,
    make_sampler,
    paired_t_test,
    save_json,
    set_seed,
    setup_matplotlib,
    TentConfig,
    TentVariant,
    tent_step,
)
from proto_absorb.tent import softmax_entropy


CONDITIONS = ("id_subbatch", "id_fullmatch", "mixed_maskedloss", "mixed")

_COLORS = {
    "id_subbatch":      "tab:blue",
    "id_fullmatch":     "tab:cyan",
    "mixed_maskedloss": "tab:orange",
    "mixed":            "tab:red",
}
_LABELS = {
    "id_subbatch":      "id_subbatch (oracle, αB)",
    "id_fullmatch":     "id_fullmatch (filled, B)",
    "mixed_maskedloss": "mixed_maskedloss (BN only, no OOD grad)",
    "mixed":            "mixed (realistic)",
}


# ---------------------------------------------------------------------------
# Extra-ID sampler for id_fullmatch
# ---------------------------------------------------------------------------

def _sample_extra_id(
    id_pool, n: int, seed: int, device: torch.device
) -> torch.Tensor:
    """Return n images sampled deterministically from id_pool."""
    if n <= 0:
        # Return empty tensor with correct trailing dims
        img0, _ = id_pool[0]
        return torch.zeros(0, *img0.shape, device=device)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(id_pool), size=n, replace=(n > len(id_pool)))
    imgs = torch.stack([id_pool[int(i)][0] for i in idxs])
    return imgs.to(device)


# ---------------------------------------------------------------------------
# Core 4-condition runner
# ---------------------------------------------------------------------------

def run_4cond_on_batch(
    ckpt_path: str,
    device: torch.device,
    batch,          # MixedBatch
    id_pool,        # dataset; supplies extra ID images for id_fullmatch
    extra_seed: int,
    n_steps: int,
    lr: float,
) -> dict[str, dict]:
    """Run all 4 conditions on the same batch. Each condition uses a fresh model.

    Returns dict[condition] → {msp_auroc, msp_fpr95, id_acc}  (arrays len n_steps+1).
    """
    cfg = TentConfig(variant=TentVariant.VANILLA, lr=lr)
    batch_dev = batch.to(device)
    id_mask = ~batch_dev.is_ood  # (B,) bool

    n_ood = int(id_mask.shape[0]) - int(id_mask.sum().item())
    extra_imgs = _sample_extra_id(id_pool, n_ood, extra_seed, device)

    results: dict[str, dict] = {}
    for cond in CONDITIONS:
        model, opt = fresh_tent_model(ckpt_path, device, cfg)

        msp_auroc_curve: list[float] = []
        msp_fpr95_curve: list[float] = []
        id_acc_curve:    list[float] = []

        for t in range(n_steps + 1):
            ev = evaluate_batch(model, batch_dev)
            msp_auroc_curve.append(auroc_from_eval(ev, "msp"))
            msp_fpr95_curve.append(fpr95_from_eval(ev, "msp"))
            id_acc_curve.append(id_accuracy_from_eval(ev))

            if t == n_steps:
                break

            if cond == "id_subbatch":
                if id_mask.any():
                    tent_step(model, opt, batch_dev.images[id_mask], cfg)

            elif cond == "id_fullmatch":
                if id_mask.any():
                    # BN sees only ID images; count matches full batch size
                    full_id = torch.cat([batch_dev.images[id_mask], extra_imgs], dim=0)
                    tent_step(model, opt, full_id, cfg)

            elif cond == "mixed_maskedloss":
                if id_mask.any():
                    # Full batch forward → BN statistics contaminated by OOD
                    # Entropy loss restricted to ID slice → no OOD gradients
                    opt.zero_grad(set_to_none=True)
                    logits = model(batch_dev.images)
                    H = softmax_entropy(logits)
                    loss = H[id_mask].mean()
                    loss.backward()
                    opt.step()

            elif cond == "mixed":
                tent_step(model, opt, batch_dev.images, cfg)

        results[cond] = {
            "msp_auroc": np.array(msp_auroc_curve),
            "msp_fpr95": np.array(msp_fpr95_curve),
            "id_acc":    np.array(id_acc_curve),
        }

    return results


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def run_cell(
    ckpt_path: str,
    data_root: str,
    corruption: str,
    alpha: float,
    ood_name: str,
    n_batches: int,
    n_steps: int,
    batch_size: int,
    severity: int,
    master_seed: int,
    lr: float,
    device: torch.device,
) -> dict:
    sampler  = make_sampler(data_root, corruption, alpha, ood_name, batch_size, severity)
    id_pool  = build_id_pool(data_root, corruption, severity)
    batches  = draw_paired_batches(sampler, n_batches, master_seed)

    per_cond: dict[str, dict[str, list]] = {
        c: {"auroc_0": [], "auroc_T": [], "delta_auroc": []}
        for c in CONDITIONS
    }

    for b, batch in enumerate(batches):
        extra_seed = master_seed * 3_999_971 + b
        res = run_4cond_on_batch(
            ckpt_path, device, batch, id_pool, extra_seed, n_steps, lr
        )
        for cond in CONDITIONS:
            a0 = float(res[cond]["msp_auroc"][0])
            aT = float(res[cond]["msp_auroc"][-1])
            per_cond[cond]["auroc_0"].append(a0)
            per_cond[cond]["auroc_T"].append(aT)
            per_cond[cond]["delta_auroc"].append(aT - a0)

    summary: dict = {}
    for cond in CONDITIONS:
        d = per_cond[cond]
        m, lo, hi = bootstrap_ci(d["delta_auroc"], seed=master_seed)
        summary[cond] = {
            "delta_auroc_mean":      m,
            "delta_auroc_lo":        lo,
            "delta_auroc_hi":        hi,
            "auroc_T_mean":          float(np.mean(d["auroc_T"])),
            "n_batches":             len(d["delta_auroc"]),
            "delta_auroc_per_batch": d["delta_auroc"],
        }

    # Paired tests between adjacent conditions
    pairs = [
        ("id_subbatch",      "id_fullmatch"),
        ("id_fullmatch",     "mixed_maskedloss"),
        ("mixed_maskedloss", "mixed"),
        ("id_subbatch",      "mixed"),
    ]
    for (a, b) in pairs:
        t_stat, p_val = paired_t_test(
            per_cond[a]["delta_auroc"], per_cond[b]["delta_auroc"]
        )
        summary[f"t_{a}_vs_{b}"] = {"t": t_stat, "p": p_val}

    # Decomposition (higher delta_auroc = better = less harm)
    d_sub  = np.array(per_cond["id_subbatch"]["delta_auroc"])
    d_full = np.array(per_cond["id_fullmatch"]["delta_auroc"])
    d_ml   = np.array(per_cond["mixed_maskedloss"]["delta_auroc"])
    d_mix  = np.array(per_cond["mixed"]["delta_auroc"])

    total_gap       = float(np.mean(d_sub - d_mix))
    sample_artifact = float(np.mean(d_sub - d_full))
    bn_contam       = float(np.mean(d_full - d_ml))
    grad_contam     = float(np.mean(d_ml - d_mix))

    def _frac(x: float) -> float:
        return x / total_gap if total_gap != 0 else float("nan")

    summary["decomposition"] = {
        "total_gap":          total_gap,
        "sample_artifact":    sample_artifact,
        "bn_contamination":   bn_contam,
        "grad_contamination": grad_contam,
        "frac_sample":        _frac(sample_artifact),
        "frac_bn":            _frac(bn_contam),
        "frac_grad":          _frac(grad_contam),
    }

    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def forest_plot(results: dict, cells: list[tuple], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(cells) * len(CONDITIONS) + 1.5))
    y = 0
    y_pos, y_lbls = [], []
    legend_done: set[str] = set()

    for ood, alpha in cells:
        cell_key = f"{ood}_a{alpha}"
        if cell_key not in results:
            continue
        cell = results[cell_key]
        for cond in CONDITIONS:
            s = cell[cond]
            m, lo, hi = s["delta_auroc_mean"], s["delta_auroc_lo"], s["delta_auroc_hi"]
            lbl = _LABELS[cond] if cond not in legend_done else None
            ax.errorbar(
                m, y,
                xerr=[[m - lo], [hi - m]],
                fmt="o", color=_COLORS[cond], capsize=4, lw=1.5, markersize=5,
                label=lbl,
            )
            if lbl:
                legend_done.add(cond)
            y_pos.append(y)
            y_lbls.append(f"{ood} α={alpha}  {cond}")
            y += 1
        y += 0.6

    ax.axvline(0, ls="--", color="k", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_lbls, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$AUROC = AUROC(T) − AUROC(0)  [95% CI]")
    ax.set_title("Step 8 — Four-Condition Confound Control")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def decomposition_figure(results: dict, cells: list[tuple], out_path: Path) -> None:
    """Two-panel paper figure.

    Panel A — grouped bar of ΔAUROC per cell (4 conditions).
    Panel B — signed horizontal bars showing each decomposition component.
              BN bar extends LEFT (negative = beneficial); gradient extends RIGHT (harmful).
    """
    import matplotlib.pyplot as plt

    valid_cells = [(o, a) for o, a in cells if f"{o}_a{a}" in results]
    n = len(valid_cells)
    cell_lbls = [f"{o}\nα={a}" for o, a in valid_cells]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, max(3.0, 0.65 * n + 1.8)))

    # ---- Panel A: grouped bar of 4 conditions ----
    bar_w = 0.18
    x = np.arange(n)
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w
    for ci, cond in enumerate(CONDITIONS):
        vals, lo_e, hi_e = [], [], []
        for ood, alpha in valid_cells:
            s = results[f"{ood}_a{alpha}"][cond]
            vals.append(s["delta_auroc_mean"])
            lo_e.append(s["delta_auroc_mean"] - s["delta_auroc_lo"])
            hi_e.append(s["delta_auroc_hi"] - s["delta_auroc_mean"])
        ax_a.bar(x + offsets[ci], vals, width=bar_w,
                 color=_COLORS[cond], label=_LABELS[cond], alpha=0.88)
        ax_a.errorbar(x + offsets[ci], vals, yerr=[lo_e, hi_e],
                      fmt="none", color="k", capsize=2.5, lw=0.9)
    ax_a.axhline(0, color="k", lw=0.8, ls="--")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cell_lbls, fontsize=8)
    ax_a.set_ylabel(r"$\Delta$AUROC (T − 0)")
    ax_a.set_title("A  Four-condition ΔAUROC", fontsize=10)
    ax_a.legend(fontsize=7, frameon=False, loc="upper left")

    # ---- Panel B: signed horizontal bars for each decomposition component ----
    comp_info = [
        ("sample_artifact",    "sample-count artifact",  "tab:gray"),
        ("bn_contamination",   "BN effect (← beneficial = negative)", "tab:blue"),
        ("grad_contamination", "OOD-gradient contamination",          "tab:red"),
    ]
    y = np.arange(n)
    bar_h = 0.22
    offsets_b = np.array([-1, 0, 1]) * bar_h
    for ci, (key, name, color) in enumerate(comp_info):
        vals = [results[f"{o}_a{a}"]["decomposition"][key] for o, a in valid_cells]
        ax_b.barh(y + offsets_b[ci], vals, height=bar_h,
                  color=color, label=name, alpha=0.88)
    ax_b.axvline(0, color="k", lw=0.8, ls="--")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(cell_lbls, fontsize=8)
    ax_b.set_xlabel(r"$\Delta$AUROC contribution  (positive = harmful)")
    ax_b.set_title("B  Decomposition", fontsize=10)
    ax_b.legend(fontsize=7, frameon=False, loc="lower right")

    fig.suptitle("Step 8 — Four-Condition Confound Control", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# Keep old name as alias so existing call-sites don't break
decomposition_bar = decomposition_figure


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli() -> None:
    parser = argparse.ArgumentParser(description="Step 8 four-condition confound control")
    parser.add_argument("--ckpt",        default=None,
                        help="Checkpoint path. Not required with --plot-only.")
    parser.add_argument("--data-root",   default="data")
    parser.add_argument("--out",         default="results/step8")
    parser.add_argument("--plot-only",   action="store_true",
                        help="Reload existing JSON and regenerate figures only.")
    parser.add_argument("--corruption",  default="gaussian_noise",
                        help="ID corruption for CIFAR-10-C.")
    parser.add_argument("--oods",        nargs="+", default=["svhn", "places365"],
                        choices=["svhn", "cifar100", "dtd", "places365"])
    parser.add_argument("--alphas",      type=float, nargs="+", default=[0.5, 0.9],
                        help="ID fraction(s) in mixed batch.")
    parser.add_argument("--severity",    type=int,   default=5)
    parser.add_argument("--steps",       type=int,   default=DEFAULT_T)
    parser.add_argument("--batches",     type=int,   default=30,
                        help="Paired batch draws per cell (target ≥30).")
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--smoke",       action="store_true",
                        help="Quick sanity check (2 batches, 3 steps, 1 cell).")
    args = parser.parse_args()

    import json

    log = get_logger("step8")
    setup_matplotlib()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out)
    log.info(f"Output dir: {out_dir}")

    cells = [(ood, alpha) for ood in args.oods for alpha in args.alphas]

    if args.plot_only:
        json_path = out_dir / "step8_results.json"
        with open(json_path) as fh:
            saved = json.load(fh)
        all_results = saved["results"]
        # keys in JSON are strings; no conversion needed
        log.info(f"Loaded {len(all_results)} cells from {json_path}")
    else:
        if args.ckpt is None:
            parser.error("--ckpt is required unless --plot-only is set")
        device = get_device()
        log.info(f"Device: {device}")

        if args.smoke:
            args.steps   = min(args.steps,   3)
            args.batches = min(args.batches, 2)
            args.oods    = args.oods[:1]
            args.alphas  = args.alphas[:1]
            cells = [(ood, alpha) for ood in args.oods for alpha in args.alphas]

        all_results: dict[str, dict] = {}
        for cell_idx, (ood, alpha) in enumerate(cells):
            cell_key  = f"{ood}_a{alpha}"
            cell_seed = args.seed * 7919 + cell_idx * 997 + int(alpha * 1000)
            log.info(f"== Cell {cell_idx+1}/{len(cells)}: OOD={ood}  alpha={alpha} ==")

            summary = run_cell(
                ckpt_path=args.ckpt,
                data_root=args.data_root,
                corruption=args.corruption,
                alpha=alpha,
                ood_name=ood,
                n_batches=args.batches,
                n_steps=args.steps,
                batch_size=args.batch_size,
                severity=args.severity,
                master_seed=cell_seed,
                lr=args.lr,
                device=device,
            )
            all_results[cell_key] = summary

            for cond in CONDITIONS:
                d = summary[cond]
                log.info(
                    f"  {cond:22s}  ΔA={d['delta_auroc_mean']:+.4f} "
                    f"[{d['delta_auroc_lo']:+.4f}, {d['delta_auroc_hi']:+.4f}]"
                )
            dc = summary["decomposition"]
            log.info(
                f"  Decomposition  total={dc['total_gap']:+.4f}  "
                f"sample={dc['frac_sample']:+.1%}  "
                f"BN={dc['frac_bn']:+.1%}  "
                f"grad={dc['frac_grad']:+.1%}"
            )

        save_json({"results": all_results, "config": vars(args)},
                  out_dir / "step8_results.json")

    forest_plot(all_results, cells, out_dir / "step8_forest.png")
    decomposition_figure(all_results, cells, out_dir / "step8_decomposition.png")
    log.info(f"Done. Results → {out_dir}")


if __name__ == "__main__":
    cli()
