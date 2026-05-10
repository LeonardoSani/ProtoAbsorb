"""Step 9 — Ranking Reversal Figure.

Two-panel bar chart showing that closed-set and open-world rankings diverge:

  Panel A  (x=method, y=ΔID accuracy)     : standard closed-set ranking
  Panel B  (x=method, y=mixed-stream AUROC + paired-gap overlay)

Both panels average over 2 OODs × 2 corruptions at α=0.9.
Rank annotations illustrate the reversal between panels.

Usage:
  python -m experiments.plot_step9_ranking_reversal \\
    --results results/step9/ranking_audit_results.json \\
    --out results/figures/step9_ranking_reversal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS      = ["no_tta", "tent", "eta", "unient_plus"]
OODS         = ["svhn", "dtd"]
CORRUPTIONS  = ["gaussian_noise", "fog"]
METHOD_LABEL = {
    "no_tta":       "No TTA",
    "tent":         "TENT",
    "eta":          "ETA",
    "unient_plus":  "UniEnt+",
}
METHOD_COLOR = {
    "no_tta":       "#9e9e9e",
    "tent":         "#4c72b0",
    "eta":          "#dd8452",
    "unient_plus":  "#55a868",
}


def _avg_over_grid(results: dict, method: str, metric: str,
                   alpha: float = 0.9) -> float:
    vals = []
    for corr in CORRUPTIONS:
        for ood in OODS:
            key = f"{corr}|{ood}|{method}|{alpha}"
            if key in results and metric in results[key]:
                vals.append(results[key][metric])
    return float(np.mean(vals)) if vals else float("nan")


def _avg_ci(results: dict, method: str, lo_key: str, hi_key: str,
            alpha: float = 0.9) -> tuple[float, float]:
    los, his = [], []
    for corr in CORRUPTIONS:
        for ood in OODS:
            key = f"{corr}|{ood}|{method}|{alpha}"
            if key in results:
                los.append(results[key].get(lo_key, float("nan")))
                his.append(results[key].get(hi_key, float("nan")))
    return float(np.mean(los)), float(np.mean(his))


def _rank_str(rank: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(rank, f"{rank}th")


def plot(results: dict, out_stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    # --- gather data ---
    methods = [m for m in METHODS if any(
        f"{c}|{o}|{m}|0.9" in results
        for c in CORRUPTIONS for o in OODS
    )]

    dac   = [_avg_over_grid(results, m, "delta_id_acc")  for m in methods]
    maur  = [_avg_over_grid(results, m, "mixed_auroc")   for m in methods]
    iaur  = [_avg_over_grid(results, m, "id_only_auroc") for m in methods]
    pgap  = [_avg_over_grid(results, m, "paired_gap")    for m in methods]

    dac_ci = [_avg_ci(results, m, "delta_id_acc_lo", "delta_id_acc_hi") for m in methods]
    mix_ci = [_avg_ci(results, m, "mixed_auroc_lo",  "mixed_auroc_hi")  for m in methods]

    # --- ranks ---
    rank_a = list(np.argsort(np.argsort(-np.array(dac))) + 1)   # higher dac = better
    rank_b = list(np.argsort(np.argsort(-np.array(maur))) + 1)  # higher auroc = better

    x = np.arange(len(methods))
    bar_w = 0.55
    colors = [METHOD_COLOR.get(m, "#888888") for m in methods]
    labels = [METHOD_LABEL.get(m, m) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # --- Panel A: ΔID accuracy ---
    dac_lo = [d - lo for d, (lo, hi) in zip(dac, dac_ci)]
    dac_hi = [hi - d for d, (lo, hi) in zip(dac, dac_ci)]
    bars_a = ax1.bar(x, dac, width=bar_w, color=colors, alpha=0.85,
                     yerr=[dac_lo, dac_hi], capsize=4, error_kw={"lw": 1.2})
    ax1.axhline(0, color="k", lw=0.8, ls="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("ΔID accuracy")
    ax1.set_title("Panel A — Closed-Set Ranking\n(ΔID accuracy, no OOD in batch)")

    for i, (bar, ra, rb) in enumerate(zip(bars_a, rank_a, rank_b)):
        h = bar.get_height()
        offset = max(abs(dac_hi[i]), 0.003) + 0.005
        sign = 1 if h >= 0 else -1
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + sign * offset,
            f"A#{ra} / B#{rb}",
            ha="center", va="bottom" if h >= 0 else "top",
            fontsize=7.5, color="#333333",
        )

    # --- Panel B: mixed AUROC + paired gap overlay ---
    mix_lo = [m - lo for m, (lo, hi) in zip(maur, mix_ci)]
    mix_hi = [hi - m for m, (lo, hi) in zip(maur, mix_ci)]
    bars_b = ax2.bar(x, maur, width=bar_w, color=colors, alpha=0.85,
                     yerr=[mix_lo, mix_hi], capsize=4, error_kw={"lw": 1.2},
                     label="mixed AUROC")

    # Overlay paired gap as hatched extension from mixed_auroc to id_only_auroc
    for i, (xi, ma, ia) in enumerate(zip(x, maur, iaur)):
        if not (np.isnan(ma) or np.isnan(ia)) and ia > ma:
            ax2.bar(xi, ia - ma, bottom=ma, width=bar_w,
                    color=colors[i], alpha=0.35,
                    hatch="///", edgecolor="white", lw=0,
                    label="ID-Oracle gap" if i == 0 else "_")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("MSP-AUROC")
    ax2.set_title("Panel B — Open-World Safety Ranking\n(mixed AUROC; hatching = gap to ID-Oracle)")

    for i, (bar, ra, rb) in enumerate(zip(bars_b, rank_a, rank_b)):
        top = iaur[i] if not np.isnan(iaur[i]) else maur[i]
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            top + 0.005,
            f"A#{ra} / B#{rb}",
            ha="center", va="bottom",
            fontsize=7.5, color="#333333",
        )

    ax2.legend(fontsize=8)

    # --- shared annotation ---
    fig.text(
        0.5, 0.01,
        "A# = rank by ΔID accuracy (Panel A)  ·  B# = rank by mixed AUROC (Panel B)  "
        "·  α=0.9, avg over SVHN+DTD × gaussian_noise+fog",
        ha="center", fontsize=7.5, color="#555555",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}")
    plt.close(fig)
    print(f"Saved → {out_stem}.{{png,pdf}}")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",
                        default="results/step9/ranking_audit_results.json")
    parser.add_argument("--out",
                        default="results/figures/step9_ranking_reversal")
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="Which α slice to average over.")
    args = parser.parse_args()

    with open(args.results) as fh:
        data = json.load(fh)
    results = data.get("results", data)  # tolerate bare dict

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot(results, str(out_path))


if __name__ == "__main__":
    cli()
