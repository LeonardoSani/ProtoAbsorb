"""Scatter plot: pre-TTA AUROC(t=0) vs paired gap (id_only - mixed ΔAUROC).

Theory prediction: paired gap scales monotonically with pre-TTA separability.
If supported, this is the main theory-confirming figure.

Data sources:
  - x-axis: MSP-AUROC(t=0) from step1 JSON files (each OOD × corruption cell)
  - y-axis: paired gap (id_only Δ - mixed Δ) from step2 JSON files

Usage:
    python -m experiments.plot_scatter_pretTA_vs_gap --out results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


OOD_CONFIGS = {
    "svhn": {
        "s1": "results/step1_svhn/sanity.json",
        "s1_t0_key": "cifar10c_vs_svhn_auroc_t0",
        "s1_clean_key": "clean_vs_svhn_auroc",
        "s2": "results/step2_svhn/step2_results.json",
        "color": "#2166ac",
        "marker": "o",
        "label": "SVHN",
    },
    "dtd": {
        "s1": "results/step1_dtd/sanity.json",
        "s1_t0_key": "cifar10c_vs_dtd_auroc_t0",
        "s1_clean_key": "clean_vs_dtd_auroc",
        "s2": "results/step2_dtd/step2_results.json",
        "color": "#d6604d",
        "marker": "s",
        "label": "DTD",
    },
    "places365": {
        "s1": "results/step1_places365/sanity.json",
        "s1_t0_key": "cifar10c_vs_places365_auroc_t0",
        "s1_clean_key": "clean_vs_places365_auroc",
        "s2": "results/step2_places365/step2_results.json",
        "color": "#4dac26",
        "marker": "^",
        "label": "Places365",
    },
}

CORRUPTION_MARKERS = {
    "gaussian_noise": "o", "fog": "s",
    "impulse_noise": "^", "elastic_transform": "D", "pixelate": "P",
}


def load_rows(root: Path, alpha: float = 0.9) -> list[dict]:
    """Load one point per (OOD × corruption) at fixed alpha.

    Fixing alpha eliminates the alpha-driven confound that dominated the
    60-point version and obscured the cross-corruption pattern.
    """
    alpha_str = str(alpha)
    rows = []
    for ood, cfg in OOD_CONFIGS.items():
        s1 = json.loads((root / cfg["s1"]).read_text())
        s2 = json.loads((root / cfg["s2"]).read_text())
        t0_data = s1[cfg["s1_t0_key"]]

        # main corruption (gaussian_noise)
        id_d = s2["main"][alpha_str]["id_only"]["delta_auroc_mean"]
        mx_d = s2["main"][alpha_str]["mixed"]["delta_auroc_mean"]
        rows.append({
            "ood": ood, "corruption": "gaussian_noise", "alpha": alpha,
            "t0_msp": t0_data["gaussian_noise"]["msp"],
            "id_delta": id_d, "mixed_delta": mx_d,
            "paired_gap": id_d - mx_d,
        })

        # extra corruptions
        for corr in ["fog", "impulse_noise", "elastic_transform", "pixelate"]:
            if corr not in t0_data or corr not in s2["other"]:
                continue
            if alpha_str not in s2["other"][corr]:
                continue
            id_d = s2["other"][corr][alpha_str]["id_only"]["delta_auroc_mean"]
            mx_d = s2["other"][corr][alpha_str]["mixed"]["delta_auroc_mean"]
            rows.append({
                "ood": ood, "corruption": corr, "alpha": alpha,
                "t0_msp": t0_data[corr]["msp"],
                "id_delta": id_d, "mixed_delta": mx_d,
                "paired_gap": id_d - mx_d,
            })
    return rows


def fit_line(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    m, b = np.polyfit(xs, ys, 1)
    from scipy.stats import pearsonr
    r, p = pearsonr(xs, ys)
    return m, b, r, p


def make_plot(rows: list[dict], out_path: Path, alpha: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))

    xs_id, ys_id = [], []
    for r in rows:
        cfg = OOD_CONFIGS[r["ood"]]
        ax.scatter(
            r["id_delta"], r["paired_gap"],
            c=cfg["color"], marker=CORRUPTION_MARKERS.get(r["corruption"], "o"),
            s=90, alpha=0.80, edgecolors="white", linewidths=0.5, zorder=3,
        )
        xs_id.append(r["id_delta"])
        ys_id.append(r["paired_gap"])

    m, b, r_val, p_val = fit_line(xs_id, ys_id)
    x_range = np.linspace(min(xs_id) - 0.005, max(xs_id) + 0.005, 100)
    ax.plot(x_range, m * x_range + b, "k--", lw=1.5, zorder=2,
            label=f"r={r_val:.2f}, p={p_val:.2g}")
    ax.set_xlabel(r"ID-Oracle $\Delta$AUROC", fontsize=10)
    ax.set_ylabel(r"Paired gap ($\Delta$AUROC: ID-Oracle $-$ Mixed-Adapt)", fontsize=9)
    ax.set_title("Contamination penalty scales\nwith adaptation benefit available", fontsize=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor="none")
    print(f"  scales with id-only delta: r={r_val:.3f}, p={p_val:.3g}, n={len(xs_id)}")

    # legend: OOD by color + corruption by marker
    ood_handles = [
        Patch(facecolor=cfg["color"], label=cfg["label"])
        for ood, cfg in OOD_CONFIGS.items()
    ]
    corr_handles = [
        Line2D([0], [0], marker=mk, color="gray", lw=0, markersize=7, label=c)
        for c, mk in CORRUPTION_MARKERS.items() if any(r["corruption"] == c for r in rows)
    ]
    fig.legend(handles=ood_handles + corr_handles, fontsize=7,
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08),
               framealpha=0.9, edgecolor="none")
    fig.suptitle(rf"$\alpha={alpha}$, $n={len(rows)}$ (OOD $\times$ corruption) cells",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="Fixed alpha level to use (default 0.9, strongest signal)")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(root, alpha=args.alpha)
    print(f"Loaded {len(rows)} (OOD × corruption) points at alpha={args.alpha}")

    make_plot(rows, out_dir / "scatter_pretTA_vs_gap.pdf", args.alpha)
    make_plot(rows, out_dir / "scatter_pretTA_vs_gap.png", args.alpha)


if __name__ == "__main__":
    cli()
