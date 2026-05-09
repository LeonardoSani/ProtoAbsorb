"""ImageNet-scale paired-gap figure.

Two-panel figure (fog | jpeg_compression). Each panel shows 4 grouped bars:
TENT α=0.9, TENT α=0.5, EATA α=0.9, EATA α=0.5.
Each group has two dots with CI: id_only Δ (blue) and mixed Δ (red).
Paired p-value annotated above each group.

Usage:
    python -m experiments.plot_imagenet_scale --out results/imagenet_scale
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


COLORS = {"id_only": "#2166ac", "mixed": "#d6604d"}
LABELS = {"id_only": "ID-only TENT/EATA", "mixed": "Mixed TENT/EATA"}

CELLS = [
    ("fog", "tent", "0.9"),
    ("fog", "tent", "0.5"),
    ("fog", "eata", "0.9"),
    ("fog", "eata", "0.5"),
    ("jpeg_compression", "tent", "0.9"),
    ("jpeg_compression", "tent", "0.5"),
    ("jpeg_compression", "eata", "0.9"),
    ("jpeg_compression", "eata", "0.5"),
]


def _fmt_p(p: float) -> str:
    if p < 1e-10:
        return f"p<1e-10"
    exp = int(np.floor(np.log10(p)))
    return f"p<1e{exp}"


def make_figure(data: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=False)
    corruption_labels = {"fog": "Fog (sev-5)", "jpeg_compression": "JPEG (sev-5)"}

    for col, corruption in enumerate(["fog", "jpeg_compression"]):
        ax = axes[col]
        cells = [(m, a) for (c, m, a) in CELLS if c == corruption]

        x_positions = np.arange(len(cells))
        offset = 0.18

        for i, (method, alpha) in enumerate(cells):
            key = f"{corruption}|ninco|{method}|{alpha}"
            r = data[key]
            p = r["paired_test_mixed_vs_id_only"]["p"]

            for j, cond in enumerate(["id_only", "mixed"]):
                c = r[cond]
                d_mean = c["delta_auroc_mean"]
                d_lo = c["delta_auroc_lo"]
                d_hi = c["delta_auroc_hi"]
                xerr_lo = d_mean - d_lo
                xerr_hi = d_hi - d_mean
                xpos = x_positions[i] + (offset if j == 0 else -offset)

                ax.errorbar(
                    xpos, d_mean,
                    yerr=[[xerr_lo], [xerr_hi]],
                    fmt="o", color=COLORS[cond],
                    markersize=7, capsize=4, linewidth=1.5,
                    zorder=3,
                )

            # connect id_only and mixed with dashed line
            id_mean = r["id_only"]["delta_auroc_mean"]
            mx_mean = r["mixed"]["delta_auroc_mean"]
            ax.plot(
                [x_positions[i] + offset, x_positions[i] - offset],
                [id_mean, mx_mean],
                color="gray", lw=0.8, linestyle="--", zorder=2, alpha=0.6,
            )

            # annotate p-value above the pair
            y_top = max(r["id_only"]["delta_auroc_hi"], r["mixed"]["delta_auroc_hi"])
            ax.text(
                x_positions[i], y_top + 0.02,
                _fmt_p(p),
                ha="center", va="bottom", fontsize=7, color="#444444",
            )

        ax.axhline(0, color="black", lw=0.8, linestyle="-", alpha=0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [f"{m.upper()}\nα={a}" for m, a in cells],
            fontsize=8,
        )
        ax.set_title(corruption_labels[corruption], fontsize=10, fontweight="bold")
        ax.set_ylabel("ΔAUROC (MSP, vs no-TTA)", fontsize=9)
        ax.set_xlim(-0.6, len(cells) - 0.4)

    # shared legend
    handles = [
        Line2D([0], [0], marker="o", color=COLORS["id_only"], lw=0,
               markersize=8, label="ID-only (oracle)"),
        Line2D([0], [0], marker="o", color=COLORS["mixed"], lw=0,
               markersize=8, label="Mixed (realistic)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, edgecolor="none",
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "ImageNet-Scale Validation: ResNet-50, ImageNet-C → NINCO (n=30 batches)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    for ext in [".png", ".pdf"]:
        p = out_path.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/imagenet_scale/imagenet_scale_results.json")
    parser.add_argument("--out", default="results/imagenet_scale/imagenet_scale_figure")
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())["results"]
    make_figure(data, Path(args.out))


if __name__ == "__main__":
    cli()
