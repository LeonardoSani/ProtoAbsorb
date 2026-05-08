"""Three-way decomposition table.

Attributes the total AUROC change (relative to clean baseline) across three
additive components:

  Δ_corruption    = AUROC(t=0, corrupted) − AUROC(clean)
  Δ_TTA           = AUROC(T, id_only)    − AUROC(t=0, corrupted)
  Δ_contamination = AUROC(T, mixed)      − AUROC(T, id_only)  [always ≤ 0]

All numbers come from existing step1 and step2 JSON files — no GPU needed.

Usage:
    python -m experiments.plot_three_way_decomp --out results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


OOD_CONFIGS = {
    "svhn": {
        "s1": "results/step1/sanity.json",
        "s1_t0_key": "cifar10c_vs_svhn_auroc_t0",
        "s1_clean_key": "clean_vs_svhn_auroc",
        "s2": "results/step2/step2_results.json",
    },
    "dtd": {
        "s1": "results/step1_dtd/sanity.json",
        "s1_t0_key": "cifar10c_vs_dtd_auroc_t0",
        "s1_clean_key": "clean_vs_dtd_auroc",
        "s2": "results/step2_dtd/step2_results.json",
    },
    "places365": {
        "s1": "results/step1_places365/sanity.json",
        "s1_t0_key": "cifar10c_vs_places365_auroc_t0",
        "s1_clean_key": "clean_vs_places365_auroc",
        "s2": "results/step2_places365/step2_results.json",
    },
}

ALPHAS = ["0.9", "0.75", "0.5", "0.25"]


def build_table(root: Path) -> list[dict]:
    rows = []
    for ood, cfg in OOD_CONFIGS.items():
        s1 = json.loads((root / cfg["s1"]).read_text())
        s2 = json.loads((root / cfg["s2"]).read_text())

        clean_msp = s1[cfg["s1_clean_key"]]["msp"]
        t0_msp = s1[cfg["s1_t0_key"]]["gaussian_noise"]["msp"]
        d_corruption = t0_msp - clean_msp

        for alpha_str in ALPHAS:
            cell = s2["main"][alpha_str]
            # id_only delta_auroc_mean = AUROC(T,id_only) - AUROC(t=0)
            d_tta = cell["id_only"]["delta_auroc_mean"]
            # mixed delta - id_only delta = Δ_contamination
            d_contam = cell["mixed"]["delta_auroc_mean"] - cell["id_only"]["delta_auroc_mean"]

            auroc_clean = clean_msp
            auroc_t0 = t0_msp
            auroc_id_only = auroc_t0 + d_tta
            auroc_mixed = auroc_id_only + d_contam

            rows.append({
                "ood": ood, "alpha": float(alpha_str),
                "AUROC_clean": auroc_clean,
                "AUROC_t0": auroc_t0,
                "AUROC_id_only_T": auroc_id_only,
                "AUROC_mixed_T": auroc_mixed,
                "delta_corruption": d_corruption,
                "delta_TTA": d_tta,
                "delta_contamination": d_contam,
            })
    return rows


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'OOD':<12} {'α':>5}  "
        f"{'AUROC_clean':>11}  {'AUROC_t0':>8}  "
        f"{'Δ_corr':>7}  {'AUROC_idT':>9}  "
        f"{'Δ_TTA':>6}  {'AUROC_mxT':>9}  {'Δ_contam':>9}"
    )
    print(header)
    print("-" * len(header))
    prev_ood = None
    for r in rows:
        if prev_ood and prev_ood != r["ood"]:
            print()
        prev_ood = r["ood"]
        print(
            f"{r['ood']:<12} {r['alpha']:>5.2f}  "
            f"{r['AUROC_clean']:>11.3f}  {r['AUROC_t0']:>8.3f}  "
            f"{r['delta_corruption']:>+7.3f}  {r['AUROC_id_only_T']:>9.3f}  "
            f"{r['delta_TTA']:>+6.3f}  {r['AUROC_mixed_T']:>9.3f}  "
            f"{r['delta_contamination']:>+9.3f}"
        )


def save_latex(rows: list[dict], out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Three-way AUROC decomposition (MSP, gaussian\_noise sev-5). "
        r"$\Delta_\text{corr}=\text{AUROC}(t{=}0,\text{corr})-\text{AUROC}(\text{clean})$, "
        r"$\Delta_\text{TTA}=\text{AUROC}(T,\text{id-only})-\text{AUROC}(t{=}0)$, "
        r"$\Delta_\text{contam}=\text{AUROC}(T,\text{mixed})-\text{AUROC}(T,\text{id-only})$.}",
        r"\label{tab:three_way_decomp}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"OOD & $\alpha$ & Clean & $t{=}0$ & $\Delta_\text{corr}$ & "
        r"$T_\text{id}$ & $\Delta_\text{TTA}$ & $T_\text{mx}$ & $\Delta_\text{contam}$ \\",
        r"\midrule",
    ]
    prev_ood = None
    for r in rows:
        if prev_ood and prev_ood != r["ood"]:
            lines.append(r"\midrule")
        prev_ood = r["ood"]
        ood_label = r["ood"] if r["alpha"] == 0.9 else ""
        lines.append(
            f"{ood_label} & {r['alpha']:.2f} & "
            f"{r['AUROC_clean']:.3f} & {r['AUROC_t0']:.3f} & "
            f"\\textbf{{{r['delta_corruption']:+.3f}}} & "
            f"{r['AUROC_id_only_T']:.3f} & "
            f"\\textbf{{{r['delta_TTA']:+.3f}}} & "
            f"{r['AUROC_mixed_T']:.3f} & "
            f"\\textbf{{{r['delta_contamination']:+.3f}}} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX: {out_path}")


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_table(root)
    print_table(rows)

    save_latex(rows, out_dir / "three_way_decomp.tex")

    import json
    (out_dir / "three_way_decomp.json").write_text(
        json.dumps(rows, indent=2)
    )
    print(f"Saved JSON: {out_dir / 'three_way_decomp.json'}")


if __name__ == "__main__":
    cli()
