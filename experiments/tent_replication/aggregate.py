"""Aggregate per-(corruption,severity) cells into Table 2 rows; diff vs paper.

Reads all *.json cell files under one or more directories, groups by
(dataset, adaptation) at the requested severity, computes the mean error
percentage across the 15 corruption types, and compares to expected.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _load_cells(cell_dirs: list[str]) -> list[dict]:
    cells: list[dict] = []
    for d in cell_dirs:
        for p in Path(d).glob("*.json"):
            cells.append(json.loads(p.read_text()))
    return cells


def aggregate(cell_dirs: list[str], expected_path: str, severity: int = 5) -> dict:
    cells = _load_cells(cell_dirs)
    expected = yaml.safe_load(Path(expected_path).read_text())
    tol_default = float(expected["tolerance"]["default"])
    tol_hard = float(expected["tolerance"]["source_hard_fail"])

    rows: list[dict] = []
    all_pass = True
    grouped: dict[tuple[str, str], list[float]] = {}
    for c in cells:
        if int(c["severity"]) != severity:
            continue
        grouped.setdefault((c["dataset"], c["adaptation"]), []).append(c["err"])

    for (dataset, adaptation), errs in sorted(grouped.items()):
        if len(errs) != 15:
            rows.append({
                "dataset": dataset, "method": adaptation,
                "n_corruptions": len(errs),
                "mean_err_pct": None, "paper_err_pct": None, "delta": None,
                "within_tol": False, "hard_fail": False,
                "note": f"expected 15 corruption types, got {len(errs)}",
            })
            all_pass = False
            continue
        # cells store err as a fraction in [0,1]; tests pass already-percent values
        mean_err_pct = (100.0 * sum(errs) / len(errs)) if max(errs) <= 1.0 else (sum(errs) / len(errs))
        paper = float(expected[dataset][adaptation])
        delta = abs(mean_err_pct - paper)
        hard_fail = adaptation == "source" and delta > tol_hard
        within = delta <= tol_default
        if not within or hard_fail:
            all_pass = False
        rows.append({
            "dataset": dataset, "method": adaptation, "n_corruptions": 15,
            "mean_err_pct": round(mean_err_pct, 3),
            "paper_err_pct": paper, "delta": round(delta, 3),
            "within_tol": within, "hard_fail": hard_fail, "note": "",
        })

    return {"pass": all_pass, "rows": rows, "severity": severity}


def _format_md(report: dict) -> str:
    lines = [
        f"# TENT replication — Table 2 reproduction (severity {report['severity']})",
        "",
        "| dataset | method | mean err % | paper % | |Δ| | within ±0.3 | source hard fail | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['dataset']} | {r['method']} | {r['mean_err_pct']} | "
            f"{r['paper_err_pct']} | {r['delta']} | {r['within_tol']} | "
            f"{r['hard_fail']} | {r['note']} |"
        )
    lines.append("")
    lines.append(f"**Overall pass:** {report['pass']}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", nargs="+", required=True,
                   help="one or more dirs of *.json cell files")
    p.add_argument("--expected", default="experiments/tent_replication/expected.yaml")
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--out", default="docs/tent_replication/reproduction_report.md")
    args = p.parse_args()

    report = aggregate(args.cells, args.expected, args.severity)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(_format_md(report))
    print(_format_md(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
