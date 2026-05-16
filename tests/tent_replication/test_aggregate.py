"""Unit tests for aggregate.py tolerance + diff logic."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


def _load_agg():
    spec = importlib.util.spec_from_file_location(
        "aggregate", "experiments/tent_replication/aggregate.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


CORRUPTIONS_15 = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
]


def _write_cells(out_dir: Path, dataset: str, adapt: str, errs: list[float]) -> None:
    assert len(errs) == 15
    for corruption, err in zip(CORRUPTIONS_15, errs):
        cell = {
            "dataset": dataset, "adaptation": adapt, "arch": "x",
            "severity": 5, "corruption": corruption,
            "num_ex": 10000, "err": err,
        }
        (out_dir / f"{adapt}_{dataset}_sev5_{corruption}.json"
         ).write_text(json.dumps(cell))


def _write_expected(path: Path) -> None:
    path.write_text(yaml.safe_dump({
        "severity": 5,
        "cifar10": {"source": 40.8, "norm": 17.3, "tent": 14.3},
        "cifar100": {"source": 67.2, "norm": 42.6, "tent": 37.3},
        "tolerance": {"default": 0.3, "source_hard_fail": 1.0},
    }))


def test_within_tolerance_passes(tmp_path):
    agg = _load_agg()
    cells_dir = tmp_path / "cells"; cells_dir.mkdir()
    _write_cells(cells_dir, "cifar10", "tent", [14.3] * 15)
    expected = tmp_path / "expected.yaml"; _write_expected(expected)
    report = agg.aggregate(
        cell_dirs=[str(cells_dir)], expected_path=str(expected), severity=5,
    )
    assert report["pass"] is True
    assert report["rows"][0]["delta"] == pytest.approx(0.0, abs=1e-6)


def test_minor_drift_within_default_tolerance_passes(tmp_path):
    agg = _load_agg()
    cells_dir = tmp_path / "cells"; cells_dir.mkdir()
    _write_cells(cells_dir, "cifar10", "tent", [14.45] * 15)
    expected = tmp_path / "expected.yaml"; _write_expected(expected)
    report = agg.aggregate(
        cell_dirs=[str(cells_dir)], expected_path=str(expected), severity=5,
    )
    assert report["pass"] is True


def test_overshoot_default_tolerance_fails(tmp_path):
    agg = _load_agg()
    cells_dir = tmp_path / "cells"; cells_dir.mkdir()
    _write_cells(cells_dir, "cifar10", "tent", [15.0] * 15)
    expected = tmp_path / "expected.yaml"; _write_expected(expected)
    report = agg.aggregate(
        cell_dirs=[str(cells_dir)], expected_path=str(expected), severity=5,
    )
    assert report["pass"] is False
    failing = [r for r in report["rows"] if not r["within_tol"]]
    assert len(failing) == 1
    assert failing[0]["method"] == "tent" and failing[0]["dataset"] == "cifar10"


def test_source_hard_fail(tmp_path):
    agg = _load_agg()
    cells_dir = tmp_path / "cells"; cells_dir.mkdir()
    _write_cells(cells_dir, "cifar10", "source", [42.0] * 15)
    expected = tmp_path / "expected.yaml"; _write_expected(expected)
    report = agg.aggregate(
        cell_dirs=[str(cells_dir)], expected_path=str(expected), severity=5,
    )
    assert report["pass"] is False
    src_row = [r for r in report["rows"] if r["method"] == "source"][0]
    assert src_row["hard_fail"] is True
