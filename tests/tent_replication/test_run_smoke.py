"""Smoke test for run_cifar_c.run_from_cfg() with patched RobustBench."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def driver_module(monkeypatch, tmp_path):
    # Build a tiny BN-only model so collect_params() finds something.
    class TinyBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(3, 10)

        def forward(self, x):
            return self.fc(self.pool(self.bn(x)).flatten(1))

    tiny = TinyBN().eval()

    import robustbench.utils as ru
    import robustbench.data as rd

    def fake_load_model(arch, ckpt_dir, dataset, threat_model):
        return tiny

    n = 8
    y = torch.zeros(n, dtype=torch.long)
    x = torch.zeros(n, 3, 32, 32)
    tiny.fc.weight.data.zero_()
    tiny.fc.bias.data.zero_()
    tiny.fc.bias.data[0] = 10.0

    def fake_load_cifar10c(num_ex, severity, data_dir, shuffle, corruptions):
        return x[:num_ex], y[:num_ex]
    fake_load_cifar100c = fake_load_cifar10c

    monkeypatch.setattr(ru, "load_model", fake_load_model)
    monkeypatch.setattr(rd, "load_cifar10c", fake_load_cifar10c, raising=False)
    monkeypatch.setattr(rd, "load_cifar100c", fake_load_cifar100c, raising=False)

    sys.path.insert(0, "experiments/tent_replication")
    spec = importlib.util.spec_from_file_location(
        "run_cifar_c", "experiments/tent_replication/run_cifar_c.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_tiny_cfg(path: Path, dataset: str, adaptation: str) -> None:
    yaml_text = f"""
MODEL:
  ADAPTATION: {adaptation}
  ARCH: dummy
TEST:
  BATCH_SIZE: 4
CORRUPTION:
  DATASET: {dataset}
  SEVERITY:
    - 5
  TYPE:
    - gaussian_noise
  NUM_EX: 8
OPTIM:
  METHOD: Adam
  STEPS: 1
  BETA: 0.9
  LR: 1e-3
  WD: 0.
"""
    path.write_text(yaml_text)


def test_source_cifar10_smoke(driver_module, tmp_path):
    cfg_path = tmp_path / "src.yaml"
    _write_tiny_cfg(cfg_path, "cifar10", "source")
    out = driver_module.run_from_cfg(str(cfg_path), out_dir=str(tmp_path / "out"))
    assert out["cells"][0]["err"] == pytest.approx(0.0, abs=1e-6)
    assert out["cells"][0]["dataset"] == "cifar10"
    assert out["cells"][0]["corruption"] == "gaussian_noise"
    assert out["cells"][0]["severity"] == 5


def test_tent_cifar100_smoke(driver_module, tmp_path):
    cfg_path = tmp_path / "tent.yaml"
    _write_tiny_cfg(cfg_path, "cifar100", "tent")
    out = driver_module.run_from_cfg(str(cfg_path), out_dir=str(tmp_path / "out"))
    assert out["cells"][0]["dataset"] == "cifar100"
    assert out["cells"][0]["err"] == pytest.approx(0.0, abs=1e-6)
