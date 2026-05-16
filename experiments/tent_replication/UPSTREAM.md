# Upstream provenance

**Repository:** https://github.com/DequanWang/tent (archived 2025-02-15, MIT license)
**Vendored commit:** e9e926a668d85244c66a6d5c006efbd2b82e83e8
**Vendor date:** 2026-05-14

## Files vendored verbatim

- `tent.py` → `experiments/tent_replication/tent_official.py`
- `norm.py` → `experiments/tent_replication/norm_official.py`
- `conf.py` → `experiments/tent_replication/conf.py`
- `cfgs/source.yaml` → `experiments/tent_replication/cfgs/source.yaml`
- `cfgs/norm.yaml`   → `experiments/tent_replication/cfgs/norm.yaml`
- `cfgs/tent.yaml`   → `experiments/tent_replication/cfgs/tent.yaml`

Only modification: a 3-line header comment at the top of each Python file
recording source URL, commit SHA, and the upstream MIT license attribution.
YAML files are byte-identical.

## CIFAR-100 architecture

- `MODEL.ARCH` (CIFAR-100): `Hendrycks2020AugMix_ResNeXt`
- RobustBench source: model_zoo['cifar100'][ThreatModel.corruptions]
- Source-only error must reproduce 67.2% ± 1.0% (Table 2 Source column).
  If it does not, choose a different key.

## RobustBench version

- Pinned via `pyproject.toml` to `robustbench>=1.1,<2`.
- Concrete installed version: 1.1.1.
