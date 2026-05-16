# TENT Paper Replication (P1)

Reproduces the CIFAR-10-C and CIFAR-100-C cells of Table 2 from Wang et al.,
*Tent: Fully Test-Time Adaptation by Entropy Minimization*, ICLR 2021
(arXiv:2006.10726v3, PDF at `doc/2006.10726v3.pdf`).

## Provenance

The Python modules `tent_official.py`, `norm_official.py`, `conf.py` and the
YAML configs `cfgs/{source,norm,tent}.yaml` are vendored verbatim from
https://github.com/DequanWang/tent (MIT). See `UPSTREAM.md` for the exact
commit SHA. The only change is a 3-line license/provenance header at the
top of each Python file. CIFAR-100 YAMLs are new and adapt the CIFAR-10
templates by changing only the dataset and architecture keys.

## One-shot reproduction

```bash
uv sync --extra tent-replication --extra dev

# CIFAR-10-C
.venv/bin/python experiments/tent_replication/run_cifar_c.py \
    --cfg experiments/tent_replication/cfgs/source.yaml \
    --out results/tent_replication/cifar10c/source
.venv/bin/python experiments/tent_replication/run_cifar_c.py \
    --cfg experiments/tent_replication/cfgs/norm.yaml \
    --out results/tent_replication/cifar10c/norm
.venv/bin/python experiments/tent_replication/run_cifar_c.py \
    --cfg experiments/tent_replication/cfgs/tent.yaml \
    --out results/tent_replication/cifar10c/tent

# CIFAR-100-C
.venv/bin/python experiments/tent_replication/run_cifar_c.py \
    --cfg experiments/tent_replication/cfgs/source_cifar100.yaml \
    --out results/tent_replication/cifar100c/source
.venv/bin/python experiments/tent_replication/run_cifar_c.py \
    --cfg experiments/tent_replication/cfgs/norm_cifar100.yaml \
    --out results/tent_replication/cifar100c/norm
.venv/bin/python experiments/tent_replication/run_cifar_c.py \
    --cfg experiments/tent_replication/cfgs/tent_cifar100.yaml \
    --out results/tent_replication/cifar100c/tent

# Aggregate + diff vs paper
.venv/bin/python experiments/tent_replication/aggregate.py \
    --cells results/tent_replication/cifar10c/source \
            results/tent_replication/cifar10c/norm \
            results/tent_replication/cifar10c/tent \
            results/tent_replication/cifar100c/source \
            results/tent_replication/cifar100c/norm \
            results/tent_replication/cifar100c/tent \
    --expected experiments/tent_replication/expected.yaml \
    --out docs/tent_replication/reproduction_report.md
```

Exit code 0 iff every cell is within ±0.3% (and Source within ±1.0%).

## Compute

Tested on 1× A100 (Leonardo). End-to-end ~1.5 h including dataset download
from Zenodo (~5 GB cached under `$ROBUSTBENCH_CACHE` if set, otherwise
`./data`).

## Tests

```bash
.venv/bin/pytest tests/tent_replication/ -v
```
