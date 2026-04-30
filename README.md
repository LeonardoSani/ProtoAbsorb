# ProtoAbsorb

**Prototype Absorption: OOD Detection Degradation under Test-Time Adaptation**

This repository implements the experiments described in [`Theory.md`](./Theory.md).
We empirically verify that under TENT-style entropy minimization on mixed batches
(ID + OOD), OOD feature representations are pulled toward the nearest in-distribution
class centroid — destroying the OOD-detection signal — and we evaluate three fixes.

---

## 1. Quick start

This project is managed with [`uv`](https://github.com/astral-sh/uv).

```bash
# install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# create venv & install deps
uv sync
```

If you have a CUDA GPU and want CUDA-enabled PyTorch, after `uv sync` install the
appropriate wheel into the env, e.g. for CUDA 12.1:

```bash
uv pip install --upgrade torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. End-to-end pipeline

```bash
# 1. download / prepare datasets (CIFAR-10, CIFAR-10-C, SVHN)
uv run python -m proto_absorb.scripts.download_data

# 2. train a ResNet-18 on clean CIFAR-10 (target >=93% test accuracy)
uv run proto-train --epochs 100 --out checkpoints/resnet18_cifar10.pt

# 3. precompute the class centroids on the clean train set
uv run proto-centroids --ckpt checkpoints/resnet18_cifar10.pt \
                       --out checkpoints/centroids.npz

# 4. run experiments
uv run proto-exp1 --ckpt checkpoints/resnet18_cifar10.pt \
                  --centroids checkpoints/centroids.npz \
                  --out results/exp1
uv run proto-exp2 --ckpt checkpoints/resnet18_cifar10.pt \
                  --centroids checkpoints/centroids.npz \
                  --out results/exp2
uv run proto-exp3 --ckpt checkpoints/resnet18_cifar10.pt \
                  --centroids checkpoints/centroids.npz \
                  --out results/exp3
uv run proto-exp7 --ckpt checkpoints/resnet18_cifar10.pt \
                  --centroids checkpoints/centroids.npz \
                  --out results/exp7
```

Use `--smoke` on any experiment script for a fast end-to-end correctness check
(few batches, few steps).

---

## 3. Project layout

```
src/proto_absorb/
  data.py         # CIFAR-10, CIFAR-10-C, SVHN loaders + mixed-batch sampler
  models.py       # ResNet-18 (CIFAR variant) with feature extractor hook
  tent.py         # TENT (BN-affine entropy minimization) + Fix A/B/C variants
  centroids.py    # Frozen + dynamic centroid computation
  scorers.py      # MSP, Energy, Mahalanobis OOD scorers
  metrics.py      # AUROC + distance utilities
  utils.py        # seeding, device, logging
  cli.py          # console entry points
  scripts/        # data download
experiments/
  exp1_failure_mode.py
  exp2_geometric_mechanism.py
  exp3_fixes.py
  exp7_vector_field.py
```

See [`Theory.md`](./Theory.md) for the full experimental protocol and motivation.
