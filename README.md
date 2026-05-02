# ProtoAbsorb

**Adapting to Shift, Forgetting to Abstain — TTA degrades open-world OOD detection.**

This repository implements the experimental protocol described in
[`doc/reframe.md`](./doc/reframe.md), which **supersedes** the original
"prototype absorption" framing in [`Theory.md`](./Theory.md).

The thesis: entropy-minimizing test-time adaptation (TENT, EATA, …) is
*class-closing* but *not novelty-aware*. On mixed test batches that contain
both shifted-ID and OOD samples, the adaptation update sharpens predictions
on every sample — including OOD — and silently degrades any score-based OOD
detector. We expose the failure mode with a paired protocol that controls for
TTA itself, and we evaluate confidence-weighted entropy as a partial fix.

---

## 1. Quick start

```bash
# install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# create venv & install deps
uv sync

# (optional, GPU users) install the CUDA build of torch
uv pip install --upgrade torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. End-to-end pipeline

```bash
# 1. download datasets (CIFAR-10, SVHN, CIFAR-100, CIFAR-10-C ~ 2.6 GB)
uv run python -m proto_absorb.scripts.download_data --ood both

# 2. train a ResNet-18 on clean CIFAR-10 (target >= 93%)
uv run proto-train --epochs 100 --out checkpoints/resnet18_cifar10.pt

# 3. precompute the class centroids + tied covariance (for Mahalanobis)
uv run proto-centroids --ckpt checkpoints/resnet18_cifar10.pt \
                       --out checkpoints/centroids.npz

# 4. run the reframed protocol (each step depends only on steps 1-3)
uv run proto-step1 --ckpt checkpoints/resnet18_cifar10.pt \
                   --centroids checkpoints/centroids.npz \
                   --out results/step1            # BLOCKING sanity baselines
uv run proto-step2 --ckpt checkpoints/resnet18_cifar10.pt \
                   --out results/step2            # CORE: contamination isolation
uv run proto-step3 --ckpt checkpoints/resnet18_cifar10.pt \
                   --centroids checkpoints/centroids.npz \
                   --out results/step3            # detector breadth
uv run proto-step4 --ckpt checkpoints/resnet18_cifar10.pt \
                   --centroids checkpoints/centroids.npz \
                   --out results/step4            # mechanism + Pareto
uv run proto-step5 --ckpt checkpoints/resnet18_cifar10.pt \
                   --out results/step5            # generalization (CIFAR-100 + EATA)
uv run proto-step6 --ckpt checkpoints/resnet18_cifar10.pt \
                   --out results/step6            # Fix A under paired protocol
```

Every step script accepts `--smoke` for a fast correctness check (few
batches, few steps), e.g.:

```bash
uv run proto-step2 --ckpt checkpoints/resnet18_cifar10.pt --smoke
```

> **Step 1 is blocking.** It exits non-zero if clean-CIFAR-10-vs-OOD MSP-AUROC
> falls below the configured threshold (default 0.70). If it fails, debug the
> checkpoint or the OOD scoring pipeline before running anything else.

### Hardware

| Step                | Compute                     | Notes                                  |
|---------------------|-----------------------------|----------------------------------------|
| Train ResNet-18     | GPU strongly recommended    | ~10–20 min on a 3050; intractable on CPU |
| Centroids           | GPU 1 min / CPU 5–10 min    | one pass + one pass for tied covariance |
| Step 1              | < 1 min on GPU              | full-set forward over CIFAR-10 + OOD   |
| Step 2 (n=30)       | ~30–60 min on a 3050        | 3 conditions × 4 alphas × 5 corruptions |
| Step 3              | ~10 min                     | 3 conditions, 1 alpha                  |
| Step 4              | ~15 min                     | 4 alphas × Pareto trace                |
| Step 5              | ~30–60 min                  | 2 OODs × 2 methods × 4 alphas          |
| Step 6              | ~20 min                     | 4 methods × 4 alphas                   |

---

## 3. Project layout

```
src/proto_absorb/
  data.py         CIFAR-10, CIFAR-10-C, SVHN/CIFAR-100, mixed-batch sampler
  models.py       ResNet-18 (CIFAR variant) with feature-extraction hook
  tent.py         TENT (BN-affine entropy minimization) + Fix A/B/C variants
  eata.py         EATA (sample-selection + reweighted entropy, no Fisher)
  centroids.py    Frozen + dynamic centroid + tied-covariance bank
  scorers.py      MSP, Energy, Mahalanobis OOD scorers
  metrics.py      AUROC + accuracy
  utils.py        seeding, device, logging
  cli.py          console-script entry points
  scripts/        data download
experiments/
  _common.py             shared paired-protocol primitives + plotting
  exp_step1_sanity.py            Step 1 (BLOCKING)
  exp_step2_contamination.py     Step 2 (CORE)
  exp_step3_detector_breadth.py  Step 3
  exp_step4_mechanism.py         Step 4
  exp_step5_generalization.py    Step 5 (EATA + CIFAR-100)
  exp_step6_fix_a.py             Step 6
  exp1_failure_mode.py           legacy (original framing)
  exp2_geometric_mechanism.py    legacy
  exp3_fixes.py                  legacy
  exp7_vector_field.py           legacy
doc/
  reframe.md      canonical narrative; supersedes Theory.md
  results.md      conservative reading of the legacy results
```

See [`doc/reframe.md`](./doc/reframe.md) for the full motivation, theory
section, and per-step protocols.

---

## 4. Legacy scripts

The original prototype-absorption protocol is preserved as `proto-exp{1,2,3,7}`
for reproducibility. It is no longer the recommended framing — see
[`doc/reframe.md`](./doc/reframe.md) §1 for why.
