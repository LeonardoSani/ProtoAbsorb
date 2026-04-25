# Prototype Absorption: OOD Detection Degradation under Test-Time Adaptation

**Authors:** Leonardo Sani, Giuseppe Stillitano, Xavier Del Giudice, Can Lin

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Central Hypothesis](#2-central-hypothesis)
3. [Setup](#3-setup)
   - 3.1 [Datasets](#31-datasets)
   - 3.2 [Model](#32-model)
   - 3.3 [Centroid Computation](#33-centroid-computation)
   - 3.4 [OOD Scoring Baseline](#34-ood-scoring-baseline)
4. [Experiment 1 — Verify the Failure Mode](#4-experiment-1--verify-the-failure-mode)
5. [Experiment 2 — Characterize the Geometric Mechanism](#5-experiment-2--characterize-the-geometric-mechanism)
6. [Experiment 3 — Proposed Fixes](#6-experiment-3--proposed-fixes)
7. [Optional — Latent Space Vector Field](#7-optional--latent-space-vector-field)
8. [Task Checklist](#8-task-checklist)
9. [References](#9-references)

---

## 1. Motivation

Test-Time Adaptation (TTA) methods — most notably **TENT** [1] — allow a pre-trained model to adapt at inference time by minimizing prediction entropy over incoming test batches. While effective against covariate shift, these methods implicitly assume that test batches consist *exclusively* of in-distribution (ID) samples.

In realistic deployment, this assumption fails: test batches routinely contain **out-of-distribution (OOD)** samples, i.e., images from semantic categories never seen during training. The test stream is therefore a mixture:

$$
X_{\text{test}} \sim \alpha \, P_{\mathrm{ID}_{\text{shifted}}} + (1-\alpha) \, P_{\mathrm{OOD}}
$$

where $\alpha \in (0,1)$ controls the proportion of shifted-ID vs. OOD content. We ask: **what does entropy minimization do to OOD samples when they are silently included in the adaptation stream?**

---

## 2. Central Hypothesis

Let $\phi(x) \in \mathbb{R}^d$ be the penultimate-layer (pre-logit) feature of sample $x$, and let

$$
\mu_c = \frac{1}{|S_c|} \sum_{x \in S_c} \phi(x), \quad c \in \{1,\ldots,C\}
$$

be the centroid of class $c$ computed on the clean training set $S_c$.

**Hypothesis (Prototype Absorption):** Under TENT updates on mixed batches, OOD feature representations are pulled monotonically toward the nearest ID class centroid:

$$\frac{d}{dt}\, \lVert\phi(x_{\mathrm{OOD}}) - \mu_{c^\ast}\rVert_2 < 0, \qquad c^\ast = \arg\min_c \lVert\phi(x_{\mathrm{OOD}}) - \mu_c\rVert_2$$

As OOD features collapse onto ID prototypes, any OOD detector relying on feature-space geometry (distance, energy, softmax confidence) will degrade — the model effectively *absorbs* OOD samples into its in-distribution manifold.

---

## 3. Setup

### 3.1 Datasets

| Role | Dataset | Notes |
|---|---|---|
| **ID (clean)** | CIFAR-10 | 10 classes, used for training and centroid computation |
| **ID (shifted)** | CIFAR-10-C | Standard corruptions from [2]: noise, blur, weather, digital (severity 1–5) |
| **OOD** | CIFAR-100 (superclasses not overlapping) / SVHN / Tiny-ImageNet | Semantically disjoint from CIFAR-10 classes |

> **Note:** At test time, each batch is constructed by sampling a fraction $\alpha$ from the shifted-ID pool and $(1-\alpha)$ from the OOD pool. The labels are **never** used during adaptation — TENT is fully unsupervised.

**Corruption types to use (CIFAR-10-C):**
- Gaussian noise, shot noise, impulse noise
- Defocus blur, glass blur, motion blur, zoom blur
- Snow, frost, fog, brightness, contrast, elastic transform, pixelate, JPEG compression

**Values of $\alpha$ to sweep:** `{0.9, 0.75, 0.5, 0.25}` (i.e., from mostly-ID to mostly-OOD streams)

**Corruption severity:** Fix at severity `5` for main experiments; ablate over `{1,3,5}` if time permits.

---

### 3.2 Model

- **Architecture:** ResNet-18
- **Pre-training:** Supervised on CIFAR-10 (clean); target accuracy ≥ 93% on clean test set
- **TENT adaptation target:** BatchNorm affine parameters $(\gamma, \beta)$ only — all other weights frozen
- **Adaptation steps per batch:** $t = 1, \ldots, T$, with $T = 20$ (or until convergence of AUROC drop)
- **Batch size:** 64 (standard TENT setting)
- **Feature extractor:** Everything up to and including the global average pooling layer, i.e., $\phi(x) \in \mathbb{R}^{512}$

---

### 3.3 Centroid Computation

Centroids $\{\mu_c\}_{c=1}^{10}$ are computed **once** on the clean CIFAR-10 training set using the frozen pre-trained encoder:

$$
\mu_c^{(0)} = \frac{1}{|S_c|} \sum_{x \in S_c} \phi_{\theta_0}(x)
$$

Two centroid regimes are tracked throughout all experiments:

| Regime | Symbol | Description |
|---|---|---|
| **Frozen** | $\mu_c^{(0)}$ | Computed once before any adaptation, never updated |
| **Dynamic** | $\mu_c^{(t)}$ | Recomputed after each TENT step using **ID-only** samples in the current batch (oracle access to ID labels used **only** for centroid update, not for TENT itself) |

> The comparison between frozen and dynamic centroids isolates whether the degradation is driven by OOD feature drift, ID feature drift, or both.

---

### 3.4 OOD Scoring Baseline

Use **Maximum Softmax Probability (MSP)** as the primary OOD score:

$$
s(x) = 1 - \max_c p_c(x)
$$

Higher $s(x)$ → more likely OOD. AUROC is computed treating ID samples as negative and OOD samples as positive.

Additionally track:
- **Energy score:** $E(x) = -\log \sum_c e^{f_c(x)}$
- **Mahalanobis distance** to nearest centroid: $d_M(x) = \min_c \sqrt{(\phi(x)-\mu_c)^\top \Sigma^{-1} (\phi(x)-\mu_c)}$

---

## 4. Experiment 1 — Verify the Failure Mode

**Goal:** Show that AUROC for OOD detection degrades monotonically as TENT adaptation steps $t$ increase on mixed batches.

### Protocol

```
For each alpha in {0.9, 0.75, 0.5, 0.25}:
    Reset model to pre-trained weights (theta_0)
    For t = 0, 1, ..., T:
        Record AUROC(t) using MSP score on current batch
        Perform one TENT gradient step on the mixed batch
```

- **Baseline (no TTA):** AUROC at $t=0$ (frozen model, mixed batch)
- **Each step:** collect logits on all samples → compute MSP → binary classification AUROC (ID=0, OOD=1)
- Repeat over multiple random batches and report **mean ± std**

### Plots

**Plot 1.1 — AUROC vs. Adaptation Steps**
- X-axis: adaptation step $t \in [0, T]$
- Y-axis: AUROC
- One curve per $\alpha$ value
- Shaded region: ±1 std across batch seeds
- Horizontal dashed line: AUROC at $t=0$ (no TTA baseline)

```
Expected shape: monotonically decreasing curves, steeper for lower alpha (more OOD contamination)
```

**Plot 1.2 — AUROC Degradation Heatmap**
- X-axis: corruption type (e.g., gaussian noise, fog, …)
- Y-axis: $\alpha$ value
- Color: $\Delta\text{AUROC} = \text{AUROC}(T) - \text{AUROC}(0)$
- Reveals which corruption types + OOD ratios are most destructive

**Plot 1.3 — Classification Accuracy vs. Adaptation Steps** *(sanity check)*
- Same setup but tracking clean ID accuracy
- Verifies that TENT is actually helping for ID samples while hurting OOD detection

---

## 5. Experiment 2 — Characterize the Geometric Mechanism

**Goal:** Confirm that OOD features geometrically approach the nearest ID centroid under TENT — the core "absorption" mechanism.

### Protocol

```
For each alpha in {0.9, 0.75, 0.5, 0.25}:
    For centroid_mode in {frozen, dynamic}:
        Reset model to pre-trained weights
        For t = 0, 1, ..., T:
            Compute phi(x) for all OOD samples in batch
            For each OOD sample x_i:
                d_i(t) = min_c || phi(x_i) - mu_c^(t) ||_2
            Record: mean(d_i(t)), std(d_i(t)), full distribution {d_i(t)}
            Perform one TENT step
```

### Metrics

| Metric | Symbol | Description |
|---|---|---|
| Mean distance | $\bar{d}(t)$ | Average $L_2$ distance OOD → nearest centroid at step $t$ |
| Per-class absorption | $\bar{d}_c(t)$ | Mean distance to centroid $c$ for OOD samples absorbed into class $c$ |
| Absorption rate | $\Delta d / \Delta t$ | Discrete derivative of mean distance |

### Plots

**Plot 2.1 — Mean OOD-to-Centroid Distance vs. Steps**
- X-axis: adaptation step $t$
- Y-axis: $\bar{d}(t)$
- Two line styles: frozen centroids (solid) vs. dynamic centroids (dashed)
- One color per $\alpha$
- Confirms hypothesis: should see monotonic decrease under frozen centroids

**Plot 2.2 — Distance Distribution Histograms (Temporal)**
- One panel per selected timestep: $t \in \{0, 5, 10, 20\}$ (or every 5 steps)
- X-axis: $L_2$ distance to nearest centroid
- Overlaid histograms for ID samples (green) and OOD samples (red)
- Shows progressive overlap of OOD distances with ID distances

> **Key insight to capture:** At $t=0$, ID and OOD distance distributions should be separable. At $t=T$, OOD histogram should overlap strongly with ID histogram → detector fails.

**Plot 2.3 — Per-Class Absorption Sanity Check**
- Bar chart: for each CIFAR-10 class $c$, how many OOD samples are absorbed (i.e., $c^* = c$)?
- Tracked at $t=0$ and $t=T$
- Reveals which ID classes act as "attractors" for OOD

**Plot 2.4 — Frozen vs. Dynamic Centroid Comparison**
- Side-by-side $\bar{d}(t)$ curves for frozen and dynamic centroids under same $\alpha$
- Quantifies how much of the absorption is due to OOD feature collapse vs. ID centroid drift

**Plot 2.5 — 2D UMAP / t-SNE Visualization** *(qualitative)*
- Project 512-dim features to 2D using UMAP fit on clean ID data
- Plot ID (by class), OOD points
- Show frames at $t = 0, T/2, T$ (can animate)
- Expected: OOD clusters migrate toward ID clusters over adaptation steps

---

## 6. Experiment 3 — Proposed Fixes

**Goal:** Mitigate prototype absorption while preserving TENT's benefit for ID covariate shift adaptation.

### Fix A — OOD-Weighted Entropy Minimization

**Idea:** Scale each sample's entropy contribution by a weight proportional to the inverse of its OOD score. Samples that look OOD contribute less to the gradient update.

$$
\mathcal{L}_{\text{weighted}} = \sum_{i=1}^{N} w_i \cdot H(p_i), \qquad w_i = 1 - s(x_i) = \max_c p_c(x_i)
$$

Equivalently, confident (low-entropy, high MSP) samples are weighted more heavily; ambiguous/OOD samples are down-weighted.

**Protocol:**
```
Replace TENT loss: L = mean(H(p)) 
With:              L = sum(w_i * H(p_i)) / sum(w_i)
where w_i = max_c softmax(f(x_i))   [MSP as confidence]
```

Compare against vanilla TENT across all $(\alpha, t)$ combinations.

**Metrics:** AUROC at $t=T$, $\bar{d}(T)$ for frozen centroids, ID classification accuracy.

---

### Fix B — Prototype-Anchored Regularization

**Idea:** Add a regularization term to the TENT objective that penalizes deviation of **ID** sample features from their initial centroids. This anchors the ID manifold while still allowing BN adaptation.

$$
\mathcal{L}_{\text{anchor}} = \mathcal{L}_{\text{TENT}} + \lambda \sum_{i : \hat{y}_i \text{ confident}} \|\phi(x_i) - \mu_{\hat{y}_i}^{(0)}\|_2^2
$$

where confidence is determined by a threshold $\tau$ on MSP, and $\hat{y}_i = \arg\max_c p_c(x_i)$.

**Hyperparameters to tune:** $\lambda \in \{0.01, 0.1, 1.0\}$, $\tau \in \{0.7, 0.8, 0.9\}$

---

### Fix C — Selective Sample Filtering (Hard Threshold)

**Idea:** Before each TENT step, filter out samples whose OOD score exceeds a threshold $\tau_{\text{OOD}}$, computing gradients only on likely-ID samples.

$$
\tilde{X} = \{x_i \in X : s(x_i) < \tau_{\text{OOD}}\}
\qquad \mathcal{L} = \frac{1}{|\tilde{X}|} \sum_{x \in \tilde{X}} H(p(x))
$$

This is the hard-threshold analogue of Fix A. The threshold can be calibrated on the clean validation set.

> **Note:** Fixes A–C can be combined. A natural ensemble is Fix A + Fix B (weighted entropy + prototype anchoring).

---

### Evaluation Protocol for Fixes (shared)

For each fix:

| Metric | Condition |
|---|---|
| AUROC at $t = T$ | Per $\alpha \in \{0.9, 0.75, 0.5, 0.25\}$ |
| $\Delta\text{AUROC} = \text{AUROC}(T) - \text{AUROC}(0)$ | Smaller magnitude = better |
| ID Classification Accuracy at $t=T$ | Should remain ≥ TENT baseline |
| $\bar{d}(T)$ OOD-to-centroid distance | Higher = OOD samples stayed separated |

**Summary Plot — Fix Comparison Table / Bar Chart:**
- X-axis: method (No TTA, TENT, Fix A, Fix B, Fix C, A+B)
- Grouped bars per $\alpha$
- Y-axis: AUROC at $t=T$
- Secondary panel: ID accuracy

---

## 7. Optional — Latent Space Vector Field

**Goal:** Characterize the *direction and magnitude* of feature drift induced by one or more TENT steps. This gives a geometric picture of the adaptation dynamics in feature space.

### Representation Residual

For a sample $x$, define the **representation residual** across steps $t$ to $t+k$:

$$
\Delta\phi^{(t \to t+k)}(x) = \phi_{\theta_{t+k}}(x) - \phi_{\theta_t}(x) \in \mathbb{R}^{512}
$$

For $k=1$ this is the per-step drift vector; for larger $k$ it captures cumulative drift.

### Protocol

```
For t in {0, 5, 10, 15}:
    For k in {1, 5}:
        Compute delta_phi(x) for all ID and OOD samples
        Project onto 2D UMAP embedding (fit at t=0)
        Draw quiver/arrow plot showing direction and magnitude of drift
```

### Plots

**Plot 7.1 — Quiver Plot in UMAP Space**
- Background: UMAP scatter of ID clusters and OOD cloud (at $t=0$)
- Arrows: $\Delta\phi^{(t \to t+1)}$ projected onto UMAP basis for each sample
- Color: ID (by class) vs. OOD
- Expected: OOD arrows pointing toward nearest ID cluster; ID arrows pointing inward/stabilizing

**Plot 7.2 — Drift Magnitude vs. Distance to Centroid**
- X-axis: $\|\phi(x) - \mu_{c^*}\|_2$ at step $t$
- Y-axis: $\|\Delta\phi^{(t \to t+1)}\|_2$ (magnitude of one-step drift)
- Separate scatter for ID (blue) and OOD (red)
- Expected: OOD samples farther from centroid drift more → attraction basin behavior

**Plot 7.3 — Cosine Alignment**
- For each OOD sample: compute cosine similarity between $\Delta\phi^{(t \to t+1)}(x)$ and $\mu_{c^*}^{(0)} - \phi_{\theta_t}(x)$ (direction toward nearest centroid)
- A value close to 1 confirms the drift is directly toward the centroid

---

## 8. Task Checklist

### Infrastructure
- [ ] Set up CIFAR-10 dataloader (train/val/test splits)
- [ ] Set up CIFAR-10-C dataloader (all 15 corruption types, severity 5)
- [ ] Set up OOD dataset loader (CIFAR-100 or SVHN)
- [ ] Implement mixed-batch sampler (parameterized by $\alpha$)
- [ ] Train or download ResNet-18 checkpoint (target ≥ 93% clean accuracy)
- [ ] Implement TENT (BN affine parameters only, entropy minimization)
- [ ] Implement centroid computation (frozen + dynamic modes)
- [ ] Implement OOD scorers: MSP, Energy, Mahalanobis

### Experiment 1
- [ ] AUROC logging loop (per step, per $\alpha$, per seed)
- [ ] Plot 1.1: AUROC vs. adaptation steps (per $\alpha$)
- [ ] Plot 1.2: AUROC degradation heatmap (corruption × $\alpha$)
- [ ] Plot 1.3: ID accuracy vs. adaptation steps (sanity check)

### Experiment 2
- [ ] Per-sample $L_2$ distance logging (frozen centroids)
- [ ] Per-sample $L_2$ distance logging (dynamic centroids)
- [ ] Plot 2.1: Mean distance vs. steps (frozen vs. dynamic, per $\alpha$)
- [ ] Plot 2.2: Distance histogram panels at $t \in \{0, 5, 10, 20\}$
- [ ] Plot 2.3: Per-class absorption bar chart
- [ ] Plot 2.4: Frozen vs. dynamic centroid comparison
- [ ] Plot 2.5: UMAP visualization (qualitative, frames at $t=0, T/2, T$)

### Experiment 3
- [ ] Implement Fix A: OOD-weighted entropy minimization
- [ ] Implement Fix B: Prototype-anchored regularization
- [ ] Implement Fix C: Hard OOD filtering before TENT step
- [ ] Hyperparameter sweep for Fix B ($\lambda$, $\tau$)
- [ ] Summary bar chart (all methods × $\alpha$)
- [ ] Ablation: Fix A+B combined

### Optional
- [ ] Compute representation residuals $\Delta\phi^{(t \to t+k)}$
- [ ] UMAP quiver plot (Plot 7.1)
- [ ] Drift magnitude scatter (Plot 7.2)
- [ ] Cosine alignment plot (Plot 7.3)

---

## 9. References

[1] Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). **TENT: Fully Test-Time Adaptation by Entropy Minimization.** *ICLR 2021.*

[2] Hendrycks, D., & Dietterich, T. (2019). **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.** *ICLR 2019.*

[3] Hendrycks, D., & Gidaris, G. (2019). **A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.** *ICLR 2017.*

[4] Liu, W., Wang, X., Owens, J., & Li, Y. (2020). **Energy-based Out-of-distribution Detection.** *NeurIPS 2020.*

[5] Lee, K., Lee, K., Lee, H., & Shin, J. (2018). **A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.** *NeurIPS 2018.* *(Mahalanobis score)*
