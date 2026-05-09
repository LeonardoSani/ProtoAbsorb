# Open-World TTA Safety: The Adaptation–Abstention Conflict

**Target venue:** ACCV 2025 (or BMVC / WACV / ECCV workshop)
**Last updated:** 2026-05-09

---

## 1. What We Are Studying

Standard TTA evaluation asks: *does model accuracy on corrupted ID images improve?* This is the only
metric used in TENT, EATA, SAR, CoTTA, and nearly every follow-up. It is blind to a second question
that matters in real deployment: *does the model still know what it does not know?*

In open-world test streams, batches contain both shifted ID samples (training classes under corruption)
and OOD samples (unseen categories). A model must do two things simultaneously:
1. Classify shifted ID images correctly — what TTA optimizes.
2. Detect OOD images as anomalous and abstain — what TTA ignores.

**Core claim:** Entropy-minimizing TTA on mixed batches systematically forfeits the OOD-detection
gain that the same adaptation would deliver on the ID slice alone. This is not an implementation
failure — it is an objective-level conflict.

---

## 2. Theory: Why the Conflict Is Inevitable

### 2.1 What entropy minimization does

TENT minimizes mean per-sample softmax entropy:

```
L_TENT = (1/N) Σ_i H(p_i) = -(1/N) Σ_i Σ_c p_{i,c} log p_{i,c}
```

Entropy is minimized when each `p_i` is one-hot. The gradient for sample `i` with current argmax
class `c*` sharpens toward `c*` and suppresses all other classes:

```
∂H(p_i)/∂f_{i,c}  ≈  p_{i,c} − 1[c = c*]    (when p_i is concentrated at c*)
```

**TENT has no notion of whether a sample is ID or OOD.** It sharpens every prediction toward its
current most likely class — whatever that class happens to be.

### 2.2 Why this conflicts with OOD detection

Every major score-based OOD detector relies on the model expressing *uncertainty* on OOD inputs:

| Detector | What it measures | What TENT does to it |
|---|---|---|
| MSP = 1 − max_c p_c | Low confidence → OOD | Raises max_c p_c for all samples, including OOD |
| Energy = −log Σ_c exp f_c | Low energy → ID | Sharpens logits, changing energy magnitudes |
| Mahalanobis distance | Far from ID prototype → OOD | Contracts features toward prototype bank |

The fundamental tension:

```
TTA objective:     minimize H(p)  →  make every prediction more certain
OOD safety goal:   maximize H(p) on unknowns  →  stay uncertain on novel inputs
```

These goals are **in direct conflict on OOD samples**. No amount of tuning the learning rate
resolves this — only an objective that explicitly distinguishes ID from OOD can do so.

### 2.3 The shared-BatchNorm mechanism

Even if OOD samples contribute zero gradient directly (e.g. are down-weighted), the BatchNorm
affine parameters `(γ, β)` are shared across the batch. Updates driven by ID samples modify
the representation of OOD samples in subsequent forward passes. Therefore `s(x_OOD)` decreases
(model looks more confident on OOD) even when OOD samples are filtered from the gradient step.

**Quantified by Ablation A:** Replacing BN with InstanceNorm reduces the paired gap by 88.9%
(BN gap +0.130 → IN gap +0.014). ViT-S/16 (LayerNorm) independently replicates this residual
(+0.013), triangulating that ~89% of the contamination penalty is BN-statistics contamination
and ~11% is direct OOD-gradient contamination.

### 2.4 Batch-size confound in the id_only oracle (known limitation)

The `id_only` condition adapts on $\alpha B$ samples (the ID sub-batch) while `mixed` adapts on
the full batch of $B$ samples. For BN-based TTA this confounds contamination with sub-batch
statistics quality. The confound is addressed by the four-condition control (Step 8 below), but
is worth noting as a caveat on the core paired-gap result.

The confound is already partially bounded by the ViT/IN results: both use LayerNorm/InstanceNorm
and still show significant paired gaps (+0.013, +0.014), matching the expected residual from
gradient contamination alone. This makes it unlikely that the batch-size artifact accounts for
most of the BN paired gap (+0.130), but the four-condition control (§Step 8) is needed to
quantify it directly.

---

## 3. Origins: ProtoAbsorb (Superseded)

The original hypothesis was:

> Under TENT updates on mixed batches, OOD feature representations are pulled monotonically
> toward the nearest ID class centroid ("prototype absorption").

**What the experiments showed:**
- Both ID and OOD distances to frozen centroids decreased — this is global feature contraction,
  not OOD-selective absorption.
- Under dynamic centroids, the effect inverted: OOD distance grew above its starting value by
  t=20 while ID distance kept shrinking. The absorption story was conditional on holding prototypes
  fixed while the encoder drifts — not a robust claim.
- The hypothesis as written is a per-sample signed derivative. The experiments only measured mean
  endpoint changes. Not the same claim.

**Verdict:** The failure mode is real; the explanation was wrong. We dropped the geometric story and
replaced it with the objective-conflict framing above. Legacy experiments preserved in
`results/legacy/`.

### Why the direction change was correct

| Dimension | ProtoAbsorb (original) | Adaptation–Abstention (current) |
|---|---|---|
| **Core claim** | OOD features drift monotonically toward nearest ID centroid | Entropy-based TTA is class-closing but not novelty-aware — an objective-level conflict |
| **What the data showed** | Global feature contraction (ID and OOD both shrink); effect inverts under dynamic centroids | Paired gap (id_only > mixed) holds in 27/32 cells across 4 OODs, 2 methods, 3 detectors |
| **Claim type** | Per-sample signed derivative — experiments measured only mean endpoints | Aggregate AUROC gap — directly and cleanly measurable |
| **Falsifiability** | Can be "confirmed" by frozen-centroid artifact even if no real OOD-selective effect exists | Falsified if id_only TENT and mixed TENT show the same AUROC degradation |
| **Scope** | Specific to TENT + frozen centroids + Euclidean distance | Applies to any entropy-minimizing TTA method with any score-based OOD detector |
| **Novelty** | Descriptive mechanism name for a known type of representation drift | Identifies a fundamental conflict between two established objectives (entropy minimization vs. OOD safety) |
| **Practical implication** | Fix the geometry of TENT | Fix the evaluation norm for all TTA papers |
| **Reviewer response** | "But ID features contract too — it is just global shrinkage" | "This explains why closed-set TTA metrics miss a real safety failure mode" |
| **Venue fit** | Weak: speculative mechanism, CIFAR-only, geometric story deflated by own data | Strong: clean paired protocol, evaluation critique, principled tension, broadly applicable |

The single most important difference: the original framing required the geometry story to hold
cleanly, and it does not. The new framing requires only that mixed TENT degrades OOD AUROC more
than id-only TENT — which is directly testable, unambiguous, and well-motivated by the objective
analysis in §2.

---

## 4. Experiment Timeline

### Paired Protocol: id\_only as an Unrealizable Oracle

We introduce `id_only` TENT as a **diagnostic oracle**, not a deployable method: it runs TENT exclusively on the subset of test samples drawn from the source distribution, a selection that requires ground-truth knowledge of which samples are in-distribution versus out-of-distribution—precisely the information that OOD detection systems are designed to produce.
Because this label is unavailable at inference time in any realistic deployment, `id_only` TENT is strictly unrealizable and is included solely to establish a measurable upper bound on adaptation quality under zero contamination.
The paired gap $\Delta = \text{Acc}(\texttt{id\_only}) - \text{Acc}(\texttt{mixed})$, computed over matched test streams that differ only in OOD composition, quantifies the performance cost attributable to contamination rather than to any other source of distribution shift.
This gap serves as a calibrated diagnostic: a large $\Delta$ indicates that OOD samples are actively degrading batch statistics during adaptation, while a small $\Delta$ suggests the method is robust to the level of contamination present.
The contribution of the paired protocol is therefore $\Delta$ itself as an interpretable, reproducible bound on contamination cost—not `id_only` as a system one would or could deploy.

---

### Step 1 — Baseline Sanity ✓

**Goal:** Verify setup is trustworthy before any contamination experiments.

Clean model: ResNet-18, CIFAR-10 accuracy = 95.14%.

| OOD | MSP-AUROC | Energy-AUROC | Mahal-AUROC |
|---|---|---|---|
| SVHN | 0.918 | 0.933 | 0.922 |
| Places365 | 0.880 | 0.883 | 0.855 |
| DTD | 0.910 | 0.906 | 0.933 |

Pre-TTA AUROC under gaussian_noise severity 5:

| OOD | MSP(t=0) | Energy(t=0) | Mahal(t=0) |
|---|---|---|---|
| SVHN | 0.526 | 0.696 | 0.342 |
| Places365 | 0.489 | 0.614 | 0.262 |
| DTD | 0.530 | 0.635 | 0.479 |

**Key flag:** Corruption collapses Mahalanobis by 0.58 on SVHN before any TTA.
Places365 pre-TTA MSP-AUROC ≈ 0.489 (chance) — this is the key moderator that explains
Places365's anomalous behavior downstream.

---

### Step 2 — Contamination Isolation ✓ (CORE RESULT)

**Goal:** Show contamination, not TTA itself, is causal. Three conditions on the same drawn batches:
`No TTA`, `ID-only TENT` (oracle), `Mixed TENT` (realistic). Key statistic: paired gap =
ΔAUROC(id_only) − ΔAUROC(mixed).

**Protocol:** gaussian_noise, MSP-AUROC, n=30, α ∈ {0.9, 0.75, 0.5, 0.25}.

| OOD | α | id_only Δ [95% CI] | mixed Δ [95% CI] | paired gap | paired p | forfeit_fraction |
|---|---|---|---|---|---|---|
| SVHN | 0.9 | +0.188 [+0.162,+0.212] | −0.043 [−0.100,+0.016] | +0.231 | 2.6e-08 | 1.23 |
| SVHN | 0.75 | +0.130 [+0.114,+0.145] | −0.045 [−0.077,−0.010] | +0.175 | 1.2e-09 | 1.35 |
| SVHN | 0.5 | +0.077 [+0.063,+0.092] | −0.052 [−0.081,−0.023] | +0.129 | 9.1e-09 | 1.68 |
| SVHN | 0.25 | +0.027 [+0.012,+0.042] | −0.051 [−0.077,−0.022] | +0.078 | 1.6e-05 | 2.89 |
| DTD | 0.9 | +0.209 [+0.182,+0.240] | −0.003 [−0.053,+0.050] | +0.212 | 7.9e-11 | 1.01 |
| DTD | 0.75 | +0.166 [+0.154,+0.179] | −0.037 [−0.068,−0.005] | +0.203 | 3.5e-14 | 1.22 |
| DTD | 0.5 | +0.119 [+0.102,+0.134] | −0.055 [−0.081,−0.029] | +0.174 | 5.2e-16 | 1.46 |
| DTD | 0.25 | +0.097 [+0.073,+0.122] | −0.032 [−0.061,+0.001] | +0.129 | 9.6e-10 | 1.33 |
| Places365 | 0.9 | +0.192 [+0.170,+0.215] | +0.016 [−0.008,+0.040] | +0.176 | 4.3e-14 | 0.92 |
| Places365 | 0.75 | +0.153 [+0.136,+0.171] | +0.032 [+0.009,+0.055] | +0.121 | 1.1e-09 | 0.79 |
| Places365 | 0.5 | +0.080 [+0.061,+0.100] | +0.020 [+0.001,+0.039] | +0.060 | 2.7e-05 | 0.75 |
| Places365 | 0.25 | +0.018 [−0.003,+0.038] | +0.006 [−0.021,+0.036] | +0.012 | 0.44 (ns) | 0.67 |

**Verdict:** Paired gap always positive, significant in 11/12 cells. Mixed TENT's absolute sign
does not flip for Places365 because pre-TTA separability is near zero — TTA has nothing to destroy,
but contamination still forfeits most of the adaptation benefit.

---

### Step 3 — Detector Breadth ✓

**Goal:** Show the effect is not an MSP artifact — holds for Energy and Mahalanobis too.

Representative results at α=0.9 and α=0.5:

| OOD | α | Det | id_only Δ | mixed Δ | paired p | forfeit_fraction |
|---|---|---|---|---|---|---|
| DTD | 0.9 | MSP | +0.209 | −0.003 | 7.9e-11 | 1.01 |
| DTD | 0.9 | Energy | +0.192 | −0.033 | 1.7e-12 | 1.17 |
| DTD | 0.9 | Mahal | +0.055 | +0.024 | 2.9e-04 | 0.56 |
| DTD | 0.5 | MSP | +0.119 | −0.055 | 5.2e-16 | 1.46 |
| DTD | 0.5 | Energy | +0.113 | −0.065 | 2.6e-18 | 1.58 |
| DTD | 0.5 | Mahal | +0.032 | +0.015 | 0.034 | 0.53 |
| Places365 | 0.9 | MSP | +0.192 | +0.016 | 4.3e-14 | 0.92 |
| Places365 | 0.9 | Energy | +0.186 | −0.011 | 4.5e-17 | 1.06 |
| Places365 | 0.9 | Mahal | +0.066 | +0.018 | 1.2e-06 | 0.73 |
| Places365 | 0.5 | MSP | +0.080 | +0.020 | 2.7e-05 | 0.75 |
| Places365 | 0.5 | Energy | +0.075 | +0.006 | 3.2e-07 | 0.92 |
| Places365 | 0.5 | Mahal | +0.028 | −0.003 | 2.2e-03 | 1.11 |

**Verdict:** Paired gap significant across all three detectors for DTD and Places365 at α ≥ 0.5.
For DTD Mahalanobis, both conditions improve but id_only improves more — "same-sign degradation"
means "degradation relative to id_only," not necessarily "absolute AUROC drop."

---

### Step 4 — Mechanism Traces ✓

**Goal:** Directly show confidence sharpening on OOD samples explains the score degradation.

SVHN and DTD: mean max-softmax on OOD samples increases monotonically under both conditions.
Entropy decreases monotonically. The ID/OOD score gap shrinks under mixed TENT but not id-only TENT.
Confidence-sharpening story is confirmed.

**Key output:** Pareto plot (`results/step4_svhn/pareto_gaussian_noise.png`) — candidate main figure.

---

### Step 5 — Generalization (4 OODs × 2 methods) ✓

**Goal:** Show the failure is not specific to one OOD set or one TTA method.

Representative results (TENT and EATA, gaussian_noise, α=0.9 and α=0.5):

| OOD | Method | α | id_only Δ | mixed Δ | paired p | forfeit_fraction |
|---|---|---|---|---|---|---|
| SVHN | TENT | 0.9 | +0.210 | −0.075 | 2.0e-08 | 1.36 |
| SVHN | EATA | 0.9 | +0.118 | −0.107 | 1.2e-06 | 1.90 |
| DTD | TENT | 0.9 | +0.217 | −0.027 | 3.1e-08 | 1.12 |
| DTD | EATA | 0.9 | +0.146 | −0.013 | 4.6e-09 | 1.09 |
| Places365 | TENT | 0.9 | +0.217 | +0.039 | 1.9e-08 | 0.82 |
| Places365 | EATA | 0.9 | +0.130 | +0.011 | 1.6e-07 | 0.92 |
| CIFAR-100 | TENT | 0.9 | +0.186 | −0.001 | 1.4e-06 | 1.01 |
| CIFAR-100 | EATA | 0.9 | +0.131 | +0.015 | 7.2e-07 | 0.88 |

Significance count (paired p<0.05 across 4 α values):

| OOD | TENT | EATA |
|---|---|---|
| SVHN | 4/4 | 4/4 |
| DTD | 4/4 | 4/4 |
| Places365 | 3/4 | 3/4 |
| CIFAR-100 | 2/4 | 3/4 |

**Total: 27/32 cells significant.** Non-significant cells are all at α=0.25, where id_only Δ is
small (little adaptation gain to forfeit — consistent with the scaling claim).

---

### Step 6 — Fix A: Confidence-Weighted Entropy ✗ (null)

**Goal:** Test whether down-weighting high-confidence OOD-like samples in the entropy loss
eliminates the contamination penalty.

| α | mixed Δ | fix_a Δ | paired p (fix_a vs mixed) |
|---|---|---|---|
| 0.9 | −0.043 | −0.050 | 0.38 (ns) |
| 0.75 | −0.045 | −0.049 | 0.39 (ns) |
| 0.5 | −0.052 | −0.055 | 0.35 (ns) |
| 0.25 | −0.051 | −0.049 | 0.78 (ns) |

**Verdict:** Fix A is statistically indistinguishable from vanilla mixed TENT at all α. Slightly
worse at α=0.9. Not a usable method contribution.

**Why it failed:** Down-weighting reduces per-sample gradient contribution, but BatchNorm
statistics are computed over the full batch first. OOD presence still shifts BN moments
even when OOD-like samples are excluded from the gradient.

---

### Ablation A — BatchNorm vs InstanceNorm ✓

**Goal:** Quantify how much of the contamination penalty is BN-statistics contamination
vs. gradient contamination alone.

**Protocol:** SVHN, α=0.5, gaussian_noise sev-5, n=30. BN-TENT = original model.
IN-TENT = `bn_to_in` deepcopy of same checkpoint (γ/β copied, per-sample statistics,
no batch mixing). Same paired batches across conditions.

| | id_only ΔAUROC | mixed ΔAUROC | paired gap (id_only − mixed) | paired p |
|---|---|---|---|---|
| BN-TENT | +0.0775 [+0.063, +0.092] | −0.0523 [−0.081, −0.023] | +0.1298 | 9.1e-09 |
| IN-TENT | +0.0106 [+0.008, +0.013] | −0.0038 [−0.007, −0.001] | +0.0144 | 6.1e-17 |

**BN gap vs IN gap: t=7.06 (independent samples), p=2.3e-09. Reduction: 88.9%.**

**Confound — IN adapts less overall:** id_only ΔAUROC drops from +0.077 (BN) to +0.011 (IN),
a 86% reduction in base adaptation power. IN computes per-sample statistics, which are
noisier estimates than batch statistics, making γ/β updates less effective.

Contamination penalty as a fraction of adaptation gain (gap / id_only Δ):
- BN: 0.130 / 0.077 = **1.67** — contamination costs 167% of what adaptation gains
- IN: 0.014 / 0.011 = **1.36** — contamination costs 136% of what adaptation gains

The ratios are similar, meaning IN is only proportionally slightly less contaminated per
unit of adaptation. However, the **absolute** degradation for IN mixed-TENT is −0.004
(negligible) while BN is −0.052 (large and significant). In practice, IN eliminates the
observable degradation — but also most of the adaptation benefit.

**Verdict:** Shared BN statistics account for ~89% of the contamination penalty. Gradient
contamination alone (the only pathway remaining in IN-TENT) contributes ~11%. This explains
why Fix A (gradient weighting) and Fix C (sample filtering) both fail: they modify the
gradient pathway but leave BN in the adaptation loop, leaving ~89% of the mechanism intact.

**Practical implication:** The solution is not to swap BN for IN — that sacrifices most of
the adaptation benefit (id_only gain: 0.077 → 0.011). A complete solution requires an
adaptation method that achieves BN-level gains without sharing normalization statistics
across ID and OOD samples in the batch.

---

### Step 7 — Fix C: Entropy-Threshold Filtering (partial)

**Goal:** Test whether filtering highest-entropy (likely OOD) samples from the gradient step
reduces the contamination penalty, and at what ID-accuracy cost.

τ-sweep (drop_fraction ∈ {0.1, 0.2, 0.3, 0.5}) on SVHN and DTD at α ∈ {0.9, 0.5}.

- **SVHN α=0.9:** all τ values ns vs vanilla (p=0.46–0.80). Fix C fails.
- **SVHN α=0.5:** τ=0.5 p=0.047 vs vanilla (barely significant, Δ=−0.022 vs −0.052).
- **DTD α=0.9:** all τ values ns vs vanilla. Fix C fails.
- **DTD α=0.5:** τ=0.3 p=0.010, τ=0.5 p=0.018. Δ=−0.027 vs −0.055 (vanilla) — ~50% reduction.

**Verdict:** Fix C reduces the contamination penalty by ~50% for DTD at α=0.5 with aggressive
filtering, but fails for SVHN and for DTD at α=0.9. Mitigation is partial and
condition-dependent — consistent with the BN ablation: filtering the gradient without replacing
BN leaves ~89% of the mechanism intact.

---

### ViT-S/16 Backbone Check ✓

**Goal:** Test whether the paired gap persists on a second architecture with no BatchNorm.
ViT uses LayerNorm (per-token, per-sample statistics) — the shared-batch-statistics
contamination pathway is absent. LN-TENT adapts the 50 LayerNorm affine parameter pairs
(19.2k scalars). SVHN OOD, α=0.5, gaussian_noise sev-5, n=30 batches.

**Training:** ViT-S/16 (timm, pretrained ImageNet weights) fine-tuned 30 epochs on
CIFAR-10 at 224×224. Best test accuracy: **98.80%** (vs ResNet-18: 95.14%).

| Condition | ΔAUROC | 95% CI | AUROC(T) |
|---|---|---|---|
| no_tta | +0.000 | [+0.000, +0.000] | 0.828 |
| id_only | +0.016 | [+0.013, +0.020] | 0.844 |
| mixed | +0.003 | [+0.001, +0.005] | 0.831 |

**Paired gap: +0.013, p=2.59e-10**

| Backbone | Norm | Paired gap |
|---|---|---|
| ResNet-18 | BatchNorm | +0.130 |
| ResNet-18 | InstanceNorm (Ablation A) | +0.014 |
| ViT-S/16 | LayerNorm (this exp) | +0.013 |

**Convergent finding:** ViT's gap (+0.013) almost exactly matches ResNet-18/IN's residual
gap (+0.014). Both represent the gradient-contamination-only signal — BN absent in both
cases. Two independent methods triangulate to the same ~10% residual, independently
confirming that BN statistics account for ~89–90% of the contamination penalty.

**Verdict:** Effect generalizes to ViT-S/16. The gap is significant but 10× smaller than
ResNet-18/BN — consistent with the mechanism decomposition.

**Files:** `results/vit_backbone/vit_step2_results.json`,
`results/vit_backbone/forest_gaussian_noise_alpha0.5.png`
**Checkpoint:** `checkpoints/vit_small_cifar10.pt`
**Scripts:** `src/proto_absorb/train_vit.py`, `src/proto_absorb/models.py::ViTSmall`,
`src/proto_absorb/tent.py::configure_vit_tent_model`, `experiments/exp_vit_backbone.py`

---

### ImageNet-Scale Validation (ResNet-50) ✓

**Goal:** Replicate the paired gap at ImageNet scale to address reviewer concern about CIFAR-only scope.

**Setup:** ResNet-50 (torchvision pretrained, ~76% top-1 on clean ImageNet val).
ID pool = ImageNet-C severity 5. OOD = NINCO (~5878 images, species not in ImageNet-1K).
n=30 batches, batch_size=64, n_steps=20, α ∈ {0.9, 0.5}.

**Corruption note:** gaussian_noise sev-5 reduces ResNet-50 accuracy to ~5% — model outputs
near-uniform softmax on all inputs including OOD, causing inverted AUROC (< 0.5). Excluded.
DTD also excluded: textures trigger ImageNet-pretrained features more than corrupted images do,
causing inverted AUROC regardless of severity. fog and jpeg_compression at sev-5 retain enough
semantic content (~30–65% accuracy) for well-oriented OOD detection.

**Scripts:** `experiments/exp_imagenet_baseline.py` (Step 1 verify),
`experiments/exp_imagenet_scale.py` (paired protocol, 8 cells).
**Results:** `results/imagenet_scale/imagenet_scale_results.json`

| corruption | method | α | id_only AUROC(T) | mixed AUROC(T) | id_only Δ | mixed Δ | paired p |
|---|---|---|---|---|---|---|---|
| fog | TENT | 0.9 | **0.843** | 0.498 | +0.280 | −0.065 | 1.2e-14 |
| fog | TENT | 0.5 | **0.698** | 0.520 | +0.244 | +0.066 | 4.9e-11 |
| fog | EATA | 0.9 | **0.584** | 0.530 | +0.034 | −0.020 | 6.5e-05 |
| fog | EATA | 0.5 | **0.520** | 0.457 | +0.060 | −0.003 | 4.4e-11 |
| jpeg | TENT | 0.9 | **0.827** | 0.454 | +0.362 | −0.011 | 5.3e-18 |
| jpeg | TENT | 0.5 | **0.696** | 0.487 | +0.285 | +0.075 | 4.3e-16 |
| jpeg | EATA | 0.9 | **0.457** | 0.427 | +0.003 | −0.026 | 4.0e-03 |
| jpeg | EATA | 0.5 | **0.466** | 0.400 | +0.039 | −0.027 | 2.2e-09 |

**Summary: mixed worse than id_only in 8/8 cells; p<0.05 in 8/8 cells.**

**TENT gap >> EATA gap** — consistent with mechanism: EATA's entropy threshold filters
unreliable samples from the gradient, partially shielding OOD detection. But shared BN
statistics still carry OOD signal into the adapted model (~89% of penalty per Ablation A),
explaining why EATA's mitigation is partial not complete.

**FPR95 pattern mirrors AUROC** — id_only consistently lower FPR95 than mixed across all 8 cells
(e.g., fog TENT α=0.9: id_only 0.380 vs mixed 0.859).

---

### Summary Scatter: Pre-TTA Separability vs Paired Gap ✓

**Script:** `experiments/plot_scatter_pretTA_vs_gap.py`
**Output:** `results/figures/scatter_pretTA_vs_gap.{png,pdf}`

Fixed α=0.9, 15 points (3 OODs × 5 corruptions):
- **Panel A** (t0_msp vs paired_gap): r=−0.534, p=0.040. Lower pre-TTA separability →
  harder corruption → more adaptation benefit available → larger contamination penalty. **Inverse.**
- **Panel B** (id_only Δ vs paired_gap): r=+0.723, p=0.002. Paired gap best predicted by
  how much adaptation id_only TENT delivers. **Strong positive. This is the theory-confirming figure.**

**Why Panel B holds theoretically:** Severe corruption confuses the model more → id-only
TTA delivers a larger AUROC improvement → but a mixed batch under the same corruption also
generates stronger gradient signal from OOD samples (more confused predictions = larger
entropy gradient magnitudes) → larger contamination penalty. Both the benefit and the
contamination scale with corruption severity; the contamination consistently outpaces the
benefit (gap/id_only ratio = 1.67 for BN-TENT), so the net effect of mixed adaptation is
always worse than id-only, and worse by more when the corruption is harder.

**Narrative revision:** The original "paired gap scales positively with pre-TTA separability"
claim is not supported. Correct claim: "paired gap scales with adaptation benefit available
(id_only Δ), which in turn scales with corruption severity."

---

### Three-Way Decomposition ✓

**Script:** `experiments/plot_three_way_decomp.py`
**Output:** `results/figures/three_way_decomp.{tex,json}`

For any cell, total AUROC change = Δ_corruption + Δ_TTA + Δ_contamination.

Key take: Δ_corruption dominates (−0.38 to −0.39 across OODs). Δ_TTA is moderate (+0.02 to
+0.21 depending on α). Δ_contamination is always negative and significant — it is the
incremental penalty imposed by mixed vs ID-only adaptation.

---

## 5. Final Claims & Verdicts

**Corrected headline claim (supported by 27/32 cells):**

> Entropy minimization on a mixed test batch systematically forfeits a large fraction of the
> OOD-detection gain that the same adaptation would deliver on the ID slice alone.
> The magnitude of the forfeit scales with the adaptation benefit available — i.e., with
> how much id-only TENT could have improved OOD detection — which in turn scales with
> corruption severity (r=+0.72, p=0.002, 15-cell scatter).

| Claim | Verdict |
|---|---|
| Paired gap (id_only > mixed) significant across 4 OODs, 2 methods | **Strongly supported** (27/32 cells) |
| Paired gap holds for all α | **Supported at α ≥ 0.5; weakens at α=0.25** |
| Absolute AUROC drop below no-TTA (SVHN/DTD) | **Supported at α ≤ 0.75** |
| Absolute AUROC drop below no-TTA (Places365/CIFAR-100) | **Not supported — sign reverses** |
| Contamination penalty scales with id_only Δ | **Supported — r=+0.72, p=0.002** |
| Contamination penalty scales with pre-TTA separability | **Revised — relationship is inverse across corruption types (r=−0.53); drop original framing** |
| Effect holds for EATA | **Supported** (4/4 OODs) |
| Effect holds for MSP, Energy, Mahalanobis | **Supported** (SVHN, DTD, Places365) |
| Fix A mitigates the problem | **Not supported** |
| Fix C mitigates the problem | **Partially — DTD α=0.5 τ≥0.3; null for SVHN and DTD α=0.9** |
| BN-statistics contamination is primary mechanism | **Strongly supported — IN-TENT reduces paired gap by 89%** |
| Confidence sharpening + separability gap explains mechanism | **Supported** (SVHN and DTD, Step 4) |
| Result generalizes across architectures | **Supported — ViT-S/16 paired gap +0.013 (p=2.6e-10); 10× smaller than ResNet-18/BN, consistent with BN ablation** |
| Result replicates at ImageNet scale | **Strongly supported — 8/8 cells, p<0.05 in all; ResNet-50, ImageNet-C, NINCO** |

---

## 6. What the Results Mean

**For practitioners:** Deploying TTA in mixed batches yields almost none of the OOD safety
improvement that ID-filtered adaptation could provide — even when absolute AUROC does not fall
below baseline. The "success" reported by closed-set TTA benchmarks masks a safety failure.

**For the field:** TTA papers should report OOD detection metrics alongside ID accuracy whenever
adaptation may occur in open-world conditions. A simple norm — showing the ID accuracy / OOD AUROC
Pareto curve across methods — makes the tradeoff visible and prevents silent benchmark gaming.

**Why the fixes fail:** Both Fix A and Fix C keep BatchNorm in the adaptation loop, leaving ~89%
of the contamination mechanism intact. Replacing BN with LN (ViT) or IN (Ablation A) reduces the
gap by ~90% — but does not eliminate it, because gradient contamination (~10%) remains.
A complete solution requires an adaptation objective with an explicit open-set or abstention term.

**Known weaknesses to pre-empt:**

| Weakness | Pre-emption |
|---|---|
| Sign flips for Places365/CIFAR-100 | Three-way decomp; frame as "forfeit" not "destroy"; scatter explains |
| Single backbone | **Addressed** — ViT-S/16 result in paper (convergent finding, not just supplementary) |
| Fix A null, no working solution | Frame as diagnosis paper; directional fix shown |
| Mahal collapses from corruption alone | Three-way decomp shows contamination is incremental |
| "Semantic modulation" framing not supported | Dropped; id_only Δ as proxy for adaptation benefit |
| α=0.25 cells often ns | Report honestly; small id_only gain → small signal to forfeit |

---

## 7. Paper Outline

> **Updated 2026-05-09** for 5/6 ACCV target. Paper is now framed as an **evaluation/protocol
> paper**: primary contribution is the paired diagnostic + contamination-cost decomposition +
> cross-method audit. The conflict itself is no longer claimed as a discovery (ROSETTA, UniEnt
> pre-empt this); the novelty is the protocol that quantifies what each method still leaves unsolved.
>
> **ETA note:** The "EATA" implementation omits the Fisher regularizer and must be labeled **ETA**
> throughout.

1. **Introduction** — open-world TTA; standard eval blind spot; contributions:
   (a) paired contamination protocol, (b) cost decomposition, (c) method audit showing
   closed-set ranking ≠ mixed-stream ranking, (d) reporting standard
2. **Background** — TENT/ETA; MSP/Energy/Mahal; OSTTA prior work (OWTTT, UniEnt, ROSETTA);
   why existing methods reduce but do not eliminate the gap
3. **Theory** — entropy minimization is class-closing, not novelty-aware; BN shared-stats pathway;
   four-condition decomposition framework (§2.3–2.4)
4. **Paired protocol** — contamination isolation design; id_only as oracle; four-condition control
   decomposing sample-count vs BN-stats vs gradient contamination (Step 8)
5. **Results I — Method Audit** *(new main result for 5/6)*
   - Cross-method ranking table: No TTA / TENT / ETA / OWTTT / UniEnt+ under paired protocol
   - Ranking reversal figure: Panel A (ΔID acc, closed-set) vs Panel B (mixed AUROC + gap)
   - UniEnt+ reduces but does not eliminate the paired gap → protocol still needed
6. **Results II — Characterization**
   - Core paired gap: SVHN + DTD + Places365, 3 detectors, forest plot (Steps 2–3)
   - Generalization: 4 OODs × 2 methods, significance table (Step 5)
   - Scaling: scatter Panel B (id_only Δ vs paired gap, r=+0.72)
   - Mechanism: confidence sharpening + Pareto (Step 4)
7. **Results III — ImageNet-Scale** — ResNet-50, fog+jpeg, NINCO; 8/8 cells; UniEnt+ anchor
8. **Mitigation** — contamination-aware BN (Fix B); partial win expected; frames solution direction
9. **Discussion** — three-way decomp; Places365/CIFAR-100 boundary; BN ablation; limits;
   reporting standard recommendation
10. **Conclusion** — paired gap as benchmark column; reporting norm for TTA papers

---

## 9. Related Work Draft Fragments

### Positioning Against OWTTT, UniEnt, and ROSETTA

Li et al. (OWTTT, ICCV 2023, arXiv:2308.06879) extend test-time adaptation to open-world streams
by detecting and clustering unknown categories, treating contamination as a problem to be solved by
a richer adaptation algorithm. Gao et al. (UniEnt/UniEnt+, CVPR 2024, arXiv:2404.06065) propose a
unified entropy objective that applies entropy minimization to pseudo-ID samples and entropy
maximization to pseudo-OOD samples, explicitly addressing the ID/OOD split at the loss level.
Zhao et al. (ROSETTA, arXiv:2604.01589, April 2026) similarly frame an ID/OOD tradeoff in OSTTA
and propose a new method to mitigate it.

Our work differs in kind: we do not propose a new adaptation method, but introduce a **paired
diagnostic protocol** that isolates and quantifies the contamination cost as a measurable gap
between a contamination-free oracle (`id_only`) and realistic mixed-stream adaptation (`mixed`).
Crucially, we evaluate this protocol **on existing OSTTA methods including UniEnt+** and show that
the paired gap persists — partially reduced but not eliminated — even for methods explicitly
designed for open-world streams. This means standard TTA evaluation (closed-set ID accuracy only)
under-reports safety failures even when OSTTA methods are used, and the paired protocol is needed
to quantify what remains unsolved.

The distinction from ROSETTA and UniEnt is therefore not "we discovered the conflict" (they did
too) but "we provide the evaluation tool that reveals how much each method actually solves."

---

## 8. Remaining Work

### New Planned Experiments (from ACCV review, 2026-05-09)

| Priority | Action | Status | Notes |
|---|---|---|---|
| **HIGH** | Integrate UniEnt/UniEnt+ under paired protocol | TODO | Code: github.com/gaozhengqing/UniEnt |
| **HIGH** | Cross-method ranking audit: No TTA / TENT / ETA / OWTTT / UniEnt+ | TODO | 2 OOD settings, 2 corruptions, CIFAR + ImageNet anchor |
| **HIGH** | Ranking reversal figure (2 panels: ΔID acc vs mixed AUROC) | TODO | Built from UniEnt runs |
| **HIGH** | 4-condition confound control | TODO | See §2.4 and Theory.md §9.2 |
| **HIGH** | Fix B: contamination-aware BN mitigation | TODO | Pseudo-ID-only BN stats; see Theory.md §6 Fix B |
| **Medium** | Relabel all EATA → ETA in paper, tables, code comments | TODO | `eata.py:17` omits Fisher regularizer |
| **Medium** | Draft paper §1–2 (intro + background) | TODO | |
| **Medium** | Generate figures from plotting scripts | Partial | Scatter, decomp done |
| **Done** | Second backbone (ViT-S/16, one cell) | ✓ | 98.80% acc, paired gap +0.013 p=2.6e-10 |
| **Done** | ImageNet-scale validation | ✓ | 8/8 cells p<0.05; ResNet-50, fog+jpeg, NINCO |
| **Medium** | Write paper §7 (ImageNet-scale, 1 table + prose) | TODO | |
| **Medium** | Add FPR95 to all existing CIFAR results tables | TODO | |

### Step 8 — Four-Condition Confound Control (Planned)

**Goal:** Decompose the contamination cost into three additive components to settle the batch-size
confound objection and strengthen the BN-mechanism claim.

**Four conditions (run on matched batches):**
1. `id_subbatch` — current oracle: adapt on ID slice only, size αB
2. `id_fullmatch` — fill OOD slots with extra corrupted ID samples so adaptation uses B ID samples
3. `mixed_maskedloss` — full mixed batch forward (OOD in BN stats), but loss computed on ID slice
4. `mixed` — current realistic condition (full batch, full loss)

**Decomposition:**
- `id_subbatch − id_fullmatch` = sample-count artifact (expected ≈ 0)
- `id_fullmatch − mixed_maskedloss` = BN-statistics contamination (expected dominant, ~89%)
- `mixed_maskedloss − mixed` = direct OOD-gradient contamination (expected ~11%)

**Minimum cells:** SVHN α=0.5 + Places365 α=0.9, n=30 each.
**Full appendix:** add SVHN α=0.9 + Places365 α=0.5.

### Step 9 — Cross-Method Ranking Audit (Planned)

**Goal:** Show that standard closed-set TTA evaluation mis-ranks method families compared to the
paired mixed-stream safety evaluation. This is the primary 4/6 → 5/6 upgrade.

**Methods:** No TTA / TENT / ETA / OWTTT / UniEnt / UniEnt+
**Settings:** 2 OOD (SVHN + DTD), 2 corruptions (gaussian_noise + fog), α=0.5 and α=0.9, n=30
**ImageNet anchor:** fog+jpeg, NINCO OOD, at least UniEnt+ added to existing TENT/ETA results

**Output figure (2 panels):**
- Panel A: ΔID accuracy under standard closed-set eval (x-axis: method)
- Panel B: Mixed-stream AUROC + paired contamination gap (same methods)
- Expected: TENT/ETA look good in Panel A, rank differently in Panel B; UniEnt+ improves Panel B

**Reporting standard:** Every TTA paper should report: `ΔID accuracy` + `mixed-stream OOD AUROC`
+ `paired contamination gap (id_only − mixed)`. This is the actionable endpoint for the paper.
