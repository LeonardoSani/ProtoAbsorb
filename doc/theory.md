# ProtoAbsorb — Theoretical Background

This is the merged theory document for the ProtoAbsorb → Adaptation–Abstention Conflict
project. It covers: (1) the formal setup and TENT mechanics, (2) the original prototype
absorption hypothesis and what the experiments revealed, (3) the proposed fixes and their
results, (4) the contamination cost decomposition framework, and (5) the critical
observations that motivated the research pivot. The concrete loss/optimizer is implemented
in [`src/proto_absorb/tent.py`](../src/proto_absorb/tent.py); experimental results are in
[`doc/research.md`](./research.md).

This document does **not** prove convergence or claim mathematical guarantees beyond what
the cited references establish. The prototype absorption hypothesis (§3) is now superseded
as the primary narrative — see §3.5 for the research pivot.

---

## 1. Setting

We have a classifier `f_θ : X → ℝ^C` factored as

```
φ_θ : X → ℝ^d        (encoder, here d = 512 for ResNet-18)
W,b : ℝ^d → ℝ^C      (linear head)
f_θ(x) = W φ_θ(x) + b
p_θ(x) = softmax(f_θ(x))   ∈ Δ^{C-1}
```

Training: supervised on a clean ID source distribution `P_ID` (CIFAR-10).
At test time, the data-generating distribution is a mixture

```
X_test  ~  α · P_{ID, shifted}  +  (1 − α) · P_OOD,         α ∈ (0,1).
```

`P_{ID, shifted}` is the ID distribution under covariate shift (here,
CIFAR-10-C at severity 5). `P_OOD` is semantically disjoint (SVHN by
default). The labels are **never** observed at test time by TENT itself;
oracle labels are only used in the *dynamic centroid* control of Exp 2,
strictly to update the prototype bank.

The OOD detector uses MSP

```
s(x) = 1 − max_c p_θ,c(x),
```

with AUROC computed treating ID = negative class, OOD = positive
([_common.py:125-129](../experiments/_common.py#L125-L129)). Higher
`s(x)` ⇒ more OOD-like. Energy `E(x) = −log Σ_c exp f_c(x)` and a
class-conditional Mahalanobis score on `φ` are also defined
([Theory.md §3.4](../Theory.md#34-ood-scoring-baseline)) but are not
plotted in any AUROC curve in the saved results.

---

## 2. TENT (Wang et al., ICLR 2021)

### 2.1 Objective

For a test batch `B = {x_1, …, x_N}` (no labels), TENT minimizes the
mean per-sample softmax entropy:

```
H(p) = − Σ_c p_c log p_c,         (entropy of one prediction)

L_TENT(θ; B) = (1/N) Σ_i H(p_θ(x_i)).
```

### 2.2 Parameter subset

Only the affine BatchNorm parameters `(γ, β)` are updated, all other
weights are frozen, BN running stats are disabled and recomputed from the
current batch
([tent.py:35-68](../src/proto_absorb/tent.py#L35-L68)). With ResNet-18
that is roughly `2 × Σ_layer C_layer ≈ 9.6k` scalar parameters out of
~11M.

### 2.3 Optimizer

SGD with learning rate `1e−3` and momentum `0.9`
([tent.py:164-165](../src/proto_absorb/tent.py#L164-L165)). One step per
batch is the canonical TENT setting; in this repository each batch is
adapted for `T = 20` steps so the within-batch trajectory `t = 0, …, T`
can be inspected.

### 2.4 What TENT does to a single example

Entropy is minimized when `p` is one-hot. The gradient
`∂H(p)/∂logits` for a sample whose argmax is `c*` is, up to a positive
scalar,

```
∂H/∂f_c  =  p_c (log p_c − Σ_k p_k log p_k)
         ≈  p_c − 1[c = c*]    when p is already concentrated at c*,
```

so the update sharpens the predicted distribution toward the current
argmax — *whatever* that argmax is. **TENT has no notion of whether a
sample is ID or OOD.** Every sample's prediction is sharpened toward
its current most likely class.

This is the core of the failure-mode worry encoded in the project: when
an OOD sample is misclassified into some ID class `c̃` (`p_{c̃}` is
already > 1/C by chance), TENT will sharpen toward `c̃`, making the
encoder more confident about that wrong assignment.

---

## 3. Class prototypes and the "absorption" hypothesis

### 3.1 Frozen prototypes

Computed once on clean CIFAR-10 with the pre-trained encoder
([centroids.py:76-130](../src/proto_absorb/centroids.py#L76-L130)):

```
μ_c^{(0)} = (1/|S_c|) Σ_{x ∈ S_c} φ_{θ_0}(x),    c = 1, …, C.
```

A tied (class-conditional) covariance `Σ̂` is also computed
([centroids.py:107-116](../src/proto_absorb/centroids.py#L107-L116)) and
saved alongside the centroids in `centroids.npz`; it is used by the
Mahalanobis scorer.

### 3.2 Dynamic prototypes (Exp 2 control)

After every TENT step, `μ_c` is updated by an EMA on **ID-only** samples
in the current batch
([centroids.py:133-157](../src/proto_absorb/centroids.py#L133-L157)):

```
μ_c^{(t+1)} = 0.9 · μ_c^{(t)} + 0.1 · mean({φ_{θ_t}(x) : (x, c) ∈ B}).
```

`momentum = 0.9` is **slow** (heavily anchored), not a fast EMA — only 10%
of the new direction is taken per step.

### 3.3 Hypothesis (as stated in [Theory.md §2](../Theory.md#2-central-hypothesis))

For an OOD sample `x_OOD` with nearest frozen prototype
`c* = argmin_c ‖φ(x_OOD) − μ_c^{(0)}‖_2`, the claim is

```
d/dt  ‖φ_{θ_t}(x_OOD) − μ_{c*}^{(0)}‖_2  <  0.
```

Read literally, this asserts a *signed monotone contraction* of the OOD
features toward the (fixed) nearest prototype during TENT adaptation. The
claim only references OOD samples; it is silent on ID samples.

### 3.4 Why one might expect this on first principles

- TENT sharpens `p` toward the current argmax. The argmax-class logit
  `f_{c̃}(x) = ⟨w_{c̃}, φ(x)⟩ + b_{c̃}` increases. For a linear head
  trained on the standard cross-entropy objective, `w_c` aligns with the
  per-class feature mean direction (in the limit of class-conditional
  Gaussians with shared covariance). So pushing up `f_{c̃}(x)` and down
  the others is, geometrically, pushing `φ(x)` along `w_{c̃}` away from
  the other class directions — i.e. toward a region the head has decided
  is "class `c̃`-like".
- Because the BN affines `(γ, β)` are shared across the entire spatial
  feature map and across the batch, an update driven by *some* sample's
  `(p, target)` will modify `φ(·)` for *every* sample passing through
  the network in subsequent forward passes, including OOD samples that
  contributed nothing to the gradient at that step (or contributed a
  noisy gradient toward whichever ID class they happened to be
  misclassified as).

The above is heuristic, not a proof. It motivates the experiments rather
than establishing them.

### 3.5 What the data actually supports

[results.md §3](./results.md#3-experiment-2--geometric-mechanism)
quantifies this. Briefly:

- **Frozen prototypes**: mean OOD distance decreases monotonically by
  ~0.9–1.0 over 20 steps at every α. **But mean ID distance also
  decreases by ~0.6–0.7.** The most defensible reading is "global feature
  contraction toward the prototype bank under TENT", not "selective OOD
  absorption."
- **Dynamic prototypes**: mean OOD distance dips for ~5 steps and then
  *grows above* its initial value by t = 20; mean ID distance keeps
  shrinking. Under this control the absorption story inverts.

The hypothesis is therefore *true under one set of assumptions* (frozen
prototypes, ID samples ignored) and *false under another* (prototypes
allowed to drift with the ID distribution, even slowly). A careful claim
would be: *"the gap between mean OOD and mean ID distances to the frozen
prototype bank shrinks under TENT,"* which is the quantity an MSP /
Mahalanobis OOD detector ultimately depends on.

**Research pivot (2026):** These limitations led to dropping the absorption framing as
the primary narrative. The failure mode is real; the geometric explanation is not
uniquely supported. The project reframed around an objective-level conflict: entropy
minimization is class-closing (it sharpens every prediction toward its current argmax),
while OOD safety requires staying uncertain on novel inputs — a direct conflict on OOD
samples regardless of feature geometry. Legacy ProtoAbsorb experiments are preserved in
`results/legacy/`; the current framing is in `doc/research.md §2`.

---

## 4. The OOD score functions

All three are computed in
[src/proto_absorb/scorers.py](../src/proto_absorb/scorers.py) and
[_common.py:91-122](../experiments/_common.py#L91-L122), but only MSP is
plotted in saved AUROC curves.

### 4.1 MSP (the primary metric)

```
s_MSP(x) = 1 − max_c p_θ,c(x).
```

Direct function of the softmax output. Sensitive to `(γ, β)` updates by
construction: TENT minimizing entropy raises `max_c p_c` for *all*
samples it sees, including OOD, which mechanically lowers `s_MSP` on OOD
and reduces the ID/OOD margin in `s_MSP`-space — exactly the failure mode
Exp 1 is built to measure.

### 4.2 Energy

```
E(x) = − log Σ_c exp f_c(x) = − T · logsumexp(f_c / T)        (T = 1).
```

Softmax-free; depends on the absolute logit magnitudes. Less direct
coupling to entropy minimization than MSP, but BN updates change logit
magnitudes too, so it is not invariant.

### 4.3 Mahalanobis on `φ`

```
d_M(x) = min_c √( (φ(x) − μ_c)ᵀ Σ̂⁻¹ (φ(x) − μ_c) ),
```

where `Σ̂` is the tied class-conditional covariance from clean CIFAR-10
training data
([centroids.py:107-123](../src/proto_absorb/centroids.py#L107-L123)).
This is the score most directly tied to the geometric picture in §3:
under frozen `μ_c, Σ̂`, anything reducing `‖φ(x_OOD) − μ_{c*}‖` reduces
`d_M(x_OOD)` and degrades AUROC. Under dynamic `μ_c` this ceases to be
the case (Exp 2 dynamic control).

---

## 5. The proposed fixes

[exp3_fixes.py](../experiments/exp3_fixes.py),
[tent.py:71-137](../src/proto_absorb/tent.py#L71-L137).

The general motivation: vanilla TENT weights every sample's entropy
equally. If "low MSP confidence" correlates with "more likely OOD," then
down-weighting low-MSP samples should reduce the gradient contribution
from OOD inputs. Each fix instantiates this differently.

### 5.1 Fix A — Confidence-weighted entropy

```
w_i = max_c p_θ,c(x_i)         (detached, no backprop through w)

L_A(θ; B) = ( Σ_i w_i H(p_i) ) / ( Σ_i w_i ).
```

A sample whose prediction is uniform contributes essentially zero gradient.
A sample already classified confidently contributes its full
entropy-sharpening gradient. Equivalently: **only confident samples are
allowed to drive adaptation.**

Caveat the experiments expose: an OOD sample that has been confidently
*misclassified* into some ID class also has high `w_i` and is therefore
not down-weighted. Fix A reduces the number of ambiguous-OOD updates but
does nothing about confident OOD misclassifications.

### 5.2 Fix B — Frozen-prototype anchor

For every sample with `max_c p_c > τ` (default `τ = 0.7`), with predicted
class `ĉ`,

```
L_B(θ; B) = L_TENT(θ; B) + λ · (1/|M|) Σ_{i ∈ M} ‖φ(x_i) − μ_{ĉ_i}^{(0)}‖^2,
            M = { i : max_c p_c(x_i) > τ },     default λ = 0.1.
```

Motivation: pin the encoder to the original prototypes for samples it is
confident about, so that the ID manifold cannot drift far. Implicitly,
this is a regularizer on the BN affines toward `θ_0`'s feature geometry
on confidently-classified samples.

Caveat the experiments expose: the gate `conf > 0.7` does **not**
selectively exclude OOD samples — confident OOD misclassifications enter
`M` and get pulled toward whichever prototype `ĉ` they were misassigned
to. The result in §4 of [results.md](./results.md) is that Fix B's
`ood_dist_T` is the *smallest* in the table — the anchor pulls OOD in
*more strongly*. The fix is mistargeted at the failure mode it tries to
prevent.

### 5.3 Fix B (revised) — Contamination-Aware BatchNorm

Ablation A (`doc/research.md §Ablation A`) shows ~89% of the contamination penalty
comes from shared BN statistics, not from the gradient. The mechanism-aligned fix is
therefore to compute BN adaptation statistics using only pseudo-ID samples, while still
evaluating on the full mixed stream:

```
μ̂_γ, σ̂_γ  ←  BatchNorm stats over { x_i ∈ B : s(x_i) < τ }
```

where `s(x_i)` is an OOD score (MSP or entropy) and `τ` selects the pseudo-ID subset of
the current batch. All other adaptation steps (gradient on the full batch) proceed normally.

**Why this is mechanism-aligned:** It targets the dominant contamination pathway (shared
BN statistics, ~89%) rather than the minor gradient pathway (~11%). Unlike Fix A and Fix C,
which filter the gradient while leaving BN intact, this fix goes after the component
Ablation A identifies as primary.

**Status (2026-05-09):** TODO. Minimum experiment: SVHN α=0.5 and DTD α=0.5 under the
paired protocol. Compare No TTA / TENT / ETA / Fix B (contamination-aware BN). Report
ΔAUROC and paired gap.

**Caveat:** At very low α (mostly OOD batches), the pseudo-ID subset may be too small for
reliable statistics estimation; may need a minimum-size floor or fallback to running stats.

**ETA labeling note:** The `eata.py` implementation omits the Fisher regularizer
([`src/proto_absorb/eata.py:17`](../src/proto_absorb/eata.py#L17)) and must be labeled
**ETA** throughout — not EATA. Full EATA requires the Fisher-importance update filter.

---

### 5.4 Fix C — Hard OOD filter

Drop the top `drop_fraction` (default 25%) of the batch by entropy and
take the vanilla TENT step on the remainder:

```
keep = topk_smallest(H(p_i), N − ⌈ρ·N⌉),     ρ = 0.25
L_C(θ; B) = (1/|keep|) Σ_{i ∈ keep} H(p_i).
```

Equivalent to Fix A with a hard 0/1 weight instead of a soft `w_i`.

Caveat: shifted-ID samples can have higher entropy than confident-OOD
samples. The filter therefore drops the wrong tail of the batch in
non-trivial cases. The Exp 3 numbers at α = 0.5 (worse than vanilla TENT)
are consistent with this.

### 5.5 Fix A+B

Combine `L_A` and `λ · (anchor)`. In the saved results this reproduces
Fix B's pathology (the anchor term dominates).

---

## 6. Why "absorption" is a plausible *but* slippery framing

Even before looking at the numbers, the framing has built-in tensions:

1. **The hypothesis as stated talks only about OOD.** A complete causal
   story has to also explain what happens to ID features. If ID features
   collapse equally fast, "absorption" is a misnomer; the right concept is
   "prototype-anchored feature contraction" or simply "BN-driven feature
   collapse under entropy minimization."

2. **The hypothesis is conditional on a frozen prototype bank.** Once the
   prototype bank itself is allowed to track the ID distribution (even
   slowly), the OOD distance can grow. The frozen-prototype assumption is
   not innocent — it bakes in the possibility that any feature drift
   reduces the OOD distance.

3. **The fixes are gated on `max_c p_c`**, which is exactly the quantity
   TENT *makes larger over time*. So a confidence gate that excludes OOD
   at `t = 0` may admit OOD at `t = T`. None of the saved code re-evaluates
   the gate against an external OOD detector — it always uses the current
   model's softmax, which is the very signal entropy minimization is
   sharpening.

These are theoretical observations, not fatal flaws. They are recorded
here so that interpretations of the results are calibrated against what
the setup actually controls for and what it does not.

---

## 7. Contamination Cost Decomposition

### 7.1 The Batch-Size Confound in the id_only Oracle

The `id_only` oracle adapts on the ID sub-batch of size `αB`, while `mixed` adapts on
the full batch of size `B`. For BN-based TTA this introduces a confound: the paired gap
conflates OOD contamination with a batch-size / statistics-quality effect (smaller batches
give noisier BN estimates).

The ViT/IN residual result partially bounds this — LayerNorm and InstanceNorm `id_only`
conditions also adapt on `αB` samples and still show a significant paired gap (+0.013,
+0.014) — but a direct batch-size control is required to isolate contamination cleanly.

### 7.2 Four-Condition Decomposition

Run these four conditions on identical matched batches:

| Condition | BN sees | Loss computed on | Adapts on |
|---|---|---|---|
| `id_subbatch` | ID slice (αB) | ID slice | αB ID samples |
| `id_fullmatch` | ID only (B, replicated) | ID only | B ID samples |
| `mixed_maskedloss` | Full batch (B) | ID slice only | ID gradient, OOD in BN |
| `mixed` | Full batch (B) | Full batch | B mixed samples |

The three additive components of the contamination cost are:

```
Δ_subbatch   = AUROC(id_subbatch) − AUROC(id_fullmatch)
             → sample-count / BN noise artifact (expected ≈ 0)

Δ_BN         = AUROC(id_fullmatch) − AUROC(mixed_maskedloss)
             → BN-statistics contamination (expected dominant, ~89%)

Δ_grad       = AUROC(mixed_maskedloss) − AUROC(mixed)
             → direct OOD-gradient contamination (expected residual, ~11%)
```

If `Δ_subbatch ≈ 0` and `Δ_BN >> Δ_grad`, then the batch-size confound is negligible,
the contamination-penalty claim is confirmed, and the BN-dominates-gradient mechanism
claim is directly quantified.

**Minimum experiment:** SVHN α=0.5 (far-OOD, absolute-failure case) + Places365 α=0.9
(forfeit-without-sign-flip case), n=30 matched batches each.

### 7.3 Implications for Fix Design

The decomposition gives a precise target for any mitigation:

- **Fix A and Fix C** filter the gradient pathway only → address `Δ_grad ≈ 11%`. This
  explains why both produce null or near-null results: they leave ~89% of the mechanism
  intact.
- **Fix B (contamination-aware BN, §5.3)** targets `Δ_BN` directly → the correct
  mechanism-aligned direction, with `id_fullmatch` as the upper bound on achievable gain.
- `id_fullmatch` itself is not deployable (requires knowing which samples are ID), but
  establishes what a perfect BN-level fix could achieve.

---

## 8. References

[1] Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021).
*TENT: Fully Test-Time Adaptation by Entropy Minimization.* ICLR 2021.

[2] Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network
Robustness to Common Corruptions and Perturbations.* ICLR 2019. *(CIFAR-10-C)*

[3] Hendrycks, D., & Gimpel, K. (2017). *A Baseline for Detecting
Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR
2017. *(MSP score)*

[4] Liu, W., Wang, X., Owens, J., & Li, Y. (2020). *Energy-based
Out-of-distribution Detection.* NeurIPS 2020.

[5] Lee, K., Lee, K., Lee, H., & Shin, J. (2018). *A Simple Unified
Framework for Detecting Out-of-Distribution Samples and Adversarial
Attacks.* NeurIPS 2018. *(Mahalanobis score)*

[6] Niu, S., Wu, J., Zhang, Y., Chen, Y., Zheng, S., Zhao, P., & Tan, M. (2022).
*Efficient Test-Time Model Adaptation without Forgetting.* ICML 2022.
*(EATA — full version includes Fisher regularizer; this codebase implements ETA only)*

[7] Li, Y., et al. (2023). *On the Robustness of Open-World Test-Time Training:
Self-Training with Dynamic Prototype Expansion.* ICCV 2023. *(OWTTT)*

[8] Gao, Z., et al. (2024). *Unified Entropy Optimization for Open-Set Test-Time
Adaptation.* CVPR 2024. *(UniEnt/UniEnt+; code: github.com/gaozhengqing/UniEnt)*

[9] Zhao, W., et al. (2026). *ROSETTA* (working title). arXiv:2604.01589.
*(Frames ID/OOD tradeoff in OSTTA; no public code as of 2026-05-09)*
