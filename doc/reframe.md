# Reframe: Adapting to Shift, Forgetting to Abstain

This document captures the new research direction following an external critical review of the
original ProtoAbsorb hypothesis. It supersedes the "prototype absorption" framing in
[../Theory.md](../Theory.md) and should be treated as the canonical narrative going forward.

---

## 1. Why the Previous Framing Was Weak

The original hypothesis stated:

> Under TENT updates on mixed batches, OOD feature representations are pulled
> monotonically toward the nearest ID class centroid.

The experiments partially supported this — but not in the way the hypothesis claimed:

- **Both ID and OOD distances to frozen centroids decreased** under TENT. This is global
  feature contraction, not OOD-selective absorption.
- **Under dynamic centroids, the effect inverted**: OOD distance grew above its starting
  value by t=20 while ID distance kept shrinking. The "absorption" story is conditional
  on holding prototypes fixed while the encoder drifts — that is not a robust claim.
- The hypothesis as written is a per-sample signed derivative. The experiments only measured
  mean endpoint changes. Those are not the same claim.
- "Prototype absorption" is a catchy name for a mechanism the data does not cleanly support.

The failure mode itself — TENT degrading OOD detection on mixed batches — **is real**. The
wrong part was the explanation.

---

## 2. The Problem We Are Actually Solving

Standard TTA evaluation asks one question: *does the model's accuracy on corrupted ID
images improve?* This is the metric used in TENT, EATA, SAR, CoTTA, and nearly every
follow-up paper.

This evaluation is blind to a second question that matters in real deployment: *does the
model still know what it does not know?*

In open-world test streams — which are the realistic deployment setting — test batches
contain both shifted ID samples (images from training classes under corruption) and OOD
samples (images from categories the model was never trained on). A model must do two things
simultaneously:

1. Classify shifted ID images correctly (what TTA optimizes for).
2. Detect OOD images as anomalous and abstain or flag them (what TTA ignores).

The problem this paper solves is: **current TTA benchmarks are closed-set and therefore
reward methods that can silently fail in open-world conditions**. A method can score well
on every published TTA leaderboard while systematically destroying its own OOD detector.

---

## 3. The Theory: An Objective Conflict

### 3.1 What entropy minimization does

TENT minimizes the mean per-sample softmax entropy over a test batch:

```
L_TENT = (1/N) Σ_i H(p_i) = -(1/N) Σ_i Σ_c p_{i,c} log p_{i,c}
```

Entropy is minimized when each `p_i` is one-hot. The gradient for sample `i` with current
argmax class `c*` sharpens the prediction toward `c*` and suppresses all other classes:

```
∂H(p_i)/∂f_{i,c}  ≈  p_{i,c} − 1[c = c*]    (when p_i is concentrated at c*)
```

**TENT has no notion of whether a sample is ID or OOD.** It sharpens every prediction
toward its current most likely class — whatever that class happens to be.

### 3.2 Why this conflicts with OOD detection

Every major score-based OOD detector relies on the model expressing *uncertainty* on
out-of-distribution inputs:

| Detector | What it measures | What TENT does to it |
|---|---|---|
| MSP = 1 − max_c p_c | Low confidence → OOD | TENT raises max_c p_c for all samples, including OOD |
| Energy = −log Σ_c exp f_c | Low energy → ID | TENT sharpens logits, changing energy magnitudes |
| Mahalanobis distance | Far from ID prototype → OOD | TENT contracts features toward the prototype bank |

In all three cases, the same update that helps the model commit to a class on corrupted ID
images also makes OOD images look more in-distribution. The objective is:

> **entropy minimization is class-closing but not novelty-aware**

It does not distinguish between "this sample is confidently ID class 3" and "this OOD
sample happens to look slightly more like class 3 than any other class." It treats both
identically and sharpens both.

### 3.3 The adaptation–abstention conflict (formal sketch)

Let `f_θ` be a softmax classifier. Define the OOD score as `s(x) = 1 − max_c p_c(x)`.
After one TENT step on a mixed batch containing OOD sample `x_OOD`:

- The gradient from `x_OOD` pushes `max_c p_c(x_OOD)` upward (toward 1).
- Even if `x_OOD` contributes zero gradient directly (e.g. it is down-weighted), the
  BatchNorm affine parameters `(γ, β)` are shared across the batch. Updates driven by ID
  samples modify the representation of `x_OOD` in subsequent forward passes.

Therefore, `s(x_OOD)` decreases (model looks more confident on OOD) and AUROC degrades.
This is not a failure of TENT's implementation — it is a consequence of its objective.

The fundamental tension is:

```
TTA objective:     minimize H(p)  →  make every prediction more certain
OOD safety goal:   maximize H(p) on unknowns  →  stay uncertain on novel inputs
```

These two goals are **in direct conflict on OOD samples**. No amount of tuning TENT's
learning rate resolves this — only an objective that explicitly distinguishes ID from OOD
can do so.

---

## 4. The Solution (Research Contribution)

This paper does not claim to fully solve the conflict. The contribution is threefold:

### 4.1 Diagnosis

We expose the blind spot: **closed-set TTA benchmarks can rank a method as successful
while that method is simultaneously and silently degrading open-world OOD detection.**

We demonstrate this with a controlled paired protocol:
- `No TTA` — evaluate at t=0, no adaptation
- `ID-only TENT` — adapt on a batch with only shifted ID samples (oracle control)
- `Mixed TENT` — adapt on a batch containing both shifted ID and OOD samples (realistic)

The key result: `Mixed TENT` degrades OOD AUROC significantly more than `ID-only TENT`,
while both leave ID accuracy roughly flat. This isolates **contamination in the adaptation
batch** as the causal factor.

### 4.2 Mechanism

The degradation is explained by two measurable phenomena:
1. **Confidence sharpening on OOD**: max softmax probability increases on OOD samples
   after adaptation, reducing their score-based OOD signal.
2. **Reduced ID/OOD separability**: the gap between mean OOD score and mean ID score
   shrinks, degrading the OOD detector's ability to separate the two populations.

This is a weaker and more honest claim than "prototype absorption." It is supported by
multiple detectors (MSP, Energy, Mahalanobis) and survives the frozen-vs-dynamic centroid
comparison.

### 4.3 Practical recommendation

TTA papers should report OOD detection metrics alongside ID accuracy whenever adaptation
may occur in open-world conditions. A simple reporting norm — showing the ID accuracy /
OOD AUROC Pareto curve across methods — would make the tradeoff visible.

As a partial mitigation, confidence-weighted entropy (Fix A) shows directional improvement
over vanilla TENT in most settings, but does not fully resolve the conflict and does not
beat the no-adaptation baseline. A complete solution requires an adaptation objective with
an explicit open-set or abstention term.

---

## 5. Experiment Roadmap

The following experiments are required, in priority order. Each is independent and can be
run in parallel after the baseline sanity check (Step 1) passes.

### Step 1 — Baseline sanity (1–2 days) [BLOCKING]

**Goal**: verify the experimental setup is trustworthy.

Run and save:
- Clean CIFAR-10 test accuracy (target ≥ 93%)
- Clean CIFAR-10 vs SVHN MSP-AUROC (expected ≫ 0.7)
- Clean CIFAR-10 vs SVHN Energy-AUROC and Mahalanobis-AUROC
- CIFAR-10-C vs SVHN pre-TTA (t=0) AUROC for all three detectors at severity 5

If the clean CIFAR-10 vs SVHN AUROC is ≤ 0.70, stop and debug the checkpoint or the
OOD scoring pipeline before proceeding. Everything downstream depends on this.

### Step 2 — Contamination isolation (3–4 days) [CORE RESULT]

**Goal**: show that contamination, not TTA itself, is the causal factor.

Protocol:
- Three conditions on the **same drawn batches**: `No TTA`, `ID-only TENT`, `Mixed TENT`
- Same random seed, same batch draws across all three conditions
- `n = 30–50` independent batch draws per (α, corruption) cell
- α ∈ {0.9, 0.75, 0.5, 0.25}, corruption = gaussian_noise for main result + 4 others
- Report: mean ΔAUROC with 95% CI for each condition, paired t-test Mixed vs ID-only

**Key figure**: forest plot of ΔAUROC with CIs for all three conditions, across α values.

### Step 3 — Detector breadth (1–2 days) [MECHANISM SUPPORT]

**Goal**: show the effect is not an MSP artifact.

- Report the same paired results for MSP, Energy, and Mahalanobis under Step 2 protocol
- All three should show same-sign degradation for Mixed TENT vs ID-only TENT

**Key figure**: 3-panel detector comparison at α = 0.5 (representative α).

### Step 4 — Mechanism without derivative claim (2–3 days) [MECHANISM SECTION]

**Goal**: support the confidence-sharpening + separability-gap explanation.

Measure per step (t = 0, …, 20) for Mixed TENT vs ID-only TENT:
- Mean max softmax on OOD samples (should increase)
- Mean entropy on OOD samples (should decrease)
- OOD score gap: mean OOD score − mean ID score (should shrink)
- Feature norm of OOD representations (controls for global BN rescaling)
- Normalized centroid margin: (‖φ(x_OOD)−μ_{c*}‖ − ‖φ(x_ID)−μ_{c_ID}‖) / ‖φ‖

**Key figure**: 4-panel mechanism plot (entropy, max-softmax, margin, feature norm).

Pareto plot (bonus, very high value):
- x-axis: ΔID accuracy, y-axis: ΔOOD AUROC
- One point per (method, α, step) combination
- Shows the adaptation–abstention tradeoff visually

### Step 5 — Generalization (3–4 days) [BREADTH]

**Goal**: show the failure mode is not specific to one OOD or one TTA method.

- Add CIFAR-100 (semantically disjoint superclasses) as a second OOD alongside SVHN
- Add EATA as a second TTA method alongside TENT
- Run the Step 2 paired protocol on both
- If compute allows: add ResNet-50 or ViT-S/16 as a second backbone on CIFAR-10-C

**Key table**: generalization table (2 OODs × 2 TTA methods × 3 detectors, aggregate Δ).

### Step 6 — Fix A revisited (1–2 days) [OPTIONAL METHOD CONTRIBUTION]

Run Fix A (confidence-weighted entropy) under the paired Step 2 protocol.
Report its position on the Pareto plot alongside No TTA, ID-only TENT, Mixed TENT.
Only include as a method contribution if it consistently outperforms Mixed TENT in ΔAUROC
while matching or exceeding it on ID accuracy.

---

## 6. Why This Approach Is Better Than the Previous One

| Dimension | Previous (ProtoAbsorb) | New (Adaptation–Abstention) |
|---|---|---|
| **Core claim** | OOD features drift monotonically toward the nearest ID centroid | Entropy-based TTA is class-closing but not novelty-aware — an objective-level conflict |
| **Claim strength** | Overstated: data shows global contraction, not OOD-selective absorption; inverts under dynamic centroids | Correctly scoped: supported by confidence traces, score gaps, and multiple detectors |
| **Falsifiability** | Hypothesis can be "confirmed" by frozen-centroid artifact even if no real OOD-specific effect exists | Hypothesis is falsified if ID-only TENT and Mixed TENT show the same AUROC degradation |
| **Novelty** | Descriptive mechanism name for a known type of representation drift | Identifies a fundamental conflict between two established objectives (entropy minimization vs. OOD safety) |
| **Scope** | Specific to TENT + frozen centroids + Euclidean distance | Applies to any entropy-minimizing TTA method with any score-based OOD detector |
| **Practical implication** | Fix the geometry of TENT | Fix the evaluation norm for all TTA papers |
| **Reviewer response** | "But ID features contract too, so it is just global shrinkage" | "This explains why closed-set TTA metrics miss a real safety failure mode" |
| **ACCV fit** | Weak: speculative mechanism, CIFAR-only, prototype story is deflated by own data | Strong: clean paired protocol, evaluation critique, principled tension, broadly applicable |

The single most important difference: the previous framing required the geometry story to
hold cleanly, and it does not. The new framing requires only that Mixed TENT degrades OOD
AUROC more than ID-only TENT — which is directly testable, unambiguous, and well-motivated
by the objective analysis in §3.
