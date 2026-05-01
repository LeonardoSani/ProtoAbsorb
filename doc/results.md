# ProtoAbsorb — Experimental Results (rigorous reading)

This document describes what each experiment in this repository **actually
does in code** and what the **saved JSON results** show, with caveats. It is
deliberately conservative: every numerical claim below is traceable to a
saved file under [results/](../results/), and every behavioral claim is
traceable to a line in the implementation. Anything that would require
re-running or opening the PNGs to verify is flagged as such.

## 1. Shared setup

Implementation: [experiments/_common.py](../experiments/_common.py),
[src/proto_absorb/tent.py](../src/proto_absorb/tent.py),
[src/proto_absorb/centroids.py](../src/proto_absorb/centroids.py).

- **Mixed batches.** Each batch is built by `MixedBatchSampler`, which draws
  `int(α·B)` shifted-ID samples (CIFAR-10-C, severity 5) and the remainder
  from an OOD pool (default SVHN). Default `B = 64`, `α ∈ {0.9, 0.75, 0.5,
  0.25}`.
- **TENT setup** ([tent.py:47-68](../src/proto_absorb/tent.py#L47-L68)): the
  ResNet-18 is set to `eval()`; only BatchNorm `(γ, β)` are trainable; BN
  layers are flipped to `train()` and have `track_running_stats = False` so
  batch statistics are recomputed each forward.
- **Adaptation step**
  ([tent.py:168-210](../src/proto_absorb/tent.py#L168-L210)): SGD,
  `lr = 1e-3`, momentum 0.9, default `T = 20` steps per batch. The model is
  reset to `θ₀` at the start of every batch
  ([_common.py:52-60](../experiments/_common.py#L52-L60)).
- **OOD score** used for AUROC plots throughout: MSP =
  `1 − max_c softmax(logits)_c`. Energy and Mahalanobis are *computed* in
  `evaluate_batch` but never plotted as AUROC curves
  ([_common.py:91-122](../experiments/_common.py#L91-L122)).
- **AUROC convention**: ID = negative class, OOD = positive class
  ([_common.py:125-129](../experiments/_common.py#L125-L129)).

> Caveat that affects every absolute number below: I have **not** verified
> the checkpoint's clean-CIFAR-10 accuracy or its clean-MSP AUROC on SVHN.
> [Theory.md](../Theory.md) targets ≥ 93% on clean CIFAR-10, but no such
> baseline is saved under [results/](../results/). The absolute AUROCs
> reported below are therefore uncalibrated — see §6.

---

## 2. Experiment 1 — Verify the failure mode

**Code:** [experiments/exp1_failure_mode.py](../experiments/exp1_failure_mode.py).

**What it does.** For each α, run `n_batches` independent batches; for each
batch, reset to `θ₀`, fix one mixed batch, and record AUROC + ID accuracy at
`t = 0, …, T` while taking a TENT step on the *same batch* between
evaluations
([exp1:39-68](../experiments/exp1_failure_mode.py#L39-L68)). Then average
the curves over batches. A second pass produces the corruption × α
ΔAUROC heatmap with `max(2, n_batches // 2)` batches per cell
([exp1:187-203](../experiments/exp1_failure_mode.py#L187-L203)).

**Defaults that matter:** `--batches 5` for the main per-α curves, **2
batches per cell** for the heatmap. At α = 0.9 with B = 64 there are only
~6 OOD samples per batch contributing to AUROC.

### Plot 1.1 — AUROC vs adaptation steps (corruption = `gaussian_noise`)

From [results/exp1/results_gaussian_noise.json](../results/exp1/results_gaussian_noise.json),
mean over 5 batches:

| α    | AUROC(0) | AUROC(T=20) | Δ      | std at T |
|------|---------:|------------:|-------:|---------:|
| 0.9  | 0.588    | 0.531       | −0.057 | 0.089    |
| 0.75 | 0.601    | 0.536       | −0.065 | 0.027    |
| 0.5  | 0.607    | 0.507       | −0.100 | 0.048    |
| 0.25 | 0.583    | 0.539       | −0.044 | 0.050    |

ID-only classification accuracy on the ID slice of the batch is roughly
flat across t (e.g. α=0.9: 0.683 → 0.690; α=0.5: 0.600 → 0.613).

### Plot 1.2 — ΔAUROC heatmap

From
[results/exp1/heatmap_delta_auroc.json](../results/exp1/heatmap_delta_auroc.json),
**2 batches per cell**:

- Most cells are negative, consistent with the hypothesis. Largest drops
  appear at α=0.25: `fog −0.232`, `pixelate −0.171`, `impulse_noise
  −0.161`, `elastic_transform −0.160`, `shot_noise −0.156`.
- **Several cells are positive** (TENT *helps* MSP-AUROC):
  - α=0.9: `shot_noise +0.099`, `impulse_noise +0.092`,
    `defocus_blur +0.006`, `snow +0.011`.
  - α=0.75: `impulse_noise +0.043`, `contrast +0.010`.
  - α=0.5: `shot_noise +0.008`.
  - α=0.25: `snow +0.025`.
- The expected monotone trend "smaller α ⇒ larger drop" is **not** clean:
  among the central αs, the worst case is α=0.5; α=0.9 and α=0.75 are
  comparable in magnitude.

### Honest reading of Exp 1

- **Direction matches the hypothesis** for `gaussian_noise` and for most
  cells of the heatmap.
- **Absolute starting AUROCs are low** (~0.58–0.61). Without a saved
  clean-CIFAR-10 AUROC sanity baseline, these are essentially "barely better
  than chance, decaying toward chance"; conclusions about "AUROC drops by Δ"
  are quantitatively weak.
- **Statistical power is thin.** 5 batches × 64 samples → ~30 OOD samples
  per batch at α = 0.5. The σ around AUROC(T) is comparable to the mean Δ,
  e.g. α=0.9: |Δ|=0.057, σ=0.089. Most ΔAUROC values cannot be cleanly
  separated from zero at this n.
- The heatmap has **2 batches per cell**, so corruption-by-corruption
  rankings within the heatmap should not be over-interpreted.

PNG plots that I did not open: `plot_1.1_auroc_vs_steps_gaussian_noise.png`,
`plot_1.2_auroc_degradation_heatmap.png`,
`plot_1.3_acc_vs_steps_gaussian_noise.png`.

---

## 3. Experiment 2 — Geometric mechanism

**Code:** [experiments/exp2_geometric_mechanism.py](../experiments/exp2_geometric_mechanism.py).

**What it does.** Same outer loop as Exp 1, but at every step it computes
`φ(x)` for every sample in the batch and the L2 distance to the nearest
class centroid
([exp2:43-100](../experiments/exp2_geometric_mechanism.py#L43-L100)). Two
centroid regimes are run separately:

- **Frozen**: `μ_c = μ_c^(0)` loaded once from
  [checkpoints/centroids.npz](../checkpoints/centroids.npz).
- **Dynamic**: after every TENT step, an EMA update on **ID-only** samples
  in the current batch with `momentum = 0.9`
  ([exp2:78-86](../experiments/exp2_geometric_mechanism.py#L78-L86),
  [centroids.py:133-157](../src/proto_absorb/centroids.py#L133-L157)),
  i.e. `μ_c ← 0.9·μ_c + 0.1·batch_mean_c`. Oracle ID labels are used
  **only** for this update, never for TENT.

**Defaults:** `--batches 4`.

### Plot 2.1 / 2.4 — Mean OOD→nearest-centroid distance

From [results/exp2/summary.json](../results/exp2/summary.json):

**Frozen centroids — primary evidence for the absorption claim**

| α    | mean_ood(0) | mean_ood(T=20) | Δ      |
|------|------------:|---------------:|-------:|
| 0.9  | 3.681       | 2.779          | −0.902 |
| 0.75 | 3.257       | 2.369          | −0.888 |
| 0.5  | 3.293       | 2.369          | −0.924 |
| 0.25 | 3.500       | 2.482          | −1.018 |

Monotone decrease at every α — the cleanest signal in the project. About
25–30% relative reduction in mean OOD distance over 20 steps.

**ID distances under frozen centroids — the necessary control**

| α    | mean_id(0) | mean_id(T) | Δ       |
|------|-----------:|-----------:|--------:|
| 0.9  | 2.790      | 2.141      | −0.649  |
| 0.75 | 2.894      | 2.238      | −0.656  |
| 0.5  | 2.913      | 2.297      | −0.616  |
| 0.25 | 3.195      | 2.527      | −0.668  |

ID features contract toward the centroid bank in roughly the same direction
and only modestly less in magnitude. The OOD-vs-ID gap (mean_ood − mean_id)
shrinks under TENT — for example at α=0.9 from 0.89 to 0.64. This is what
the OOD detector "sees", and it is consistent with MSP-AUROC dropping. But
calling this **prototype absorption of OOD specifically** is not what the
data supports — it is a **global feature contraction toward the prototype
bank**, with OOD shrinking marginally more in absolute distance than ID.

**Dynamic centroids — the falsifiability test**

| α    | mean_ood(0) | min along trace (≈ t=5) | mean_ood(T) |
|------|------------:|------------------------:|------------:|
| 0.9  | 3.825       | 3.513 (t=5)             | 3.572       |
| 0.75 | 3.406       | 3.066 (t=6)             | 3.568       |
| 0.5  | 3.441       | 2.954 (t=6)             | 3.584       |
| 0.25 | 3.593       | 3.090 (t=4)             | 3.923       |

Under dynamic centroids the OOD distance **dips for ~5 steps and then grows
above its starting value**. Meanwhile `mean_id(t)` continues to shrink
monotonically (e.g. α=0.25: 3.465 → 1.876). Mechanistically: when the
prototypes are allowed to track the ID feature drift, the prototypes
**out-run** the OOD samples; OOD ends up *farther* from the prototype bank
at t=T than at t=0.

### Plots 2.2, 2.3, 2.5

Histograms (Plot 2.2), per-class absorption bar chart (Plot 2.3) and UMAP
snapshots (Plot 2.5) are produced but only the JSON summary above is saved
as numerical aggregates. The corresponding PNGs exist
([plot_2.2_distance_histograms.png](../results/exp2/plot_2.2_distance_histograms.png),
[plot_2.3_per_class_absorption.png](../results/exp2/plot_2.3_per_class_absorption.png),
[plot_2.5_umap_snapshots.png](../results/exp2/plot_2.5_umap_snapshots.png));
I have not opened them, so I make no claims about their content beyond what
the code produces by construction.

### Honest reading of Exp 2

- **The hypothesis "OOD distance to frozen prototypes decreases under TENT"
  is supported.** Magnitude ~−0.9 to −1.0 in 20 steps, consistent across
  α.
- **But the same effect is observed for ID samples.** This means the
  observed AUROC degradation in Exp 1 is best explained as global feature
  contraction reducing the ID-vs-OOD distance gap, not as OOD-selective
  absorption.
- **Under dynamic prototypes the absorption story inverts.** This is a
  meaningful negative control: the apparent attraction of OOD to prototypes
  in the frozen regime is conditional on holding the prototypes fixed while
  the encoder drifts. With even a slow EMA on the prototypes
  (`momentum = 0.9`), OOD distance grows above its starting value by t=20.

---

## 4. Experiment 3 — Proposed fixes

**Code:** [experiments/exp3_fixes.py](../experiments/exp3_fixes.py).

**Methods compared**
([exp3:51-58](../experiments/exp3_fixes.py#L51-L58)):

- **No TTA** — evaluate at t=0 only.
- **TENT** — vanilla mean-entropy minimization.
- **Fix A** — confidence-weighted entropy:
  `Σᵢ wᵢ H(pᵢ) / Σᵢ wᵢ`, with `wᵢ = max_c p_c(xᵢ)` detached
  ([tent.py:87-96](../src/proto_absorb/tent.py#L87-L96)).
- **Fix B** — TENT loss + `λ · ‖φ(x) − μ_{ĉ}‖²` for samples whose softmax
  confidence exceeds `0.7`, with frozen `μ`
  ([tent.py:99-121](../src/proto_absorb/tent.py#L99-L121)). Default
  `λ = 0.1`.
- **Fix C** — drop the top `0.25` fraction by entropy and run vanilla TENT
  on the remainder
  ([tent.py:124-137](../src/proto_absorb/tent.py#L124-L137)).
- **Fix A+B** — weighted entropy + anchor.

**Defaults:** `--batches 4`, `T = 20`.

### Numerical results

From [results/exp3/fix_results.json](../results/exp3/fix_results.json).

**AUROC at T (higher is better)**

| α    | NoTTA | TENT  | FixA  | FixB  | FixC  | A+B   |
|------|------:|------:|------:|------:|------:|------:|
| 0.9  | 0.678 | 0.522 | 0.623 | 0.443 | 0.549 | 0.524 |
| 0.75 | 0.650 | 0.636 | 0.521 | 0.499 | 0.553 | 0.482 |
| 0.5  | 0.560 | 0.487 | 0.590 | 0.561 | 0.518 | 0.489 |
| 0.25 | 0.581 | 0.456 | 0.583 | 0.512 | 0.517 | 0.527 |

**ΔAUROC = AUROC(T) − AUROC(0) (closer to zero is better)**

| α    | NoTTA | TENT   | FixA   | FixB   | FixC   | A+B    |
|------|------:|-------:|-------:|-------:|-------:|-------:|
| 0.9  | 0.000 | −0.152 | +0.050 | −0.128 | −0.088 | −0.152 |
| 0.75 | 0.000 | −0.001 | −0.078 | −0.045 | −0.007 | −0.096 |
| 0.5  | 0.000 | −0.088 | −0.072 | −0.082 | −0.127 | −0.113 |
| 0.25 | 0.000 | −0.177 | −0.076 | −0.046 | −0.008 | −0.049 |

**OOD distance to *frozen* centroids at T**

| α    | NoTTA | TENT  | FixA  | FixB  | FixC  | A+B   |
|------|------:|------:|------:|------:|------:|------:|
| 0.9  | 3.718 | 2.867 | 3.038 | 1.961 | 3.224 | 2.221 |
| 0.75 | 3.494 | 2.535 | 2.575 | 1.918 | 2.970 | 1.969 |
| 0.5  | 3.331 | 2.512 | 2.505 | 1.961 | 2.714 | 1.991 |
| 0.25 | 3.477 | 2.509 | 2.557 | 1.962 | 2.805 | 2.202 |

### Honest reading of Exp 3

- **No TTA wins on AUROC(T) at three of four α values** (0.9, 0.75, 0.25).
  Only at α = 0.5 does any TTA method (Fix A) match or beat the no-adapt
  baseline. The simplest faithful summary of this experiment is "in the
  presence of OOD contamination, not adapting beats every adaptation
  variant tested here, except Fix A at α = 0.5 and α = 0.25."
- **Fix A** is the only fix with a directional improvement over vanilla
  TENT in three of four cells. Its ΔAUROC is the smallest in
  α ∈ {0.9, 0.5, 0.25}; at α = 0.9 it is positive (+0.05). At α = 0.75 it
  is worse than vanilla TENT, but vanilla TENT in that cell barely degrades
  to start with.
- **Fix B is actively counterproductive** at α = 0.9 (AUROC(T) = 0.443,
  worst in the table). Its `ood_dist_T` collapses to ≈ 1.96 at every α —
  the smallest OOD distance of any method. The confidence gate
  (`conf > 0.7`) does not gate strongly enough; the anchor pulls OOD
  features in alongside ID.
- **Fix C is a wash.** Its ΔAUROC at α = 0.5 is *worse* than vanilla TENT
  (−0.127 vs −0.088), suggesting that dropping the top-25% by entropy is
  also dropping useful shifted-ID samples, not selectively the OOD ones.
- **Fix A+B inherits Fix B's pathology** — anchor term dominates, recovers
  worse-than-TENT behavior at every α.
- **Methodological caveat that limits all comparisons.** Each method runs
  with its own seed
  ([exp3:197](../experiments/exp3_fixes.py#L197):
  `seed = args.seed + int(α*1000) + abs(hash(name)) % 1000`), so methods
  see *different sampled batches*. Within a single α column,
  `auroc_0` varies across methods (e.g. at α = 0.9: 0.678, 0.674, 0.573,
  0.570, 0.636, 0.676), confirming the draws differ. ΔAUROC partially
  controls for this by anchoring on each method's own t=0; absolute
  AUROC(T) cross-method comparisons are confounded.
- **Standard deviations are large** (e.g. Fix A+B at α = 0.9 has σ = 0.233
  over 4 batches). Most pairwise differences in this table are not
  significant at the n run.

PNG: [plot_3_summary.png](../results/exp3/plot_3_summary.png) (not opened).

---

## 5. Experiment 7 — Latent vector field (no numerical results saved)

**Code:** [experiments/exp7_vector_field.py](../experiments/exp7_vector_field.py).

The script collects per-step features for one batch
([exp7:40-58](../experiments/exp7_vector_field.py#L40-L58)) at α = 0.5,
`gaussian_noise`, T = 20, and renders three plots for each requested
snapshot:

- 7.1 quiver of `Δφ^(t→t+k)` in 2D UMAP
  ([exp7:61-99](../experiments/exp7_vector_field.py#L61-L99)),
- 7.2 scatter of `‖Δφ^(t→t+1)‖` vs `‖φ(x) − μ_{c*}‖`
  ([exp7:102-126](../experiments/exp7_vector_field.py#L102-L126)),
- 7.3 histogram of `cos(Δφ, μ_{c*} − φ(x))`
  ([exp7:129-156](../experiments/exp7_vector_field.py#L129-L156)).

Only [results/exp7/config.json](../results/exp7/config.json) is saved as
numerical output (run config, no metrics). The 16 PNG files exist but I
have not opened them, so I make **no quantitative claim** about Exp 7.

---

## 6. Cross-cutting caveats and what I would ask before publishing

1. **Sanity baselines are missing.** No saved clean-CIFAR-10 accuracy and
   no clean-CIFAR-10-vs-SVHN MSP-AUROC. Without these, the absolute AUROCs
   in Exp 1 and Exp 3 are uncalibrated. A typical pre-trained CIFAR-10
   ResNet on SVHN scores MSP-AUROC ≫ 0.7 in published baselines; the t=0
   numbers here are 0.56–0.68, which deserves a sanity audit before
   over-interpreting absolute drops.
2. **Sample sizes are too small.** 4–5 batches per (α, method) and 2
   batches per heatmap cell. The σs around the AUROC summaries are often
   comparable to the mean Δ. For a publication-grade comparison I would
   want ≥ 20 batches.
3. **Per-method seeding in Exp 3 confounds the comparison.** Methods should
   share the same drawn batches so that AUROC(T) at fixed α is an
   apples-to-apples comparison, not just ΔAUROC.
4. **MSP is the only AUROC reported.**
   [_common.py](../experiments/_common.py) computes Energy and Mahalanobis;
   the Mahalanobis curve in particular would be the right control for the
   prototype-absorption story (Exp 2 distance is a Euclidean stand-in for
   it). They are not plotted.
5. **The "absorption" framing is partially deflated by Exp 2.** Both ID and
   OOD distances shrink under frozen centroids; under dynamic centroids,
   OOD distance grows. The most defensible mechanistic claim from the
   current data is "TENT induces global feature contraction toward
   whichever prototypes are held fixed", not "OOD features are selectively
   absorbed into ID prototypes."

---

## 7. One-paragraph summary

The qualitative phenomenon — TENT entropy minimization on mixed batches
degrading OOD detection — is visible in this data, most cleanly at α = 0.5
on `gaussian_noise`. The geometric piece — frozen-prototype OOD distance
shrinks monotonically under TENT — is the strongest signal in the
repository. But the framing as "prototype absorption *of OOD*" is only
partially supported: ID features collapse together with OOD under frozen
prototypes, and the effect inverts under dynamic prototypes. The proposed
fixes are inconsistent and underpowered as evaluated; only confidence-
weighted entropy (Fix A) shows a reproducible directional improvement, and
even that does not beat doing no test-time adaptation at all in three of
four α settings.
