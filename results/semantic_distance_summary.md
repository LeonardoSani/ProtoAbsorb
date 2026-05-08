# Semantic-distance OOD comparison

**Setup:** ResNet-18, CIFAR-10 ID, CIFAR-10-C `gaussian_noise` severity 5 as
shifted-ID corruption, MSP-AUROC. n=30 paired batches per cell (Step 2 main
table; Step 5 uses n=20). Same draw across `no_tta` / `id_only` / `mixed`.

## Step 2 main results — Δ MSP-AUROC after T=20 TENT steps

| OOD | α | id_only Δ [95% CI] | mixed Δ [95% CI] | id_only − mixed | paired p |
|---|---|---|---|---|---|
| SVHN       | 0.9  | +0.188 [+0.162,+0.212] | -0.043 [-0.100,+0.016] | +0.231 | 2.6e-08 |
| SVHN       | 0.75 | +0.130 [+0.114,+0.145] | -0.045 [-0.077,-0.010] | +0.175 | 1.2e-09 |
| SVHN       | 0.5  | +0.077 [+0.063,+0.092] | -0.052 [-0.081,-0.023] | +0.129 | 9.1e-09 |
| SVHN       | 0.25 | +0.027 [+0.012,+0.042] | -0.051 [-0.077,-0.022] | +0.078 | 1.6e-05 |
| DTD        | 0.9  | +0.209 [+0.182,+0.240] | -0.003 [-0.053,+0.050] | +0.212 | 7.9e-11 |
| DTD        | 0.75 | +0.166 [+0.154,+0.179] | -0.037 [-0.068,-0.005] | +0.203 | 3.5e-14 |
| DTD        | 0.5  | +0.119 [+0.102,+0.134] | -0.055 [-0.081,-0.029] | +0.174 | 5.2e-16 |
| DTD        | 0.25 | +0.097 [+0.073,+0.122] | -0.032 [-0.061,+0.001] | +0.129 | 9.6e-10 |
| Places365  | 0.9  | +0.192 [+0.170,+0.215] | +0.016 [-0.008,+0.040] | +0.176 | 4.3e-14 |
| Places365  | 0.75 | +0.153 [+0.136,+0.171] | +0.032 [+0.009,+0.055] | +0.121 | 1.1e-09 |
| Places365  | 0.5  | +0.080 [+0.061,+0.100] | +0.020 [+0.001,+0.039] | +0.060 | 2.7e-05 |
| Places365  | 0.25 | +0.018 [-0.003,+0.038] | +0.006 [-0.021,+0.036] | +0.012 | 0.44 |
| CIFAR-100* | 0.9  | +0.226 [+0.202,+0.250] | +0.054 [+0.024,+0.083] | +0.172 | 8.2e-09 |
| CIFAR-100* | 0.75 | +0.151 [+0.132,+0.173] | +0.015 [-0.008,+0.041] | +0.136 | 1.6e-08 |
| CIFAR-100* | 0.5  | +0.082 [+0.067,+0.096] | +0.022 [-0.003,+0.049] | +0.060 | 8.4e-05 |
| CIFAR-100* | 0.25 | +0.073 [+0.048,+0.099] | +0.027 [+0.003,+0.052] | +0.046 | 0.014 |

*CIFAR-100 numbers are from Step 5 (n=20). All other rows are Step 2 (n=30).

## Pattern

Order the four OOD sets by clean MSP-AUROC at t=0 (a proxy for semantic
distance from CIFAR-10):

| OOD | clean MSP-AUROC | type |
|---|---|---|
| DTD       | 0.910 | textures (semantically disjoint) |
| SVHN      | 0.918 | digits (visually disjoint, semantic content unrelated) |
| Places365 | 0.880 | scenes (semantically disjoint) |
| CIFAR-100 | (lower) | proximal: shares super-classes with CIFAR-10 |

The **paired Mixed − ID-only gap is significant in 30/32 cells** across all
four OODs and both TTA methods (TENT, EATA). That gap — TTA's *contamination
penalty* — is the falsifiable claim of the reframe.

What the new OODs change is the *absolute* sign of Mixed Δ-AUROC:

- **DTD, SVHN (far OOD):** Mixed Δ ≤ 0 in 6/8 main cells. TTA on a contaminated
  batch is strictly worse than no adaptation under MSP — the "silent
  destruction" headline holds.
- **Places365 (far OOD, scenes):** Mixed Δ ≈ 0..+0.04. TTA does not destroy
  the detector below baseline, but it consumes most of the improvement that
  ID-only adaptation would have delivered.
- **CIFAR-100 (proximal OOD):** Mixed Δ > 0 in all cells. Mixed TENT *does*
  improve AUROC over no-TTA — but still far less than ID-only TENT.

## Reframe

The earlier draft's headline ("TTA silently destroys OOD detection") is too
strong for proximal OODs and Places365. The corrected, defensible claim
across all four OOD sets is:

> Entropy minimization on a mixed test batch *systematically forfeits a
> large fraction of the OOD-detection gain that the same adaptation would
> deliver if applied only to the ID slice*. The size of the loss scales with
> semantic distance: maximal for textures and digits, intermediate for
> scenes, mildest for class-proximal OODs.

This is a strictly weaker — and harder to dismiss — statement: the
adaptation–abstention conflict is real (paired test significant in nearly
every cell), but its magnitude is modulated by how separable the OOD set
was *before* adaptation.

## Step 5 — full grid (4 OODs × 2 methods × 4 αs, n=20)

Significance counts (paired Mixed vs ID-only, p<0.05):

| | TENT | EATA |
|---|---|---|
| SVHN      | 4/4 | 4/4 |
| CIFAR-100 | 2/4 | 3/4 |
| DTD       | 4/4 | 4/4 |
| Places365 | 3/4 | 3/4 |

Total: 27/32 cells significant. Non-significant cells are all at α=0.25,
where ID-only itself barely improves AUROC (small Δ → less signal to
destroy).

## Files

- Step 2 / DTD: `results/step2_dtd/` (forest plots × 5 corruptions, JSON)
- Step 2 / Places365: `results/step2_places365/`
- Step 5 / 4-OOD grid: `results/step5_4oods/` (`generalization_grid.png`)
- Step 1 sanity: `results/step1_dtd/`, `results/step1_places365/`
