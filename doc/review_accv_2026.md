# ACCV 2026 Review Session — gpt-5.4 xhigh
**Date:** 2026-05-09 | **ThreadID:** 019e0cc2-036b-7cb1-98d9-9abdd63d614e | **Rounds:** 3

---

## Mock Review Score

**Score: 3/6 (borderline reject / weak reject as written)**
**Confidence: 4/5**

---

## Summary (Reviewer Voice)

This paper studies TTA under open-world test streams and argues that entropy-minimizing TTA can improve shifted-ID accuracy while harming OOD detection. The authors propose a paired contamination protocol comparing an oracle `id_only` condition to realistic `mixed` adaptation and show a consistent paired gap across several datasets, detectors, and methods. The evidence breadth is good, but novelty is vulnerable relative to recent OSTTA literature, the oracle control is confounded, and there is no comparison against dedicated OSTTA baselines.

---

## Strengths

1. Practically relevant problem with clean framing.
2. Paired experimental design is stronger than most TTA papers.
3. Broad empirical sweep: multiple OODs, detectors, TTA methods, ImageNet-scale validation.
4. Unusually honest claim calibration — sign reversals, null fixes, α=0.25 weakness all reported.
5. Strong figure potential and interpretable paired-gap narrative.

---

## Weaknesses (rejection risks)

1. **Novelty eroded.** ROSETTA (arXiv 2604.01589, April 2, 2026) explicitly frames an ID/OOD tradeoff in OSTTA and proposes a fix. The conflict itself is no longer a fresh discovery in mid-2026. Defensible novelty: the paired contamination protocol and evaluation audit.

2. **Batch-size confound in `id_only` oracle.** `id_only` adapts on αB samples, `mixed` adapts on B samples. For BN-based TTA, this confounds "contamination effect" with "sub-batch size / statistics quality effect." The BN→IN/LN ablation helps but does not fully remove this confound (LN `id_only` still uses αB samples).

3. **EATA is not full EATA.** The Fisher regularizer is omitted (confirmed in `src/proto_absorb/eata.py:17`). Must relabel as **ETA** before submission — reviewers will catch this.

4. **No OSTTA baseline comparison.** Without running UniEnt / UniEnt+ under the same paired protocol, the paper cannot answer: "do existing open-set TTA methods already solve this?" That is the most dangerous reviewer question.

5. **No ranking reversal figure.** The strongest version of the protocol story requires showing that standard closed-set ranking of methods differs from mixed-stream safety ranking. Currently missing.

---

## Questions from Reviewer

- Does the paired gap persist under a **batch-size-matched** oracle control (`id_fullmatch`)?
- How do UniEnt / ROSETTA rank under the paired protocol?
- Why should reviewers view this as more than re-benchmarking a known OSTTA issue?
- Can you show the full corruption grid, not just gaussian-noise-centered evidence?

---

## Critical Action Plan (Priority Order, July 5 Deadline)

### Priority 1: UniEnt/UniEnt+ comparison (HIGHEST LIFT)
**Why:** Answers the novelty attack directly. If UniEnt+ still has a nontrivial paired gap, the protocol is useful even for OSTTA methods. If it closes the gap entirely, paper survives as evaluation-norm paper.

**Minimum:** Run `No TTA / TENT / ETA / UniEnt+` on 2 OOD settings under the paired protocol.
- Code: https://github.com/gaozhengqing/UniEnt
- Label the EATA implementation as **ETA** throughout.

**Expected pattern:** UniEnt+ should reduce the contamination gap substantially but not eliminate it (due to pseudo-partition errors and shared BN statistics). This is the "protocol still matters" story.

**Ranking reversal figure (2 panels, same method set):**
- Panel A: standard closed-set metric (ΔID accuracy)
- Panel B: mixed-stream safety metric (mixed AUROC or mixed FPR95) + paired gap annotation

### Priority 2: 4-Condition Confound Control
**Why:** Settles the batch-size confound objection. Decomposes contamination cost cleanly.

**Four conditions:**
1. `id_subbatch` — current oracle (adapt on ID slice, size αB)
2. `id_fullmatch` — replace OOD slots with extra ID samples from corrupted pool (size B, ID-only)
3. `mixed_maskedloss` — full mixed batch forward (BN sees all), but loss computed on ID slice only
4. `mixed` — current realistic condition

**Decomposition:**
- `id_subbatch − id_fullmatch` = batch-size / sample-count artifact
- `id_fullmatch − mixed_maskedloss` = BN/statistics contamination (this should be the big gap)
- `mixed_maskedloss − mixed` = direct OOD-gradient contamination (~11%)

**Minimum cells:** 2 cells sufficient:
- `SVHN α=0.5` (absolute-failure case) + `Places365 α=0.9` (forfeit-without-sign-flip case)
- n=30 each. Safe appendix: add `SVHN α=0.9` + `Places365 α=0.5`.

### Priority 3: Relabeling (Quick Fix)
Replace all instances of "EATA" with "ETA" in paper text, tables, and code comments. Add explicit note that Fisher regularizer is omitted.

---

## 4-6 Week Timeline

| Week | Action |
|---|---|
| 1 | Integrate UniEnt/UniEnt+; run anchor cell (1 OOD × TENT/ETA/UniEnt+) |
| 2 | Expand to 2-panel ranking figure on 2 OOD settings; relabel ETA |
| 3 | Implement 4-condition control (id_fullmatch + mixed_maskedloss) |
| 4 | Run 2 must-run control cells (SVHN α=0.5 + Places365 α=0.9), n=30 each |
| 5 | Expand control to 4 cells if needed; ImageNet anchor if time |
| 6 | Finalize claims, tables, paper draft; submission |

**Decision gate after Week 2:**
- If UniEnt+ has nontrivial paired gap → protocol paper is alive, proceed as planned
- If UniEnt+ fully closes gap → reframe as evaluation-norm paper; still viable but narrower

---

## ROSETTA Status (as of 2026-05-09)

- arXiv 2604.01589 (April 2, 2026) — frames ID/OOD tradeoff in OSTTA, proposes new method
- **No public code confirmed** as of May 2026 (CatalyzeX shows "Request Code")
- Does NOT appear to use the matched-batch `id_only` vs `mixed` diagnostic protocol
- **Do not make ROSETTA a dependency.** UniEnt+ is sufficient for the comparison story.

Key prior art to cite:
- TENT (Wang et al. ICLR 2021)
- EATA/ETA (Niu et al. ICML 2022)
- CoTTA (Wang et al. CVPR 2022)
- SAR (Niu et al. ICLR 2023)
- OWTTT / Li et al. (ICCV 2023) — arXiv 2308.06879
- UniEnt / Gao et al. (CVPR 2024) — arXiv 2404.06065
- ROSETTA / Zhao et al. (2026) — arXiv 2604.01589
- MSP (Hendrycks 2017), Mahalanobis (Lee 2018), Energy (Liu 2020)

---

## Reframe: Paper Contribution (Accepted Framing)

**NOT:** "We discover that entropy-minimizing TTA conflicts with OOD safety."
**YES:** "We introduce a paired contamination protocol and contamination-cost decomposition that reveals a systematic gap in current TTA evaluation practice, and show that standard closed-set TTA benchmarks can mask safety regressions that affect even dedicated OSTTA methods."

Primary contributions:
1. A reproducible paired diagnostic for measuring contamination cost in TTA
2. A decomposition separating adaptation benefit from contamination penalty (three-way + four-condition)
3. An evaluation audit demonstrating that standard TTA reporting is insufficient for open-world deployment claims
4. Evidence that the gap persists even under OSTTA methods (UniEnt+), quantifying what remains unsolved

---

## What Would Push Score to 4/6

1. UniEnt/UniEnt+ comparison showing partial but not full gap closure
2. 4-condition confound control on 2 cells
3. Ranking reversal figure (2 panels)
4. Relabeling ETA correctly

No single paper fix is sufficient — all three experiments are needed.
