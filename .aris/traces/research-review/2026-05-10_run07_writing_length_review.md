# Writing & Length Review — ProtoAbsorb / Open-World TTA Safety
**Date:** 2026-05-10  
**Reviewer model:** GPT-5.4 via Codex MCP (xhigh reasoning)  
**Thread ID:** 019e1243-f5b2-7a70-9319-e85bda9c67da  
**Focus:** Page length, writing quality, structural critique  
**Target venue:** ACCV 2026

---

## Overall Assessment

- **Score:** 6/10 (Weak Accept, conditional on tightening)
- **Confidence:** Medium
- **Core diagnosis:** Paper reads like a long technical report, not a crisp conference paper. Diagnostic/evaluation-only contribution is publishable only if extremely crisp. Currently dense, repetitive, and over-instrumented.

---

## Page Length Critique

Section 4 has too many "secondary validations" crammed into one section:
- Core gap characterization
- Detector breadth (5 detectors)
- Correlation analysis
- ImageNet replication
- Streaming (100-batch + nonstationary)
- Frozen-detector recalibration
- Office-Home domain-shift

That's 8 sub-experiments in one section. For a 9-page ACCV paper, all but 1-2 of these must move to appendix.

---

## Concrete 9-Page Outline

| Section | Pages | What Stays | What Moves to Appendix |
|---------|-------|-----------|----------------------|
| §1 Introduction | 0.8–1.0 | Problem, Fig 1, contributions (tightened), scope | Exact p-values, cell counts in bullet text |
| §2 Related Work | 0.7–0.9 | 3 paragraphs: TTA family, OOD detection, open-world TTA | Detector mechanistic speculation (ViM null-space) |
| §3 Protocol | 0.9–1.1 | Contract box, 3 conditions, GAP metric, oracle explanation | Full stat protocol details, bootstrap details |
| §4 Characterization | 1.7–2.1 | Fig 2 (one OOD, one detector, all α), Table 1 (compact sig summary), one-sentence pointers to appendix tables | Tables 2–5 (ImageNet, Streaming, Recalibration, Office-Home) |
| §5 Mechanism | 0.9–1.1 | 4-condition decomp table (2 representative cells), headline BN/gradient finding, why >100% happens | BN vs IN ablation, ViT results, Fix A/Fix C |
| §6 Audit | 0.9–1.1 | Table 7 (methods × ΔID × mixed AUROC × gap at α=0.9), one Pareto plot | Extended Table 8 (both α), mitigation heuristics |
| §7 Conclusion | 0.6–0.9 | 5-7 sentences + 4-6 limitations | Long future-work list |
| Buffer | — | White space, visuals breathe | — |

---

## Which Tables to Move

**Keep in main:**
- Table 1: significance summary (restricted to primary detector only, not all 5)

**Move to appendix:**
- Table 2: ImageNet-scale 8-cell replication → one sentence + pointer
- Table 3: 100-batch streaming → one sentence + pointer  
- Table 4: frozen vs recalibrated detector validation → one sentence + pointer
- Table 5: Office-Home domain-shift → one sentence + pointer

---

## Writing Quality Issues

### Abstract
**Problem:** Overstuffed, reads like a miniature results section. Too many parentheticals, too many numbers, multiple stacked claims per sentence.

**Rewritten abstract (from GPT-5.4):**

> Test-time adaptation (TTA) is typically evaluated by closed-set accuracy on shifted in-distribution (ID) data, but real deployments often process mixed streams that include out-of-distribution (OOD) inputs. In such streams, entropy-minimizing TTA sharpens predictions on all samples, creating a direct tension with OOD detection, which relies on preserving low confidence on OOD inputs.
> We introduce a *paired contamination protocol* to quantify this tension under a frozen-detector evaluation contract: we compare (i) an **ID-only oracle** that adapts on ID samples only (diagnostic, not deployable) to (ii) **mixed-stream adaptation** on the same test streams, while evaluating a pre-deployed OOD detector that is not updated during adaptation. The resulting *paired gap* isolates the OOD-detection benefit lost specifically to OOD contamination in the adaptation loss.
> Across four OOD datasets, multiple ID fractions, and five detectors, mixed-batch entropy minimization frequently forfeits most of the OOD-detection gain available under gradient-clean adaptation, and in many tested settings leaves OOD detection worse than no adaptation. A four-condition control further attributes the gap primarily to **OOD gradient contamination**, while exposing BatchNorm statistics to OOD inputs *without* including them in the loss is consistently beneficial. We replicate the effect at ImageNet scale and in sequential streams. Finally, a cross-method audit shows that closed-set accuracy improvements can mis-rank methods under mixed-stream OOD safety, motivating reporting the paired gap alongside standard TTA metrics for open-world deployment.

### Introduction
**Problem:** Repeats the conflict (entropy sharpens OOD) three times in different words. Say it once, move to protocol.

### Terminology overload
- "paired contamination protocol," "frozen-detector contract," "forfeit fraction," "gradient-clean benefit," "ml condition" — too many terms
- **"ml" condition name is cryptic** — reads like "machine learning condition." Must rename.

### Section 4
**Problem:** Reads like a lab notebook, not a paper section. No hierarchy between "core" and "supporting" claims.

**Rewritten §4 opening (from GPT-5.4):**

> In Section 3 we defined the paired contamination gap as a diagnostic for a concrete deployment failure: when adaptation is driven by entropy minimization on mixed ID/OOD batches, the same updates that improve closed-set classification can erase the uncertainty signal needed for OOD detection. We now quantify how large this penalty is in practice, and how it varies with contamination level.
> Our goal in this section is not to exhaustively benchmark every detector and dataset combination, but to answer a simple question: **how much OOD-detection benefit is available under gradient-clean adaptation, and how much of it is lost when the same method adapts on mixed batches?**
> We instantiate the protocol with entropy-minimizing BatchNorm adaptation (TENT and ETA) and evaluate OOD detection using pre-calibrated, frozen detectors. We vary the ID fraction α to control contamination severity and report ΔAUROC relative to the pretrained model, together with the paired gap between the ID-only oracle and mixed-stream adaptation. The remaining sections and appendix extend this core result across detectors, datasets, sequential streams, and ImageNet-scale settings.

### Statistical framing
- n=4 Spearman/Kendall correlations: must be framed as "case study illustration," NOT as a robust statistical finding. Current language ("reveals rank discordance") is too strong.
- ">100% of gap" percentages from mechanism decomposition: mathematically valid (BN term is negative/cancelling) but reads like statistical puffery. Must explain cancellation explicitly.

---

## Condition Name Suggestions

| Current | Suggested | Notes |
|---------|-----------|-------|
| no_tta | **No-Adapt** | Clear in captions and prose |
| id_only | **ID-Oracle** | Makes "oracle/diagnostic" status clear |
| mixed | **Mixed-Adapt** | Self-explanatory |
| id_sub | **ID-Subsample** | Clarifies it's the subsample oracle |
| id_full | **ID-Padded** | Clarifies it's count-matched |
| ml | **BN-Expose (Grad-Clean)** or **All-Fwd / ID-Loss** | "ml" is unreadable |

---

## Contribution #1 Rewrite

**Current (too dense with numbers):**
"We define a contamination cost GAP = DeltaA(id_only) - DeltaA(mixed)... Across 4 OOD datasets, 2 TTA methods, 4 ID fractions alpha, and 3 OOD detectors, the gap is significant in 27/32 cells (p<0.05 BH-corrected); FORFEIT>1 in most cells..."

**Rewritten (from GPT-5.4):**
> **Paired contamination protocol.** We propose a simple diagnostic to measure open-world OOD-safety under mixed test streams. For each test stream, we run the same adaptation method in two paired conditions: an **ID-only oracle** that computes the adaptation loss on ID samples only (diagnostic, not deployable), and a **mixed-stream** condition that computes the same loss on the full ID+OOD batch. Under a frozen-detector evaluation contract—where the OOD detector is calibrated once pre-deployment and not updated during adaptation—we report the resulting *paired contamination gap* in OOD detection (ΔAUROC(ID-Oracle) − ΔAUROC(Mixed-Adapt)) as a direct measure of OOD contamination cost. In our experiments this gap is positive in most tested settings, indicating that mixed-batch entropy minimization forfeits substantial OOD-detection gain available under gradient-clean adaptation.

---

## Top 5 Impactful Writing Improvements (Priority Order)

1. **Rewrite abstract** (highest ROI) — use GPT-5.4 rewrite above
2. **Kill intro repetition** — state conflict once; make contributions one-line each; push numerics to captions
3. **Rename condition labels** — especially "ml" → "BN-Expose (Grad-Clean)"; adopt consistent readable names throughout
4. **Restructure §4 as hierarchical** — one primary experiment + pointers to appendix; use rewritten opening above
5. **Fix statistical framing** — label n=4 correlations as case study; explain >100% percentages explicitly

---

## Claim Caution Flags

- **"Consistently outperforms"**: has explicit exceptions (α=0.25, ViM mostly null). Use "in most tested settings."
- **"relevant contract" (frozen detector)**: needs 1 paragraph on when this contract is realistic vs not.
- **Rank discordance at n=4**: treat as illustrative case study, not a finding with statistical weight.
- **ml beats id_only (4.4×)**: currently undersold. This is the most interesting mechanistic insight — BN exposure is helpful context, gradient is the hazard. Could be a clearer headline.
