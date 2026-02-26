# Reject Inference Project — Deep Conclusion, Reconciled Theory, and Next Steps (Max Detail)

**Date:** 2026-02-11  
**Scope:** Home Credit Default Risk (Kaggle) — methodological comparison of reject inference routes  
**Primary outcome:** Retrieve “diamonds” (good borrowers) among the rejected population with maximal precision at top-k selection.

---

## Table of Contents
1. [Project Context and Objective](#1-project-context-and-objective)  
2. [Data, Split Logic, and Populations](#2-data-split-logic-and-populations)  
3. [Routes Compared (A–D) — What Was Built](#3-routes-compared-a–d--what-was-built)  
4. [Results Summary — What Won and by How Much](#4-results-summary--what-won-and-by-how-much)  
5. [EDA: Why Preprocessing Hurt (Root Cause Evidence)](#5-eda-why-preprocessing-hurt-root-cause-evidence)  
6. [Critique Resolution: What Was Overstated and How We Fix It](#6-critique-resolution-what-was-overstated-and-how-we-fix-it)  
7. [Final Integrated Conclusion](#7-final-integrated-conclusion)  
8. [Production Recommendations](#8-production-recommendations)  
9. [Research-Grade Extensions (Fair Neural Comparison)](#9-research-grade-extensions-fair-neural-comparison)  
10. [Concrete Next Steps Plan (Engineering + Science)](#10-concrete-next-steps-plan-engineering--science)  
11. [Risks, Failure Modes, and Mitigations](#11-risks-failure-modes-and-mitigations)  
12. [Appendix A — “Problematic Claims” → “Corrected Wording”](#appendix-a--problematic-claims--corrected-wording)  
13. [Appendix B — Reproducibility Checklist](#appendix-b--reproducibility-checklist)  

---

## 1. Project Context and Objective

### 1.1 The real-world problem being simulated
In credit, we observe labels (default / non-default) mostly for **approved** applicants. For **rejected** applicants, we typically do not observe outcomes (because they never received a loan). This creates a bias: the training data is not representative of the full applicant population.

**Reject inference** attempts to use the approved-labeled population to infer outcomes for rejects, improving selection decisions and risk estimation.

### 1.2 Objective (operational metric)
The goal is not “highest AUC on a test set” in isolation; it is:

- **Identify as many truly good borrowers (“diamonds”) among rejects as possible**
- at a fixed review/approval budget (e.g., top 20% of rejects),
- maximizing **Precision@k** and lift over random.

This is explicitly implemented in the project’s “diamond recovery” analysis workflow. (See working plan phases and evaluation section.) fileciteturn0file3L1-L1

---

## 2. Data, Split Logic, and Populations

### 2.1 Dataset
- **Home Credit Default Risk** (Kaggle)  
- ~307k applications, ~132 primary features (mix of numerical + categorical)  
- ~8% default rate overall (class imbalance)

### 2.2 Populations and scorecard-based reject simulation
The project simulates approval vs rejection using a **scorecard proxy** based on `EXT_SOURCE_2` cutoff (30th percentile), splitting the population into:

- **D_L (Approved / Labeled):** outcomes visible for training/validation/testing  
- **D_R (Rejected / Unlabeled in training):** labels hidden during training, used only for final evaluation of “diamond recovery”

The working plan specifies the exact split validation steps and leakage checks. fileciteturn0file3L42-L42

### 2.3 Empirically used sizes (from the implemented routes)
From diamond recovery analysis in the plan:
- **D_R size:** 92,715 rejected  
- **Composition:** ~86.6% diamonds, 13.4% defaults  

(These values are referenced in the Route A/B summary and diamond recovery sections.) fileciteturn0file3L165-L165

---

## 3. Routes Compared (A–D) — What Was Built

This project compares multiple methodological routes for reject inference. The key difference is whether representation learning (VIME) improves reject inference vs a classical strong baseline (CatBoost on raw features).

### 3.1 Route A — VIME embeddings + CatBoost teacher/student + Isolation Forest filtering
**Pipeline:**
1. Preprocess for VIME:
   - add missing indicators
   - median imputation
   - one-hot encoding categoricals
   - MinMax scaling  
2. Self-supervised VIME training on unlabeled population (D_L_train + D_R)
3. Extract 128-d embeddings for all splits
4. Train CatBoost teacher on D_L embeddings + labels
5. Pseudo-label D_R, filter with Isolation Forest, tune pseudo-label threshold
6. Train student CatBoost on (D_L + filtered D_R)

**Key implementation details** are enumerated phase-by-phase in the working plan. fileciteturn0file3L105-L105

### 3.2 Route B — Classical baseline: CatBoost on raw features (native categorical + missing handling)
**Pipeline:**
1. Use raw tabular features (132)
2. Train CatBoost teacher on D_L raw features + labels
3. Pseudo-label all D_R (no IF filtering in classical variant)
4. Train final CatBoost on augmented set (D_L + all D_R pseudo-labels)

This is the strongest “practical baseline” and wins consistently. fileciteturn0file3L189-L189

### 3.3 Route C — Hybrid: raw features + VIME embeddings
**Hypothesis:** embeddings add complementary signal beyond raw features.

**Implementation:** concatenate 128 embedding features with 132 raw features → 260 total.

Outcome: embeddings slightly hurt D_L performance; no meaningful gain on D_R. fileciteturn0file3L233-L233

### 3.4 Route D — VIME as learned imputer for income (AMT_INCOME_TOTAL)
**Hypothesis:** VIME reconstruction can impute income for rejects to improve downstream modeling.

Outcome: income reconstruction correlation on D_R is extremely weak (0.07), therefore not viable. fileciteturn0file3L251-L251

---

## 4. Results Summary — What Won and by How Much

### 4.1 Core headline
**Route B (raw features + CatBoost) wins decisively.**  
Routes that depend on VIME preprocessing and embeddings underperform.

### 4.2 Consolidated results table (as recorded)
| Route | Approach | Test AUC (D_L) | AUC on D_R | Outcome |
|---|---|---:|---:|---|
| A | CatBoost on VIME embeddings | 0.6912 | 0.6947 | Baseline VIME |
| B | CatBoost on raw features | **0.7391** | **0.7206** | **Best overall** |
| C | Raw + embeddings | 0.7331 | 0.7216 | Embeddings not helpful |
| D | VIME income imputation | N/A | N/A | Not viable on D_R |

Source: extended routes summary in the working plan. fileciteturn0file3L257-L257

### 4.3 Diamond recovery: precision at top 20%
- Route A Top 20% precision: **95.5%** (1.10x lift), **17,700 diamonds** recovered  
- Route B Top 20% precision: **96.5%** (1.11x lift), **17,885 diamonds** recovered  
- Route B wins at all k% thresholds (per plan summary). fileciteturn0file3L208-L208

---

## 5. EDA: Why Preprocessing Hurt (Root Cause Evidence)

This section explains *why* VIME-based routes underperformed, using evidence from the preprocessing impact EDA.

### 5.1 The central issue: informative missingness + median imputation
Key features with high predictive value also have high missingness. Median imputation destroys the relationship between:
- “value is missing” (often meaningfully related to applicant risk profile)
- and default probability.

EDA quantified correlation loss after VIME-style preprocessing:
- EXT_SOURCES_PROD: ~49% correlation loss
- EXT_SOURCE_1: ~41% correlation loss
- APARTMENTS_AVG: ~62% correlation loss

These are presented explicitly as “most damaged features” in the EDA. fileciteturn0file2L17-L17

### 5.2 Why missing indicators alone didn’t fix it (for key predictors)
Missing indicators preserved *some* signal, but for the strongest features (EXT_SOURCE family) they only recovered a small fraction:
- Indicator recovers ~10–20% of lost signal in key EXT features
- leaving a large net loss

EDA’s “Signal Recovery Analysis (Value + Indicator vs Raw)” shows this pattern. fileciteturn0file2L61-L61

### 5.3 MinMax scaling + extreme outliers = variance collapse
Example: `AMT_INCOME_TOTAL` has extreme max (~117M) vs median (~135k).
MinMax scaling compresses 99% of observations into a tiny band near zero, causing:
- very low variance
- reduced reconstruction learning signal under MSE
- downstream model ignoring income-like features

EDA gives the concrete outlier ratio (max/99th ~226x) and notes variance rank near bottom. fileciteturn0file2L94-L94

### 5.4 One-hot encoding harms representation learning on high-cardinality categoricals
Categorical features like `ORGANIZATION_TYPE` (58 categories) become 58 sparse binary columns.
This spreads signal across many low-variance columns and makes reconstruction harder.

EDA compares one-hot vs CatBoost-style target/categorical handling. fileciteturn0file2L170-L170

### 5.5 Summary: why Route B’s input pipeline is inherently advantaged
Route B avoids the three big distortions:
1. No median imputation of informative NaNs *as values*
2. No MinMax scaling collapse from outliers
3. No sparse one-hot explosion for categoricals

This is exactly aligned with the Route B win observed empirically. fileciteturn0file2L229-L229

---

## 6. Critique Resolution: What Was Overstated and How We Fix It

The critique identified that the earlier conclusion document made **sweeping theoretical claims** about neural networks that are not accurate in modern deep learning literature.

### 6.1 The specific overclaim
Original doc claimed (paraphrased):
- “NNs cannot process NaN; imputation necessarily destroys information; trees are structurally superior.”

The critique correctly notes:
- This is true for simple MLP + median imputation pipelines
- but false for modern tabular DL that uses tokenization/masking/learnable missing embeddings
- and the correct framing is “structurally fit” and “effort-to-accuracy ratio,” not impossibility.

See critique sections: “NNs Cannot Process NaN Fallacy” and “Nuance on Structural Superiority.” fileciteturn0file1L14-L14

### 6.2 What we keep (still correct)
- The practical conclusion “use CatBoost” for this project is correct.
- The empirical argument “our implemented NN route underperforms” is correct.
- The EDA that shows preprocessing damage is correct.

### 6.3 What we change (corrected scientific framing)
We replace:
- “NNs are structurally incapable”  
with:
- “Standard NN pipelines require careful missingness-aware modeling; absent that, their preprocessing can destroy signal and make them less data-efficient than GBDTs in this regime.”

This aligns with the critique’s recommendation to soften language and cite pragmatics. fileciteturn0file1L94-L94

---

## 7. Final Integrated Conclusion

### 7.1 Project-level conclusion (what the project proves)
On Home Credit reject inference under the implemented methodology:
- **CatBoost on raw features (Route B) is the best-performing approach.**
- VIME embeddings do not improve reject inference and can degrade performance.
- Hybridizing embeddings with raw features does not yield significant benefit.

This is backed by both AUC and diamond recovery metrics. fileciteturn0file3L257-L257

### 7.2 Scientific conclusion (how to interpret the win)
The win is best explained as:
- **signal preservation + inductive bias + data efficiency**, not “NN impossibility.”

**Why trees win here:**
- They exploit missingness patterns without forcing median imputation into the value channel
- They handle categorical features without sparse explosion
- They work extremely well in medium-scale tabular regimes with limited tuning

**Why the implemented NN route lost:**
- preprocessing damaged high-value predictors (EDA)
- embeddings were frozen/bottlenecked (128 dims)
- SSL objective (reconstruction) may denoise precisely the “missingness is risk” signal

Critique supports this interpretation explicitly. fileciteturn0file1L39-L39

### 7.3 Updated “final conclusion” text (safe to publish)
> For medium-scale tabular credit-risk data with informative missingness, a CatBoost model trained on minimally processed raw features provides superior reject inference performance versus a VIME-based representation learning pipeline, primarily because the latter requires preprocessing transformations (median imputation, scaling, and one-hot encoding) that measurably degrade predictive signal in key features.  
>  
> This result should be interpreted as a practical advantage in this regime (inductive bias and efficiency), rather than a general structural impossibility of neural networks, since modern missingness-aware tabular DL architectures can model missingness natively when implemented appropriately.

This replaces the overgeneralized phrasing in the earlier conclusion. fileciteturn0file0L1-L1

---

## 8. Production Recommendations

### 8.1 Recommended production model
- **Route B: CatBoost on raw features**
- Use native categorical handling and missingness handling
- Calibrate probability outputs on D_L_test
- Monitor PSI drift and default rate calibration on deployed population

### 8.2 Recommended production metrics
- ROC-AUC + PR-AUC (class imbalance)
- Precision@k for reject ranking (business)
- Lift over random
- Calibration curves / Brier score
- PSI for drift

### 8.3 Operational thresholding
Pick threshold by business constraints:
- max default rate tolerated
- review capacity
- cost of false positives vs false negatives

---

## 9. Research-Grade Extensions (Fair Neural Comparison)

If the project wants to make a defensible claim about “NN vs GBDT,” the neural baseline must be modern and end-to-end.

### 9.1 Replace VIME preprocessing with missingness-native modeling
Options:
- Feature tokenization with a learnable “missing token” embedding
- Masked modeling where missing features are *masked*, not median-filled
- Robust scaling (RobustScaler / log transforms) instead of MinMax with extreme outliers

### 9.2 End-to-end fine-tuning (no frozen bottleneck)
Instead of using VIME as a fixed feature extractor:
- attach classification head
- fine-tune encoder on D_L labels (and optionally pseudo-labels)

This addresses the critique’s main methodological point. fileciteturn0file1L47-L47

### 9.3 Objective alignment
Reconstruction may conflict with “missingness is predictive.” Try:
- contrastive objectives
- masked feature prediction with weighting
- auxiliary loss that preserves missingness signal explicitly

---

## 10. Concrete Next Steps Plan (Engineering + Science)

This is a “do-next” list aligned to the working plan phases.

### 10.1 Finish Phase 6 evaluation pack (mandatory)
Deliverables:
1. PR curves and PR-AUC for D_R
2. Lift curves at multiple k%
3. SHAP analysis (Route B)
4. Consolidated comparison report tables/figures

Reference: Phase 6 tasks in the working plan. fileciteturn0file3L286-L286

### 10.2 Refactor documentation into final-report structure (Phase 7)
Deliverables:
- Final report with: Abstract → Method → Results → Discussion → Conclusion
- Replace overclaims with corrected wording (Appendix A below)
- Add “limitations” section explicitly describing what was and wasn’t tested

Reference: Phase 7 tasks in the working plan. fileciteturn0file3L309-L309

### 10.3 Optional: run “fair NN” baseline (research extension)
Deliverables:
- Implement missingness-token tabular transformer baseline
- End-to-end fine-tune vs CatBoost
- Compare on identical splits and diamond metrics

This should be framed as “Extension study,” not required for the core project conclusion.

---

## 11. Risks, Failure Modes, and Mitigations

### 11.1 Reject simulation mismatch
Risk: using EXT_SOURCE_2 cutoff may not match real bank reject policy.  
Mitigation: test alternative reject policies or multi-feature scorecard simulation.

### 11.2 Pseudo-label noise
Risk: training on pseudo-labels can propagate teacher errors.  
Mitigation: confidence filtering, soft labels, calibration, or semi-supervised techniques.

### 11.3 Leakage via encoding
Risk: target encoding can leak if not properly done.  
Mitigation: use CatBoost’s built-in handling or strict CV target encoding.

### 11.4 Overclaiming scientific results
Risk: reviewers reject sweeping claims about deep learning.  
Mitigation: keep claims scoped to “implemented routes” and “this regime.”

---

## Appendix A — “Problematic Claims” → “Corrected Wording”

### A.1 “NNs cannot process NaN”
**Problematic:** “Neural networks cannot process NaN values.”  
**Corrected:** “Standard MLP pipelines cannot ingest NaNs directly and often rely on imputation; however, modern tabular DL can represent missingness via masking or learnable missing embeddings.” fileciteturn0file1L14-L14

### A.2 “Imputation necessarily destroys information”
**Problematic:** “Imputation necessarily destroys information.”  
**Corrected:** “Median imputation in this project measurably reduced predictive signal in key features; missingness-aware modeling can preserve the signal more faithfully.” fileciteturn0file2L17-L17

### A.3 “Trees are structurally superior”
**Problematic:** “Tree-based models are structurally superior.”  
**Corrected:** “GBDTs are structurally well-suited and more data-efficient for medium-scale tabular data with informative missingness; NNs may be competitive with specialized architectures and additional tuning/scale.” fileciteturn0file1L79-L79

---

## Appendix B — Reproducibility Checklist

### B.1 Data & splits
- [ ] Dataset version pinned and hashed
- [ ] EXT_SOURCE_2 cutoff value stored in metadata
- [ ] D_L/D_R sizes + default rates logged
- [ ] No ID overlap between train/val/test

### B.2 Training & evaluation
- [ ] Seeds fixed (numpy/torch/catboost)
- [ ] Exact feature list versioned
- [ ] Metrics computed on:
  - D_L_test
  - D_R_ground_truth
  - full population union

### B.3 Artifacts
- [ ] Models serialized + configs saved
- [ ] Figures saved (ROC, PR, lift, SHAP)
- [ ] Final report references exact artifact filenames

(Structure and QA items correspond to “⚙️ QA Checks” in the working plan.) fileciteturn0file3L79-L79
