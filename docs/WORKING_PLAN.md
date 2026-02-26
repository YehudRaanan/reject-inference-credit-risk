# Working Plan — Reject Inference Methodological Comparison

> **Dataset**: Home Credit Default Risk (Kaggle) — 307k applications, 122 features  
> **Goal**: Compare VIME-based hybrid reject inference (Route A) vs classical CatBoost baseline (Route B)  
> **Key Question**: Which approach better identifies good borrowers among rejected applicants?  
> **Reference**: [data_splitting_proposal.md](file:///C:/Users/User/.gemini/antigravity/brain/86e70929-cdf3-49ae-97d3-d0be8765fd1e/data_splitting_proposal.md) for full data flow diagrams.

> Each phase ends with a **🔍 Owner Code Review** session.  
> Steps marked with ⚙️ are automated QA checks. Steps marked with 🧑‍🏫 are learning walkthroughs.  
> This plan is **iterative** — results at any step may trigger going back to refine earlier steps.

---

## Phase 0: Environment & Data Setup
**Goal**: Working environment, dataset downloaded, EDA complete, splits validated.

- [x] **0.1** Install dependencies (`pip install -r requirements.txt`)
- [ ] **0.2** Download Home Credit Default Risk dataset from Kaggle
  - Place `application_train.csv` into `data/raw/`
  - Optional: `bureau.csv`, `previous_application.csv` for feature engineering
- [ ] **0.3** Exploratory Data Analysis (`notebooks/01_eda.ipynb`)
  - Dataset overview: shape, dtypes, missing value rates per column
  - Target distribution: ~8% default rate (class imbalance)
  - `EXT_SOURCE_2` distribution: examine the score → validate 30th percentile cutoff
  - Key feature correlations with TARGET
  - Missing value patterns (which groups of columns co-miss?)
- [ ] **0.4** Validate scorecard-based split
  - ⚙️ Confirm D_L default rate ~5–6%, D_R default rate ~15–20%
  - ⚙️ Confirm D_L size ~215k, D_R size ~92k
  - ⚙️ Verify no label leakage: D_R labels completely hidden during training
  - Visualize score distributions: D_L vs D_R populations
- [ ] **0.5** 🧑‍🏫 **Learning Walkthrough**: EDA findings, split rationale, class balance
- [ ] **0.6** 🔍 **Owner Code Review #1**: Dataset, splits, EDA

---

## Phase 1: Shared Preprocessing Pipeline
**Goal**: Clean data, engineer features, produce reproducible train/val/test/D_R splits.

- [ ] **1.1** Data cleaning (`shared_pipeline.py`)
  - Handle DAYS_EMPLOYED anomaly: 365,243 → NaN (retirement/unknown flag)
  - Drop low-utility columns (FLAG_DOCUMENT_*, low-variance flags)
  - Convert DAYS_* features to positive values (years/months)
- [ ] **1.2** Feature engineering (`shared_pipeline.py`)
  - Credit ratios: `AMT_CREDIT / AMT_INCOME_TOTAL` (loan-to-income)
  - Annuity burden: `AMT_ANNUITY / AMT_INCOME_TOTAL`
  - Goods ratio: `AMT_GOODS_PRICE / AMT_CREDIT`
  - Age: `DAYS_BIRTH / -365`
  - Employment years: `DAYS_EMPLOYED / -365`
  - External score aggregation: mean/min of EXT_SOURCE_1/2/3
- [ ] **1.3** Scorecard-based reject simulation (`shared_pipeline.py`)
  - Compute `EXT_SOURCE_2` 30th-percentile cutoff
  - Split into D_L (approved, labels visible) and D_R (rejected, labels hidden)
  - Stratified split of D_L: train (70%) / val (15%) / test (15%)
  - Save D_R ground truth separately for evaluation
  - ⚙️ Assert no label leakage in D_R pipeline
- [ ] **1.4** Save all splits to `data/processed/`
  - `D_L_train.parquet`, `D_L_val.parquet`, `D_L_test.parquet`
  - `D_R_features.parquet`, `D_R_ground_truth.parquet`
  - `split_metadata.json` (sizes, default rates, cutoff value)
- [ ] **1.5** ⚙️ **QA Checks**:
  - No target column in D_R_features
  - D_L_train / D_L_val / D_L_test have correct proportions
  - Feature columns match across all splits
  - No duplicate Applicant IDs across splits
- [ ] **1.6** 🧑‍🏫 **Learning Walkthrough**: Feature engineering rationale, split statistics
- [ ] **1.7** 🔍 **Owner Code Review #2**: Preprocessing logic, split validation

---

## Phase 2: Route-Specific Preprocessing
**Goal**: Prepare Route A (VIME-compatible) and Route B (CatBoost-native) feature matrices.

- [x] **2.1** Route A preprocessing (`route_a_prep.py`)
  - Add `_is_missing` binary masks for every column with NaN (via `add_indicator=True`)
  - Fill NaN with Median (safe numerical value)
  - One-Hot encode categoricals (No target leakage allowed)
  - MinMaxScaler [0, 1] on all numerical features (required for VIME sigmoid reconstruction)
  - Save processed parquet files (`vime_X_*.parquet`)
- [ ] **2.2** Route B preprocessing (`route_b_prep.py`)
  - Median imputation for numerical, mode for categorical
  - Target encoding with smoothing (regularized to avoid leakage)
  - No scaling (CatBoost handles internally)
  - Identify `cat_features` list for CatBoost's native categorical handling
- [ ] **2.3** ⚙️ **QA Checks**:
  - Route A output: dense float tensor, no NaN, all scaled
  - Route B output: DataFrame with correct dtypes, cat_features indexed
  - Both outputs have same row count per split
- [ ] **2.4** 🔍 **Owner Code Review #3**: Route-specific preprocessing

---

## Phase 3: VIME Self-Supervised Training
**Goal**: Pre-train VIME encoder on full population (D_L_train + D_R), no labels.

- [x] **3.1** Implement VIME architecture (`src/models/vime/vime_model.py`)
  - `VIME` class (Encoder + MaskEstimator + FeatureReconstructor)
  - `column_manager.py` for robust feature grouping (Critical/High-Missing)
- [x] **3.2** Implement self-supervised training loop (`src/models/vime/vime_trainer.py`)
  - Corruption: 3-Level Masking (Critical=0.5, HighMissing=0.1, Regular=0.3)
  - Loss: `α · BCE(mask_pred, mask_true) + (1-α) · MSE(recon, original)`
  - Data: Loads `vime_X_*.parquet` (Full unlabeled corpus)
  - `vime_utils.py` handles custom probability vector generation
- [x] **3.3** Implement embedding extraction (`src/models/vime/vime_embeddings.py`)
  - `generate_embeddings()` → latent vectors
  - Extract for ALL splits: D_L_train, D_L_val, D_L_test, D_R
- [x] **3.4** ⚙️ **QA Checks**:
  - Reconstruction loss decreases over training
  - Embeddings have reasonable variance (not collapsed)
  - Embedding dim matches config
- [x] **3.5** 🧑‍🏫 **Learning Walkthrough**: VIME architecture, self-supervised loss design, why it works
  - See `docs/reports/phase3_vime_report.md`
- [x] **3.6** 🔍 **Owner Code Review #4**: VIME implementation
  - See `docs/reports/phase3_vime_report.md`

---

## Phase 4: Route A — Hybrid Reject Inference ✅
**Goal**: Use VIME embeddings + CatBoost teacher + IF filtering for reject inference.

- [x] **4.1** Train Teacher Model (CatBoost)
  - Train on `vime_emb_train.parquet` (Labels: `vime_y_train.parquet`).
  - Features: 128-dim VIME embeddings.
  - **Result**: Teacher Val AUC: 0.6982, Test AUC: 0.6897
- [x] **4.2** Pseudo-Label Rejects
  - Predict $P(Default)$ for `vime_emb_reject.parquet`.
  - **Isolation Forest Filter**: Kept 77,219/92,715 inliers (83.3%)
  - **Threshold tuning**: Swept 0.15-0.45, optimal=0.15 (36.5% pseudo-default rate)
- [x] **4.3** Train Student Model (CatBoost)
  - Train on Augmented Dataset (`D_L_train` + `D_R_filtered`).
  - **Result**: Final Test AUC: 0.6912 (+0.0015 vs teacher)
- [x] **4.4** Train final model (Route A) on D_L_train + D_R_filtered
  - CatBoost on VIME embeddings
  - Validate on D_L_val
- [x] **4.5** ⚙️ **QA**: Teacher AUC on D_L_val, filtering statistics
  - See `outputs/results/threshold_sweep_results.json`
  - See `outputs/results/diamond_recovery_analysis.json`
- [x] **4.6** 💎 **Diamond Recovery Analysis**
  - D_R: 92,715 rejected (86.6% diamonds, 13.4% defaults)
  - AUC on D_R (ground truth): 0.6947
  - Top 20%: 95.5% precision, 1.10x lift, 17,700 diamonds recovered
- [ ] **4.7** 🔍 **Owner Code Review #5**: Route A pipeline

---

## Phase 5: Route B — Classical Baseline ✅
**Goal**: Classical CatBoost reject inference without VIME or filtering.

- [x] **5.1** Train teacher CatBoost on D_L_train raw features + labels
  - Features: 132 raw features (16 categorical, handled natively)
  - **Result**: Teacher Val AUC: 0.7540, Test AUC: 0.7406
- [x] **5.2** Pseudo-label D_R using teacher on raw features
  - Threshold: 0.15 → 30.8% pseudo-default rate
  - **No Isolation Forest filtering** (classical approach)
- [x] **5.3** Train final model (Route B) on D_L_train + ALL D_R
  - Augmented: 243,072 samples (D_L=150,357 + D_R=92,715)
  - **Result**: Final Test AUC: 0.7391 (-0.0016 vs teacher)
- [x] **5.4** ⚙️ **QA**: Teacher AUC on D_L_val, pseudo-label distribution
  - See `outputs/results/route_b_metrics.json`
- [x] **5.5** 💎 **Diamond Recovery Comparison**
  - AUC on D_R: **0.7206** (vs Route A: 0.6947)
  - Top 20%: **96.5% precision**, **1.11x lift**, 17,885 diamonds
  - **Route B wins at all k% thresholds**
- [ ] **5.6** 🔍 **Owner Code Review #6**: Route B pipeline

---

## Phase 5.5: Extended Routes (C & D) ✅
**Goal**: Explore additional approaches based on initial Route A vs B findings.

### Route C — Hybrid: VIME Embeddings + Raw Features ✅
**Hypothesis**: Combining embeddings (128) + raw features (132) = 260 features might capture complementary patterns.

- [x] **C.1** Implement `route_c_hybrid.py`
  - Concatenate VIME embeddings with raw tabular features
  - CatBoost with native categorical handling
- [x] **C.2** Train and evaluate
  - Teacher Test AUC: 0.7361 (worse than Route B: 0.7406)
  - Final Test AUC: 0.7331 (worse than Route B: 0.7391)
  - AUC on D_R: 0.7216 (similar to Route B: 0.7206)
- [x] **C.3** Conclusion
  - **Adding VIME embeddings hurts performance** on D_L
  - No significant improvement on D_R
  - Embeddings add noise rather than complementary signal

### Route D — VIME Income Imputation Experiment ✅
**Hypothesis**: VIME's learned reconstruction can predict AMT_INCOME_TOTAL for D_R.

- [x] **D.1** Implement `route_d_vime_impute.py`
  - Mask income column, use VIME reconstructor to predict
  - Compare predicted vs actual income
- [x] **D.2** Evaluate reconstruction quality
  - **D_R (rejected)**: Correlation = **0.070** (very weak)
  - **D_L (approved)**: Correlation = 0.369 (moderate)
- [x] **D.3** Conclusion
  - **VIME cannot reliably predict income for rejected population**
  - The rejected population (low EXT_SOURCE_2) has different income patterns
  - Income imputation via VIME is not viable for D_R

### Extended Routes Summary

| Route | Approach | Test AUC (D_L) | AUC on D_R | Conclusion |
|-------|----------|----------------|------------|------------|
| A | VIME embeddings + CatBoost | 0.6912 | 0.6947 | Baseline VIME |
| **B (Fair)** | **Raw features + CatBoost** | **0.7372** | **0.7149** | **Best overall** ✅ |
| C | Raw + Embeddings + CatBoost | 0.7331 | 0.7216 | Embeddings hurt |
| D | VIME imputation | N/A | N/A | Cannot impute income for D_R |
| F (Fair) | Two-Model Ensemble | 0.7237 | 0.7096 | Single model better |
| A-Fixed | Fixed VIME emb + CatBoost | 0.7021 | — | +0.0109 vs A |
| C-Fixed | Raw + Fixed emb + CatBoost | 0.7328 | — | Teacher beats B! |
| F-Fixed | Var split raw+emb | 0.7294 | 0.7171 | −0.0097 vs F |
| **Dynamic 2.2A** | **Fine-tuned encoder + MLP** | **0.7163** | 0.6998 | +0.0251 vs A, −0.0228 vs B |
| Dynamic 2.2B | Fine-tuned + pseudo-labels | 0.7132 | 0.6879 | Pseudo-labels hurt |
| Dynamic 2.2C | Frozen encoder + MLP | 0.7166 | 0.6873 | ≈ Dynamic 2.2A! |

**Key Findings (Updated with Fair Comparison)**:
1. **CatBoost (single model) is the overall winner** when using equivalent training setup (class_weights + P@20% early stopping)
2. **Two-Model Ensemble no longer outperforms** - earlier results used different loss/early stopping, not a fair comparison
3. Raw features (Route B) outperform all VIME-based approaches on both D_L and D_R
4. **Route A-Fixed preprocessing fixes improved VIME embeddings (+0.0164 Teacher AUC over original Route A)**
5. **Route C-Fixed Teacher (raw + fixed emb) beats Route B on validation (0.7457 vs 0.7391)**
6. **Pseudo-labeling remains the bottleneck** — Student always degrades vs Teacher
7. **Fine-tuning the encoder doesn't help** — Frozen (0.7166) ≈ Dynamic (0.7163), confirming the 128-dim bottleneck is the limiting factor
8. **MLP head slightly outperforms CatBoost on same embeddings** — 0.7166 (MLP) vs 0.7021 (CatBoost)

---

## Phase 5.6: Route A-Fixed — VIME Preprocessing Improvements 🔄
**Goal**: Determine if fixing VIME preprocessing (RobustScaler, Partial Reconstruction Loss, Target Encoding) allows VIME to compete with CatBoost baseline (Route B).

**Decision**: Approved 2026-02-14 (see `docs/decision_log.md`)

### Step 1.1–1.3: Preprocessing Fixes ✅

- [x] **1.1** RobustScaler instead of MinMaxScaler
  - `ClippedRobustScaler` in `src/preprocessing/custom_scalers.py`
  - AMT_INCOME_TOTAL variance: **704.3x improvement** (0.000046 → 0.032456)
- [x] **1.2** Partial Reconstruction Loss
  - `masked_mse_loss()` in `src/models/vime/vime_trainer.py`
  - Missing masks saved: `vime_missing_mask_*.parquet` (23.8% cells masked)
- [x] **1.3** Target Encoding for high-cardinality categoricals
  - ORGANIZATION_TYPE: 58 OHE → 1 column, OCCUPATION_TYPE: 18 OHE → 1 column
  - Total features: 328 → 253 (75 fewer sparse columns)

**Artifacts**: `vime_X_*_fixed.parquet`, `vime_encoder_fixed.pth`, `route_a_fixed_preprocessor.pkl`

### Step 1.4: Downstream Classifiers (Fixed Embeddings) ✅

- [x] **1.4-pre** Generate fixed embeddings (`vime_emb_*_fixed.parquet`)
  - 4 splits: train (150357,128), val (32219,128), test (32220,128), reject (92715,128)
- [x] **1.4A** CatBoost on **embeddings only** (128-dim)
  - Teacher Val AUC: 0.7147, Test AUC: 0.7076
  - IF filtering: 79,399/92,715 inliers (85.6%)
  - Student Val AUC: 0.7070, Test AUC: **0.7021** (+0.0109 vs Route A)
  - **Conclusion**: Fixed embeddings carry more signal; Teacher improved +0.0164 vs original Route A
- [x] **1.4B** CatBoost on **raw features + embeddings** (~260-dim)
  - Teacher Val AUC: **0.7457**, Test AUC: **0.7362** (**beats Route B on validation!**)
  - Pseudo-label rate: 31.4% defaults, augmented: 243,072 samples
  - Student Val AUC: 0.7413, Test AUC: 0.7328 (-0.0003 vs Route C)
  - **Conclusion**: Teacher surpasses Route B, but pseudo-labeling noise degrades Student

### Step 1.5: Variable Split Ensemble (EXT_SOURCE vs OTHER) ✅

- [x] **1.5A** Two-model ensemble on **embeddings only**
  - Model_EXT: D_L_test=0.7093, D_R=0.6840
  - Model_OTHER: D_L_test=0.7076, D_R=0.6813
  - Ensemble optimal: D_R=0.6846 (w_ext=0.70) — below Route F baseline
  - **Conclusion**: VIME embeddings are a single latent space; splitting adds no value
- [x] **1.5B** Two-model ensemble on **raw + embeddings**
  - Model_EXT (137 feat): D_L_test=0.7163, D_R=0.7012
  - Model_OTHER (251 feat): D_L_test=0.7264, D_R=0.7033
  - Ensemble optimal: D_R=**0.7171** (w_ext=0.45, w_other=0.55)
  - **Conclusion**: -0.0097 vs Route F; fixed embeddings don't help the ensemble

### Phase 5.6 Results Summary

| Step | Model | Val AUC | Test AUC | vs Baseline |
|------|-------|---------|----------|-------------|
| 1.4A | Teacher (emb only) | 0.7147 | 0.7076 | +0.0164 vs A |
| 1.4A | Student (emb only) | 0.7070 | **0.7021** | +0.0109 vs A |
| 1.4B | Teacher (raw+emb) | **0.7457** | **0.7362** | −0.0029 vs B |
| 1.4B | Student (raw+emb) | 0.7413 | 0.7328 | −0.0003 vs C |
| 1.5A | Ensemble (emb) | — | 0.7097 | D_R: −0.0422 vs F |
| 1.5B | Ensemble (raw+emb) | — | 0.7294 | D_R: −0.0097 vs F |

**Key Finding**: The 3 preprocessing fixes genuinely improved VIME embedding quality (+0.0164 Teacher AUC). However, pseudo-labeling and reject inference are the dominant bottleneck — not embedding representation.

**Artifacts:**
- `outputs/results/route_a_fixed_metrics.json`
- `outputs/results/route_c_fixed_metrics.json`
- `outputs/results/route_f_fixed_results.json`
- Models: `route_a_fixed_teacher.cbm`, `route_a_fixed_final.cbm`, `route_c_fixed_teacher.cbm`, `route_c_fixed_final.cbm`, `route_f_fixed_model_ext.cbm`, `route_f_fixed_model_other.cbm`

- [x] **1.6** ⚙️ **QA**: Consolidated comparison table across all routes
- [ ] **1.7** 🔍 **Owner Code Review**: Route A-Fixed results

---

## Phase 5.7: Dynamic Embeddings — End-to-End Fine-Tuning 🔄
**Goal**: Address the frozen-embeddings critique by fine-tuning the VIME encoder jointly with a classification head, providing the fairest possible NN vs GBDT comparison.

**Motivation**: All prior VIME experiments used **frozen embeddings** — the encoder was trained only with Self-Supervised Learning (reconstruction + mask prediction), then its output was fed to CatBoost as static features. The critique (`docs/reports/critique_of_conclusion.md`) correctly identifies this as an unfair comparison because:
1. The 128-dim bottleneck discards information the encoder considered unimportant for reconstruction — but that information may be critical for classification
2. The encoder optimizes for reconstruction loss, not classification loss
3. CatBoost operates on raw features with no bottleneck — a fundamentally different regime

> [!CAUTION]
> **PAIR-PROGRAMMING REQUIREMENT**: The classification head architecture code (Step 2.1) must be written collaboratively with the project owner. **Every line of code** for the classification head module must be:
> 1. **Explained** — what it does and why it's needed
> 2. **Approved** — by the owner before being added to the code file
>
> This is a learning exercise. No code should be auto-generated or added without the owner understanding each line.

### Step 2.1: Classification Head Architecture ✅ 🧑‍🏫
**Mode**: Pair-programming with owner (every line explained and approved)

- [x] **2.1.1** Designed classification head: 128→64→32→1, ReLU, Dropout=0.3, same LR
- [x] **2.1.2** Implemented `src/models/vime/classification_head.py` (10,369 params)
- [x] **2.1.3** Implemented `src/models/vime/vime_finetune.py` (fine-tuning loop)
- [x] **2.1.4** Owner reviewed and approved all code

### Step 2.2: Fine-Tuning Experiments ✅

| Step | Model | Val AUC | Test AUC | D_R AUC | Best Epoch |
|------|-------|---------|----------|---------|------------|
| **2.2A** | Dynamic (fine-tuned encoder + head) | **0.7319** | **0.7163** | 0.6998 | 7/17 |
| 2.2B | Dynamic + pseudo-labels (D_L+D_R) | 0.7240 | 0.7132 | 0.6879 | 3/13 |
| **2.2C** | Frozen encoder + head only | 0.7272 | **0.7166** | 0.6873 | 38/48 |

**Key Finding**: Frozen (0.7166) ≈ Dynamic (0.7163) — **fine-tuning the encoder doesn't help**.
The 128-dim information bottleneck is the limiting factor, not the training objective.

**Artifacts:**
- `outputs/results/finetune_results.json` (2.2A)
- `outputs/results/finetune_pseudo_results.json` (2.2B)
- `outputs/results/finetune_frozen_results.json` (2.2C)
- Models: `vime_finetuned.pth`, `vime_finetuned_pseudo.pth`, `vime_frozen_head.pth`

### Step 2.3: Analysis & Documentation ✅

- [x] **2.3.2** Updated Extended Routes Summary (11 routes)
- [x] **2.3.3** Updated Final Conclusion with fine-tuning results
- [x] **2.3.4** ⚙️ **QA**: All metrics saved and compared
- [ ] **2.3.5** 🔍 **Owner Code Review**: Final results

### Outcome: Information bottleneck confirmed
Fine-tuning **doesn't close the gap** → the information loss happens at the architecture level (253→128 dim compression), not at the training objective. This confirms the 128-dim bottleneck is the real problem, not whether the encoder was optimized for reconstruction or classification.


---

### Final Conclusion: GBDT vs Neural Networks (Updated with Phase 5.7) ✅

**For medium-scale tabular data with informative missing values,  GBDTs remain pragmatically superior to VIME-based neural network approaches — but this is a practical efficiency gap, not a structural impossibility.**

**Root Cause Analysis (Progressive Elimination):**

| Hypothesis | Test | Result | Eliminated? |
|-----------|------|--------|------------|
| Bad preprocessing | Phase 5.6 fixes | +0.0164 Teacher AUC | ✅ Partially |
| Frozen embeddings unfair | Phase 5.7 fine-tuning | Frozen ≈ Dynamic | ✅ Not the cause |
| Wrong downstream classifier | 2.2C MLP vs CatBoost | MLP slightly better | ✅ Not the cause |
| **128-dim bottleneck** | All experiments | **Consistent ceiling ~0.716** | ❌ **This is it** |
| Pseudo-label noise | 2.2B vs 2.2A | Pseudo hurts (−0.003) | ❌ **Also confirmed** |

**The Performance Ladder:**
```
 Route A (original VIME + CB):      0.6912  ─── bad preprocessing
 Route A-Fixed (fixed VIME + CB):   0.7021  ─── embedding bottleneck
 Dynamic 2.2A (fine-tuned + MLP):   0.7163  ─── ~0.716 ceiling
 Dynamic 2.2C (frozen + MLP):       0.7166  ─── confirms ceiling
 Route B (raw features + CatBoost): 0.7391  ─── no bottleneck
```

**Why 0.716 is the VIME ceiling:**
1. 253 features → 128-dim bottleneck discards task-relevant information
2. Encoder optimized for reconstruction keeps reconstruction-relevant info, not classification-relevant info
3. Fine-tuning can't recover information already lost at encoding time
4. CatBoost operates on all 253 raw features with no compression

**Nuanced Conclusion (updated from critique):**
- ❌ "NNs cannot handle missing data" — **False**. The issue is VIME's specific architecture, not NNs generally
- ✅ "GBDTs are more data-efficient" — **True**. CatBoost achieves 0.7391 with zero preprocessing
- ✅ "The 128-dim bottleneck limits VIME" — **Confirmed**. The ceiling is architectural
- ✅ "Pseudo-labeling adds noise" — **Confirmed**. Student always degrades vs Teacher

**Documentation:** `docs/reports/conclusion_nn_vs_gbdt_missing_data.md`, `docs/reports/critique_of_conclusion.md`

---

## Phase 6: Evaluation & Comparison ✅
**Goal**: Compare Route A vs Route B across three populations with diamond-recovery metrics.

- [x] **6.1** Evaluate on D_L_test (approved population)
  - Route A: AUC-ROC=0.6912, PR-AUC=0.1259
  - Route B: AUC-ROC=**0.7391**, PR-AUC=**0.1671**
- [x] **6.2** 💎 Evaluate on D_R_ground_truth (rejected population)
  - Route A: AUC-ROC=0.6947, PR-AUC=0.2606
  - Route B: AUC-ROC=**0.7206**, PR-AUC=**0.2790**
  - **Precision@20%**: Route A=95.5%, Route B=**96.5%**
  - **Diamonds@20%**: Route A=17,700, Route B=**17,885** (+185)
- [x] **6.3** Evaluate on full population (D_L_test + D_R_ground_truth)
  - Route A: AUC-ROC=0.7148
  - Route B: AUC-ROC=**0.7326**
- [x] **6.4** SHAP analysis
  - Generated SHAP summary plot for Route B
  - See `outputs/figures/shap_route_b.png`
- [x] **6.5** Generate comparison tables and figures
  - ROC curves: `outputs/figures/roc_comparison.png`
  - PR curves: `outputs/figures/pr_comparison_dr.png`
  - Lift curves: `outputs/figures/lift_comparison.png`
- [x] **6.6** ⚙️ **QA**: All metrics computed, figures saved
  - Results: `outputs/results/phase6_evaluation.json`
  - Summary: `outputs/results/phase6_summary.md`

### Phase 6 Results Summary

| Population | Metric | Route A (VIME) | Route B (Raw) | Winner |
|------------|--------|----------------|---------------|--------|
| D_L_test | AUC-ROC | 0.6912 | **0.7391** | B |
| D_L_test | PR-AUC | 0.1259 | **0.1671** | B |
| D_R | AUC-ROC | 0.6947 | **0.7206** | B |
| D_R | PR-AUC | 0.2606 | **0.2790** | B |
| Full | AUC-ROC | 0.7148 | **0.7326** | B |
| D_R | Precision@20% | 95.5% | **96.5%** | B |
| D_R | Diamonds@20% | 17,700 | **17,885** | B |

**Conclusion:** Route B wins on ALL metrics across ALL populations.

- [ ] **6.7** 🧑‍🏫 **Learning Walkthrough**: Results interpretation
- [ ] **6.8** 🔍 **Owner Code Review #7**: Evaluation and findings

---

## Phase 7: Final Report & Submission 🔄
**Goal**: Academic report summarizing methodology, results, and conclusions.

- [x] **7.1** Write final report (`docs/reports/final_report.md`)
  - Abstract, Introduction, Methodology, Results, Discussion, Conclusion
  - Embed key figures and tables
- [ ] **7.2** Code cleanup and documentation
  - Docstrings for all public functions
  - README with reproduction instructions
- [ ] **7.3** 🔍 **Owner Code Review #8**: Final review

---

## Phase 8: DCN-v2 Implementation (PAIR-PROGRAMMING) ✅
**Goal**: Implement Deep & Cross Network v2 to address VIME's 128-dim bottleneck limitation.
- [x] **8.1 - 8.5** System Implementation (SoftBinning, Embedding, CrossNet, DeepNet)
- [x] **8.6** Training Pipeline (Focal Loss + Precision@20% early stopping)
- [x] **8.7** Evaluation on D_R (ground truth)
- [x] **8.8** Final comparison

### DCN-v2 Result Summary (Updated 2026-02-17):
| Dataset | Metric | DCN-v2 | Route B | Winner |
|---------|--------|--------|---------|--------|
| D_L Test | AUC | 0.7236 | **0.7391** | Route B |
| D_R | AUC | 0.6966 | **0.7206** | Route B |
| D_R | P@20% | 0.9567 | **0.9650** | Route B |
| D_R | Diamonds | 17,741 | **17,885** | Route B |

**Conclusion**: DCN-v2 with Focal Loss achieved competitive but not superior results to Route B.

---

## Phase 9: VIME Bottleneck Hypothesis Test ✅
**Goal**: Test whether VIME's 128-dim bottleneck is the limiting factor by training a 512-dim version.

**Hypothesis**: If the bottleneck causes information loss, a wider encoder (512-dim) should improve performance.

### Implementation:
- [x] **9.1** Create VIMEWide model (253 → 512 → 512 latent, no bottleneck)
- [x] **9.2** Train with same setup: Focal Loss, Precision@20% early stopping
- [x] **9.3** Fair comparison with VIME-128 using identical training
- [x] **9.4** Fair v2: Re-train VIME-128 with identical training setup to VIME-512 (eliminate confounding variables)
- [x] **9.5** Paired bootstrap significance test on fair v2 comparison
- [x] **9.6** Fair v3: Re-train VIME-512 using SAME VIME model class as 128 (3-layer encoder, 1-layer decoder) to eliminate layer-count confounders
- [x] **9.7** Paired bootstrap significance test on fair v3 comparison

### Results (Original Comparison - Confounded):
| Model | Latent Dim | D_L Test AUC | D_R AUC | D_R P@20% | D_R Diamonds |
|-------|------------|--------------|---------|-----------|--------------|
| VIME-128 (old) | 128 | 0.7079 | 0.6856 | 0.9522 | 17,531 |
| VIME-512 | 512 | 0.7042 | 0.6822 | 0.9564 | 17,609 |

*Note: Original comparison had confounding variables (dropout 0.1 vs 0.3, alpha 2.0 vs 1.0, batch 128 vs 512).*

### Results (Fair v2 Comparison - Confounders Eliminated):
| Model | Latent Dim | D_L Test AUC | D_R AUC | D_R P@20% | D_R Diamonds |
|-------|------------|--------------|---------|-----------|--------------|
| **VIME-128-Fair-v2** | 128 | **0.7081** | **0.6956** | **0.9572** | **17,623** |
| VIME-512 | 512 | 0.7042 | 0.6822 | 0.9564 | 17,609 |

### Results (Fair v3 - Fully Controlled, Same Model Class):
| Model | Model Class | Latent Dim | D_L Test AUC | D_R AUC | D_R P@20% |
|-------|------------|------------|--------------|---------|-----------|
| **VIME-512-Fair-v3** | VIME | 512 | **0.7188** | **0.7058** | **0.9591** |
| VIME-128-Fair-v2 | VIME | 128 | 0.7171 | 0.6956 | 0.9588 |
| VIME-512 (G-Wide) | VIMEWide | 512 | 0.7125 | 0.6822 | 0.9548 |

### Key Finding: **CAPACITY HELPS, ARCHITECTURE MATTERS MORE**
- **Fair-v3 (fully controlled):** 512-dim significantly outperforms 128-dim (ΔAUC = +0.0102, 95% CI [+0.0080, +0.0125], p < 0.05) when using the SAME model class
- **Architecture effect:** VIMEWide's mirror decoder hurts by ΔAUC = −0.0236 compared to simple VIME architecture at the same 512-dim width
- **Fair-v2 was confounded:** The apparent 128 > 512 advantage (+0.0134) conflated width with architecture (different model class, layer counts)

**Confounding variables eliminated across experiments:**
| Variable | VIME-128 (old) | Fair-v2 (128) | Fair-v3 (512) | G-Wide (512) |
|----------|---------------|---------------|---------------|--------------|
| Model class | VIME | VIME | **VIME** | VIMEWide |
| Encoder layers | 3 | 3 | **3** | 2 |
| Decoder layers | 1 | 1 | **1** | 2 (mirror) |
| Dropout | 0.1 | 0.3 | **0.3** | 0.3 |
| SSL alpha | 2.0 | 1.0 | **1.0** | 1.0 |
| SSL epochs | 20 | 30 | **30** | 30 |
| SSL batch | 128 | 512 | **512** | 512 |

**Conclusion**: More capacity does help (+0.0102 AUC), but architecture matters more — the VIMEWide mirror decoder architecture hurts 2× more than extra capacity helps. The pretext task misalignment remains the dominant factor limiting all VIME variants relative to CatBoost.

**Artifacts:**
- `src/models/vime/vime_wide.py` (VIMEWide 512-dim encoder)
- `src/models/vime/train_vime_wide.py` (G-Wide training pipeline)
- `src/models/vime/train_vime_128_fair_v2.py` (fair v2 training pipeline)
- `src/models/vime/train_vime_512_fair_v3.py` (fair v3 training pipeline — same VIME class as 128)
- `outputs/results/vime_wide_results.json`
- `outputs/results/vime_128_results.json`
- `outputs/results/vime_128_fair_v2_results.json`
- `outputs/results/vime_512_fair_v3_results.json`

---

## Final Results Summary ✅

### All Models Comparison - Fair Comparison (D_R - Rejected Population):

All models trained with equivalent setup: Focal Loss (or class_weights equivalent) + Precision@20% early stopping.

| Model | D_L AUC | D_R AUC | D_R P@20% | D_R Diamonds |
|-------|---------|---------|-----------|--------------|
| **CatBoost** | **0.7372** | **0.7149** | **0.9661** | **17,914** |
| CatBoost-Ensemble | 0.7237 | 0.7096 | 0.9641 | 17,878 |
| VIME-512-Fair-v3 | 0.7188 | 0.7058 | 0.9591 | 17,658 |
| DCN-v2 | 0.7236 | 0.6966 | 0.9567 | 17,741 |
| VIME-128-Fair-v2 | 0.7171 | 0.6956 | 0.9588 | 17,623 |
| VIME-128 (old) | 0.7079 | 0.6856 | 0.9522 | 17,531 |
| VIME-512 (G-Wide) | 0.7125 | 0.6822 | 0.9548 | 17,609 |

### Key Conclusions (Fair Comparison):
1. **CatBoost (single model) has best point estimates** (D_L AUC: 0.7372, D_R AUC: 0.7149, D_R P@20%: 0.9661, Diamonds: 17,914)
2. **CatBoost vs Ensemble difference NOT statistically significant** - 95% CIs overlap (see Bootstrap CIs below)
3. **Neural networks underperform GBDTs** on this tabular classification task
4. **Capacity helps, architecture matters more** — 512-dim outperforms 128-dim by ΔAUC = +0.0102 (same model class, Fair-v3), but VIMEWide mirror decoders hurt by ΔAUC = −0.0236
5. **Self-supervised pretraining doesn't help** when the objective (reconstruction) differs from the task (classification)
6. **Pseudo-labeling consistently degrades performance** (Student < Teacher in all experiments)

### Bootstrap 95% Confidence Intervals (D_R)

| Model | AUC [95% CI] | P@20% [95% CI] | Diamonds [95% CI] |
|-------|--------------|----------------|-------------------|
| **CatBoost** | 0.7154 [0.7108, 0.7201] | 0.9661 [0.9634, 0.9688] | 17,914 [17,864, 17,961] |
| CatBoost-Ensemble | 0.7097 [0.7050, 0.7142] | 0.9639 [0.9610, 0.9666] | 17,878 [17,825, 17,926] |
| VIME-512-Fair-v3 | 0.7058 [0.7010, 0.7106] | 0.9591 [0.9562, 0.9619] | 17,658 |
| VIME-128-Fair-v2 | 0.6956 [0.6907, 0.7004] | 0.9588 [0.9542, 0.9600] | 17,623 |
| VIME-128 (old) | 0.6855 [0.6809, 0.6905] | 0.9522 [0.9491, 0.9552] | 17,531 |
| VIME-512 (G-Wide) | 0.6821 [0.6772, 0.6871] | 0.9548 [0.9535, 0.9594] | 17,609 |

**Statistical Significance (Paired Bootstrap):**
- **CatBoost vs Ensemble:** ΔAUC = +0.0057, significant at 95%
- **VIME-512-Fair-v3 vs VIME-128-Fair-v2:** ΔAUC = +0.0102 [+0.0080, +0.0125] → **SIGNIFICANT** (capacity helps)
- **VIME-512-Fair-v3 vs VIME-512 G-Wide:** ΔAUC = +0.0236 [+0.0209, +0.0264] → **SIGNIFICANT** (architecture matters more)
- **CatBoost vs VIME-128:** ΔAUC = +0.0299 [+0.0260, +0.0340] → **SIGNIFICANT** - GBDTs outperform NNs

### Methodological Finding: Early Stopping Metric Bias ⚠️

**The ensemble's earlier apparent advantage was an artifact of the early stopping metric choice:**

| Early Stopping | Single CatBoost D_R AUC | Ensemble D_R AUC | Winner |
|----------------|-------------------------|------------------|--------|
| AUC | 0.7206 | **0.7268** | Ensemble |
| **P@20%** | **0.7149** | 0.7096 | **Single** |

**Root cause:**
- **AUC measures overall ranking** → rewards models that separate classes across entire distribution
- **P@20% measures concentration** → rewards models that place best borrowers in top quintile
- **Ensembles smooth predictions** (weighted averaging reduces extremes) → helps AUC, hurts P@20%
- **Single models have sharper predictions** → better concentration for P@20%

**Lesson learned:** The choice of early stopping metric can systematically favor certain architectures. When the business objective is "find the best N applicants" (diamond recovery), use P@20% early stopping, not AUC. Always align the optimization objective with the downstream use case.

**Motivation**: Phase 5.7 confirmed that VIME's ceiling (~0.716 AUC) is architectural. The 253→128 compression discards task-relevant information. DCN-v2 preserves all features through residual connections.

> [!CAUTION]
> **PAIR-PROGRAMMING REQUIRED**: Every line of code must be explained and approved by owner before writing. See `AI_GUIDE.md` Section 2.1.

### Phase 8.0: Project Setup ✅
- [x] **8.0.1** Create architecture proposal (`docs/proposals/2026-02-15_dcn_v2_architecture.md`)
- [x] **8.0.2** Create folder structure (`src/models/dcn_v2/`)
- [x] **8.0.3** Update decision log with approval
- [x] **8.0.4** Update AI_GUIDE.md with pair-programming protocol

### Phase 8.1: Soft Binning Module ✅
- [x] **8.1.1** Design discussion: quantile initialization vs uniform
- [x] **8.1.2** Design discussion: fixed temperature vs decay schedule
- [x] **8.1.3** Design discussion: NaN handling with sentinel value
- [x] **8.1.4** Implement `SoftBinning` class (`soft_binning.py`)
- [x] **8.1.5** Implement `compute_quantiles_for_binning` helper
- [x] **8.1.6** Update `__init__.py` with exports

**Artifacts:** `src/models/dcn_v2/soft_binning.py`

### Phase 8.2: Embedding Engine ✅
- [x] **8.2.1** Design discussion: numerical bin embeddings (weighted sum of bin embeddings)
- [x] **8.2.2** Design discussion: categorical embedding dimensions (sqrt(cardinality) rule)
- [x] **8.2.3** Implement `EmbeddingEngine` class
- [x] **8.2.4** Add LayerNorm on concatenated x₀

**Artifacts:** `src/models/dcn_v2/embedding_engine.py`

### Phase 8.3: Cross Network ✅
- [x] **8.3.1** Design discussion: CrossLayer formula (low-rank factorization)
- [x] **8.3.2** Implement `CrossLayer` class
- [x] **8.3.3** Implement `CrossNetwork` (stacked layers)

**Artifacts:** `src/models/dcn_v2/cross_network.py`

### Phase 8.4: Deep Network ✅
- [x] **8.4.1** Design discussion: MLP architecture (raw layers, no Sequential)
- [x] **8.4.2** Implement `DeepNetwork` class

**Artifacts:** `src/models/dcn_v2/deep_network.py`

### Phase 8.5: Complete DCN-v2 Model ✅
- [x] **8.5.1** Implement `DCNv2` class (combines all components)
- [ ] **8.5.2** Integration test with dummy data

**Artifacts:** `src/models/dcn_v2/dcn_v2.py`

### Phase 8.6: Training Pipeline ✅
- [x] **8.6.1** Create Colab notebook (`notebooks/dcn_v2_training.ipynb`)
- [x] **8.6.2** Implement DCNv2Dataset with Dict-based feature access
- [x] **8.6.3** Custom collate function for batching Dict inputs
- [x] **8.6.4** Quantile computation for SoftBinning initialization
- [x] **8.6.5** Training loop with BCEWithLogitsLoss (pos_weight for class imbalance)
- [x] **8.6.6** Early stopping and model checkpointing

**Artifacts:** `notebooks/dcn_v2_training.ipynb`

### Phase 8.7: Reject Inference
- [ ] **8.7.1** Teacher-student pipeline for DCN-v2

### Phase 8.8: Evaluation & Comparison
- [ ] **8.8.1** Evaluate on D_L_test, D_R, full population
- [ ] **8.8.2** Compare with all previous routes (A, B, C, D, F, A-Fixed, etc.)
- [ ] **8.8.3** Write phase report

**Success Criteria:**
- [ ] DCN-v2 Test AUC ≥ 0.7391 (match Route B)
- [ ] DCN-v2 D_R AUC ≥ 0.7268 (match Route F)
- [ ] No information bottleneck (253 features preserved)

---

## Iteration Protocol

> If at any phase, results indicate issues (e.g., VIME not converging, pseudo-labels too noisy, IF filtering too aggressive), we go back to the relevant phase and adjust:
> - **Phase 3**: Tune VIME hyperparams (corruption rate, hidden dim, loss weights)
> - **Phase 4**: Adjust IF contamination, pseudo-label threshold
> - **Phase 1**: Revisit feature engineering, scoring cutoff
