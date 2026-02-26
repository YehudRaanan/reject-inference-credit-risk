# Reject Inference for Credit Risk: Neural Networks vs Gradient Boosted Decision Trees

**Final Report - Deep Learning Course Project**
**Date:** 2026-02-26 (Updated - Fair v3 controlled comparison, NaN EXT_SOURCE_2 excluded)

---

## Abstract

This study investigates reject inference methodologies for credit risk assessment, comparing neural network approaches (VIME, DCN-v2) against gradient boosted decision trees (CatBoost). Using the Home Credit Default Risk dataset (307K applications, 122 features), we simulate a realistic reject inference scenario where approximately 30% of applicants are rejected based on an external credit score threshold.

**Key Findings:**

1. **GBDT outperforms neural networks in this study:** CatBoost achieves AUC of 0.715 with 95% CI [0.711, 0.720]. Paired bootstrap analysis indicates statistically significant differences compared to VIME-128 (ΔAUC = +0.030, 95% CI entirely above zero) and VIME-512 (ΔAUC = +0.033).

2. **Capacity helps, but architecture matters more:** A controlled three-stage comparison reveals that (a) VIME-128 beats the original VIME-512/VIMEWide (ΔAUC = +0.0134, significant) — but this was confounded by different layer structure; (b) when using the **same model class** (3-layer encoder, 1-layer decoder), VIME-512-Fair-v3 (AUC: 0.706) **significantly outperforms** VIME-128-Fair-v2 (AUC: 0.696) with ΔAUC = +0.0102, 95% CI [+0.0080, +0.0125]. More capacity does help, but the original VIMEWide's mirror decoder architecture was the real bottleneck.

3. **Pretext-task misalignment:** Self-supervised reconstruction objectives do not appear to transfer effectively to discriminative credit risk classification tasks. Both VIME variants substantially underperform CatBoost.

4. **Ensemble regularization trade-off:** Single CatBoost (AUC: 0.715) outperforms the ensemble variant (AUC: 0.710), suggesting that ensemble averaging introduces smoothing that reduces discriminative sharpness without improving generalization on this dataset.

**Methodological note:** All models trained with equivalent setup (Focal Loss for neural networks / class weights for CatBoost + Precision@20% early stopping) for fair comparison. Statistical significance assessed via **paired bootstrap** (n=1000 resamples, 95% CI for ΔAUC). Calibration assessed via Brier score and ECE. Neural network results are from single-seed runs.

---

## 1. Introduction

### 1.1 Problem Statement

Credit institutions face a fundamental challenge: they can only observe outcomes (default/non-default) for applicants they approve. Rejected applicants represent an information gap that biases models toward the approved population. **Reject inference** attempts to leverage rejected applicant data to improve model generalization and find "diamonds in the mud" - good borrowers among rejected applicants.

### 1.2 Research Questions

1. Can self-supervised learning (VIME) improve reject inference by learning robust representations?
2. Does the VIME 128-dim bottleneck limit performance? Would removing it help?
3. Can modern deep learning architectures (DCN-v2) match or exceed GBDT performance?
4. What is the practical best approach for reject inference?

### 1.3 Dataset

- **Source:** Home Credit Default Risk (Kaggle)
- **Size:** 307,511 applications, 122 features (16 categorical, 106 numerical)
- **Target:** Binary classification (default=1, non-default=0)
- **Class imbalance:** ~8% default rate

### 1.4 Reject Simulation

We simulate rejection using the `EXT_SOURCE_2` external credit score:
- **D_L (Approved):** Top 70% by score → 214,796 applications (labels visible)
- **D_R (Rejected):** Bottom 30% by score → 92,055 applications (labels hidden during training)
- **NaN Handling:** Applicants with missing `EXT_SOURCE_2` (660 rows, 7.9% default rate) are **excluded** from the study to avoid structural correlation between missingness and rejection label

The rejected population has a higher default rate (13.5% vs 5.8%) as expected.

**Methodological Note on Single-Feature Rejection:**

This simulation uses a single-feature threshold (`EXT_SOURCE_2 >= 30th percentile`) to mimic real-world scorecard behavior where banks often apply cutoffs on external bureau scores. This design choice has implications:

1. **Axis-aligned decision boundary:** The rejection boundary is a single threshold on one feature, which tree-based methods (CatBoost) can capture with a single split. Neural networks must learn this boundary through gradient descent, potentially requiring more capacity.

2. **NaN exclusion rationale:** We exclude 660 applicants with missing `EXT_SOURCE_2` from the study. Analysis showed these applicants have a lower default rate (7.9%) than score-based rejected applicants (13.5%), suggesting they are rejected due to data availability rather than credit risk. Including them would create structural correlation between missingness and rejection label, potentially confounding model comparisons.

3. **Alternative approaches not explored:** A "shadow model" rejection using logistic regression on multiple features would create a smoother decision boundary that may be more favorable to neural network architectures. This represents a limitation of the current study.

**Data Splits:**

| Split | Count | Default Rate | Purpose |
|-------|-------|--------------|---------|
| D_L_train | 150,357 | 5.8% | Model training |
| D_L_val | 32,219 | 5.8% | Early stopping, hyperparameter tuning |
| D_L_test | 32,220 | 5.8% | Final evaluation on approved population |
| D_R | 92,055 | 13.5% | Reject inference evaluation (labels hidden during training) |

---

## 2. Methodology

### 2.0 Data Preprocessing Pipeline

All models share a common preprocessing foundation, with model-specific adaptations. Understanding this pipeline is critical for interpreting results.

#### 2.0.1 Shared Pipeline (All Models)

**Data Loading & Cleaning:**
- Load `application_train.csv` (307,511 rows, 122 features)
- Convert DAYS_* columns from negative to positive values (e.g., DAYS_BIRTH = -12000 → 12000 days)
- Identify column types: 16 categorical, 106 numerical

**Feature Engineering (10 New Features):**

| Feature | Formula | Purpose |
|---------|---------|---------|
| `EXT_SOURCES_MEAN` | mean(EXT_SOURCE_1, 2, 3) | Combined external score |
| `EXT_SOURCES_WEIGHTED` | 0.5×EXT_1 + 0.35×EXT_2 + 0.15×EXT_3 | Weighted combination |
| `EXT_SOURCES_PROD` | EXT_1 × EXT_2 × EXT_3 | Multiplicative interaction |
| `CREDIT_INCOME_RATIO` | AMT_CREDIT / AMT_INCOME_TOTAL | Debt burden indicator |
| `ANNUITY_INCOME_RATIO` | AMT_ANNUITY / AMT_INCOME_TOTAL | Payment burden indicator |
| `CREDIT_TERM` | AMT_CREDIT / AMT_ANNUITY | Loan duration proxy |
| `DAYS_EMPLOYED_RATIO` | DAYS_EMPLOYED / DAYS_BIRTH | Employment stability |
| `INCOME_PER_PERSON` | AMT_INCOME_TOTAL / CNT_FAM_MEMBERS | Per-capita income |
| `AGE_YEARS` | DAYS_BIRTH / 365 | Age in years |
| `EMPLOYMENT_YEARS` | DAYS_EMPLOYED / 365 | Employment tenure |

*Note: These engineered features are available to all models equally and do not explain performance differences.*

#### 2.0.2 Model-Specific Preprocessing

| Aspect | VIME | CatBoost | DCN-v2 |
|--------|------|----------|--------|
| **Numerical NaN** | Median + binary mask | Median | Median |
| **Categorical NaN** | Sentinel `'_MISSING_'` + mask | Sentinel `'_MISSING_'` | Index 0 (embedding) |
| **Scaling** | ClippedRobustScaler [0,1] | None (scale-invariant) | ClippedRobustScaler [0,1] |
| **Categorical Encoding** | Target encoding + One-hot | **Native (raw categories)** | Integer indices for embeddings |
| **Missing Signal** | **Preserved** (253 mask features) | **Partial** (via `_MISSING_` sentinel) | **Partial** (index 0 embedding) |
| **Output Dimensions** | 253 features | 132 features | 106 numerical + 16 categorical |

**Important Clarification on CatBoost Preprocessing:**

CatBoost does **NOT** receive target encoding. It receives original categorical values with `_MISSING_` sentinel for NaN values, and handles encoding internally via its native `cat_features` parameter with ordered target statistics. This is intentional: each model receives preprocessing appropriate to its architecture.

```python
# From train_catboost_fair.py, lines 93-96
cat_features = [col for col in config.CATEGORICAL_COLS if col in other_cols]
for col in cat_features:
    for df in [X_train, X_val, X_test, X_reject]:
        df[col] = df[col].fillna('_MISSING_')

# Lines 126-128: Native categorical indices passed to CatBoost
cat_indices = [X_train.columns.get_loc(col) for col in cat_features
               if col in X_train.columns]
```

**Important Clarification on Target Encoding (VIME only):**

Target encoding for VIME high-cardinality categoricals (ORGANIZATION_TYPE, OCCUPATION_TYPE) follows proper out-of-fold discipline:
- Encoder is **fit only on D_L_train** with training labels
- Val/test/reject sets receive **transform-only** (no leakage)
- Bayesian smoothing (smoothing=10.0) prevents overfitting to rare categories

```python
# From route_a_prep.py, lines 188-191
X_train_high_enc = target_encoder.fit_transform(X_train_high, y_train)  # fit on train
X_val_high_enc = target_encoder.transform(X_val_high)    # transform only
X_test_high_enc = target_encoder.transform(X_test_high)  # transform only
X_reject_high_enc = target_encoder.transform(X_reject_high)  # transform only
```

No target leakage occurs in the preprocessing pipeline.

**ClippedRobustScaler:** Standard MinMaxScaler with extreme outliers (e.g., AMT_INCOME_TOTAL max = 117M vs median = 135K) compresses 99% of values into [0, 0.005]. ClippedRobustScaler applies RobustScaler (center=median, scale=IQR), clips at 1st/99th percentiles, then normalizes to [0, 1]. This spreads values across the full range while limiting outlier impact.

*Note: The 1%/99% clipping threshold was not tuned. Alternative thresholds (0.5%/99.5% or 2%/98%) may yield marginal improvements for neural networks.*

#### 2.0.3 Missing Value Handling: A Critical Design Choice

**The Missingness Paradox:** VIME explicitly preserves missingness through 253 binary mask features and partial reconstruction loss (ignoring imputed positions during SSL). CatBoost uses simple median/mode imputation that removes the direct missingness signal. Despite this, CatBoost outperforms VIME in our experiments.

**Explanation:** Preserving missingness information is necessary but not sufficient. VIME's reconstruction objective optimizes for features predictable from neighbors, not features predictive of default. Features with high missingness (e.g., EXT_SOURCE_1 at 56%) are often highly predictive of default but hard to reconstruct, so VIME may learn to ignore them. CatBoost, through gradient boosting and feature interactions, can implicitly capture correlated missingness patterns even after imputation.

### 2.1 VIME-based Approach

**VIME (Value Imputation and Mask Estimation)** is a self-supervised learning framework:

1. **Self-supervised pretraining:** Train encoder on reconstruction + mask prediction
2. **Embedding extraction:** Generate latent representations (128-dim or 512-dim)
3. **Classification:** Train MLP head with Focal Loss

**Preprocessing fixes applied:**
- ClippedRobustScaler (handles outliers)
- Partial reconstruction loss (ignores missing values)
- Target encoding for high-cardinality categoricals

### 2.2 CatBoost Baseline

Direct CatBoost training on raw features:
- Native categorical handling (no one-hot encoding)
- Native missing value handling (NaN as separate category)
- No dimensionality reduction

### 2.3 CatBoost-Ensemble

Two-model ensemble to mitigate EXT_SOURCE feature dominance:
- **Model EXT:** CatBoost on EXT_SOURCE features only (9 features)
- **Model OTHER:** CatBoost on all OTHER features (123 features)
- **Ensemble:** Weighted combination (optimal: EXT=0.3, OTHER=0.7)

### 2.4 Bottleneck Hypothesis Test

To test whether VIME's 128-dim bottleneck limits performance:
- **VIME-512:** 512-dim encoder (253 → 512 → 512, no compression)
- **VIME-128:** 128-dim encoder (253 → 256 → 256 → 128, compression)
- **Same training setup:** Focal Loss, Precision@20% early stopping, dropout=0.3

### 2.5 DCN-v2: Deep & Cross Network

Modern architecture designed for tabular data:
- **SoftBinning:** Learnable feature discretization
- **Cross Network:** Explicit polynomial feature interactions
- **Deep Network:** Implicit nonlinear interactions
- **Native categorical embeddings**

### 2.6 Evaluation Metrics

- **AUC-ROC:** Overall discrimination ability
- **Precision@20%:** Among top 20% lowest-risk predictions, what fraction are truly good borrowers?
- **Diamonds Found:** Absolute count of good borrowers in top 20%

### 2.7 Detailed Model Architectures

#### 2.7.1 VIME Architecture

**VIME-128 Encoder (Primary):**
```
Input (253 features)
  ↓ Linear(253 → 256) + BatchNorm + ReLU
  ↓ Linear(256 → 256) + BatchNorm + ReLU + Dropout(0.1)
  ↓ Linear(256 → 128) + BatchNorm + ReLU  [Latent embedding]
```

**VIME-512 Encoder (Wide variant, no bottleneck):**
```
Input (253 features)
  ↓ Linear(253 → 512) + BatchNorm + ReLU + Dropout(0.3)
  ↓ Linear(512 → 512) + BatchNorm + ReLU + Dropout(0.3)  [Latent = 512-dim]
```

**Classification Head:**
```
Latent (128 or 512-dim)
  ↓ Linear(128 → 64) + ReLU + Dropout(0.3)
  ↓ Linear(64 → 32) + ReLU + Dropout(0.3)
  ↓ Linear(32 → 1) + Sigmoid
  ↓ P(default) ∈ [0, 1]
```

**Self-Supervised Pretraining (VIME):**

| Component | Description |
|-----------|-------------|
| **Corruption Rate** | 30% of features randomly masked |
| **Corruption Method** | Feature values shuffled across samples |
| **Mask Prediction Loss** | BCE between predicted and true corruption mask |
| **Reconstruction Loss** | MSE on original values (partial: only non-missing positions) |
| **Combined Loss** | Loss = α × Loss_mask + Loss_recon (α = 2.0) |

**VIME Training Hyperparameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate | 1e-3 | Adam optimizer |
| Batch Size | 128 (SSL), 256 (fine-tune) | |
| Epochs | 20 (SSL), 50 (fine-tune) | |
| Early Stopping | 10 epochs patience | |
| Dropout | 0.1 (encoder), 0.3 (head) | Regularization |

**VIME-128 vs VIME-512 Comparison:**

| Aspect | VIME-128 | VIME-512 |
|--------|----------|----------|
| Latent Dimension | 128 (compression) | 512 (no compression) |
| Hidden Dimension | 256 | 512 |
| Total Parameters | ~183K | ~850K |
| Dropout Rate | 0.1 | 0.3 (stronger regularization) |
| Bottleneck | Yes | No |

**VIME Decoder Architecture (Reconstruction Heads):**

The VIME autoencoder has two decoder heads that share the same latent representation:

**VIME-128 Decoder (Single-Layer Heads):**
```
Latent (128-dim)
    ├─→ [Mask Head] Linear(128 → 253) + Sigmoid
    │       ↓ Predicted corruption mask ∈ [0, 1]^253
    │
    └─→ [Reconstruction Head] Linear(128 → 253)
            ↓ Reconstructed feature values ∈ ℝ^253
```

**VIME-512 Decoder (Mirror Architecture Heads):**
```
Latent (512-dim)
    ├─→ [Mask Head]
    │     Linear(512 → 512) + BatchNorm + ReLU + Dropout(0.3)
    │     Linear(512 → 253) + Sigmoid
    │       ↓ Predicted corruption mask ∈ [0, 1]^253
    │
    └─→ [Reconstruction Head]
          Linear(512 → 512) + BatchNorm + ReLU + Dropout(0.3)
          Linear(512 → 253)
            ↓ Reconstructed feature values ∈ ℝ^253
```

| Decoder Aspect | VIME-128 | VIME-512 |
|----------------|----------|----------|
| Mask Head Layers | 1 (128 → 253) | 2 (512 → 512 → 253) |
| Recon Head Layers | 1 (128 → 253) | 2 (512 → 512 → 253) |
| Mask Activation | Sigmoid | Sigmoid |
| Recon Activation | None (linear) | None (linear) |
| Decoder Parameters | ~65K | ~525K |

**Why Mirror Architecture (VIME-512):** The VIME-512 decoder mirrors the encoder architecture (2 layers instead of 1) to avoid "decoder bottleneck" where a shallow decoder limits expressiveness. This ensures the capacity increase in the encoder can be fully utilized.

#### 2.7.2 DCN-v2 Architecture

**Architecture Overview:**
```
Raw Features (106 numerical + 16 categorical)
    ↓
[EmbeddingEngine]
├─ Numerical: SoftBinning → Bin Embeddings (16-dim each)
└─ Categorical: Lookup Embeddings (sqrt(cardinality) dim)
    ↓
x₀ (dense embedding vector, ~700-dim)
    ↓
    ├─→ [CrossNetwork] → explicit polynomial interactions
    └─→ [DeepNetwork] → implicit non-linear transformations
    ↓
concat(cross_out, deep_out)
    ↓
[Classification Head] (Linear)
    ↓
sigmoid → P(default)
```

**SoftBinning Layer (Numerical Features):**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Number of Bins | 10 | Learnable bin boundaries |
| Temperature | 5.0 | Controls soft-to-hard transition |
| Missing Bin | Index 0 reserved | Learnable missing embedding |
| Initialization | Data quantiles | [10%, 20%, ..., 90%] |

**Process:**
1. Compute distance from learnable cutpoints
2. Apply sigmoid for cumulative probabilities
3. Convert to per-bin membership probabilities
4. Missing values → Index 0 with probability 1.0
5. Weighted sum of bin embeddings

**Cross Network (Explicit Polynomial Interactions):**

```
Per CrossLayer: x_{l+1} = x₀ ⊙ (V(U(x_l)) + b) + x_l
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Number of Layers | 3 | 2nd to 4th order interactions |
| Rank | 64 | Low-rank factorization |
| U Matrix | (d, 64) | Project down |
| V Matrix | (64, d) | Project back up |
| Residual | Yes | Preserve lower-order interactions |

**Deep Network:**

```
Input (d-dim) → FC(d → 512) → ReLU → Dropout(0.1)
             → FC(512 → 256) → ReLU → Dropout(0.1)
             → FC(256 → 128) → ReLU → Dropout(0.1)
             → Output (128-dim)
```

**DCN-v2 Training Hyperparameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Loss Function | Focal Loss | Class imbalance handling |
| Focal α | 0.25 | Positive class weight |
| Focal γ | 2.0 | Focus on hard examples |
| Learning Rate | 1e-3 | AdamW optimizer |
| Weight Decay | 1e-5 | L2 regularization |
| Batch Size | 1024 | |
| Epochs | 100 (max) | With early stopping |
| Early Stopping | Precision@20%, patience=10 | |

#### 2.7.3 CatBoost Configuration

**Single Model (Route B) Hyperparameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Iterations | 2000 | Maximum boosting rounds |
| Depth | 6 | Tree depth |
| Learning Rate | 0.05 | Shrinkage rate |
| Early Stopping | 200 rounds | Based on validation AUC |
| Loss | Logloss | Binary cross-entropy |
| Class Weights | {0: 0.75, 1: 0.25} | Class imbalance handling (see Section 5.3.2.7 for comparison with Focal Loss) |

**Categorical Feature Handling:**
- 16 categorical columns identified
- Missing values → `"_MISSING_"` sentinel string
- CatBoost uses **native ordered target statistics** (no one-hot encoding, no target encoding)
- Raw categorical values passed to model with `cat_features` parameter
- Automatic optimal split finding per category via CatBoost's internal encoding

**Ensemble Variant (Route F) Configuration:**

| Component | Model EXT | Model OTHER |
|-----------|-----------|-------------|
| Features | 9 (EXT_SOURCE_*) | 123 (all others) |
| Iterations | 500 | 1000 |
| Depth | 4 | 6 |
| Early Stopping | 50 rounds | 100 rounds |
| Categorical Features | None (all numeric) | 16 columns |
| Ensemble Weight | 0.3 | 0.7 |

**Why These Ensemble Weights:**
- EXT_SOURCE_2 is truncated for D_R (rejection threshold)
- OTHER features provide complementary signal for rejected population
- Weight optimization via grid search on validation set

#### 2.7.4 Loss Functions Comparison

| Model | Loss Function | Parameters | Class Balance Handling |
|-------|--------------|------------|------------------------|
| VIME | BCE Loss | — | Focal Loss in fine-tuning |
| DCN-v2 | Focal Loss | α=0.25, γ=2.0 | Down-weights easy examples |
| CatBoost | Logloss | class_weights={0:0.75, 1:0.25} | Static reweighting (not equivalent to Focal Loss) |

**Focal Loss Formula:**
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

Where:
- p_t = model's predicted probability for the true class
- α_t = class weight (0.25 for positive, 0.75 for negative)
- γ = focusing parameter (γ=2 means hard examples get 4x weight)

#### 2.7.5 Early Stopping Strategies

**AUC-based (Standard):**
- Monitor validation AUC after each epoch/iteration
- Stop when no improvement for N rounds
- Used in standard CatBoost training

**Precision@20%-based (Fair Comparison):**
- Monitor Precision@20% on validation set
- More aligned with business objective (diamond recovery)
- Used in fair comparison experiments for all models
- CatBoost: Chunked training (50 iterations) with manual P@20% check
- Neural Networks: Post-epoch evaluation

**Impact of Early Stopping Metric:**

| Metric | Favors | Explanation |
|--------|--------|-------------|
| AUC | Ensembles | Measures overall ranking; smoothing helps |
| Precision@20% | Single models | Measures concentration; sharp predictions help |

This explains why CatBoost-Ensemble won under AUC early stopping but lost under P@20% early stopping.

#### 2.7.6 Model Parameter Summary

| Model | Total Parameters | Architecture Highlights |
|-------|------------------|------------------------|
| VIME-128 | ~193K | 253→256→256→128 bottleneck encoder |
| VIME-512 | ~850K | 253→512→512 wide encoder (no compression) |
| DCN-v2 | ~800K-1M | Embedding + Cross (3 layers) + Deep (3 layers) |
| CatBoost | N/A (trees) | 2000 trees × depth 6 |
| CatBoost-Ensemble | N/A (trees) | EXT (500×4) + OTHER (1000×6) |

### 2.8 Feature Missingness Analysis

Understanding feature missingness is critical for interpreting model behavior, particularly why VIME's explicit missingness handling did not translate to better performance.

#### 2.8.1 Missing Rate by Feature Category

| Feature Group | Count | Missing Rate Range | Key Examples |
|--------------|-------|-------------------|--------------|
| **EXT_SOURCE features** | 3 | 0% - 56% | EXT_SOURCE_1: 56%, EXT_SOURCE_2: 0.3%, EXT_SOURCE_3: 20% |
| **Categorical features** | 16 | 0% - 31% | OCCUPATION_TYPE: 31%, ORGANIZATION_TYPE: 0% |
| **Numerical (income/amount)** | 8 | 0% - 0.4% | AMT_INCOME_TOTAL: 0%, AMT_ANNUITY: 0.4% |
| **FLAG features** | 42 | 0% | All binary flags complete |
| **DAYS features** | 6 | 0% - 0.02% | DAYS_BIRTH: 0%, DAYS_EMPLOYED: ~0.02% (anomalies) |
| **Housing features** | 10 | 47% - 50% | OWN_CAR_AGE: 66% (conditional), WALLSMATERIAL_MODE: 50% |
| **Document flags** | 19 | 0% | All document flags complete |

#### 2.8.2 High-Missingness Features (>20%)

| Feature | Missing % | Predictive Importance | VIME Treatment |
|---------|-----------|----------------------|----------------|
| `OWN_CAR_AGE` | 66% | Low (conditional on car ownership) | Median impute + mask |
| `EXT_SOURCE_1` | 56% | **High** (external credit score) | Median impute + mask |
| `WALLSMATERIAL_MODE` | 50% | Low | Sentinel + mask |
| `HOUSETYPE_MODE` | 50% | Low | Sentinel + mask |
| `FONDKAPREMONT_MODE` | 50% | Low | Sentinel + mask |
| `EMERGENCYSTATE_MODE` | 47% | Low | Sentinel + mask |
| `OCCUPATION_TYPE` | 31% | Medium | Sentinel + mask |
| `EXT_SOURCE_3` | 20% | **High** (external credit score) | Median impute + mask |

**Key Insight:** The features with highest missingness are split between:
1. **High predictive power** (EXT_SOURCE_1, EXT_SOURCE_3): Hard to reconstruct, critical for default prediction
2. **Low predictive power** (housing modes): Easy to reconstruct from correlated features

VIME's reconstruction objective naturally focuses on the easier-to-reconstruct features, potentially underweighting the critical EXT_SOURCE features.

#### 2.8.3 Missingness Correlation with Target

| Missingness Pattern | D_L Default Rate | D_R Default Rate | Interpretation |
|--------------------|------------------|------------------|----------------|
| `EXT_SOURCE_2` missing | N/A (rejected) | 13.4% | All NaN → D_R by design |
| `EXT_SOURCE_1` missing | 5.8% | 14.1% | Higher risk if missing |
| `OCCUPATION_TYPE` missing | 6.1% | 12.8% | Slightly higher risk |
| All EXT_SOURCE present | 4.2% | 10.2% | Lower risk |

The correlation between missingness and default supports the hypothesis that missingness itself contains predictive signal. However, VIME's reconstruction loss may not effectively transfer this signal to the classification task.

---

## 3. Results

### 3.1 Main Comparison Table (Fair Comparison)

All models trained with equivalent setup: Focal Loss (or class_weights equivalent) + Precision@20% early stopping.

| Model | D_L Test AUC | D_R AUC | D_R P@20% | D_R Diamonds |
|-------|--------------|---------|-----------|--------------|
| **CatBoost** | **0.7372** | **0.7154** | **0.9660** | **17,785** |
| CatBoost-Ensemble | 0.7327 | 0.7097 | 0.9641 | 17,750 |
| VIME-512-Fair-v3 | 0.7188 | 0.7058 | 0.9591 | 17,658 |
| VIME-128-Fair-v2 | 0.7171 | 0.6956 | 0.9588 | 17,623 |
| VIME-128 (old) | 0.7079 | 0.6856 | 0.9522 | 17,531 |
| VIME-512 (G-Wide) | 0.7125 | 0.6822 | 0.9548 | 17,609 |

*VIME-128-Fair-v2 / VIME-512-Fair-v3: Both use the same VIME model class (3-layer encoder, 1-layer decoder) and identical training setup. The ONLY difference is width (128/256 vs 512/512). VIME-512 (G-Wide) uses a different model class (VIMEWide: 2-layer encoder, 2-layer mirror decoder).*
*Note: DCN-v2 was not re-trained after NaN exclusion; results from previous run may not be directly comparable.*

### 3.1.1 Bootstrap 95% Confidence Intervals (D_R)

| Model | AUC [95% CI] | P@20% [95% CI] |
|-------|--------------|----------------|
| **CatBoost** | 0.7154 [0.7108, 0.7201] | 0.9661 [0.9634, 0.9688] |
| CatBoost-Ensemble | 0.7097 [0.7050, 0.7142] | 0.9639 [0.9610, 0.9666] |
| VIME-512-Fair-v3 | 0.7058 [0.7010, 0.7106] | 0.9591 [0.9562, 0.9619] |
| VIME-128-Fair-v2 | 0.6956 [0.6907, 0.7004] | 0.9572 [0.9542, 0.9600] |
| VIME-128 (old) | 0.6855 [0.6809, 0.6905] | 0.9522 [0.9491, 0.9552] |
| VIME-512 (G-Wide) | 0.6821 [0.6772, 0.6871] | 0.9564 [0.9535, 0.9594] |

**Statistical Significance (Paired Bootstrap - Proper Comparison):**

*Paired bootstrap (n=1000): Same indices used for all models in each iteration, CI computed for ΔAUC directly.*

| Comparison | ΔAUC [95% CI] | Significant? |
|------------|---------------|--------------|
| CatBoost vs VIME-128 | +0.0299 [+0.0260, +0.0340] | **YES** |
| CatBoost vs VIME-512 | +0.0333 [+0.0295, +0.0371] | **YES** |
| CatBoost vs CatBoost-Ensemble | +0.0057 [+0.0026, +0.0089] | **YES** |
| VIME-128 (old) vs VIME-512 | +0.0034 [-0.0004, +0.0074] | **NO** |
| VIME-128-Fair-v2 vs VIME-512 (G-Wide) | +0.0134 [+0.0107, +0.0162] | **YES** |
| VIME-128-Fair-v2 vs VIME-128 (old) | +0.0100 [+0.0073, +0.0127] | **YES** |
| **VIME-512-Fair-v3 vs VIME-128-Fair-v2** | **+0.0102 [+0.0080, +0.0125]** | **YES** |
| VIME-512-Fair-v3 vs VIME-512 (G-Wide) | +0.0236 [+0.0209, +0.0264] | **YES** |

**Key findings:**
1. **CatBoost vs all VIME variants:** Statistically significant advantage in all cases (ΔAUC +0.01–0.03)
2. **VIME-512-Fair-v3 vs VIME-128-Fair-v2 (fully controlled):** When using the **same model class** and training settings, 512-dim **significantly outperforms** 128-dim (ΔAUC = +0.0102, CI [+0.0080, +0.0125]). More capacity does help.
3. **VIME-128-Fair-v2 vs VIME-512/G-Wide:** 128 beats the original G-Wide 512 (ΔAUC = +0.0134) — but this comparison confounded width with architecture (different model class, layer counts, mirror decoders). Fair-v3 resolves this.
4. **VIME-512-Fair-v3 vs VIME-512/G-Wide:** The same 512-dim width performs **dramatically better** with VIME's 3-layer encoder + shallow decoder (ΔAUC = +0.0236) than with VIMEWide's 2-layer encoder + mirror decoder. Architecture > width.
5. **CatBoost vs CatBoost-Ensemble:** Single model performs better with small but significant effect (ΔAUC = +0.006)

### 3.2 Critical Finding: Capacity Helps, Architecture Matters More (Fair v2 + v3 Comparison)

**Three-stage controlled comparison:** The original VIME-128 vs VIME-512 comparison had multiple confounders. We systematically eliminated them:

1. **Fair-v2:** Unified all training variables (dropout, loss, hyperparameters) but kept VIME-128's architecture (3-layer encoder, 1-layer decoder) vs VIME-512's VIMEWide architecture (2-layer encoder, 2-layer mirror decoder).
2. **Fair-v3:** Used the **same VIME model class** for both 128 and 512 (3-layer encoder, 1-layer decoder). Now the **only** difference is width.

| Model | Model Class | Latent Dim | Encoder | Decoder | D_R AUC [95% CI] |
|-------|------------|------------|---------|---------|------------------|
| **VIME-512-Fair-v3** | VIME | 512 | 3-layer | shallow | **0.7058** [0.7010, 0.7106] |
| VIME-128-Fair-v2 | VIME | 128 | 3-layer | shallow | 0.6956 [0.6907, 0.7004] |
| VIME-512 (G-Wide) | VIMEWide | 512 | 2-layer | mirror | 0.6822 [0.6772, 0.6871] |
| VIME-128 (old) | VIME | 128 | 3-layer | shallow | 0.6855 [0.6809, 0.6905] |

**Key results (paired bootstrap):**

| Comparison | What it tests | ΔAUC [95% CI] | Significant? |
|---|---|---|---|
| **512-Fair-v3 vs 128-Fair-v2** | **Pure capacity effect** | **+0.0102 [+0.0080, +0.0125]** | **YES** |
| 128-Fair-v2 vs 512 G-Wide | Capacity + architecture confounded | +0.0134 [+0.0107, +0.0162] | YES |
| 512-Fair-v3 vs 512 G-Wide | Pure architecture effect (same width) | +0.0236 [+0.0209, +0.0264] | YES |

**Interpretation:**
- **More capacity helps (ΔAUC = +0.0102):** When architecture is fully controlled (same model class, same layers), 512-dim significantly outperforms 128-dim. The additional capacity captures useful signal.
- **Architecture matters more (ΔAUC = +0.0236):** The VIMEWide mirror decoder architecture hurts 512-dim performance far more than the capacity helps it. VIME-512-Fair-v3 (simple architecture) outperforms VIME-512/G-Wide (mirror decoders) by +0.0236 — more than double the capacity effect.
- **Fair-v2 result was confounded:** The apparent 128 > 512 advantage (+0.0134) conflated width with architecture. When architecture is controlled, the direction reverses (512 > 128).
- All VIME variants still substantially underperform CatBoost, confirming pretext task misalignment as the dominant limitation.

**Confounding variables eliminated across comparisons:**

| Variable | VIME-128 (old) | Fair-v2 (128) | Fair-v3 (512) | G-Wide (512) |
|----------|---------------|---------------|---------------|--------------|
| Model class | VIME | VIME | **VIME** | VIMEWide |
| Encoder layers | 3 | 3 | **3** | 2 |
| Decoder layers | 1 (shallow) | 1 (shallow) | **1 (shallow)** | 2 (mirror) |
| Dropout | 0.1 | 0.3 | **0.3** | 0.3 |
| SSL alpha | 2.0 | 1.0 | **1.0** | 1.0 |
| SSL epochs | 20 | 30 | **30** | 30 |
| SSL batch | 128 | 512 | **512** | 512 |
| Recon loss | Full MSE | Masked MSE | **Masked MSE** | Masked MSE |
| Class loss | BCE | Focal | **Focal** | Focal |

### 3.3 Diamond Recovery Analysis

On the rejected population (D_R = 92,055 applicants):

| Model | Diamonds in Top 20% | Difference vs Best |
|-------|--------------------|-----------------------|
| **CatBoost** | **17,785** | — |
| CatBoost-Ensemble | 17,750 | -35 |
| VIME-512 | 17,609 | -176 |
| VIME-128 | 17,531 | -254 |

### 3.4 Training Convergence Analysis

#### 3.4.1 Neural Network Training Dynamics

**VIME Self-Supervised Pretraining:**

| Epoch Range | Mask Loss | Reconstruction Loss | Total Loss | Observations |
|-------------|-----------|---------------------|------------|--------------|
| 1-5 | 0.65 → 0.42 | 0.28 → 0.15 | 1.58 → 0.99 | Rapid initial learning |
| 5-10 | 0.42 → 0.38 | 0.15 → 0.12 | 0.99 → 0.88 | Gradual convergence |
| 10-20 | 0.38 → 0.35 | 0.12 → 0.10 | 0.88 → 0.80 | Plateau reached |

**VIME Fine-Tuning (Classification):**

| Model | Best Epoch | Val AUC at Best | Val P@20% at Best | Convergence Pattern |
|-------|------------|-----------------|-------------------|---------------------|
| VIME-128 | 11 | 0.7244 | 0.9880 | Early convergence, stable |
| VIME-512 | 16 | 0.7167 | 0.9882 | Later convergence, slight overfitting |

**DCN-v2 End-to-End Training:**

Training monitored with Precision@20% early stopping (patience = 10 epochs):
- Best epoch typically reached between epochs 15-25
- Validation P@20% stabilizes around 0.955-0.960
- Training loss continues decreasing but validation metrics plateau

**CatBoost Training:**

| Model | Best Iteration | Iterations Run | Val P@20% at Best | Training Pattern |
|-------|----------------|----------------|-------------------|------------------|
| CatBoost (Single) | 750 | 800 | 0.9847 | Strong initial gain, gradual improvement |
| Model-EXT (Ensemble) | 200 | 250 | N/A | Fast convergence on 9 features |
| Model-OTHER (Ensemble) | 500 | 600 | N/A | Slower convergence on 123 features |

#### 3.4.2 Convergence Observations

1. **VIME-128 converges faster than VIME-512:** Fewer parameters to optimize, simpler loss landscape
2. **CatBoost requires fewer effective iterations:** Each tree is a strong learner; 750 trees sufficient
3. **Neural networks more sensitive to early stopping metric:** P@20% early stopping critical for fair comparison
4. **No divergence observed:** All models stable during training (no NaN gradients, no loss explosion)

### 3.5 Feature Importance Analysis

#### 3.5.1 CatBoost Feature Importance (Top 20)

Based on SHAP-style permutation importance from the trained CatBoost model:

| Rank | Feature | Importance Score | Category |
|------|---------|------------------|----------|
| 1 | `EXT_SOURCE_2` | 100.0 | External score |
| 2 | `EXT_SOURCE_3` | 72.3 | External score |
| 3 | `EXT_SOURCE_1` | 58.1 | External score |
| 4 | `EXT_SOURCES_MEAN` | 45.2 | Engineered |
| 5 | `EXT_SOURCES_WEIGHTED` | 41.8 | Engineered |
| 6 | `DAYS_BIRTH` | 28.4 | Demographics |
| 7 | `DAYS_EMPLOYED` | 25.1 | Employment |
| 8 | `AMT_CREDIT` | 22.7 | Loan amount |
| 9 | `AMT_ANNUITY` | 20.3 | Payment amount |
| 10 | `CREDIT_INCOME_RATIO` | 18.9 | Engineered |
| 11 | `DAYS_ID_PUBLISH` | 16.4 | Document age |
| 12 | `AMT_GOODS_PRICE` | 15.2 | Goods value |
| 13 | `DAYS_REGISTRATION` | 14.1 | Registration age |
| 14 | `ANNUITY_INCOME_RATIO` | 13.5 | Engineered |
| 15 | `REGION_POPULATION_RELATIVE` | 12.8 | Geographic |
| 16 | `DAYS_LAST_PHONE_CHANGE` | 11.2 | Contact stability |
| 17 | `CODE_GENDER` | 10.5 | Demographics |
| 18 | `ORGANIZATION_TYPE` | 9.8 | Categorical |
| 19 | `NAME_EDUCATION_TYPE` | 8.7 | Categorical |
| 20 | `FLAG_OWN_CAR` | 7.9 | Asset ownership |

**Key Observations:**
1. **EXT_SOURCE dominance:** Top 5 features are all EXT_SOURCE-related (original + engineered)
2. **Engineered features valuable:** 4 of top 15 are engineered features available to all models
3. **Categorical features less important:** ORGANIZATION_TYPE and NAME_EDUCATION_TYPE rank 18th and 19th
4. **Temporal features important:** DAYS_* features capture stability signals

#### 3.5.2 Feature Importance Implications for Model Comparison

| Model | EXT_SOURCE Handling | Expected Impact |
|-------|---------------------|-----------------|
| CatBoost | Native trees, exact splits | Optimal extraction |
| DCN-v2 | SoftBinning (10 bins) | Good, but discretized |
| VIME | Scaled [0,1], reconstruction-based | May underweight (hard to reconstruct) |

The dominance of EXT_SOURCE features explains CatBoost's advantage: tree-based methods can find exact threshold splits on these continuous features, while neural networks must approximate these decision boundaries.

### 3.6 Calibration Analysis

Calibration metrics assess whether predicted probabilities match observed frequencies. A well-calibrated model that predicts 20% default probability should see ~20% actual defaults among such predictions.

#### 3.6.1 Calibration Metrics (D_R Population)

| Model | Brier Score | ECE | Interpretation |
|-------|-------------|-----|----------------|
| **CatBoost** | **0.1152** | **0.0724** | **Moderately calibrated (best)** |
| VIME-512-Fair-v3 | 0.1186 | 0.0896 | Moderately calibrated |
| VIME-128-Fair-v2 | 0.1211 | 0.0990 | Moderately calibrated |
| VIME-512 (G-Wide) | 0.1219 | 0.0959 | Moderately calibrated |
| CatBoost-Ensemble | 0.1225 | 0.0920 | Moderately calibrated |
| VIME-128 (old) | 0.1229 | 0.1045 | Poorly calibrated |

**Metrics explained:**
- **Brier Score:** Mean squared error of probability predictions (lower is better). Range [0, 1].
- **ECE (Expected Calibration Error):** Weighted average of |predicted - observed| across probability bins (lower is better). ECE < 0.05 is well calibrated, 0.05-0.10 is moderate, >0.10 is poor.

#### 3.6.2 Calibration Findings

1. **CatBoost achieves best calibration:** Lowest Brier score (0.1152) and ECE (0.0724). The logloss objective with class weights produces relatively well-calibrated probabilities.

2. **VIME-512-Fair-v3 achieves best VIME calibration:** Brier=0.1186, ECE=0.0896 — best among all VIME variants, consistent with its best AUC. The wider latent space with the simple VIME architecture produces better-calibrated probabilities.

3. **Training improvements drive calibration:** The unified training setup (dropout 0.3, masked MSE, focal loss) improved calibration across both 128-dim and 512-dim variants. VIME-128-Fair-v2 (ECE=0.0990) improved over old VIME-128 (ECE=0.1045).

4. **Practical implication:** For credit scoring where probability estimates are used for pricing or regulatory capital, CatBoost's superior calibration is an additional advantage beyond AUC.

#### 3.6.3 Reliability Diagrams

Reliability diagrams (saved to `outputs/figures/reliability_diagrams.png`) show predicted probability bins vs actual default rates. Perfect calibration would follow the diagonal. CatBoost tracks the diagonal most closely, while VIME-128 shows systematic overconfidence in low-risk predictions.

### 3.7 Computational Cost Comparison

#### 3.7.1 Training Time

| Model | Hardware | Training Time | Notes |
|-------|----------|---------------|-------|
| VIME-128 SSL | CPU | ~15 min | 20 epochs, 307K samples |
| VIME-128 Fine-tune | CPU | ~8 min | 50 epochs, 150K samples, early stop at ~11 |
| VIME-512 SSL | CPU | ~25 min | 20 epochs, 307K samples |
| VIME-512 Fine-tune | CPU | ~12 min | 50 epochs, 150K samples, early stop at ~16 |
| DCN-v2 | CPU | ~20 min | End-to-end, 100 epochs max |
| CatBoost (Single) | CPU | ~5 min | 750 iterations |
| CatBoost-Ensemble | CPU | ~8 min | EXT + OTHER models combined |

#### 3.7.2 Inference Time (per 10,000 samples)

| Model | Inference Time | Throughput |
|-------|----------------|------------|
| VIME-128 | ~50 ms | 200K samples/sec |
| VIME-512 | ~80 ms | 125K samples/sec |
| DCN-v2 | ~120 ms | 83K samples/sec |
| CatBoost | ~30 ms | 333K samples/sec |
| CatBoost-Ensemble | ~50 ms | 200K samples/sec |

#### 3.7.3 Memory Requirements

| Model | Model Size (disk) | Peak Memory (training) |
|-------|-------------------|------------------------|
| VIME-128 | ~0.8 MB | ~500 MB |
| VIME-512 | ~3.4 MB | ~1.2 GB |
| DCN-v2 | ~4 MB | ~2 GB |
| CatBoost | ~15 MB | ~2 GB |
| CatBoost-Ensemble | ~25 MB | ~3 GB |

**Summary:** In this study, CatBoost demonstrated favorable performance-to-cost characteristics: shortest training time, fastest inference, competitive memory usage, and highest accuracy among the models tested.

### 3.8 Bootstrap Methodology Details

#### 3.8.1 Bootstrap Procedure

The bootstrap confidence intervals were computed using the following methodology:

**Algorithm:**
```
1. For each model, obtain predictions on D_R (92,715 samples)
2. For i = 1 to n_bootstrap (1000):
   a. Sample indices with replacement from [0, 92714]
   b. Create bootstrap sample: y_boot = y_DR[indices], pred_boot = pred_DR[indices]
   c. Skip if only one class in sample (rare edge case)
   d. Compute AUC(y_boot, pred_boot)
   e. Compute Precision@20%(y_boot, pred_boot)
   f. Compute Diamonds@20%(y_boot, pred_boot)
3. Compute 2.5th and 97.5th percentiles for each metric
```

#### 3.8.2 Bootstrap Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| n_bootstrap | 1000 | Standard for 95% CI; stable percentile estimates |
| Confidence Level | 95% | Standard scientific threshold |
| Sampling | With replacement | Classic bootstrap (Efron, 1979) |
| Stratification | None | Preserves observed class distribution |
| Random Seed | 42 | Reproducibility |

#### 3.8.3 Statistical Interpretation

**Confidence Interval Width Analysis:**

| Model | AUC CI Width | Interpretation |
|-------|--------------|----------------|
| CatBoost | 0.0092 | Tight CI → stable performance |
| CatBoost-Ensemble | 0.0091 | Similar stability to single model |
| VIME-128 | 0.0100 | Slightly wider → more variance |
| VIME-512 | 0.0093 | Similar to CatBoost |

**Non-Overlapping CI Interpretation:**
- CatBoost CI: [0.7103, 0.7195]
- VIME-128 CI: [0.6852, 0.6952]
- Gap: 0.7103 - 0.6952 = 0.0151 (>0, non-overlapping)
- **Note:** Non-overlapping individual CIs suggest statistical significance, but the more rigorous paired bootstrap analysis (Section 3.1.1) provides formal confirmation

#### 3.8.4 Paired Bootstrap Comparison (Updated)

After initial analysis, we implemented proper **paired bootstrap** comparison:
- Same indices used for all models in each bootstrap iteration
- ΔAUC and ΔP@20% computed within each iteration
- 95% CI reported for the delta directly

This provides rigorous statistical testing. See Section 3.1.1 for paired bootstrap results.

#### 3.8.5 Remaining Limitations

1. **Point estimate for DCN-v2:** Full bootstrap requires model reload; DCN-v2 not included in paired comparison

2. **Assumes IID sampling:** May not fully capture temporal dependencies in credit data

3. **No multiple testing correction:** Multiple comparisons increase false positive rate (4 comparisons made)

---

## 4. Discussion

### 4.1 Potential Explanations for CatBoost Performance

Several factors may contribute to CatBoost's stronger performance in this study:

1. **Native categorical handling:** Processes categoricals directly without information loss from encoding
2. **Missing value handling:** CatBoost can leverage missingness patterns even after imputation through correlated feature interactions
3. **No compression:** All 132 features remain accessible without dimensionality reduction
4. **Weak signal aggregation:** Gradient boosting may efficiently capture sparse, distributed signals across features
5. **Automatic feature interactions:** Tree splits naturally learn conditional patterns (e.g., IF income < X AND age > Y THEN...)

**Clarification on Feature Engineering:** The 10 engineered features (EXT_SOURCES_*, ratios, etc.) were available to all models equally. CatBoost's advantage does not stem from having access to better features, but from its inductive bias for tabular data—automatic interaction detection, ordered boosting for categoricals, and efficient gradient utilization with limited data.

**Clarification on Imputation:** Despite using simple median/mode imputation (which "destroys" the direct missingness signal), CatBoost still outperforms VIME (which preserves missingness through 253 mask features). This suggests CatBoost captures correlated missingness patterns implicitly through feature interactions, or that the reconstruction objective in VIME fails to transfer the preserved missingness signal to classification.

### 4.2 Why CatBoost Outperforms the Ensemble (Fair Comparison)

With equivalent training setup (class weights + P@20% early stopping), single CatBoost outperforms the two-model ensemble:
1. **Information sharing:** Single model can learn feature interactions across EXT_SOURCE and OTHER features
2. **Regularization benefit:** Class weights reduce overfitting more effectively in a unified model
3. **Gradient flow:** Joint optimization captures complementary signals better than weighted averaging

### 4.2.1 Methodological Finding: Early Stopping Metric Bias

**Critical Insight:** The ensemble's earlier apparent advantage was an artifact of using AUC-based early stopping versus P@20%-based early stopping.

| Early Stopping | Single CatBoost | Ensemble | Winner |
|----------------|-----------------|----------|--------|
| AUC | 0.7206 | **0.7268** | Ensemble |
| **P@20%** | **0.7149** | 0.7096 | **Single** |

**Why this happens:**

1. **AUC measures overall ranking** - rewards models that separate classes across the entire score distribution
2. **P@20% measures concentration** - rewards models that place the best borrowers in the top quintile
3. **Ensembles smooth predictions** - weighted averaging reduces extreme scores, which helps AUC (fewer rank inversions) but hurts P@20% (less concentration at the top)
4. **Single models have sharper predictions** - more extreme scores create better concentration for P@20%

**Implication for practitioners:** The choice of early stopping metric can systematically favor certain model architectures. When the business objective is "find the best N applicants" (diamond recovery), P@20% early stopping is more appropriate than AUC. This should be aligned with the downstream use case from the start.

### 4.3 Potential Explanations for VIME Performance (And Why Width Did Not Help)

The bottleneck hypothesis assumed that compressing 253 features to 128 dimensions loses information. Our experiments suggest this may **not be the primary limiting factor**:

1. **Potential objective mismatch:** Reconstruction loss may optimize for different features than those most relevant to classification
2. **Capacity without direction:** Increased parameters (VIME-512) without task-aligned objectives may lead to overfitting rather than improved performance
3. **Pretext task relevance:** Learning to reconstruct corrupted features may not directly benefit default prediction
4. **Missingness signal transfer:** While VIME's masks preserve missingness information, the reconstruction objective may not effectively transfer this signal to the classification task

**Pretext-Task Misalignment (Detailed Explanation):**

VIME's self-supervised objective consists of two components:
- **Mask prediction:** BCE loss on identifying which features were corrupted
- **Feature reconstruction:** MSE loss on predicting original values from corrupted inputs

This objective optimizes for features that are **predictable from neighbors** (low reconstruction error), not features that are **predictive of default**. Consider:

| Feature Type | Reconstruction Ease | Default Predictiveness | VIME Focus |
|--------------|---------------------|------------------------|------------|
| Demographic (age, gender) | Easy (correlated) | Low | High |
| EXT_SOURCE_1 (56% missing) | Hard (sparse correlations) | **High** | Low |
| Housing features | Easy (clustered) | Low | High |

Features like `EXT_SOURCE_1` are highly predictive of default but hard to reconstruct due to high missingness. VIME may learn to ignore these features during pretraining, producing representations that underweight the most important predictors.

**Why VIMEWide Hurts Performance (Fair v3 Update):**

A three-stage comparison disentangled capacity from architecture:

| Model | Architecture | D_R AUC |
|---|---|---|
| VIME-512-Fair-v3 | VIME class (3-layer enc, 1-layer dec) | **0.7058** |
| VIME-128-Fair-v2 | VIME class (3-layer enc, 1-layer dec) | 0.6956 |
| VIME-512 (G-Wide) | VIMEWide class (2-layer enc, 2-layer mirror dec) | 0.6822 |

The original G-Wide's poor performance was caused by its **mirror decoder architecture**, not insufficient capacity. When 512-dim uses the same simple architecture as 128-dim (Fair-v3), it significantly outperforms both 128-dim (+0.0102) and the original G-Wide (+0.0236). The mirror decoders likely overfit the reconstruction task, producing representations that are optimized for reconstruction fidelity rather than downstream classification.

The pretext task misalignment remains the dominant factor — even the best VIME variant (512-Fair-v3, AUC 0.706) substantially underperforms CatBoost (AUC 0.715).

### 4.4 Why DCN-v2 Doesn't Match CatBoost

Despite preserving all features without compression:
1. **Categorical embedding limitations:** Learned embeddings may not capture category semantics as well
2. **Optimization challenges:** Deeper networks harder to optimize on tabular data
3. **Data efficiency:** GBDTs extract more signal from limited training data

**Numerical vs. Categorical Feature Importance:**

DCN-v2 excels at learning categorical embeddings (Index 0 reserved for missing values, learnable embedding per category). However, the Home Credit dataset is numerically dominated:
- 106 numerical features vs. 16 categorical features
- Key predictors (`EXT_SOURCE_*`, `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`) are numerical
- Categorical features (`ORGANIZATION_TYPE`, `OCCUPATION_TYPE`) have weaker predictive power

| Aspect | DCN-v2 | CatBoost |
|--------|--------|----------|
| **Numerical splits** | SoftBinning (learned bins) | Exact splits (any threshold) |
| **Interaction order** | Fixed polynomial order in Cross Network | Unlimited depth per tree |
| **Optimization** | Gradient descent (local minima risk) | Gradient boosting (greedy global) |

DCN-v2's Cross Network learns explicit polynomial feature interactions, but CatBoost's tree ensemble can find arbitrary split points and capture higher-order conditional patterns that the Cross Network cannot express efficiently.

**SoftBinning Limitation:** DCN-v2's SoftBinning layer discretizes numerical features into learned bins. While this enables gradient-based optimization, it may not find the optimal split thresholds that CatBoost discovers through exhaustive search.

### 4.5 Pseudo-Labeling Consistently Fails

In all experiments, Student models (trained with pseudo-labels) underperform Teacher models:
- Pseudo-labels add noise, not signal
- Rejected population has different feature distributions
- Confirmation bias: model learns its own mistakes

---

## 5. Conclusions

### 5.1 Practical Recommendations

Based on the findings of this study, practitioners working on reject inference for tabular credit data may consider:

1. **Gradient boosted decision trees (CatBoost/XGBoost/LightGBM)** may be a strong baseline, as they outperformed neural network approaches in our experiments
2. **Preserving raw features** rather than applying dimensionality reduction, as compression may lose predictive information
3. **Caution with pseudo-labeling approaches**, which did not improve performance in our tests
4. **Investing in feature engineering**, which may yield greater returns than architectural modifications

*Note: These recommendations are based on a single dataset and specific experimental conditions. Practitioners should validate these findings on their own data.*

### 5.2 Theoretical Insights

1. **Capacity helps, but architecture matters more:** A three-stage controlled comparison shows that (a) 512-dim significantly outperforms 128-dim when architecture is controlled (ΔAUC = +0.0102, p < 0.05); (b) the VIMEWide mirror decoder architecture hurts performance far more (ΔAUC = −0.0236 vs simple decoder). The original VIME-128 > VIME-512 result was confounded by architectural differences. The pretext task misalignment remains the dominant factor limiting all VIME variants relative to CatBoost.

2. **Self-supervised pretraining may be domain-dependent:** Reconstruction-based pretext tasks effective for images and text may not transfer directly to tabular classification, where feature semantics and structure differ fundamentally. This hypothesis warrants further investigation.

3. **GBDT data efficiency:** For medium-scale tabular data (~200K samples), gradient boosted decision trees demonstrated superior sample efficiency compared to the neural network approaches tested in this study. Whether this advantage holds at larger scales remains an open question.

4. **Informative missingness:** Neural network pipelines requiring imputation may remove predictive signal encoded in missing value patterns. However, in our experiments, this did not fully explain the performance gap, as VIME preserved missingness information yet underperformed CatBoost.

### 5.3 Limitations

#### 5.3.1 Dataset Scope
- **Single dataset:** Results may not generalize to other credit risk datasets or domains
- **Dataset size:** 307K samples is medium-scale; neural networks may perform better on larger datasets
- **Feature characteristics:** Heavy reliance on EXT_SOURCE features; datasets with different feature structures may yield different results

#### 5.3.2 Methodological Limitations

1. **Single-feature rejection simulation:** The `EXT_SOURCE_2` threshold creates an axis-aligned decision boundary that may favor tree-based methods. Tree models can capture this boundary with a single split, while neural networks must approximate it through gradient descent. This is a significant limitation: conclusions about "GBDT superiority for reject inference" should be qualified as applying to this specific simulation design. A multi-feature "shadow model" rejection (e.g., logistic regression on multiple features) would create a smoother decision boundary and provide a fairer comparison.

2. **NaN exclusion:** We excluded 660 applicants with missing `EXT_SOURCE_2` from the study to avoid structural correlation between missingness and rejection label. These applicants had a lower default rate (7.9%) than score-based rejected applicants (13.5%), suggesting they were rejected due to data availability rather than credit risk. This exclusion eliminates one potential confound but reduces the realism of the simulation (real banks do reject applicants with missing bureau scores).

3. **Imputation for CatBoost:** CatBoost was trained with median/mode imputation for fair comparison with VIME, rather than its native NaN handling. This may have underestimated CatBoost's true potential.

4. **ClippedRobustScaler thresholds:** The 1%/99% clipping was not tuned. Alternative thresholds may improve neural network performance by preserving more tail information.

5. **No contrastive pretraining:** VIME uses reconstruction-based SSL. Contrastive approaches (SimCLR, SCARF) may produce representations better aligned with classification.

6. **Single random seed:** All experiments used a single random seed (42). Neural network training is stochastic, and results can vary across initializations. We did not report variance across multiple seeds, which weakens the reliability of performance comparisons for neural network models. Future work should run each neural network configuration 3-5 times and report mean ± standard deviation.

7. **Focal Loss vs Class Weights not equivalent:** Neural networks (VIME, DCN-v2) used Focal Loss (α=0.25, γ=2.0), while CatBoost used class weights {0: 0.75, 1: 0.25}. These are not mathematically equivalent: Focal Loss dynamically down-weights easy examples via `(1-pt)^γ`, while class weights apply static reweighting. Although both address class imbalance, this creates some asymmetry in optimization objectives between model families.

#### 5.3.3 Statistical Limitations

1. **Paired bootstrap implemented:** We implemented proper paired bootstrap comparison (Section 3.1.1), using the same resampled indices across all models and computing CIs for ΔAUC/ΔP@20% directly. This provides rigorous statistical testing for model comparisons.

2. **No multiple testing correction:** Multiple model comparisons (4 pairs) increase false positive rate. No Bonferroni or FDR correction was applied. With 4 comparisons at α=0.05, family-wise error rate is approximately 1-(0.95)^4 ≈ 0.19.

3. **Point estimate only for DCN-v2:** Full bootstrap for DCN-v2 requires model reload; DCN-v2 was not included in paired bootstrap comparison.

#### 5.3.4 Hyperparameter Tuning
- Limited tuning for neural network architectures (VIME, DCN-v2)
- CatBoost used default parameters with class weighting
- Systematic hyperparameter search may narrow the performance gap

### 5.4 Future Directions

#### 5.4.1 Alternative Pretext Tasks for Tabular SSL

The reconstruction-based pretext task in VIME optimizes for features predictable from neighbors, not features predictive of default. Alternative approaches:

1. **Contrastive learning (SimCLR, SCARF):** Learn representations where similar samples (by label or feature similarity) are close in embedding space. This aligns the pretext task with discrimination rather than reconstruction.

2. **Feature-shuffling detection:** Train encoder to detect which feature columns have been shuffled across samples. This may better capture feature-level dependencies relevant to classification.

3. **Masked feature prediction with classification head:** Instead of pure reconstruction, use a classification-aware auxiliary loss during pretraining.

#### 5.4.2 Modern Tabular Architectures

1. **TabTransformer / FT-Transformer:** Attention mechanisms over feature embeddings may capture interactions differently than Cross Networks.

2. **Missingness-aware embeddings:** Learnable "missing token" embeddings (as in DCN-v2's Index 0) combined with attention masks that explicitly model missingness patterns.

3. **Two-stage training:** AUC-based early stopping for initial convergence, then P@20% fine-tuning for business-aligned optimization.

#### 5.4.3 Domain Adaptation for Reject Inference

The D_L → D_R distribution shift (5.8% vs 13.5% default rate) represents a domain adaptation problem:

1. **Importance sampling:** Re-weight D_L samples to match D_R feature distribution.

2. **DANN-style adversarial training:** Train encoder to be invariant to whether sample is from D_L or D_R.

3. **Calibrated soft pseudo-labels:** Instead of hard 0/1 pseudo-labels, use calibrated probabilities with temperature scaling.

#### 5.4.4 Rejection Simulation Improvements

1. **Shadow model rejection:** Use logistic regression on multiple features to create a smoother decision boundary, potentially reducing bias toward axis-aligned methods (GBDTs).

2. **Multi-threshold rejection:** Combine multiple external scores with different thresholds to create a more realistic rejection mechanism.

#### 5.4.5 Hyperparameter and Architecture Search

1. **ClippedRobustScaler thresholds:** Test (0.5%, 99.5%) and (2%, 98%) to assess impact on neural network performance.

2. **VIME reconstruction loss weights:** The balance between mask prediction and feature reconstruction may affect which features the encoder prioritizes.

3. **DCN-v2 SoftBinning bins:** The number of bins per numerical feature affects the granularity of learned thresholds.

---

## 6. Conclusions

**For credit risk reject inference under equivalent training conditions:**

| Use Case | Recommendation | Evidence |
|----------|----------------|----------|
| **Primary scoring (D_L)** | CatBoost | Best D_L AUC: 0.737 |
| **Diamond recovery (D_R)** | CatBoost | Best P@20%: 96.6% [96.3%, 96.9%] |
| **Overall D_R discrimination** | CatBoost | Best AUC: 0.715 [0.711, 0.720] |

**Summary:** Within this experimental setup, CatBoost outperformed all neural network approaches tested, with paired bootstrap analysis indicating statistically significant differences:
- ΔAUC (CatBoost - VIME-128) = +0.0299 [+0.0260, +0.0340]
- ΔAUC (CatBoost - VIME-512) = +0.0333 [+0.0295, +0.0371]

CatBoost also achieved the best calibration among models tested (Brier=0.1152, ECE=0.072). Among VIME variants, VIME-512-Fair-v3 achieved the best calibration (Brier=0.1186, ECE=0.090).

**Finding on capacity vs architecture (three-stage controlled comparison):** When using the same model class (VIME, 3-layer encoder + 1-layer decoder) and identical training, 512-dim **significantly outperforms** 128-dim (ΔAUC = +0.0102, 95% CI [+0.0080, +0.0125]). However, the original VIMEWide architecture (2-layer encoder + mirror decoder) performs **dramatically worse** than the simple VIME architecture at the same 512-dim width (ΔAUC = −0.0236). Architecture choice matters more than capacity. The pretext task misalignment remains the dominant factor limiting all VIME variants relative to CatBoost.

**Important caveats:**
1. The rejection simulation uses a uni-dimensional threshold on EXT_SOURCE_2, which may favor tree-based methods that can capture axis-aligned boundaries efficiently
2. Neural network results are from single-seed runs without variance estimates across multiple initializations
3. DCN-v2 was not re-trained after NaN exclusion and was not included in the updated paired bootstrap comparison
4. No multiple testing correction was applied (4 comparisons, family-wise error rate ≈ 0.19)

**Interpretation:** For this specific dataset and methodology, the performance gap between CatBoost and neural network approaches stems primarily from pretext-task misalignment in VIME's self-supervised objective. The three-stage fair comparison (v2 + v3) reveals that: (1) more capacity does help (512 > 128 by ΔAUC = +0.0102 when architecture is controlled); (2) architecture matters more than capacity — the VIMEWide mirror decoder hurts by ΔAUC = −0.0236; (3) even the best VIME configuration (512-Fair-v3, AUC 0.706) still substantially underperforms CatBoost (AUC 0.715). These conclusions should be validated on additional datasets and with alternative rejection simulation mechanisms before generalization.

---

## Appendix: Artifacts

### Code
- `src/models/vime/` - VIME implementation (128-dim and 512-dim)
- `src/models/dcn_v2/` - DCN-v2 implementation
- `scripts/train_vime_128_fair.py` - Fair comparison script (neural networks)
- `scripts/train_catboost_fair.py` - Fair comparison script (CatBoost)

### Models
- `outputs/models/vime_encoder_fixed.pth` - VIME 128-dim (original)
- `outputs/models/vime_128_fair_v2_encoder.pth` - VIME 128-dim (Fair-v2)
- `outputs/models/vime_512_fair_v3_encoder.pth` - VIME 512-dim (Fair-v3, same VIME class)
- `outputs/models/vime_wide_encoder.pth` - VIME 512-dim (G-Wide, VIMEWide class)
- `outputs/models/dcn_v2_cat_best.pth` - DCN-v2
- `outputs/models/catboost_fair.cbm` - CatBoost (Fair)
- `outputs/models/catboost_ensemble_ext_fair.cbm` - CatBoost-Ensemble EXT model
- `outputs/models/catboost_ensemble_other_fair.cbm` - CatBoost-Ensemble OTHER model

### Results
- `outputs/results/vime_128_results.json`
- `outputs/results/vime_128_fair_v2_results.json`
- `outputs/results/vime_512_fair_v3_results.json`
- `outputs/results/vime_wide_results.json`
- `outputs/results/dcn_v2_DR_results.json`
- `outputs/results/catboost_fair_results.json` - Fair comparison results

---

## References

### Primary Model References

1. **VIME (Value Imputation and Mask Estimation)**
   - Yoon, J., Zhang, Y., Jordon, J., & van der Schaar, M. (2020). *VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain*. Advances in Neural Information Processing Systems (NeurIPS), 33.
   - Key contribution: Self-supervised pretraining for tabular data via reconstruction and mask prediction.

2. **DCN-v2 (Deep & Cross Network v2)**
   - Wang, R., Shivanna, R., Cheng, D., Jain, S., Lin, D., Hong, L., & Chi, E. (2021). *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems*. Proceedings of the Web Conference 2021.
   - Key contribution: Low-rank cross network for explicit feature interactions.

3. **CatBoost**
   - Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: unbiased boosting with categorical features*. Advances in Neural Information Processing Systems (NeurIPS), 31.
   - Key contribution: Ordered boosting and native categorical feature handling.

### Supporting Methodology References

4. **Focal Loss**
   - Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection*. Proceedings of the IEEE International Conference on Computer Vision (ICCV).
   - Used for: Class imbalance handling in neural network training.

5. **Bootstrap Confidence Intervals**
   - Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife*. The Annals of Statistics, 7(1), 1-26.
   - Used for: Statistical significance testing of model comparisons.

6. **RobustScaler**
   - Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
   - Used for: Outlier-robust feature scaling in ClippedRobustScaler.

### Dataset Reference

7. **Home Credit Default Risk Dataset**
   - Home Credit Group. (2018). *Home Credit Default Risk*. Kaggle Competition.
   - URL: https://www.kaggle.com/c/home-credit-default-risk
   - Contains: 307,511 loan applications with 122 features.

### Related Work on Tabular Deep Learning

8. **TabNet**
   - Arik, S. Ö., & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. Proceedings of the AAAI Conference on Artificial Intelligence.
   - Alternative: Attention-based tabular learning (not implemented in this study).

9. **SCARF (Self-supervised Contrastive Tabular Learning)**
   - Bahri, D., Jiang, H., Tay, Y., & Metzler, D. (2022). *SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption*. International Conference on Learning Representations (ICLR).
   - Alternative: Contrastive pretraining for tabular data (discussed in Future Directions).

10. **FT-Transformer**
    - Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. Advances in Neural Information Processing Systems (NeurIPS), 34.
    - Alternative: Transformer architecture for tabular data (discussed in Future Directions).

### Reject Inference Literature

11. **Reject Inference Overview**
    - Banasik, J., & Crook, J. (2005). *Credit Scoring, Augmentation and Lean Models*. Journal of the Operational Research Society, 56(9), 1072-1081.
    - Background: Traditional reject inference methodologies.

12. **Domain Adaptation for Credit Risk**
    - Ganin, Y., & Lempitsky, V. (2015). *Unsupervised Domain Adaptation by Backpropagation*. Proceedings of the 32nd International Conference on Machine Learning (ICML).
    - Related: DANN-style adversarial training (discussed in Future Directions).

---

*Report completed: 2026-02-18*
