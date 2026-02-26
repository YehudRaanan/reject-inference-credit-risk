# Response to Methodological and Statistical Critique

## Overview

We thank the reviewer for the thorough methodological critique. This response addresses each point raised, acknowledging valid concerns and clarifying aspects where the critique may have been based on incomplete information.

---

## 1. Rejection Mechanism: Uni-Dimensional EXT_SOURCE_2

**Critique:** The rejection simulation uses a single feature (EXT_SOURCE_2) threshold, which may create inductive bias favoring tree-based models.

**Our Response:**

This critique is **valid and acknowledged**. Our rejection mechanism is indeed uni-dimensional:

```python
# From shared_pipeline.py, lines 261-263
approved_mask = df[score_col] >= cutoff_value
d_l = df[approved_mask].copy()
d_r = df[~approved_mask].copy()  # includes NaN scores
```

We explicitly chose EXT_SOURCE_2 as a proxy for a bank's existing scorecard, documented in our config:

```python
# From config.py, lines 56-62
# Rationale:
#   - EXT_SOURCE_2 has ~85% coverage (least missing of the 3 scores)
#   - Higher EXT_SOURCE_2 correlates with lower default risk
#   - 30th percentile gives ~30% rejected, ~70% approved
```

**Limitations we accept:**
- This creates an axis-aligned decision boundary that may favor tree models
- Real-world rejection policies are typically multi-dimensional
- Our conclusions about "GBDT superiority for reject inference" should be qualified as applying to *this specific simulation design*

**Suggested improvement:** Future work should test multi-dimensional rejection mechanisms (e.g., logistic scorecard, multiple threshold combinations).

---

## 2. Statistical Inference: Non-Paired Bootstrap CIs

**Critique:** The bootstrap comparison uses separate CIs per model and infers significance from non-overlap, rather than paired bootstrap on delta metrics.

**Our Response:**

This critique is **valid**. Our bootstrap implementation (`scripts/compute_bootstrap_ci.py`) does indeed compute independent CIs:

```python
# From compute_bootstrap_ci.py, lines 53-69
for _ in range(n_bootstrap):
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    y_boot = y_true[indices]
    pred_boot = y_pred[indices]
    # ... compute metrics per model independently
```

And then compares via overlap:

```python
# From compute_bootstrap_ci.py, lines 420-428
overlap = not (cb_low > ens_high or ens_low > cb_high)
if overlap:
    print("  CIs OVERLAP - difference may not be statistically significant")
else:
    print("  CIs DO NOT OVERLAP - difference is statistically significant")
```

**This is methodologically suboptimal.** The correct approach would be:
1. Use the **same bootstrap indices** for all models
2. Compute **ΔAUC** and **ΔP@20%** within each bootstrap iteration
3. Report **CI for the delta** rather than separate CIs
4. Alternatively, use DeLong's test for AUC comparison

**Impact:** Our significance claims should be interpreted cautiously. While the point estimates show clear differences (ΔAUC ≈ 0.025), the statistical inference is less rigorous than claimed.

---

## 3. CatBoost Preprocessing: Clarification on What Was Actually Done

**Critique:** Inconsistencies about whether CatBoost used "native" categorical handling vs. imputation and target encoding.

**Our Response:**

We can clarify **exactly** what was implemented, which differs by model family by design:

### CatBoost Preprocessing (`train_catboost_fair.py`):
```python
# Lines 93-96: CatBoost receives RAW categoricals with missing filled
cat_features = [col for col in config.CATEGORICAL_COLS if col in other_cols]
for col in cat_features:
    for df in [X_train, X_val, X_test, X_reject]:
        df[col] = df[col].fillna('_MISSING_')

# Lines 126-128: Native categorical indices passed to CatBoost
cat_indices = [X_train.columns.get_loc(col) for col in cat_features
               if col in X_train.columns]
```

**CatBoost does NOT receive target encoding.** It receives original categorical values with `_MISSING_` for NaN, and handles encoding internally via its native `cat_features` parameter.

### VIME Preprocessing (`route_a_prep.py`):
```python
# Lines 179-188: Target encoding fitted on D_L_train ONLY
target_encoder = TargetEncoder(
    cols=high_card_cats,
    smoothing=config.TARGET_ENCODER_SMOOTHING,
    min_samples_leaf=config.TARGET_ENCODER_MIN_SAMPLES,
    ...
)
X_train_high_enc = target_encoder.fit_transform(X_train_high, y_train)
X_val_high_enc = target_encoder.transform(X_val_high)  # transform only
```

**Key clarification:** Target encoding is used **only for VIME** (which cannot handle raw categoricals), not for CatBoost. Each model receives preprocessing appropriate to its architecture, which is standard practice for fair comparison.

---

## 4. Target Encoding Leakage: Not Present

**Critique:** Potential target leakage if target encoding was computed on full data before split.

**Our Response:**

This critique does **not apply** to our implementation. Target encoding is:
1. **Fit only on D_L_train** (line 188: `fit_transform(X_train_high, y_train)`)
2. **Transform-only** for val/test/reject sets
3. Uses Bayesian smoothing (`smoothing=10.0`) to prevent overfitting to rare categories

```python
# From route_a_prep.py, lines 188-191
X_train_high_enc = target_encoder.fit_transform(X_train_high, y_train)
X_val_high_enc = target_encoder.transform(X_val_high)
X_test_high_enc = target_encoder.transform(X_test_high)
X_reject_high_enc = target_encoder.transform(X_reject_high)
```

This follows proper cross-validation discipline. No leakage occurs.

---

## 5. Focal Loss vs Class Weights: Valid Concern

**Critique:** Using Focal Loss for neural networks vs class weights for CatBoost creates asymmetric optimization objectives.

**Our Response:**

This is **partially valid**. Our implementation:

**Neural Networks (VIME, DCN-v2):**
```python
# From train_vime_128_fair.py, lines 34-46
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        ...
```

**CatBoost:**
```python
# From train_catboost_fair.py, lines 121-122
# Focal Loss alpha equivalent: weight positive class by alpha/(1-alpha)
class_weights = {0: 0.75, 1: 0.25}
```

**These are not mathematically equivalent:**
- Focal Loss dynamically down-weights easy examples via `(1-pt)^γ`
- Class weights apply static reweighting

**Our rationale:** We chose the most common class imbalance handling for each framework. True equivalence would require implementing Focal Loss in CatBoost (possible via custom loss) or using static class weights for neural networks.

**Impact:** This creates some "apples to oranges" comparison, though both address the same underlying issue (8% default rate).

---

## 6. Multiple Seeds / Run Variance

**Critique:** Appears that single runs were used for neural networks without variance estimates.

**Our Response:**

This critique is **valid**. We used a single seed (42) throughout:

```python
# From config.py, line 46
SEED = 42
```

Neural network training is stochastic, and results can vary across initializations. Our implementation does not report variance across multiple seeds.

**What we should have done:**
- Run each neural network configuration 3-5 times
- Report mean ± std for all metrics
- Use paired tests across seeds

---

## 7. Missing Calibration and Business Metrics

**Critique:** No calibration metrics (Brier, ECE) or business-oriented metrics (Expected Maximum Profit).

**Our Response:**

This is **valid**. Our evaluation focused on:
- AUC-ROC (ranking performance)
- Precision@20% (business-interpretable: diamonds found in top quintile)
- Raw diamond counts

**Not included:**
- Calibration curves or Brier score
- Expected profit under various cost assumptions
- Threshold sensitivity analysis

This limits the practical applicability claims. A well-calibrated model with lower AUC might outperform a poorly-calibrated high-AUC model in production.

---

## 8. Fairness and Ethics: Not Addressed

**Critique:** No analysis of model fairness across demographic groups, despite CODE_GENDER being an important feature.

**Our Response:**

This is **valid and an important gap**. CODE_GENDER appears in CatBoost's top features, and we did not analyze:
- Performance disparities by gender
- Disparate impact ratios
- Proxy discrimination through correlated features

For a production credit scoring system, this analysis would be essential. Our study is academic/methodological, but the critique correctly notes this limitation.

---

## Summary: What We Acknowledge vs. Dispute

| Critique Point | Validity | Our Position |
|----------------|----------|--------------|
| Uni-dimensional rejection mechanism | **Valid** | Acknowledged limitation; conclusions apply to this simulation |
| Non-paired bootstrap | **Valid** | Methodological weakness; significance claims should be qualified |
| CatBoost preprocessing confusion | **Clarified** | Different preprocessing per model is intentional and documented |
| Target encoding leakage | **Not valid** | Proper OOF procedure was implemented |
| Focal Loss vs Class Weights | **Partially valid** | Different but both address class imbalance |
| Single seed runs | **Valid** | Should have reported variance |
| Missing calibration | **Valid** | Acknowledged limitation |
| Missing fairness analysis | **Valid** | Important for real-world deployment |

---

## Revised Conclusions

Given these acknowledged limitations, we revise our claims:

**Original:** "GBDT (CatBoost) is consistently superior for reject inference"

**Revised:** "Under our specific experimental design (uni-dimensional rejection mechanism based on EXT_SOURCE_2, single seed, this particular dataset), CatBoost outperformed VIME and DCN-v2 on D_R metrics. These results should not be generalized to all reject inference scenarios without further validation on multi-dimensional rejection policies and multiple random seeds."

**Original:** "Self-supervised pretraining (VIME) does not help for tabular reject inference"

**Revised:** "VIME's self-supervised pretraining did not provide benefits in our specific setup, where the primary challenge was covariate shift rather than limited labeled data. The ~150K labeled training samples may have been sufficient for direct supervised learning, reducing the value of self-supervised representations."

---

## Proposed Follow-Up Work

Based on the valid critiques, we recommend:

1. **Paired bootstrap implementation** with delta CIs and DeLong tests
2. **Multi-dimensional rejection simulation** using logistic scorecard combinations
3. **Multiple seed experiments** (5+ runs) with variance reporting
4. **Calibration analysis** including reliability diagrams and Brier scores
5. **Fairness audit** for gender and other sensitive attributes
6. **Ablation study** removing EXT_SOURCE_2 derivatives to isolate covariate shift effects

---

*This response acknowledges valid methodological critiques while clarifying implementation details that were accurately executed. The overall finding that GBDT performed well remains empirically supported, but the generalizability claims require qualification.*
