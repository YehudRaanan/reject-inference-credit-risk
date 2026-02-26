# Conclusion: Neural Networks vs GBDT for Tabular Data with Informative Missingness

> **Corrected Version** — Addresses overclaims in the original document with scientifically defensible framing.

## Executive Summary

**For medium-scale tabular credit-risk data with informative missingness, a CatBoost model trained on minimally processed raw features provides superior reject inference performance versus a VIME-based representation learning pipeline.**

This result should be interpreted as a **practical advantage in this regime** (inductive bias and data efficiency), rather than a general structural impossibility of neural networks.

---

## 1. Project-Level Conclusion (Fair Comparison Results)

On Home Credit reject inference under equivalent training conditions (Focal Loss + P@20% early stopping):

| Model | D_R AUC [95% CI] | D_R P@20% | Diamonds | Statistical Significance |
|-------|------------------|-----------|----------|--------------------------|
| **CatBoost** | **0.715** [0.710, 0.719] | **96.6%** | **17,914** | Reference |
| CatBoost-Ensemble | 0.710 [0.705, 0.714] | 96.4% | 17,878 | CIs overlap with single |
| DCN-v2 | 0.697 | 95.7% | 17,741 | Below CatBoost |
| VIME-128 | 0.690 [0.685, 0.695] | 95.5% | 17,701 | Significantly below CatBoost |
| VIME-512 | 0.689 [0.684, 0.694] | 95.8% | 17,760 | Significantly below CatBoost |

*Bootstrap 95% confidence intervals from n=1000 resamples*

**Statistical conclusions:**
- CatBoost's CI [0.710, 0.719] does not overlap with any neural network CI → **statistically significant** difference
- VIME-128 and VIME-512 CIs overlap completely → **no significant difference** between bottleneck variants
- This supports the hypothesis that architectural bottleneck is not the primary limitation

---

## 2. Scientific Conclusion (How to Interpret the Win)

The win is best explained as **signal preservation + inductive bias + data efficiency**, not "NN impossibility."

### 2.1 Why Trees Win Here

1. **Exploit missingness patterns** without forcing median imputation into the value channel
2. **Handle categorical features** without sparse one-hot explosion
3. **Work extremely well** in medium-scale tabular regimes with limited tuning
4. **Native NaN handling** — learn optimal split direction per node

### 2.2 Why Neural Network Approaches Underperformed

1. **Pretext-task misalignment:** Self-supervised reconstruction objectives optimize for different features than classification (supported by VIME-512 not improving over VIME-128)
2. **Preprocessing signal loss:** Median imputation and scaling measurably reduced predictive signal in key features (40-60% correlation loss in EDA)
3. **Bottleneck not the limiting factor:** VIME-512 (no compression) performed numerically worse than VIME-128, with overlapping CIs indicating no significant difference
4. **Data efficiency gap:** GBDTs extract more signal from ~200K samples than current neural architectures

---

## 3. Root Cause Evidence (EDA Summary)

### 3.1 Median Imputation Damage

| Feature | Missing% | Raw Corr | After Imputation | Signal Lost |
|---------|----------|----------|------------------|-------------|
| EXT_SOURCE_1 | 56% | -0.151 | -0.072 | **52%** |
| EXT_SOURCES_PROD | 64% | -0.217 | -0.087 | **60%** |
| EXT_SOURCES_WEIGHTED | 64% | -0.201 | -0.085 | **58%** |
| APARTMENTS_AVG | 51% | -0.035 | -0.012 | **66%** |

### 3.2 Missing Indicators: Partial Recovery Only

Missing indicators preserved some signal but only recovered **10-20%** of what was lost for key EXT_SOURCE features. Housing features (low raw correlation) were fully compensated.

### 3.3 MinMax Scaling + Extreme Outliers

AMT_INCOME_TOTAL: max/99th percentile ratio = **226x**, compressing 99% of values into [0, 0.005] range. VIME's MSE loss effectively ignored income.

---

## 4. Corrected Scientific Framing

### 4.1 What Was Overstated (Original Claims)

| Original Claim | Problem |
|----------------|---------|
| "NNs cannot process NaN" | False for modern tabular DL with masking/embeddings |
| "Imputation necessarily destroys information" | Only true for naive imputation; missingness-aware methods exist |
| "Trees are structurally superior" | Overgeneralized; should be scoped to this regime |

### 4.2 Corrected Wording

**Original:** "Neural networks cannot process NaN values."
**Corrected:** "Standard MLP pipelines cannot ingest NaNs directly and often rely on imputation; however, modern tabular DL can represent missingness via masking or learnable missing embeddings."

**Original:** "Imputation necessarily destroys information."
**Corrected:** "Median imputation in this project measurably reduced predictive signal in key features; missingness-aware modeling can preserve the signal more faithfully."

**Original:** "Tree-based models are structurally superior."
**Corrected:** "GBDTs are structurally well-suited and more data-efficient for medium-scale tabular data with informative missingness; NNs may be competitive with specialized architectures and additional tuning/scale."

---

## 5. Final Integrated Conclusion

> For medium-scale tabular credit-risk data with informative missingness, CatBoost demonstrates **statistically significant** superiority over neural network approaches (non-overlapping 95% confidence intervals). The performance gap (ΔAUC ≈ 0.018-0.026) is robust across bootstrap resamples.
>
> The bottleneck hypothesis (that VIME's 128-dimensional compression caused information loss) is **not supported**: VIME-512 with no compression performs equivalently to VIME-128 (overlapping CIs). This suggests the limitation stems from **pretext-task misalignment** - reconstruction objectives do not produce representations optimal for credit risk classification.
>
> These findings should be interpreted as specific to this dataset, methodology, and model implementations. Modern missingness-aware tabular DL architectures with classification-aligned pretraining may narrow or close this gap.

---

## 6. Production Recommendations

### 6.1 Recommended Model
- **Route B: CatBoost on raw features**
- Use native categorical handling and missingness handling
- Calibrate probability outputs on D_L_test

### 6.2 Metrics to Monitor
- ROC-AUC + PR-AUC (class imbalance)
- Precision@k for reject ranking (business metric)
- Lift over random
- PSI for drift detection

---

## 7. Research Extensions (Fair NN Comparison)

If future work wants to make a defensible claim about "NN vs GBDT," the neural baseline must be modern:

### 7.1 Missingness-Native Modeling
- Feature tokenization with learnable "missing token" embedding
- Masked modeling where missing features are *masked*, not median-filled
- Robust scaling (RobustScaler / log transforms) instead of MinMax

### 7.2 End-to-End Fine-Tuning
- Attach classification head to VIME encoder
- Fine-tune on D_L labels (and optionally pseudo-labels)
- Don't freeze embeddings as a bottleneck

### 7.3 Objective Alignment
- Contrastive objectives instead of pure reconstruction
- Auxiliary loss that preserves missingness signal explicitly

---

## 8. Limitations

This conclusion is scoped to:
- **This dataset:** Home Credit Default Risk (307k samples, 132 features)
- **This methodology:** VIME with median imputation + MinMax scaling + one-hot encoding
- **This regime:** Medium-scale tabular data with 40-60% missing rates in key features

It should **not** be generalized to:
- Large-scale tabular data where NNs may have more capacity advantage
- Datasets with low/random missingness
- Modern missingness-aware NN architectures (TabTransformer, FT-Transformer with masking, etc.)

---

*Report generated: Phase 5.5 Extended Routes Analysis*
*Dataset: Home Credit Default Risk (Kaggle)*
*Full project documentation: See `docs/reports/reject_inference_deep_conclusion_next_steps.md`*
