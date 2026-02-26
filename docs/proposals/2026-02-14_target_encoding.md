# Proposal: Target Encoding for High-Cardinality Categoricals

**STATUS: APPROVED**

## What

Replace One-Hot Encoding with smoothed Target Encoding for high-cardinality categorical features:
- ORGANIZATION_TYPE (58 categories)
- OCCUPATION_TYPE (18 categories)

Low-cardinality categoricals (CODE_GENDER, FLAG_OWN_CAR, etc.) remain One-Hot encoded.

## Why

**Problem Identified in EDA** (`docs/reports/eda_preprocessing_impact.md`):

One-Hot Encoding spreads signal across many sparse binary columns:
- ORGANIZATION_TYPE: 58 unique values → 58 sparse binary columns
- Each binary column has low variance (most values are 0)
- VIME reconstruction struggles with sparse features
- Signal diluted across many columns

**CatBoost Advantage:**
CatBoost uses native target encoding internally, preserving signal in a single column.
Route B outperforms Route A partly because of this.

## Alternatives

1. **Keep One-Hot**: Current approach, but signal dilution problem.
2. **Frequency Encoding**: Replace category with count. No target leakage but less predictive.
3. **Entity Embeddings**: Learnable vectors per category. Requires architectural changes.
4. **Target Encoding (selected)**: Replace with smoothed default rate per category.

## Implementation Details

**Smoothed Target Encoding:**
```python
from category_encoders import TargetEncoder

encoder = TargetEncoder(
    smoothing=10.0,      # Bayesian smoothing parameter
    min_samples_leaf=20  # Minimum samples for reliable estimate
)
```

**Critical Constraints:**
- Fit ONLY on D_L_train (where labels exist)
- D_R and test sets use transform() only (no leakage)
- This is acceptable because target encoding is applied BEFORE VIME SSL training

## Impact

**Files Modified:**
- `src/preprocessing/route_a_prep.py` - Add TargetEncoder for high-cardinality cats
- `config.py` - Add HIGH_CARDINALITY_CATS, TARGET_ENCODER_SMOOTHING
- `requirements.txt` - Add category_encoders package

**Feature Dimension:**
- Before: ~328 features (58 + 18 = 76 sparse one-hot columns)
- After: ~200 features (2 target-encoded columns instead)

**Expected Improvement:**
- Concentrated signal in fewer columns
- Better VIME reconstruction of categorical information
- Estimated AUC improvement: +0.3-0.8%

**Risks:**
- Target leakage if encoder is fit on test/reject data (mitigated by fit on D_L_train only)
- Rare categories may have noisy estimates (mitigated by smoothing parameter)
