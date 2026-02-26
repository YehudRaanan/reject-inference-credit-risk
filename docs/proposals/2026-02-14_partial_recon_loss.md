# Proposal: Partial Reconstruction Loss (Masked MSE)

**STATUS: APPROVED**

## What

Modify VIME's reconstruction loss to exclude originally-missing positions. Only compute MSE where data was genuinely present, not where we imputed median values.

Implementation:
```python
def masked_mse_loss(x_hat, x_clean, missing_mask):
    valid_mask = 1.0 - missing_mask  # 1 where data was present
    squared_error = (x_hat - x_clean) ** 2
    masked_error = squared_error * valid_mask
    return masked_error.sum() / (valid_mask.sum() + 1e-8)
```

## Why

**Problem Identified in EDA** (`docs/reports/eda_preprocessing_impact.md`):

Current approach computes MSE on ALL positions including imputed values:
- High-missing features (EXT_SOURCE_1: 54.6% missing) have median imputed
- VIME learns to reconstruct median values, not real patterns
- Correlation loss: 40-60% for high-missing features

**Example - EXT_SOURCE_1:**
- Raw correlation with TARGET: -0.151
- After imputation: -0.072 (52% lost)
- Missing indicator recovery: +0.007 (only 4.6% recovered)

The model is rewarded for predicting medians where data was missing, rather than learning meaningful patterns.

## Alternatives

1. **Keep current MSE**: Simple but loses signal from missing patterns.
2. **Weighted MSE**: Weight down imputed positions (partial solution).
3. **Masked MSE (selected)**: Completely ignore imputed positions in loss.
4. **Separate reconstruction heads**: Different outputs for present vs missing (complex).

## Impact

**Files Modified:**
- `src/models/vime/vime_trainer.py` - Add masked_mse_loss, update training loop
- `src/preprocessing/route_a_prep.py` - Save missing masks separately
- `src/models/vime/vime_utils.py` - Missing mask handling utilities

**Data Changes:**
- New files: `vime_missing_mask_*.parquet` alongside feature files

**Expected Improvement:**
- Better reconstruction of present values
- Embeddings capture actual feature distributions, not median artifacts
- Estimated AUC improvement: +0.5-1.5%

**Risks:**
- If a feature is >90% missing, very few positions to learn from
- Training may be slower due to mask operations
