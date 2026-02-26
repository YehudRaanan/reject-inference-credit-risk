# Proposal: Replace MinMaxScaler with RobustScaler

**STATUS: APPROVED**

## What

Replace `MinMaxScaler` with a custom `ClippedRobustScaler` in the Route A preprocessing pipeline (`src/preprocessing/route_a_prep.py`).

The `ClippedRobustScaler`:
1. Applies `RobustScaler` (IQR-based scaling, ignores outliers)
2. Then normalizes to [0, 1] range using min-max on the robust-scaled output
3. Clips to [0, 1] to ensure VIME sigmoid reconstruction works

## Why

**Problem Identified in EDA** (`docs/reports/eda_preprocessing_impact.md`):

AMT_INCOME_TOTAL has extreme outliers:
- Min: 25,650
- Max: 117,000,000 (outlier)
- Median: 135,000
- 99th percentile: 517,500

MinMaxScaler compressed 99% of income values to [0, 0.005] range:
```
X_scaled = (135,000 - 25,650) / 116,974,350 = 0.00093
```

**Impact:**
- Income variance dropped to 0.000046 (rank 315/328 - bottom 4%)
- VIME's MSE loss effectively ignored income during reconstruction
- Income reconstruction correlation on D_R: only 0.07

## Alternatives

1. **StandardScaler**: Mean/std normalization. Still affected by outliers, though less than MinMax.
2. **Winsorization + MinMax**: Cap at 99th percentile before scaling. Loses information above cap.
3. **Log Transform + MinMax**: Reduces outlier impact but changes interpretation.
4. **RobustScaler (selected)**: Uses median and IQR, naturally ignores outliers.

## Impact

**Files Modified:**
- `src/preprocessing/route_a_prep.py` - Use ClippedRobustScaler
- `src/preprocessing/custom_scalers.py` - NEW: ClippedRobustScaler class
- `config.py` - Add ROBUST_SCALER config section

**Expected Improvement:**
- AMT_INCOME_TOTAL variance rank: improve from 315/328 to < 100
- Better feature representation in VIME latent space
- Estimated AUC improvement: +0.5-1.0%

**Risks:**
- Values outside training range could clip to 0 or 1 (acceptable for VIME)
- Need to rerun VIME training after preprocessing change
