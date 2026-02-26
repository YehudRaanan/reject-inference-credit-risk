# EDA: Preprocessing Impact Analysis

> **Purpose**: Compare raw data predictive power vs VIME-transformed data to identify transformations that damage predictive ability.

## Executive Summary

**Key Finding**: VIME preprocessing causes significant information loss due to:
1. **MinMax scaling with extreme outliers** - Compresses 99% of data to narrow ranges
2. **Median imputation** - Destroys "missing = signal" information
3. **One-hot encoding** - Spreads categorical signal across sparse columns

**Impact**: Route B (raw features with CatBoost native handling) outperforms Route A (VIME embeddings) by ~4.5% AUC on rejected population.

---

## 1. Numerical Columns Analysis (116 features)

### 1.1 Critical Findings: Most Damaged Features

| Feature | Missing% | Raw Corr | VIME Corr | Loss% | Root Cause |
|---------|----------|----------|-----------|-------|------------|
| EXT_SOURCES_PROD | 62.5% | -0.1891 | -0.0966 | **48.9%** | Median imputation destroys missing signal |
| EXT_SOURCES_WEIGHTED | 62.5% | -0.1843 | -0.0939 | **49.0%** | Median imputation destroys missing signal |
| EXT_SOURCE_1 | 54.6% | -0.1469 | -0.0872 | **40.6%** | Median imputation destroys missing signal |
| EXT_SOURCE_3 | 19.8% | -0.1728 | -0.1468 | 15.0% | Minor loss |
| APARTMENTS_AVG | 48.9% | -0.0703 | -0.0267 | **62.1%** | High missing + median impute |
| LIVINGAREA_AVG | 48.9% | -0.0700 | -0.0262 | **62.6%** | High missing + median impute |

### 1.2 Best Preserved Features

| Feature | Missing% | Raw Corr | VIME Corr | Loss% | Notes |
|---------|----------|----------|-----------|-------|-------|
| EXT_SOURCES_MEAN | 0.0% | -0.1835 | -0.1835 | 0.0% | Fully imputed aggregate |
| EXT_SOURCE_2 | 14.7% | -0.1576 | -0.1547 | 1.8% | Low missing rate |
| DAYS_BIRTH (Age) | 0.0% | -0.0698 | -0.0698 | 0.0% | No missing data |
| REGION_RATING_CLIENT | 0.0% | 0.0578 | 0.0578 | 0.0% | No missing data |

### 1.3 Missing Indicators Analysis (66 columns)

The preprocessing adds `missingindicator_*` binary columns for features with missing values. Do these recover the lost signal?

**Top Missing Indicators by Correlation with TARGET:**

| Indicator | Missing% | Corr with TARGET | p-value |
|-----------|----------|------------------|---------|
| EMPLOYED_YEARS | 17.5% | -0.0362 | 9.4e-45 |
| FLOORSMAX_AVG | 46.7% | +0.0310 | 3.4e-33 |
| ENTRANCES_AVG | 47.3% | +0.0309 | 3.4e-33 |
| TOTALAREA_MODE | 45.2% | +0.0306 | 1.6e-32 |
| ELEVATORS_AVG | 50.3% | +0.0302 | 9.3e-32 |
| APARTMENTS_AVG | 47.7% | +0.0294 | 4.5e-30 |

**Signal Recovery Analysis (Value + Indicator vs Raw):**

| Feature | Raw Corr | VIME Value | Indicator | Combined | Net Effect |
|---------|----------|------------|-----------|----------|------------|
| EXT_SOURCE_1 | -0.151 | -0.072 | +0.007 | 0.080 | **-0.071 lost** |
| EXT_SOURCE_3 | -0.166 | -0.119 | +0.012 | 0.131 | **-0.036 lost** |
| EXT_SOURCES_PROD | -0.217 | -0.087 | +0.010 | 0.097 | **-0.120 lost** |
| APARTMENTS_AVG | -0.035 | -0.012 | +0.029 | 0.042 | +0.007 gained |
| FLOORSMAX_AVG | -0.046 | -0.025 | +0.031 | 0.056 | +0.010 gained |

**Key Findings:**
1. Missing indicators **do preserve some signal** (positive = missing correlates with higher default)
2. For housing features (low raw correlation), indicators **fully compensate** for median imputation
3. For EXT_SOURCE features (high raw correlation), indicators only recover **10-20%** of lost signal
4. The **most predictive features suffer the most net loss**

**Conclusion:** Missing indicators help but cannot compensate for the value distribution damage caused by median imputation on high-correlation features.

### 1.4 AMT_INCOME_TOTAL (Critical Feature)

**Problem**: Extreme outlier (117M vs median 135K) caused MinMax scaling to compress 99% of income values to [0, 0.03].

| Metric | Value |
|--------|-------|
| Missing Rate | 0.0% |
| Min | 25,650 |
| Max | 117,000,000 |
| Median | 135,000 |
| 99th Percentile | 517,500 |
| Max / 99th | 226x |
| VIME Variance | 0.000046 |
| VIME Variance Rank | 315/328 (bottom 4%) |

**Result**: VIME learned to ignore income because MSE loss prioritizes high-variance features. Income reconstruction correlation on D_R: **0.07** (essentially noise).

### 1.4 Pattern: Missing Rate vs Correlation Loss

```
Missing Rate    Avg Correlation Loss
-----------    ---------------------
0-10%          ~5% loss
10-30%         ~15% loss
30-50%         ~35% loss
50-70%         ~45-50% loss
```

**Root Cause**: Median imputation replaces NaN with population median, destroying the "missing data is informative" signal. In credit scoring, missing data often indicates:
- Self-employed / informal income (no documentation)
- Recent immigrants / thin-file applicants
- Higher-risk applicants who don't provide all information

---

## 2. Categorical Columns Analysis (16 features)

### 2.1 Predictive Power Summary

| Column | Missing% | Unique | Cramer's V | One-Hot Cols | Max Corr | Notes |
|--------|----------|--------|------------|--------------|----------|-------|
| OCCUPATION_TYPE | 31.3% | 18 | 0.0815 | 19 | 0.0307 | High miss, Predictive |
| ORGANIZATION_TYPE | 0.0% | 58 | 0.0723 | 58 | 0.0362 | High cardinality |
| NAME_INCOME_TYPE | 0.0% | 8 | 0.0638 | 8 | 0.0438 | Predictive |
| NAME_EDUCATION_TYPE | 0.0% | 5 | 0.0576 | 5 | 0.0457 | Predictive |
| CODE_GENDER | 0.0% | 3 | 0.0547 | 3 | 0.0429 | Predictive |
| NAME_FAMILY_STATUS | 0.0% | 6 | 0.0405 | 6 | 0.0167 | Moderate |
| NAME_HOUSING_TYPE | 0.0% | 6 | 0.0370 | 6 | 0.0223 | Moderate |
| NAME_CONTRACT_TYPE | 0.0% | 2 | 0.0309 | 2 | 0.0277 | Low |

### 2.2 Default Rate Ranges by Category

| Feature | Min Default Rate | Max Default Rate | Spread |
|---------|------------------|------------------|--------|
| NAME_INCOME_TYPE | 0.0% (Student) | 40.0% (Maternity) | 40.0% |
| ORGANIZATION_TYPE | 3.1% (Industry 12) | 15.8% (Transport 3) | 12.6% |
| OCCUPATION_TYPE | 4.8% (Accountant) | 17.2% (Low-skill) | 12.3% |
| CODE_GENDER | 0.0% (XNA) | 10.1% (Male) | 10.1% |
| NAME_EDUCATION_TYPE | 1.8% (Academic) | 10.9% (Lower secondary) | 9.1% |

### 2.3 One-Hot Encoding Impact

**Problem**: High-cardinality categoricals become sparse binary columns.

| Feature | Raw Representation | VIME Representation |
|---------|-------------------|---------------------|
| ORGANIZATION_TYPE | Single column, 58 values | 58 binary columns (mostly 0) |
| OCCUPATION_TYPE | Single column, 18 values | 19 binary columns (+ missing) |

**Result**:
- Signal spread across many sparse columns
- Each binary column has low variance
- VIME reconstruction struggles with sparse features
- CatBoost target encoding preserves signal in single column

---

## 3. Root Cause Analysis

### 3.1 MinMax Scaling + Outliers

```
Formula: X_scaled = (X - X_min) / (X_max - X_min)

Income Example:
- X_min = 25,650
- X_max = 117,000,000
- Range = 116,974,350

For median income (135,000):
scaled = (135,000 - 25,650) / 116,974,350 = 0.00093

Result: 99% of income values compressed to [0, 0.005]
```

### 3.2 Median Imputation

**Before imputation** (raw data):
- Feature has NaN for some rows
- NaN itself is informative (correlates with target)

**After median imputation**:
- All NaN replaced with population median
- "Missing" information lost
- Exception: `_is_missing` binary indicator added
- But: Binary indicator loses magnitude of missingness patterns

### 3.3 One-Hot Encoding vs Target Encoding

| Aspect | One-Hot (VIME) | Target Encoding (CatBoost) |
|--------|----------------|---------------------------|
| Dimensions | 1 column → N columns | 1 column → 1 column |
| Sparsity | Very sparse (N-1 zeros per row) | Dense |
| Signal | Spread across columns | Concentrated |
| Tree splits | Many splits needed | Single split captures |

---

## 4. Recommendations

### 4.1 For Future VIME Implementations

1. **Use robust scaling instead of MinMax**
   - RobustScaler: Uses IQR, ignores outliers
   - StandardScaler: Uses mean/std, less affected by outliers

2. **Preserve missing indicators with more information**
   - Use missingness patterns (which columns co-miss)
   - Consider missingness as categorical feature

3. **Handle high-cardinality categoricals differently**
   - Target encoding before VIME
   - Entity embeddings for categoricals
   - Frequency encoding

4. **Cap extreme outliers before scaling**
   - Winsorize at 99th percentile
   - Log transform for highly skewed features

### 4.2 For This Project

Given the preprocessing damage identified:
- **Route B (raw + CatBoost)** is the correct choice
- VIME provides no advantage when preprocessing destroys signal
- For diamond recovery, raw features with native categorical handling is optimal

---

## 5. Data Quality Summary

### Overall Statistics

| Category | Count | Avg Correlation Loss |
|----------|-------|---------------------|
| Features with >50% missing | 45 | 48.2% |
| Features with 30-50% missing | 18 | 35.1% |
| Features with <30% missing | 53 | 8.4% |

### Most Predictive Features (Raw)

| Rank | Feature | Raw Correlation | After VIME | Status |
|------|---------|-----------------|------------|--------|
| 1 | EXT_SOURCES_PROD | -0.189 | -0.097 | DAMAGED |
| 2 | EXT_SOURCES_WEIGHTED | -0.184 | -0.094 | DAMAGED |
| 3 | EXT_SOURCES_MEAN | -0.184 | -0.184 | PRESERVED |
| 4 | EXT_SOURCE_3 | -0.173 | -0.147 | MINOR LOSS |
| 5 | EXT_SOURCE_2 | -0.158 | -0.155 | PRESERVED |
| 6 | EXT_SOURCE_1 | -0.147 | -0.087 | DAMAGED |

---

## Appendix: Full Column Analysis

### A. Numerical Columns (116 total)

```
Column                          | Miss%  | Raw Corr | VIME Corr | Loss%
---------------------------------------------------------------------
EXT_SOURCES_PROD                | 62.5%  | -0.1891  | -0.0966   | 48.9%
EXT_SOURCES_WEIGHTED            | 62.5%  | -0.1843  | -0.0939   | 49.0%
EXT_SOURCES_MEAN                |  0.0%  | -0.1835  | -0.1835   |  0.0%
EXT_SOURCE_3                    | 19.8%  | -0.1728  | -0.1468   | 15.0%
EXT_SOURCE_2                    | 14.7%  | -0.1576  | -0.1547   |  1.8%
EXT_SOURCE_1                    | 54.6%  | -0.1469  | -0.0872   | 40.6%
DAYS_BIRTH                      |  0.0%  | -0.0698  | -0.0698   |  0.0%
ANNUITY_INCOME_RATIO            |  0.0%  | -0.0699  | -0.0699   |  0.0%
APARTMENTS_AVG                  | 48.9%  | -0.0703  | -0.0267   | 62.1%
LIVINGAREA_AVG                  | 48.9%  | -0.0700  | -0.0262   | 62.6%
REGION_RATING_CLIENT            |  0.0%  |  0.0578  |  0.0578   |  0.0%
REGION_RATING_CLIENT_W_CITY     |  0.0%  |  0.0608  |  0.0608   |  0.0%
AMT_INCOME_TOTAL                |  0.0%  | -0.0054  | -0.0054   |  0.0%
```

*(See full numerical analysis in previous EDA output)*

### B. Categorical Columns (16 total)

```
Column                          | Miss%  | Unique | Cramer's V | One-Hot
------------------------------------------------------------------------
OCCUPATION_TYPE                 | 31.3%  |   18   |   0.0815   |   19
ORGANIZATION_TYPE               |  0.0%  |   58   |   0.0723   |   58
NAME_INCOME_TYPE                |  0.0%  |    8   |   0.0638   |    8
NAME_EDUCATION_TYPE             |  0.0%  |    5   |   0.0576   |    5
CODE_GENDER                     |  0.0%  |    3   |   0.0547   |    3
NAME_FAMILY_STATUS              |  0.0%  |    6   |   0.0405   |    6
NAME_HOUSING_TYPE               |  0.0%  |    6   |   0.0370   |    6
NAME_CONTRACT_TYPE              |  0.0%  |    2   |   0.0309   |    2
WALLSMATERIAL_MODE              | 50.8%  |    7   |   0.0303   |    8
FLAG_OWN_CAR                    |  0.0%  |    2   |   0.0218   |    2
HOUSETYPE_MODE                  | 50.2%  |    3   |   0.0134   |    4
FONDKAPREMONT_MODE              | 68.4%  |    4   |   0.0131   |    5
EMERGENCYSTATE_MODE             | 47.4%  |    2   |   0.0121   |    3
NAME_TYPE_SUITE                 |  0.4%  |    7   |   0.0104   |    8
WEEKDAY_APPR_PROCESS_START      |  0.0%  |    7   |   0.0071   |    7
FLAG_OWN_REALTY                 |  0.0%  |    2   |   0.0061   |    2
```

---

*Report generated: Phase 5.5 Extended Routes Analysis*
*Dataset: Home Credit Default Risk (307k applications, 132 features)*
