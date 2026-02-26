# Data Preprocessing & Validation Guide

This document details the distinct data preprocessing pipelines applied before modeling. The project features a **Shared Base Pipeline** followed by **Model-Specific Branching** tailored to the inductive biases of different architectures (CatBoost, VIME, and DCN-v2).

The purpose of this guide is to provide complete transparency for outside validation and reproducibility.

---

## 1. Phase 1: Shared Base Preprocessing (`src/preprocessing/shared_pipeline.py`)

All data undergoes this shared pipeline first, ensuring a unified feature engineering and train/test/reject split mechanism.

### 1.1 Data Cleaning
* **Anomaly Handling**: Replaces the sentinel value `365243` in `DAYS_EMPLOYED` with `NaN` (affects ~18% of the data).
* **Time Conversion**: Converts negative `DAYS_*` features (e.g., `DAYS_BIRTH`, `DAYS_ID_PUBLISH`) into positive `*_YEARS` formats.

### 1.2 Feature Engineering
* **Document Stats**: Aggregates `FLAG_DOCUMENT_*` columns into `DOCUMENT_COUNT` and `NEW_DOC_KURT` (kurtosis).
* **External Sources (`EXT_SOURCE_1, 2, 3`)**: Computes `MIN`, `MAX`, `MEAN`, `VAR`, `PROD`, and a `WEIGHTED` sum.
* **Financial Ratios**: Connects loan attributes to applicant resources (e.g., `CREDIT_TO_ANNUITY_RATIO`, `ANNUITY_TO_INCOME_RATIO`, `CREDIT_TO_GOODS_RATIO`, `CREDIT_TO_INCOME_RATIO`).
* **Time & Efficiency Ratios**: E.g., `INCOME_TO_EMPLOYED_RATIO`, `INCOME_TO_BIRTH_RATIO`, `EMPLOYED_TO_BIRTH_RATIO`.
* **Dropped Columns**: ID flags or non-informative fields (as defined in `config.DROP_COLS`) are removed.

### 1.3 Scorecard Reject Simulation & Data Splitting
* **Simulation Rule**: A simulated banking reject threshold is applied using the 30th percentile of `EXT_SOURCE_2`. 
  * Scores $\ge$ Cutoff $\rightarrow$ **$D_L$ (Approved)**. Labels are kept.
  * Scores $<$ Cutoff $\rightarrow$ **$D_R$ (Rejected)**. Treated as unlabeled during training. (Rows with `NaN` in `EXT_SOURCE_2` are excluded entirely).
* **Internal $D_L$ Splits**: The approved set $D_L$ is split into Train (70%), Validation (15%), and Test (15%) using stratified sampling on the `TARGET`.

---

## 2. Phase 2: Model-Specific Preprocessing Variations

Because neural networks and tree-based methods process data differently, the shared output diverges into three distinct pipelines.

### Route A: VIME (Self-Supervised Neural Network)
**Implementation file:** `src/preprocessing/route_a_prep.py`

VIME requires fully standardized numeric inputs and handles missingness via partial reconstruction.
* **Missing Value Indicators**: A distinct binary mask (`missing_*`) is generated for all numerical columns *before* imputation. This allows VIME's self-supervised loss to focus on reconstructing artificially corrupted data rather than naturally missing data.
* **Imputation**:
  * *Numerical*: Median imputation + built-in scikit-learn Missing Indicator.
  * *Low-Cardinality Categorical*: Most-frequent (Mode) imputation.
  * *High-Cardinality Categorical*: Constant string `"_MISSING_"` imputation.
* **Scaling**:
  * *Numerical*: `ClippedRobustScaler`. Clips extreme outliers to predefined percentiles, then scales using Interquartile Range (IQR). Ensures extreme outliers (e.g., `AMT_INCOME_TOTAL`) do not compress the activation space.
* **Categorical Encoding**:
  * *Low-Cardinality*: Standard One-Hot Encoding (`OneHotEncoder`).
  * *High-Cardinality*: Target Encoding fitted **strictly on $D_L\_train$** (`category_encoders.TargetEncoder` with smoothing). Prevents explosion of the feature space (reduces ~328 columns to ~200).

### Route B: CatBoost (Gradient Boosted Trees)
**Implementation files:** `src/models/route_b_classical.py` (and `route_b_prep.py`)

*Validation Note: There are two methodologies for CatBoost present in the code. The primary modeling script (`route_b_classical.py`) bypasses the `route_b_prep.py` script to use CatBoost's native handlers.*

* **Missing Value Indicators**: None.
* **Imputation**:
  * *Numerical*: Left as `NaN`. CatBoost uses dynamic symmetric/asymmetric branch handling for missing values.
  * *Categorical*: Replaced with the literal string `"_MISSING_"`. (CatBoost requires non-null categorical inputs).
* **Scaling**: None. Tree-based learners are scale-invariant.
* **Categorical Encoding**: No pre-encoding is applied. The indices of categorical columns are explicitly mapped and passed to the `cat_features` argument in `CatBoostClassifier`, utilizing its internal Ordered Target Statistics. 

### Route DCN: Deep & Cross Network v2 (Tabular DL)
**Implementation file:** `src/preprocessing/route_dcn_prep.py`

DCN-v2 utilizes native PyTorch categorical `Embedding` layers, requiring separate ingestion of continuous and categorical features.
* **Missing Value Indicators**: None.
* **Imputation**:
  * *Numerical*: Median Imputation.
  * *Categorical*: Not explicitly imputed with strings. Instead, the `LabelEncoder` process reserves index `0` for missing or unknown values (mapping to PyTorch's `padding_idx`).
* **Scaling**:
  * *Numerical*: `ClippedRobustScaler` (identical to VIME). Cast explicitly to `float32`.
* **Categorical Encoding**:
  * No Target Encoding or One-Hot Encoding.
  * Processed via a custom `CategoricalIndexer` (Label Encoding). Values are mapped to strict integers: `1, 2, 3...` up to the cardinality size.
* **Data Output**: Writes numerical and categorical tensors to separate `.parquet` files (`_num.parquet` and `_cat.parquet`), generating a `cardinalities.json` artifact to initialize the PyTorch Embedding dimensions.

---

## 3. Preprocessing Comparison at a Glance

| Feature Treatment | VIME (Route A) | CatBoost (Route B native) | DCN-v2 (Route DCN) |
| :--- | :--- | :--- | :--- |
| **Numerical Imputation** | Median | None (`NaN` passed to model) | Median |
| **Numerical Scaling** | `ClippedRobustScaler` | None | `ClippedRobustScaler` |
| **Missing Masking** | Explicit Target + Separate Mask arrays | None | None |
| **Low-Card Categoricals**| Mode + One-Hot Encoded | `"_MISSING_"` + Internal Handling | Label Encoded (int) |
| **High-Card Categoricals**| `_MISSING_` + Target Encoded (Train only) | `"_MISSING_"` + Internal Handling | Label Encoded (int) |
| **Categorical Missing** | Imputed or `_MISSING_` | `_MISSING_` | Integer index `0` |
| **Data Output Type** | Single dense flat matrix (`float64`) | DataFrames with mixed types (`float/object`) | Split numerical (`float32`) & categorical (`int32`) matrices |

**Methodological Risk for Validation:** Because VIME and CatBoost receive fundamentally different data transformations, isolating the performance difference caused strictly by the *architectures* versus the *preprocessing workflows* requires careful ablation (e.g., evaluating CatBoost on VIME's exactly preprocessed matrix).
