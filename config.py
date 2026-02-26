"""
Central Configuration — Reject Inference Methodological Comparison
================================================================

Dataset: Home Credit Default Risk (Kaggle)
Goal:    Compare VIME-based hybrid reject inference (Route A)
         vs classical CatBoost baseline (Route B).

Key Idea:
    We simulate a bank's rejection process using EXT_SOURCE_2
    (external credit score) as a scorecard cutoff. Applicants in
    the bottom 30% by score are "rejected" — their labels are hidden.
    Both routes try to recover "diamonds in the mud": good borrowers
    who were wrongly rejected by the blunt score threshold.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# 1. PATHS
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
MODEL_DIR    = OUTPUT_DIR / "models"
FIGURE_DIR   = OUTPUT_DIR / "figures"
RESULT_DIR   = OUTPUT_DIR / "results"

# Raw data files (place these in data/raw/)
RAW_TRAIN    = DATA_RAW / "application_train.csv"
RAW_TEST     = DATA_RAW / "application_test.csv"   # Kaggle test set, unused by us
RAW_BUREAU   = DATA_RAW / "bureau.csv"              # Optional enrichment

# Processed split files
SPLIT_DL_TRAIN      = DATA_PROCESSED / "D_L_train.parquet"
SPLIT_DL_VAL        = DATA_PROCESSED / "D_L_val.parquet"
SPLIT_DL_TEST       = DATA_PROCESSED / "D_L_test.parquet"
SPLIT_DR_FEATURES   = DATA_PROCESSED / "D_R_features.parquet"
SPLIT_DR_TRUTH      = DATA_PROCESSED / "D_R_ground_truth.parquet"
SPLIT_METADATA      = DATA_PROCESSED / "split_metadata.json"

# ──────────────────────────────────────────────
# 2. REPRODUCIBILITY
# ──────────────────────────────────────────────
SEED = 42

# ──────────────────────────────────────────────
# 3. REJECT SIMULATION
# ──────────────────────────────────────────────
# We use EXT_SOURCE_2 as a proxy for a bank's existing scorecard.
# Applicants whose EXT_SOURCE_2 falls in the bottom N-th percentile
# are "rejected" (D_R) — their labels are hidden during training.
#
# Rationale:
#   - EXT_SOURCE_2 has ~85% coverage (least missing of the 3 scores)
#   - Higher EXT_SOURCE_2 correlates with lower default risk
#   - 30th percentile gives ~30% rejected, ~70% approved
#   - This produces D_R with ~15-20% default rate vs ~5-6% in D_L
#   - In real banking, rejected pop IS higher-risk on average
#   - But contains "diamonds": ~80% of D_R would have been fine
REJECT_SCORE_FEATURE = "EXT_SOURCE_2"
REJECT_CUTOFF_PERCENTILE = 30  # Bottom 30% by score → rejected

# ──────────────────────────────────────────────
# 4. DATA SPLITS (within D_L only)
# ──────────────────────────────────────────────
# D_L (approved, labeled) is split into train/val/test:
#   - D_L_train: train teacher models
#   - D_L_val:   validation / early stopping
#   - D_L_test:  final evaluation on approved population
D_L_TRAIN_RATIO = 0.70
D_L_VAL_RATIO   = 0.15
D_L_TEST_RATIO  = 0.15

# ──────────────────────────────────────────────
# 5. TARGET
# ──────────────────────────────────────────────
TARGET_COL = "TARGET"  # 1 = default, 0 = repaid

# ──────────────────────────────────────────────
# 6. FEATURE CONFIGURATION
# ──────────────────────────────────────────────

# Columns to drop (IDs, flags, leakage)
DROP_COLS = [
    "SK_ID_CURR",           # Unique ID — not a feature
]

# Categorical columns (for CatBoost native handling & target encoding)
CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE",        # Cash / Revolving
    "CODE_GENDER",               # M / F / XNA
    "FLAG_OWN_CAR",              # Y / N
    "FLAG_OWN_REALTY",           # Y / N
    "NAME_TYPE_SUITE",           # Who accompanied client
    "NAME_INCOME_TYPE",          # Working / Pensioner / etc.
    "NAME_EDUCATION_TYPE",       # Higher education / Secondary / etc.
    "NAME_FAMILY_STATUS",        # Married / Single / etc.
    "NAME_HOUSING_TYPE",         # House/apartment / Renting / etc.
    "OCCUPATION_TYPE",           # Laborers / Sales / etc. (high-cardinality)
    "ORGANIZATION_TYPE",         # Business / Industry / etc. (high-cardinality)
    "WEEKDAY_APPR_PROCESS_START",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]

# Known data anomalies to fix
DAYS_EMPLOYED_ANOMALY = 365243  # Sentinel for retired/unknown → replace with NaN

# ──────────────────────────────────────────────
# 7. FEATURE ENGINEERING
# ──────────────────────────────────────────────
# New features computed from raw columns
ENGINEERED_FEATURES = {
    "CREDIT_INCOME_RATIO":   ("AMT_CREDIT",      "AMT_INCOME_TOTAL"),
    "ANNUITY_INCOME_RATIO":  ("AMT_ANNUITY",     "AMT_INCOME_TOTAL"),
    "GOODS_CREDIT_RATIO":    ("AMT_GOODS_PRICE",  "AMT_CREDIT"),
}
# Age and employment computed from DAYS_* fields (negative days → positive years)

# ──────────────────────────────────────────────
# 8. VIME HYPERPARAMETERS
# ──────────────────────────────────────────────
VIME_HIDDEN_DIM      = 256       # Hidden layer dimension (encoder FC layers)
VIME_EMBEDDING_DIM   = 128       # Embedding (latent) dimension (output of encoder)
VIME_CORRUPTION_RATE = 0.3       # Default mask rate for regular features
VIME_ALPHA           = 2.0       # Weight for mask loss vs reconstruction loss
VIME_LR              = 1e-3      # Learning rate
VIME_EPOCHS          = 20        # Training epochs (used in trained model)
VIME_BATCH_SIZE      = 128       # Batch size (used in trained model)
VIME_PATIENCE        = 10        # Early stopping patience (not yet implemented)

# 3-Level Masking Probabilities
VIME_P_CRITICAL      = 0.5       # Mask rate for critical features (AMT_INCOME_TOTAL)
VIME_P_HIGH_MISSING  = 0.1       # Mask rate for high-missing features (>50% missing)
VIME_P_REGULAR       = 0.3       # Mask rate for regular features

# Critical features to mask more aggressively
VIME_CRITICAL_FEATURES = ["AMT_INCOME_TOTAL"]

# ──────────────────────────────────────────────
# 9. CATBOOST HYPERPARAMETERS
# ──────────────────────────────────────────────
CB_ITERATIONS    = 2000
CB_DEPTH         = 6
CB_LEARNING_RATE = 0.05
CB_EARLY_STOP    = 200      # Early stopping rounds
CB_EVAL_METRIC   = "AUC"
CB_VERBOSE       = 250

# Dict wrapper for CatBoostClassifier
CATBOOST = {
    "iterations": CB_ITERATIONS,
    "depth": CB_DEPTH,
    "learning_rate": CB_LEARNING_RATE,
    "early_stopping_rounds": CB_EARLY_STOP,
    "eval_metric": CB_EVAL_METRIC,
    "verbose": CB_VERBOSE,
}

# ──────────────────────────────────────────────
# 10. ISOLATION FOREST (Route A only)
# ──────────────────────────────────────────────
# Used to filter noisy pseudo-labels from D_R
IF_CONTAMINATION = 0.15     # Expected fraction of anomalous pseudo-labels
IF_N_ESTIMATORS  = 200
IF_RANDOM_STATE  = SEED

# Dict wrapper for IsolationForest
ISOLATION_FOREST = {
    "contamination": IF_CONTAMINATION,
    "n_estimators": IF_N_ESTIMATORS,
    "random_state": IF_RANDOM_STATE,
}

# ──────────────────────────────────────────────
# 11. PSEUDO-LABELING
# ──────────────────────────────────────────────
PSEUDO_LABEL_THRESHOLD = 0.15  # P(default) threshold for hard labels (lowered to capture more defaults in D_R)

# ──────────────────────────────────────────────
# 12. ROUTE A-FIXED PREPROCESSING (Phase 1 Improvements)
# ──────────────────────────────────────────────
# Step 1.1: RobustScaler instead of MinMaxScaler
ROBUST_SCALER_QUANTILE_RANGE = (25.0, 75.0)  # IQR range for RobustScaler
ROBUST_SCALER_CLIP_PERCENTILES = (1.0, 99.0) # Percentiles for clipping (removes top/bottom 1%)

# Step 1.3: Target Encoding for high-cardinality categoricals
HIGH_CARDINALITY_CATS = [
    "ORGANIZATION_TYPE",  # 58 categories
    "OCCUPATION_TYPE",    # 18 categories
]
TARGET_ENCODER_SMOOTHING = 10.0       # Bayesian smoothing parameter
TARGET_ENCODER_MIN_SAMPLES = 20       # Minimum samples for reliable estimate

# EXT_SOURCE feature list (for Step 1.5: Variable Split)
EXT_SOURCE_COLS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCES_MEAN",
    "EXT_SOURCES_MIN",
    "EXT_SOURCES_MAX",
    "EXT_SOURCES_STD",
    "EXT_SOURCES_PROD",
    "EXT_SOURCES_WEIGHTED",
]

# ──────────────────────────────────────────────
# 13. PHASE 5.7 — FINE-TUNING (Dynamic Embeddings)
# ──────────────────────────────────────────────
# Classification head: 128 → 64 → 32 → 1 (approved by owner 2026-02-15)
FINETUNE_HEAD_DROPOUT = 0.3       # Dropout rate in classification head
FINETUNE_LR           = 1e-3      # Same LR for encoder + head (starting point)
FINETUNE_EPOCHS       = 50        # Max epochs (early stopping expected earlier)
FINETUNE_BATCH_SIZE   = 256       # Batch size for fine-tuning
FINETUNE_PATIENCE     = 10        # Early stopping patience (epochs)
