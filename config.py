"""
Central Configuration - Reject Inference for Credit Risk
=========================================================

Dataset: Home Credit Default Risk (Kaggle)
Goal:    Compare neural network approaches (VIME, DCN-v2) against
         gradient boosted decision trees (CatBoost) for reject inference.

Key Idea:
    We simulate a bank's rejection process using EXT_SOURCE_2
    (external credit score) as a scorecard cutoff. Applicants in
    the bottom 30% by score are "rejected" - their labels are hidden.
    All methods try to recover "diamonds in the mud": good borrowers
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

# Raw data files (download from Kaggle - see data/raw/README.md)
RAW_TRAIN    = DATA_RAW / "application_train.csv"

# Processed split files
SPLIT_DIR       = DATA_PROCESSED / "splits"
SPLIT_DL_TRAIN  = SPLIT_DIR / "D_L_train.parquet"
SPLIT_DL_VAL    = SPLIT_DIR / "D_L_val.parquet"
SPLIT_DL_TEST   = SPLIT_DIR / "D_L_test.parquet"
SPLIT_DR_FEATURES = SPLIT_DIR / "D_R_features.parquet"
SPLIT_DR_TRUTH    = SPLIT_DIR / "D_R_ground_truth.parquet"
SPLIT_METADATA    = DATA_PROCESSED / "metadata" / "split_metadata.json"

# Model-specific data
VIME_DATA_DIR = DATA_PROCESSED / "vime"
DCN_DATA_DIR  = DATA_PROCESSED / "dcn"

# ──────────────────────────────────────────────
# 2. REPRODUCIBILITY
# ──────────────────────────────────────────────
SEED = 42

# ──────────────────────────────────────────────
# 3. REJECT SIMULATION
# ──────────────────────────────────────────────
REJECT_SCORE_FEATURE = "EXT_SOURCE_2"
REJECT_CUTOFF_PERCENTILE = 30  # Bottom 30% by score -> rejected

# ──────────────────────────────────────────────
# 4. DATA SPLITS (within D_L only)
# ──────────────────────────────────────────────
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
DROP_COLS = ["SK_ID_CURR"]

CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]

DAYS_EMPLOYED_ANOMALY = 365243

ENGINEERED_FEATURES = {
    "CREDIT_INCOME_RATIO":   ("AMT_CREDIT",      "AMT_INCOME_TOTAL"),
    "ANNUITY_INCOME_RATIO":  ("AMT_ANNUITY",     "AMT_INCOME_TOTAL"),
    "GOODS_CREDIT_RATIO":    ("AMT_GOODS_PRICE", "AMT_CREDIT"),
}

# ──────────────────────────────────────────────
# 7. VIME HYPERPARAMETERS
# ──────────────────────────────────────────────
VIME_HIDDEN_DIM      = 256
VIME_EMBEDDING_DIM   = 128
VIME_CORRUPTION_RATE = 0.3
VIME_ALPHA           = 2.0
VIME_LR              = 1e-3
VIME_EPOCHS          = 20
VIME_BATCH_SIZE      = 128
VIME_PATIENCE        = 10

# 3-Level Masking Probabilities
VIME_P_CRITICAL      = 0.5
VIME_P_HIGH_MISSING  = 0.1
VIME_P_REGULAR       = 0.3
VIME_CRITICAL_FEATURES = ["AMT_INCOME_TOTAL"]

# ──────────────────────────────────────────────
# 8. CATBOOST HYPERPARAMETERS
# ──────────────────────────────────────────────
CB_ITERATIONS    = 2000
CB_DEPTH         = 6
CB_LEARNING_RATE = 0.05
CB_EARLY_STOP    = 200
CB_EVAL_METRIC   = "AUC"
CB_VERBOSE       = 250

CATBOOST = {
    "iterations": CB_ITERATIONS,
    "depth": CB_DEPTH,
    "learning_rate": CB_LEARNING_RATE,
    "early_stopping_rounds": CB_EARLY_STOP,
    "eval_metric": CB_EVAL_METRIC,
    "verbose": CB_VERBOSE,
}

# ──────────────────────────────────────────────
# 9. PREPROCESSING
# ──────────────────────────────────────────────
ROBUST_SCALER_QUANTILE_RANGE = (25.0, 75.0)
ROBUST_SCALER_CLIP_PERCENTILES = (1.0, 99.0)

HIGH_CARDINALITY_CATS = [
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
]
TARGET_ENCODER_SMOOTHING = 10.0
TARGET_ENCODER_MIN_SAMPLES = 20

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
# 10. CLASSIFICATION HEAD
# ──────────────────────────────────────────────
FINETUNE_HEAD_DROPOUT = 0.3
FINETUNE_LR           = 1e-3
FINETUNE_EPOCHS       = 50
FINETUNE_BATCH_SIZE   = 256
FINETUNE_PATIENCE     = 10
