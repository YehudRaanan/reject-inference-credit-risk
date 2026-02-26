"""
Shared Preprocessing Pipeline — Home Credit Default Risk
========================================================

This module handles data loading, cleaning, feature engineering,
and the scorecard-based reject simulation that splits the data
into D_L (approved, labeled) and D_R (rejected, labels hidden).

Data Flow:
    application_train.csv
        → clean_data()           → handle anomalies, drop low-utility cols
        → engineer_features()    → credit ratios, age, employment years
        → simulate_reject()      → D_L (score ≥ cutoff) + D_R (score < cutoff)
        → split_approved()       → D_L_train / D_L_val / D_L_test
        → save all splits        → data/processed/*.parquet

The reject simulation uses EXT_SOURCE_2 as a proxy for a bank's
existing scorecard. In real banking, applicants below a certain
credit score are automatically declined. We replicate this by
setting a cutoff at the 30th percentile of EXT_SOURCE_2.

The D_R population is higher-risk on average (~15-20% default rate
vs ~5-6% in D_L), but ~80% of them are actually good borrowers —
"diamonds in the mud" that our models aim to identify.
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

def load_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the Home Credit application_train.csv dataset.

    Args:
        path: Path to CSV file. Defaults to config.RAW_TRAIN.

    Returns:
        Raw DataFrame with all columns and rows as-is.
    """
    path = path or config.RAW_TRAIN
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


# ──────────────────────────────────────────────
# 2. DATA CLEANING
# ──────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Home Credit data.

    Steps:
        1. Fix DAYS_EMPLOYED anomaly: the value 365,243 appears for
           retired or unknown employment status. Replace with NaN
           so it doesn't corrupt statistics or model training.

        2. Convert DAYS_* columns from negative days-before-application
           to positive years (more intuitive for analysis).

        3. Encode TARGET as int (0/1) if it isn't already.

    Args:
        df: Raw DataFrame from load_data().

    Returns:
        Cleaned DataFrame with anomalies fixed and days → years.

    Note:
        We intentionally do NOT drop columns here — that's handled
        by the route-specific preprocessors or config.DROP_COLS.
    """
    df = df.copy()

    # --- Fix DAYS_EMPLOYED anomaly ---
    # 365,243 days ≈ 1,000 years — clearly a sentinel value
    # Affects ~55k rows in training data
    anomaly_mask = df["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_ANOMALY
    n_anomalies = anomaly_mask.sum()
    logger.info(
        f"DAYS_EMPLOYED anomaly (={config.DAYS_EMPLOYED_ANOMALY}): "
        f"{n_anomalies} rows ({n_anomalies / len(df):.1%}) → NaN"
    )
    df.loc[anomaly_mask, "DAYS_EMPLOYED"] = np.nan

    # --- Convert DAYS_* to positive years ---
    # Home Credit encodes these as negative integers (days before application)
    # e.g., DAYS_BIRTH = -12000 means born ~32.9 years ago
    days_cols = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
                 "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]
    for col in days_cols:
        if col in df.columns:
            df[col] = df[col] / -365.25  # Convert to positive years
            # Rename to make meaning clear
            new_name = col.replace("DAYS_", "") + "_YEARS"
            df = df.rename(columns={col: new_name})
            logger.info(f"Converted {col} → {new_name} (positive years)")

    # --- Ensure TARGET is int ---
    if config.TARGET_COL in df.columns:
        df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)

    return df


# ──────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw columns.

    Engineered features:
        - CREDIT_INCOME_RATIO:  AMT_CREDIT / AMT_INCOME_TOTAL
          → How many years of income does the loan represent?
          → Higher ratio = more leveraged = higher risk.

        - ANNUITY_INCOME_RATIO: AMT_ANNUITY / AMT_INCOME_TOTAL
          → What fraction of monthly income goes to repayment?
          → Banks typically cap this at 30-40%.

        - GOODS_CREDIT_RATIO:   AMT_GOODS_PRICE / AMT_CREDIT
          → How much of the credit actually goes to the goods?
          → Ratio < 1 suggests fees/insurance bundled in.

        - EXT_SOURCE_MEAN:      Mean of EXT_SOURCE_1/2/3
          → Aggregate external credit assessment.
          → Handles NaN by taking mean of available scores.

    Args:
        df: Cleaned DataFrame from clean_data().

    Returns:
        DataFrame with additional engineered columns.
    """
    df = df.copy()

    # --- Document features (Source: js-aguiar) ---
    docs = [f for f in df.columns if "FLAG_DOCUMENT" in f]
    df["DOCUMENT_COUNT"] = df[docs].sum(axis=1)
    df["NEW_DOC_KURT"] = df[docs].kurtosis(axis=1)

    # --- External source interactions (Source: js-aguiar) ---
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    if all(c in df.columns for c in ext_cols):
        df["EXT_SOURCES_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        df["EXT_SOURCES_WEIGHTED"] = df["EXT_SOURCE_1"] * 2 + df["EXT_SOURCE_2"] * 1 + df["EXT_SOURCE_3"] * 3
        
    # Statistical aggregates for external sources
    # Note: We use np.nan functions to handle missing values
    df["EXT_SOURCES_MIN"] = df[ext_cols].min(axis=1)
    df["EXT_SOURCES_MAX"] = df[ext_cols].max(axis=1)
    df["EXT_SOURCES_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCES_VAR"] = df[ext_cols].var(axis=1)

    # --- Financial Ratios (Source: js-aguiar) ---
    # 1. Credit Length Ratios
    if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
        df["CREDIT_TO_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
        df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        
    # 2. Goods Coverage
    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
        
    # 3. Leverage
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        
    # --- Time & Efficiency Ratios (Source: js-aguiar) ---
    # Note: clean_data() renames DAYS_X -> DAYS_X_YEARS (positive)
    
    # 4. Income Efficiency
    if "AMT_INCOME_TOTAL" in df.columns:
        if "DAYS_EMPLOYED_YEARS" in df.columns:
            df["INCOME_TO_EMPLOYED_RATIO"] = df["AMT_INCOME_TOTAL"] / df["DAYS_EMPLOYED_YEARS"].replace(0, np.nan)
        if "DAYS_BIRTH_YEARS" in df.columns:
            df["INCOME_TO_BIRTH_RATIO"] = df["AMT_INCOME_TOTAL"] / df["DAYS_BIRTH_YEARS"]
        
    # 5. Time Ratios
    if "DAYS_BIRTH_YEARS" in df.columns:
        if "DAYS_EMPLOYED_YEARS" in df.columns:
            df["EMPLOYED_TO_BIRTH_RATIO"] = df["DAYS_EMPLOYED_YEARS"] / df["DAYS_BIRTH_YEARS"]
        if "OWN_CAR_AGE" in df.columns:
             df["CAR_TO_BIRTH_RATIO"] = df["OWN_CAR_AGE"] / df["DAYS_BIRTH_YEARS"]
        if "ID_PUBLISH_YEARS" in df.columns:
            df["ID_TO_BIRTH_RATIO"] = df["ID_PUBLISH_YEARS"] / df["DAYS_BIRTH_YEARS"]

    if "OWN_CAR_AGE" in df.columns and "DAYS_EMPLOYED_YEARS" in df.columns:
        df["CAR_TO_EMPLOYED_RATIO"] = df["OWN_CAR_AGE"] / df["DAYS_EMPLOYED_YEARS"]

    logger.info("Engineered features (Source: js-aguiar): Docs, ExtSource stats, Financial/Time Ratios")

    return df


# ──────────────────────────────────────────────
# 4. SCORECARD-BASED REJECT SIMULATION
# ──────────────────────────────────────────────

def simulate_reject(
    df: pd.DataFrame,
    score_col: str | None = None,
    cutoff_percentile: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Simulate a bank's reject decision using an external credit score.

    In real banking, applicants below a scorecard cutoff are automatically
    declined — their true creditworthiness is never observed. We replicate
    this using EXT_SOURCE_2:

        - Applicants with score ≥ cutoff → D_L ("approved", labels visible)
        - Applicants with score < cutoff → D_R ("rejected", labels hidden)
        - Applicants with NaN score      → EXCLUDED from study (to avoid
          structural correlation between missingness and rejection label)

    The resulting D_R population is:
        - Higher-risk on average (~13-14% default rate vs ~5-6% in D_L)
        - BUT ~86% of them are actually good borrowers ("diamonds")
        - Finding these diamonds is the whole point of reject inference

    Args:
        df:                  Full dataset with TARGET column.
        score_col:           Column to use as scorecard. Default: config.REJECT_SCORE_FEATURE.
        cutoff_percentile:   Percentile cutoff. Default: config.REJECT_CUTOFF_PERCENTILE.

    Returns:
        d_l:             Approved population (labels visible). Contains TARGET.
        d_r:             Rejected population (labels present but should be hidden
                         during training; kept for evaluation).
        cutoff_value:    The actual score value used as cutoff.
    """
    score_col = score_col or config.REJECT_SCORE_FEATURE
    cutoff_percentile = cutoff_percentile or config.REJECT_CUTOFF_PERCENTILE

    # Exclude rows with NaN score to avoid structural missingness-rejection correlation
    nan_mask = df[score_col].isna()
    n_excluded = nan_mask.sum()
    nan_default_rate = df.loc[nan_mask, config.TARGET_COL].mean() if n_excluded > 0 else 0
    logger.info(
        f"Excluding {n_excluded} rows with NaN {score_col} "
        f"(default rate: {nan_default_rate:.1%}) to avoid missingness-rejection correlation"
    )
    df_valid = df[~nan_mask].copy()

    # Compute cutoff from valid scores
    cutoff_value = np.percentile(df_valid[score_col], cutoff_percentile)
    logger.info(
        f"Reject simulation: {score_col} cutoff = {cutoff_value:.4f} "
        f"({cutoff_percentile}th percentile of {len(df_valid)} valid scores)"
    )

    # Split based on score threshold only (no NaN handling needed)
    approved_mask = df_valid[score_col] >= cutoff_value
    d_l = df_valid[approved_mask].copy()
    d_r = df_valid[~approved_mask].copy()

    # Log statistics
    d_l_rate = d_l[config.TARGET_COL].mean()
    d_r_rate = d_r[config.TARGET_COL].mean()
    d_r_diamonds = 1 - d_r_rate  # fraction of "good" borrowers in rejected pop

    logger.info(f"D_L (approved): {len(d_l)} rows, default rate = {d_l_rate:.1%}")
    logger.info(f"D_R (rejected): {len(d_r)} rows, default rate = {d_r_rate:.1%}")
    logger.info(f"D_R 'diamonds': {d_r_diamonds:.1%} of rejected are actually good")

    return d_l, d_r, cutoff_value


# ──────────────────────────────────────────────
# 5. SPLIT APPROVED POPULATION
# ──────────────────────────────────────────────

def split_approved(
    d_l: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the approved (labeled) population into train / val / test.

    Uses stratified splitting to maintain class balance across splits.

    Splits (of D_L):
        - D_L_train (70%): Train teacher models (both routes)
        - D_L_val   (15%): Validation / early stopping / hyperparameter tuning
        - D_L_test  (15%): Final evaluation on approved population

    Args:
        d_l: Approved population DataFrame with TARGET column.

    Returns:
        d_l_train, d_l_val, d_l_test DataFrames.
    """
    target = d_l[config.TARGET_COL]

    # First split: train vs (val + test)
    val_test_ratio = config.D_L_VAL_RATIO + config.D_L_TEST_RATIO
    d_l_train, d_l_rest = train_test_split(
        d_l, test_size=val_test_ratio,
        stratify=target, random_state=config.SEED,
    )

    # Second split: val vs test (equal halves of the remainder)
    val_fraction = config.D_L_VAL_RATIO / val_test_ratio
    d_l_val, d_l_test = train_test_split(
        d_l_rest, test_size=(1 - val_fraction),
        stratify=d_l_rest[config.TARGET_COL],
        random_state=config.SEED,
    )

    logger.info(
        f"D_L splits: train={len(d_l_train)}, "
        f"val={len(d_l_val)}, test={len(d_l_test)}"
    )
    for name, split in [("train", d_l_train), ("val", d_l_val), ("test", d_l_test)]:
        rate = split[config.TARGET_COL].mean()
        logger.info(f"  D_L_{name}: default rate = {rate:.2%}")

    return d_l_train, d_l_val, d_l_test


# ──────────────────────────────────────────────
# 6. PREPARE D_R (UNLABELED + GROUND TRUTH)
# ──────────────────────────────────────────────

def prepare_dr(
    d_r: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate D_R into unlabeled features and ground truth.

    During training:
        - D_R_features is used for VIME pre-training (no labels)
          and for pseudo-labeling by teacher models
        - D_R_ground_truth is NEVER seen by any model

    During evaluation:
        - D_R_ground_truth provides the true labels to evaluate
          how well each route identified "diamonds in the mud"

    Args:
        d_r: Rejected population DataFrame with TARGET column.

    Returns:
        d_r_features:      Same rows, TARGET column removed.
        d_r_ground_truth:  Same rows, TARGET column kept.
    """
    d_r_ground_truth = d_r.copy()
    d_r_features = d_r.drop(columns=[config.TARGET_COL]).copy()

    logger.info(
        f"D_R prepared: features={d_r_features.shape}, "
        f"ground_truth={d_r_ground_truth.shape}"
    )
    return d_r_features, d_r_ground_truth


# ──────────────────────────────────────────────
# 7. FULL PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────

def run_pipeline(save: bool = True) -> dict:
    """
    Execute the full preprocessing pipeline:
        1. Load raw data
        2. Clean (fix anomalies, convert days → years)
        3. Engineer features (ratios, score aggregation)
        4. Simulate reject (EXT_SOURCE_2 cutoff → D_L + D_R)
        5. Split D_L into train / val / test
        6. Prepare D_R (features vs ground truth)
        7. Save all splits to data/processed/

    Args:
        save: If True, save splits to parquet files.

    Returns:
        Dictionary with all splits and metadata:
        {
            "d_l_train": DataFrame,
            "d_l_val": DataFrame,
            "d_l_test": DataFrame,
            "d_r_features": DataFrame,
            "d_r_ground_truth": DataFrame,
            "metadata": {
                "cutoff_value": float,
                "d_l_size": int,
                "d_r_size": int,
                "d_l_default_rate": float,
                "d_r_default_rate": float,
                ...
            }
        }
    """
    # Load & clean
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)

    # Drop ID columns (not features)
    for col in config.DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info(f"Dropped column: {col}")

    # Simulate reject
    d_l, d_r, cutoff_value = simulate_reject(df)

    # Split approved
    d_l_train, d_l_val, d_l_test = split_approved(d_l)

    # Prepare D_R
    d_r_features, d_r_ground_truth = prepare_dr(d_r)

    # Compile metadata
    metadata = {
        "cutoff_feature": config.REJECT_SCORE_FEATURE,
        "cutoff_percentile": config.REJECT_CUTOFF_PERCENTILE,
        "cutoff_value": float(cutoff_value),
        "seed": config.SEED,
        "d_l_total": len(d_l),
        "d_l_train_size": len(d_l_train),
        "d_l_val_size": len(d_l_val),
        "d_l_test_size": len(d_l_test),
        "d_r_size": len(d_r),
        "d_l_default_rate": float(d_l[config.TARGET_COL].mean()),
        "d_r_default_rate": float(d_r[config.TARGET_COL].mean()),
        "d_l_train_default_rate": float(d_l_train[config.TARGET_COL].mean()),
        "total_features": len(d_l_train.columns) - 1,  # minus TARGET
    }

    # Save
    if save:
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

        d_l_train.to_parquet(config.SPLIT_DL_TRAIN, index=False)
        d_l_val.to_parquet(config.SPLIT_DL_VAL, index=False)
        d_l_test.to_parquet(config.SPLIT_DL_TEST, index=False)
        d_r_features.to_parquet(config.SPLIT_DR_FEATURES, index=False)
        d_r_ground_truth.to_parquet(config.SPLIT_DR_TRUTH, index=False)

        with open(config.SPLIT_METADATA, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"All splits saved to {config.DATA_PROCESSED}")

    return {
        "d_l_train": d_l_train,
        "d_l_val": d_l_val,
        "d_l_test": d_l_test,
        "d_r_features": d_r_features,
        "d_r_ground_truth": d_r_ground_truth,
        "metadata": metadata,
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_pipeline(save=True)
