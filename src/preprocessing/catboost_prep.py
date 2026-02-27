"""
Route B Preprocessing — Classical CatBoost Baseline
====================================================

Prepares data for the classical CatBoost pipeline (no VIME).
This is the baseline approach: train a scorecard on approved data,
pseudo-label rejected applicants, then retrain on the combined dataset.

Key design decisions:
    1. Median/mode imputation: Unlike Route A (which uses sentinel + masks),
       Route B uses traditional imputation. CatBoost can handle NaN natively,
       but for a fair comparison we impute so both routes receive complete data.

    2. Target encoding with smoothing: Same regularized target encoding as
       Route A, fitted on D_L_train only to prevent leakage.

    3. No scaling: CatBoost is tree-based and invariant to feature scale.
       This is a legitimate advantage of gradient boosting over neural nets.

    4. CatBoost categorical features: We pass a list of categorical column
       indices to CatBoost.Pool, letting it handle ordered target statistics
       natively. This is CatBoost's key differentiator over XGBoost/LightGBM.

Data flow:
    D_L_train (features + labels)  ──┐
    D_L_val   (features + labels)  ──┤──→ impute_missing()
    D_L_test  (features + labels)  ──┤    → target_encode()
    D_R_features (features only)   ──┘    → identify cat_feature indices
                                          → CatBoost-ready DataFrames
"""

import logging
import pickle

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def impute_missing(
    df: pd.DataFrame,
    fill_values: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values using median (numerical) and mode (categorical).

    Why median/mode instead of MICE or KNN?
        - Simpler, faster, and sufficient for CatBoost
        - CatBoost handles NaN natively, but we impute for consistency
        - Route B is the "classical" baseline — simplicity is the point
        - More sophisticated imputation would blur the comparison with Route A

    IMPORTANT: fill_values are computed from D_L_train only.
    Val/test/D_R use the pre-computed values to avoid leakage.

    Args:
        df:          DataFrame with potential NaN.
        fill_values: Pre-computed {col: fill_value} dict.
                     If None, computes from df (D_L_train).

    Returns:
        (imputed_df, fill_values) tuple.
    """
    df = df.copy()

    if fill_values is None:
        fill_values = {}
        for col in df.columns:
            if col == config.TARGET_COL:
                continue
            if df[col].dtype in ("object", "category"):
                # Mode for categoricals (most frequent value)
                mode_vals = df[col].mode()
                fill_values[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
            else:
                # Median for numericals (robust to outliers)
                fill_values[col] = df[col].median()
        logger.info(f"Computed fill values for {len(fill_values)} columns")

    # Apply
    n_nan_before = df.isnull().sum().sum()
    for col, val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    n_nan_after = df.isnull().sum().sum()
    logger.info(f"Imputed {n_nan_before - n_nan_after} NaN values (remaining: {n_nan_after})")

    return df, fill_values


def target_encode_categoricals(
    df: pd.DataFrame,
    target: pd.Series | None = None,
    encoding_map: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Target-encode categorical columns using D_L_train statistics.

    Same smoothing formula as Route A to ensure fair comparison.
    The only difference: Route B doesn't add missing masks.

    Args:
        df:           DataFrame with categorical columns.
        target:       TARGET series for fitting (D_L_train only).
        encoding_map: Pre-fitted map. If None, fits from target.

    Returns:
        (encoded_df, encoding_map) tuple.
    """
    df = df.copy()
    smoothing = 10

    if encoding_map is None:
        if target is None:
            raise ValueError("Must provide target when fitting encoding map")
        encoding_map = {}
        global_mean = target.mean()
        for col in config.CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            stats = df.groupby(col)[config.TARGET_COL].agg(["mean", "count"])
            smoothed = (
                (stats["count"] * stats["mean"] + smoothing * global_mean)
                / (stats["count"] + smoothing)
            )
            encoding_map[col] = smoothed.to_dict()
        logger.info(
            f"Fitted target encoding for {len(encoding_map)} categorical columns"
        )

    for col, mapping in encoding_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0.5)

    return df, encoding_map


def get_cat_feature_indices(df: pd.DataFrame) -> list[int]:
    """
    Get column indices of categorical features for CatBoost.Pool.

    CatBoost can handle categoricals natively using ordered target
    statistics. We identify which columns are categorical so CatBoost
    can apply its internal encoding during training.

    Note: After target encoding, categoricals become float, so this
    function should be called BEFORE target encoding if using CatBoost's
    native handling. If using target encoding (our default), this is
    informational only.

    Args:
        df: DataFrame to inspect.

    Returns:
        List of column indices for categorical features.
    """
    cat_indices = []
    for i, col in enumerate(df.columns):
        if col in config.CATEGORICAL_COLS and col != config.TARGET_COL:
            cat_indices.append(i)
    return cat_indices


def prepare_route_b(splits: dict, save_artifacts: bool = True) -> dict:
    """
    Run the full Route B preprocessing pipeline on all splits.

    Pipeline per split:
        1. impute_missing()        → median/mode fill
        2. target_encode()         → categoricals → risk scores

    No scaling (CatBoost is scale-invariant).
    No missing masks (classical approach doesn't track missingness).

    Args:
        splits: Dict with keys "d_l_train", "d_l_val", "d_l_test",
                "d_r_features" (DataFrames from shared_pipeline).
        save_artifacts: If True, pickle fill_values and encoding_map.

    Returns:
        Dict with same keys, values replaced by processed DataFrames.
        Also includes "fill_values" and "encoding_map" keys.
    """
    result = {}

    # Process D_L_train first (fit imputer and encoder)
    d_l_train = splits["d_l_train"].copy()
    d_l_train, fill_values = impute_missing(d_l_train)
    d_l_train, encoding_map = target_encode_categoricals(
        d_l_train, target=d_l_train[config.TARGET_COL]
    )
    result["d_l_train"] = d_l_train

    # Process remaining splits (transform only)
    for key in ["d_l_val", "d_l_test", "d_r_features"]:
        df = splits[key].copy()
        df, _ = impute_missing(df, fill_values=fill_values)
        df, _ = target_encode_categoricals(df, encoding_map=encoding_map)
        result[key] = df
        logger.info(f"Route B preprocessed: {key} → {df.shape}")

    # Save artifacts
    if save_artifacts:
        artifact_dir = config.MODEL_DIR / "route_b"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        with open(artifact_dir / "fill_values.pkl", "wb") as f:
            pickle.dump(fill_values, f)
        with open(artifact_dir / "encoding_map.pkl", "wb") as f:
            pickle.dump(encoding_map, f)
        logger.info(f"Route B artifacts saved to {artifact_dir}")

    result["fill_values"] = fill_values
    result["encoding_map"] = encoding_map
    return result
