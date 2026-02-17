"""
Route A Preprocessing — VIME-Compatible Feature Matrix (Fixed Version)
======================================================================

Prepares data for the VIME self-supervised encoder with improved preprocessing.

Phase 1 Improvements (Route A-Fixed):
    1.1 RobustScaler: Uses ClippedRobustScaler instead of MinMaxScaler
        - Handles outliers (AMT_INCOME_TOTAL max=117M vs median=135K)
        - Prevents 99% of values from being compressed to [0, 0.005]

    1.2 Missing Mask Export: Saves original missing positions separately
        - Enables partial reconstruction loss (mask missing in loss)
        - File: vime_missing_mask_*.parquet

    1.3 Target Encoding: For high-cardinality categoricals
        - ORGANIZATION_TYPE (58 cats → 1 col)
        - OCCUPATION_TYPE (18 cats → 1 col)
        - Reduces feature dimension from ~328 to ~200

Data flow:
    D_L_train (features + labels) ──→ Fit Imputer/Scaler/Encoder (target enc uses labels)
    D_L_val, Test, Reject ──→ Transform only
    Output ──→ Dense arrays (Parquet) + Missing masks for VIME training
"""

import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

from src.preprocessing.custom_scalers import ClippedRobustScaler
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_missing_mask(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """
    Create binary missing mask for numerical columns.

    Args:
        df: DataFrame with original (unimputed) values
        num_cols: List of numerical column names

    Returns:
        DataFrame with binary missing indicators (1 = missing, 0 = present)
    """
    mask_df = df[num_cols].isna().astype(int)
    mask_df.columns = [f"missing_{col}" for col in num_cols]
    return mask_df


def prepare_route_a_fixed():
    """
    Route A-Fixed Preprocessing for VIME with Phase 1 improvements.

    Improvements over original Route A:
    1. RobustScaler instead of MinMaxScaler (handles outliers)
    2. Saves missing masks for partial reconstruction loss
    3. Target encoding for high-cardinality categoricals

    Steps:
    1. Load splits (D_L_train, D_L_val, D_L_test, D_R_features).
    2. Create missing masks BEFORE imputation.
    3. Fit preprocessing on D_L_train (including target encoder with labels).
    4. Transform all datasets.
    5. Save processed features AND missing masks.
    """
    logger.info("Starting Route A-Fixed Preprocessing (VIME with improvements)...")
    logger.info("Improvements: RobustScaler, Missing Masks, Target Encoding")

    # 1. Load Data
    dl_train = pd.read_parquet(config.SPLIT_DL_TRAIN)
    dl_val = pd.read_parquet(config.SPLIT_DL_VAL)
    dl_test = pd.read_parquet(config.SPLIT_DL_TEST)
    dr_feat = pd.read_parquet(config.SPLIT_DR_FEATURES)

    # Separate Features/Targets
    X_train = dl_train.drop(columns=["TARGET"])
    y_train = dl_train["TARGET"]

    X_val = dl_val.drop(columns=["TARGET"])
    y_val = dl_val["TARGET"]

    X_test = dl_test.drop(columns=["TARGET"])
    y_test = dl_test["TARGET"]

    X_reject = dr_feat.copy()  # No label

    logger.info(f"Loaded Data: Train={X_train.shape}, Reject={X_reject.shape}")

    # 2. Identify Column Types
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Separate high-cardinality and low-cardinality categoricals
    high_card_cats = [c for c in config.HIGH_CARDINALITY_CATS if c in cat_cols]
    low_card_cats = [c for c in cat_cols if c not in high_card_cats]

    logger.info(f"Numeric Columns: {len(num_cols)}")
    logger.info(f"High-Cardinality Cats (Target Encoding): {high_card_cats}")
    logger.info(f"Low-Cardinality Cats (One-Hot): {low_card_cats}")

    # 3. Create Missing Masks BEFORE imputation
    logger.info("Creating missing masks...")
    mask_train = create_missing_mask(X_train, num_cols)
    mask_val = create_missing_mask(X_val, num_cols)
    mask_test = create_missing_mask(X_test, num_cols)
    mask_reject = create_missing_mask(X_reject, num_cols)

    # 4. Define Transformers (VIME compliant with improvements)

    # Numeric: Median Impute + Missing Indicator + ClippedRobustScaler
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', ClippedRobustScaler(
            clip_percentiles=config.ROBUST_SCALER_CLIP_PERCENTILES,
            quantile_range=config.ROBUST_SCALER_QUANTILE_RANGE
        ))
    ])

    # High-Cardinality Categorical: Target Encoding
    # Fit on D_L_train with labels, transform others
    high_card_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='_MISSING_')),
        ('encoder', TargetEncoder(
            cols=high_card_cats,
            smoothing=config.TARGET_ENCODER_SMOOTHING,
            min_samples_leaf=config.TARGET_ENCODER_MIN_SAMPLES,
            handle_unknown='value',
            return_df=False
        ))
    ])

    # Low-Cardinality Categorical: Mode Impute + One-Hot Encode
    low_card_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    # 5. Process each column group separately due to TargetEncoder needing y

    logger.info("Fitting preprocessor on D_L_train...")

    # Numerical features
    num_pipeline.fit(X_train[num_cols])
    X_train_num = num_pipeline.transform(X_train[num_cols])
    X_val_num = num_pipeline.transform(X_val[num_cols])
    X_test_num = num_pipeline.transform(X_test[num_cols])
    X_reject_num = num_pipeline.transform(X_reject[num_cols])

    # Get numerical feature names (including missing indicators)
    num_feature_names = list(num_cols)
    # Add missing indicator names
    imputer = num_pipeline.named_steps['imputer']
    if hasattr(imputer, 'indicator_') and imputer.indicator_ is not None:
        indicator_features = imputer.indicator_.features_
        for idx in indicator_features:
            num_feature_names.append(f"missingindicator_{num_cols[idx]}")

    logger.info(f"Numeric features after processing: {X_train_num.shape[1]}")

    # High-cardinality categoricals with target encoding
    if high_card_cats:
        # Fill missing first, then target encode
        X_train_high = X_train[high_card_cats].fillna('_MISSING_')
        X_val_high = X_val[high_card_cats].fillna('_MISSING_')
        X_test_high = X_test[high_card_cats].fillna('_MISSING_')
        X_reject_high = X_reject[high_card_cats].fillna('_MISSING_')

        target_encoder = TargetEncoder(
            cols=high_card_cats,
            smoothing=config.TARGET_ENCODER_SMOOTHING,
            min_samples_leaf=config.TARGET_ENCODER_MIN_SAMPLES,
            handle_unknown='value',
            return_df=True
        )

        # Fit with labels (only D_L_train has labels)
        X_train_high_enc = target_encoder.fit_transform(X_train_high, y_train)
        X_val_high_enc = target_encoder.transform(X_val_high)
        X_test_high_enc = target_encoder.transform(X_test_high)
        X_reject_high_enc = target_encoder.transform(X_reject_high)

        # Convert to arrays
        X_train_high_arr = X_train_high_enc.values
        X_val_high_arr = X_val_high_enc.values
        X_test_high_arr = X_test_high_enc.values
        X_reject_high_arr = X_reject_high_enc.values

        high_card_feature_names = high_card_cats
        logger.info(f"High-cardinality features (target encoded): {len(high_card_cats)}")
    else:
        X_train_high_arr = np.empty((len(X_train), 0))
        X_val_high_arr = np.empty((len(X_val), 0))
        X_test_high_arr = np.empty((len(X_test), 0))
        X_reject_high_arr = np.empty((len(X_reject), 0))
        high_card_feature_names = []

    # Low-cardinality categoricals with one-hot encoding
    if low_card_cats:
        low_card_pipeline.fit(X_train[low_card_cats])
        X_train_low = low_card_pipeline.transform(X_train[low_card_cats])
        X_val_low = low_card_pipeline.transform(X_val[low_card_cats])
        X_test_low = low_card_pipeline.transform(X_test[low_card_cats])
        X_reject_low = low_card_pipeline.transform(X_reject[low_card_cats])

        # Get one-hot feature names
        ohe = low_card_pipeline.named_steps['encoder']
        low_card_feature_names = list(ohe.get_feature_names_out(low_card_cats))
        logger.info(f"Low-cardinality features (one-hot): {X_train_low.shape[1]}")
    else:
        X_train_low = np.empty((len(X_train), 0))
        X_val_low = np.empty((len(X_val), 0))
        X_test_low = np.empty((len(X_test), 0))
        X_reject_low = np.empty((len(X_reject), 0))
        low_card_feature_names = []

    # 6. Concatenate all features
    X_train_proc = np.hstack([X_train_num, X_train_high_arr, X_train_low])
    X_val_proc = np.hstack([X_val_num, X_val_high_arr, X_val_low])
    X_test_proc = np.hstack([X_test_num, X_test_high_arr, X_test_low])
    X_reject_proc = np.hstack([X_reject_num, X_reject_high_arr, X_reject_low])

    # Combine feature names
    feature_names = num_feature_names + high_card_feature_names + low_card_feature_names

    logger.info(f"Total processed features: {len(feature_names)}")
    logger.info(f"  - Numeric (with indicators): {len(num_feature_names)}")
    logger.info(f"  - High-card categorical: {len(high_card_feature_names)}")
    logger.info(f"  - Low-card categorical: {len(low_card_feature_names)}")

    # 7. Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    X_val_df = pd.DataFrame(X_val_proc, columns=feature_names, index=X_val.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)
    X_reject_df = pd.DataFrame(X_reject_proc, columns=feature_names, index=X_reject.index)

    # 8. Save Processed Data
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Save features (Route A-Fixed version)
    X_train_df.to_parquet(config.DATA_PROCESSED / "vime_X_train_fixed.parquet")
    X_val_df.to_parquet(config.DATA_PROCESSED / "vime_X_val_fixed.parquet")
    X_test_df.to_parquet(config.DATA_PROCESSED / "vime_X_test_fixed.parquet")
    X_reject_df.to_parquet(config.DATA_PROCESSED / "vime_X_reject_fixed.parquet")

    # Save missing masks (for partial reconstruction loss)
    mask_train.to_parquet(config.DATA_PROCESSED / "vime_missing_mask_train.parquet")
    mask_val.to_parquet(config.DATA_PROCESSED / "vime_missing_mask_val.parquet")
    mask_test.to_parquet(config.DATA_PROCESSED / "vime_missing_mask_test.parquet")
    mask_reject.to_parquet(config.DATA_PROCESSED / "vime_missing_mask_reject.parquet")

    # Save targets (unchanged)
    pd.DataFrame(y_train).to_parquet(config.DATA_PROCESSED / "vime_y_train.parquet")
    pd.DataFrame(y_val).to_parquet(config.DATA_PROCESSED / "vime_y_val.parquet")
    pd.DataFrame(y_test).to_parquet(config.DATA_PROCESSED / "vime_y_test.parquet")

    # Save preprocessor artifacts for later use
    artifacts = {
        'num_pipeline': num_pipeline,
        'target_encoder': target_encoder if high_card_cats else None,
        'low_card_pipeline': low_card_pipeline if low_card_cats else None,
        'num_cols': num_cols,
        'high_card_cats': high_card_cats,
        'low_card_cats': low_card_cats,
        'feature_names': feature_names,
    }
    with open(config.DATA_PROCESSED / "route_a_fixed_preprocessor.pkl", 'wb') as f:
        pickle.dump(artifacts, f)

    logger.info(f"Saved processed files to {config.DATA_PROCESSED}/vime_*_fixed.parquet")
    logger.info(f"Saved missing masks to {config.DATA_PROCESSED}/vime_missing_mask_*.parquet")

    # Print summary statistics
    logger.info("\n=== Preprocessing Summary ===")
    logger.info(f"Original features: {len(num_cols)} numeric + {len(cat_cols)} categorical")
    logger.info(f"Processed features: {len(feature_names)}")
    logger.info(f"  - Reduction from one-hot: {58 + 18 - 2} fewer sparse columns")

    # Check income variance improvement
    income_idx = num_feature_names.index('AMT_INCOME_TOTAL') if 'AMT_INCOME_TOTAL' in num_feature_names else None
    if income_idx is not None:
        income_var = X_train_df['AMT_INCOME_TOTAL'].var()
        logger.info(f"AMT_INCOME_TOTAL variance: {income_var:.6f}")

    return X_train_df, X_val_df, X_test_df, X_reject_df


# Keep original function for backwards compatibility
def prepare_route_a():
    """
    Original Route A Preprocessing (deprecated).

    Use prepare_route_a_fixed() for improved preprocessing.
    """
    logger.warning("Using original prepare_route_a(). Consider using prepare_route_a_fixed() instead.")

    # Original implementation preserved for comparison
    from sklearn.preprocessing import MinMaxScaler

    dl_train = pd.read_parquet(config.SPLIT_DL_TRAIN)
    dl_val = pd.read_parquet(config.SPLIT_DL_VAL)
    dl_test = pd.read_parquet(config.SPLIT_DL_TEST)
    dr_feat = pd.read_parquet(config.SPLIT_DR_FEATURES)

    X_train = dl_train.drop(columns=["TARGET"])
    y_train = dl_train["TARGET"]
    X_val = dl_val.drop(columns=["TARGET"])
    y_val = dl_val["TARGET"]
    X_test = dl_test.drop(columns=["TARGET"])
    y_test = dl_test["TARGET"]
    X_reject = dr_feat.copy()

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', MinMaxScaler(feature_range=(0, 1)))
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], verbose_feature_names_out=False)

    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    X_reject_proc = preprocessor.transform(X_reject)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"feat_{i}" for i in range(X_train_proc.shape[1])]

    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train.index)
    X_val_df = pd.DataFrame(X_val_proc, columns=feature_names, index=X_val.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test.index)
    X_reject_df = pd.DataFrame(X_reject_proc, columns=feature_names, index=X_reject.index)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    X_train_df.to_parquet(config.DATA_PROCESSED / "vime_X_train.parquet")
    X_val_df.to_parquet(config.DATA_PROCESSED / "vime_X_val.parquet")
    X_test_df.to_parquet(config.DATA_PROCESSED / "vime_X_test.parquet")
    X_reject_df.to_parquet(config.DATA_PROCESSED / "vime_X_reject.parquet")

    pd.DataFrame(y_train).to_parquet(config.DATA_PROCESSED / "vime_y_train.parquet")
    pd.DataFrame(y_val).to_parquet(config.DATA_PROCESSED / "vime_y_val.parquet")
    pd.DataFrame(y_test).to_parquet(config.DATA_PROCESSED / "vime_y_test.parquet")

    logger.info(f"Saved original processed files to {config.DATA_PROCESSED}/vime_*.parquet")


if __name__ == "__main__":
    prepare_route_a_fixed()
