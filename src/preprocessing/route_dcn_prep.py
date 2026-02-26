"""
DCN-v2 Preprocessing — Categorical Embeddings + Numerical Soft Binning
=======================================================================

Prepares data for DCN-v2 with NATIVE categorical embedding support.

Key Differences from Route A-Fixed:
    - Categorical features kept as INTEGER INDICES (not one-hot or target-encoded)
    - Numerical features scaled with ClippedRobustScaler (same as Route A-Fixed)
    - Missing values: Index 0 for categoricals, NaN preserved for numericals
    - Outputs SEPARATE files for numerical and categorical features

Data flow:
    D_L_train, val, test, D_R
        → Numerical: median impute → ClippedRobustScaler → float32
        → Categorical: LabelEncoder (0 = missing) → int32
        → Output: dcn_X_*_num.parquet, dcn_X_*_cat.parquet

# Purpose: Prepare data for DCN-v2 with native categorical embeddings
# Why: DCN-v2 has built-in embedding layers that learn better representations
# Approved: 2026-02-16
"""

import json
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.custom_scalers import ClippedRobustScaler
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CategoricalIndexer:
    """
    Converts categorical columns to integer indices with missing handling.

    Index mapping:
        0 = missing/unknown (reserved for padding_idx in PyTorch Embedding)
        1, 2, ... = actual category values

    # Purpose: Create integer indices for categorical features
    # Why: PyTorch Embedding layers require integer indices, not strings
    # Approved: 2026-02-16
    """

    def __init__(self):
        self.encoders = {}
        self.cardinalities = {}

    def fit(self, df: pd.DataFrame, cat_cols: list):
        """
        Fit LabelEncoders on categorical columns.

        Args:
            df: DataFrame with categorical columns
            cat_cols: List of categorical column names
        """
        for col in cat_cols:
            # Get all unique values (excluding NaN)
            unique_vals = df[col].dropna().unique()

            # Create encoder with sorted values for reproducibility
            le = LabelEncoder()
            le.fit(sorted(unique_vals.astype(str)))

            self.encoders[col] = le
            # Cardinality = number of unique values (excluding missing)
            self.cardinalities[col] = len(le.classes_)

            logger.info(f"  {col}: {self.cardinalities[col]} categories")

        return self

    def transform(self, df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
        """
        Transform categorical columns to integer indices.

        Missing values → 0
        Known values → 1, 2, 3, ...
        Unknown values (not in training) → 0

        Args:
            df: DataFrame with categorical columns
            cat_cols: List of categorical column names

        Returns:
            DataFrame with integer indices (int32)
        """
        result = pd.DataFrame(index=df.index)

        for col in cat_cols:
            le = self.encoders[col]

            # Start with zeros (missing indicator)
            indices = np.zeros(len(df), dtype=np.int32)

            # Find non-missing values
            mask_present = df[col].notna()

            if mask_present.any():
                values = df.loc[mask_present, col].astype(str).values

                # Handle unknown categories (map to 0)
                known_mask = np.isin(values, le.classes_)

                # Encode known values (add 1 to shift from 0-indexed to 1-indexed)
                known_values = values[known_mask]
                if len(known_values) > 0:
                    encoded = le.transform(known_values) + 1  # Shift by 1

                    # Place encoded values back
                    present_indices = np.where(mask_present)[0]
                    known_indices = present_indices[known_mask]
                    indices[known_indices] = encoded

            result[col] = indices

        return result

    def fit_transform(self, df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(df, cat_cols)
        return self.transform(df, cat_cols)


def prepare_dcn_data():
    """
    DCN-v2 Preprocessing with native categorical support.

    Outputs:
        - dcn_X_*_num.parquet: Numerical features (float32, scaled)
        - dcn_X_*_cat.parquet: Categorical features (int32, indices)
        - dcn_y_*.parquet: Target labels
        - dcn_preprocessor.pkl: Fitted transformers
        - dcn_cardinalities.json: Category cardinalities for model init

    # Purpose: Create train/val/test/reject data for DCN-v2
    # Why: DCN-v2 needs separate numerical and categorical inputs
    # Approved: 2026-02-16
    """
    logger.info("=" * 60)
    logger.info("Starting DCN-v2 Preprocessing")
    logger.info("=" * 60)

    # 1. Load Data
    logger.info("\n[Step 1] Loading data splits...")
    dl_train = pd.read_parquet(config.SPLIT_DL_TRAIN)
    dl_val = pd.read_parquet(config.SPLIT_DL_VAL)
    dl_test = pd.read_parquet(config.SPLIT_DL_TEST)
    dr_feat = pd.read_parquet(config.SPLIT_DR_FEATURES)

    # Separate features and targets
    X_train = dl_train.drop(columns=["TARGET"])
    y_train = dl_train["TARGET"]
    X_val = dl_val.drop(columns=["TARGET"])
    y_val = dl_val["TARGET"]
    X_test = dl_test.drop(columns=["TARGET"])
    y_test = dl_test["TARGET"]
    X_reject = dr_feat.copy()

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Reject: {X_reject.shape}")

    # 2. Identify Column Types
    logger.info("\n[Step 2] Identifying column types...")
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in X_train.columns]

    logger.info(f"  Numerical columns: {len(num_cols)}")
    logger.info(f"  Categorical columns: {len(cat_cols)}")

    # 3. Process Numerical Features
    logger.info("\n[Step 3] Processing numerical features...")

    # Numerical pipeline: Median impute + ClippedRobustScaler
    num_imputer = SimpleImputer(strategy='median')
    num_scaler = ClippedRobustScaler(
        clip_percentiles=config.ROBUST_SCALER_CLIP_PERCENTILES,
        quantile_range=config.ROBUST_SCALER_QUANTILE_RANGE
    )

    # Fit on training data
    X_train_num_imputed = num_imputer.fit_transform(X_train[num_cols])
    X_train_num_scaled = num_scaler.fit_transform(X_train_num_imputed)

    # Transform other splits
    X_val_num_scaled = num_scaler.transform(num_imputer.transform(X_val[num_cols]))
    X_test_num_scaled = num_scaler.transform(num_imputer.transform(X_test[num_cols]))
    X_reject_num_scaled = num_scaler.transform(num_imputer.transform(X_reject[num_cols]))

    # Convert to DataFrames
    X_train_num = pd.DataFrame(X_train_num_scaled, columns=num_cols, index=X_train.index).astype(np.float32)
    X_val_num = pd.DataFrame(X_val_num_scaled, columns=num_cols, index=X_val.index).astype(np.float32)
    X_test_num = pd.DataFrame(X_test_num_scaled, columns=num_cols, index=X_test.index).astype(np.float32)
    X_reject_num = pd.DataFrame(X_reject_num_scaled, columns=num_cols, index=X_reject.index).astype(np.float32)

    logger.info(f"  Numerical output shape: {X_train_num.shape}")

    # 4. Process Categorical Features
    logger.info("\n[Step 4] Processing categorical features...")

    cat_indexer = CategoricalIndexer()

    # Fit on training data and transform all
    X_train_cat = cat_indexer.fit_transform(X_train, cat_cols)
    X_val_cat = cat_indexer.transform(X_val, cat_cols)
    X_test_cat = cat_indexer.transform(X_test, cat_cols)
    X_reject_cat = cat_indexer.transform(X_reject, cat_cols)

    logger.info(f"  Categorical output shape: {X_train_cat.shape}")
    logger.info(f"  Total cardinality: {sum(cat_indexer.cardinalities.values())}")

    # 5. Save Processed Data
    logger.info("\n[Step 5] Saving processed data...")
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Numerical features
    X_train_num.to_parquet(config.DATA_PROCESSED / "dcn_X_train_num.parquet")
    X_val_num.to_parquet(config.DATA_PROCESSED / "dcn_X_val_num.parquet")
    X_test_num.to_parquet(config.DATA_PROCESSED / "dcn_X_test_num.parquet")
    X_reject_num.to_parquet(config.DATA_PROCESSED / "dcn_X_reject_num.parquet")

    # Categorical features
    X_train_cat.to_parquet(config.DATA_PROCESSED / "dcn_X_train_cat.parquet")
    X_val_cat.to_parquet(config.DATA_PROCESSED / "dcn_X_val_cat.parquet")
    X_test_cat.to_parquet(config.DATA_PROCESSED / "dcn_X_test_cat.parquet")
    X_reject_cat.to_parquet(config.DATA_PROCESSED / "dcn_X_reject_cat.parquet")

    # Targets
    pd.DataFrame(y_train).to_parquet(config.DATA_PROCESSED / "dcn_y_train.parquet")
    pd.DataFrame(y_val).to_parquet(config.DATA_PROCESSED / "dcn_y_val.parquet")
    pd.DataFrame(y_test).to_parquet(config.DATA_PROCESSED / "dcn_y_test.parquet")

    # 6. Save Preprocessor Artifacts
    logger.info("\n[Step 6] Saving preprocessor artifacts...")

    artifacts = {
        'num_imputer': num_imputer,
        'num_scaler': num_scaler,
        'cat_indexer': cat_indexer,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
    }
    with open(config.DATA_PROCESSED / "dcn_preprocessor.pkl", 'wb') as f:
        pickle.dump(artifacts, f)

    # Save cardinalities as JSON for easy loading in notebooks
    with open(config.DATA_PROCESSED / "dcn_cardinalities.json", 'w') as f:
        json.dump(cat_indexer.cardinalities, f, indent=2)

    # 7. Print Summary
    logger.info("\n" + "=" * 60)
    logger.info("DCN-v2 Preprocessing Complete")
    logger.info("=" * 60)
    logger.info(f"\nOutput Files:")
    logger.info(f"  Numerical: dcn_X_*_num.parquet ({len(num_cols)} features)")
    logger.info(f"  Categorical: dcn_X_*_cat.parquet ({len(cat_cols)} features)")
    logger.info(f"  Cardinalities: dcn_cardinalities.json")
    logger.info(f"\nCardinality Summary:")
    for col, card in cat_indexer.cardinalities.items():
        logger.info(f"  {col}: {card}")

    return {
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'cardinalities': cat_indexer.cardinalities,
    }


if __name__ == "__main__":
    prepare_dcn_data()
