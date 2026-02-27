"""
CatBoost Training Module for Credit Risk Classification.

Implements training with:
- Class weighting (Focal Loss equivalent)
- Manual Precision@20% early stopping
- Native categorical handling
"""
import logging
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import config

logger = logging.getLogger(__name__)


def precision_at_k(y_true, y_pred, k=0.20):
    """
    Precision@k%: Among top k% lowest-risk predictions, what fraction are truly good?

    Args:
        y_true: True labels (1=default, 0=non-default)
        y_pred: Predicted probabilities of default
        k: Top k% to evaluate (default 20%)

    Returns:
        precision: Fraction of selected that are truly good (y=0)
        n_diamonds: Count of good borrowers found
    """
    n_top_k = int(len(y_true) * k)
    sorted_indices = np.argsort(y_pred)
    top_k_indices = sorted_indices[:n_top_k]
    top_k_labels = y_true[top_k_indices]
    n_diamonds = (top_k_labels == 0).sum()
    precision = n_diamonds / n_top_k
    return precision, int(n_diamonds)


def load_data():
    """
    Load all data for CatBoost training.

    Returns:
        data: Dict with features, labels, and metadata
    """
    logger.info("Loading data...")

    df_train = pd.read_parquet(config.SPLIT_DL_TRAIN)
    df_val = pd.read_parquet(config.SPLIT_DL_VAL)
    df_test = pd.read_parquet(config.SPLIT_DL_TEST)
    df_reject = pd.read_parquet(config.SPLIT_DR_FEATURES)
    dr_truth = pd.read_parquet(config.SPLIT_DR_TRUTH)

    y_train = df_train[config.TARGET_COL].values
    y_val = df_val[config.TARGET_COL].values
    y_test = df_test[config.TARGET_COL].values
    y_reject = dr_truth[config.TARGET_COL].values

    X_train = df_train.drop(columns=[config.TARGET_COL])
    X_val = df_val.drop(columns=[config.TARGET_COL])
    X_test = df_test.drop(columns=[config.TARGET_COL])
    X_reject = df_reject.copy()

    # Feature groups
    ext_cols = [c for c in X_train.columns if 'EXT_SOURCE' in c.upper()]
    other_cols = [c for c in X_train.columns if c not in ext_cols]

    # Handle categorical NaN
    cat_features = [col for col in config.CATEGORICAL_COLS if col in other_cols]
    for col in cat_features:
        for df in [X_train, X_val, X_test, X_reject]:
            df[col] = df[col].fillna('_MISSING_')

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Reject: {X_reject.shape}")
    logger.info(f"  Train default rate: {y_train.mean():.2%}")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'X_reject': X_reject,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test, 'y_reject': y_reject,
        'ext_cols': ext_cols, 'other_cols': other_cols, 'cat_features': cat_features
    }


def train_catboost_with_precision_stopping(X_train, y_train, X_val, y_val,
                                           cat_features=None, name="CatBoost",
                                           iterations=2000, depth=6, patience=50):
    """
    Train CatBoost with:
    - Class weights equivalent to Focal Loss alpha (0.25 for positive)
    - Manual early stopping based on Precision@20%

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        cat_features: List of categorical column names
        name: Model name for logging
        iterations: Max iterations
        depth: Tree depth
        patience: Early stopping patience

    Returns:
        best_model: Trained CatBoostClassifier
        best_val_prec: Best validation Precision@20%
        best_iteration: Best iteration number
    """
    logger.info(f"\nTraining {name} with Precision@20% early stopping...")

    # Focal Loss alpha equivalent via class weights
    class_weights = {0: 0.75, 1: 0.25}

    # Get categorical indices
    cat_indices = None
    if cat_features:
        cat_indices = [X_train.columns.get_loc(col) for col in cat_features
                       if col in X_train.columns]

    # Train with manual early stopping based on Precision@20%
    best_val_prec = 0.0
    best_iteration = 0
    patience_counter = 0
    chunk_size = 50

    model = None
    best_model = None

    for chunk in range(iterations // chunk_size):
        if model is None:
            model = CatBoostClassifier(
                iterations=chunk_size,
                depth=depth,
                learning_rate=0.05,
                class_weights=class_weights,
                eval_metric='AUC',
                verbose=0,
                random_seed=config.SEED,
                cat_features=cat_indices,
                allow_writing_files=False
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            new_model = CatBoostClassifier(
                iterations=chunk_size,
                depth=depth,
                learning_rate=0.05,
                class_weights=class_weights,
                eval_metric='AUC',
                verbose=0,
                random_seed=config.SEED,
                cat_features=cat_indices,
                allow_writing_files=False
            )
            new_model.fit(X_train, y_train, eval_set=(X_val, y_val), init_model=model)
            model = new_model

        total_iterations = (chunk + 1) * chunk_size

        # Evaluate Precision@20%
        val_pred = model.predict_proba(X_val)[:, 1]
        val_prec, _ = precision_at_k(y_val, val_pred, k=0.20)
        val_auc = roc_auc_score(y_val, val_pred)

        if val_prec > best_val_prec:
            best_val_prec = val_prec
            best_iteration = total_iterations
            patience_counter = 0
            best_model = model.copy()
            logger.info(f"  Iter {total_iterations}: Val P@20%={val_prec:.4f}, AUC={val_auc:.4f} *best*")
        else:
            patience_counter += 1
            if total_iterations % 200 == 0:
                logger.info(f"  Iter {total_iterations}: Val P@20%={val_prec:.4f}, AUC={val_auc:.4f}")

        if patience_counter >= (patience // chunk_size):
            logger.info(f"  Early stopping at iteration {total_iterations}")
            break

    logger.info(f"  Best iteration: {best_iteration}, Best Val P@20%: {best_val_prec:.4f}")

    return best_model, best_val_prec, best_iteration


def evaluate_model(model, X_test, y_test, X_reject, y_reject, name="Model"):
    """
    Evaluate model on test and reject sets.

    Returns:
        results: Dict with AUC, P@20%, and Diamonds for both sets
    """
    pred_test = model.predict_proba(X_test)[:, 1]
    pred_reject = model.predict_proba(X_reject)[:, 1]

    auc_test = roc_auc_score(y_test, pred_test)
    auc_reject = roc_auc_score(y_reject, pred_reject)

    prec_test, diamonds_test = precision_at_k(y_test, pred_test, k=0.20)
    prec_reject, diamonds_reject = precision_at_k(y_reject, pred_reject, k=0.20)

    logger.info(f"\n{name} Results:")
    logger.info(f"  D_L Test: AUC={auc_test:.4f}, P@20%={prec_test:.4f}, Diamonds={diamonds_test:,}")
    logger.info(f"  D_R:      AUC={auc_reject:.4f}, P@20%={prec_reject:.4f}, Diamonds={diamonds_reject:,}")

    return {
        'D_L_test': {'AUC': float(auc_test), 'P@20%': float(prec_test), 'Diamonds': int(diamonds_test)},
        'D_R': {'AUC': float(auc_reject), 'P@20%': float(prec_reject), 'Diamonds': int(diamonds_reject)}
    }
