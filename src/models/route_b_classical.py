"""
Route B — Classical Reject Inference Pipeline (Phase 5)

Uses raw tabular features + CatBoost with native categorical handling.
NO VIME embeddings, NO Isolation Forest filtering.

Pipeline:
    5.1 Train Teacher on D_L_train raw features (validate on D_L_val)
    5.2 Pseudo-label D_R using teacher (NO filtering)
    5.3 Train Final model on D_L_train + ALL D_R
    5.4 Evaluate on D_L_test

Reference: WORKING_PLAN.md Phase 5
"""
import logging
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data():
    """
    Load raw tabular data for Route B pipeline.

    Returns:
        X_train, X_val, X_test, X_reject: Feature DataFrames
        y_train, y_val, y_test: Label arrays
        cat_features: List of categorical column indices for CatBoost
    """
    logger.info("Loading raw tabular data...")

    # Load D_L splits (with labels)
    df_train = pd.read_parquet(config.SPLIT_DL_TRAIN)
    df_val = pd.read_parquet(config.SPLIT_DL_VAL)
    df_test = pd.read_parquet(config.SPLIT_DL_TEST)

    # Load D_R features (no labels during training)
    df_reject = pd.read_parquet(config.SPLIT_DR_FEATURES)

    # Separate features and labels
    y_train = df_train[config.TARGET_COL].values
    y_val = df_val[config.TARGET_COL].values
    y_test = df_test[config.TARGET_COL].values

    X_train = df_train.drop(columns=[config.TARGET_COL])
    X_val = df_val.drop(columns=[config.TARGET_COL])
    X_test = df_test.drop(columns=[config.TARGET_COL])
    X_reject = df_reject.copy()

    # Identify categorical columns (for CatBoost native handling)
    cat_features = [col for col in config.CATEGORICAL_COLS if col in X_train.columns]
    cat_indices = [X_train.columns.get_loc(col) for col in cat_features]

    # Fill NaN in categorical columns (CatBoost requires non-null for cat features)
    for col in cat_features:
        X_train[col] = X_train[col].fillna("_MISSING_")
        X_val[col] = X_val[col].fillna("_MISSING_")
        X_test[col] = X_test[col].fillna("_MISSING_")
        X_reject[col] = X_reject[col].fillna("_MISSING_")

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Reject: {X_reject.shape}")
    logger.info(f"  Categorical features: {len(cat_features)}")

    return X_train, X_val, X_test, X_reject, y_train, y_val, y_test, cat_indices


def train_catboost_raw(X_train, y_train, X_val, y_val, cat_features, model_name="model"):
    """
    Train CatBoost classifier on raw features with native categorical handling.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data for early stopping
        cat_features: List of categorical column indices
        model_name: Name for logging

    Returns:
        model: Trained CatBoostClassifier
        val_auc: Validation AUC-ROC
    """
    model = CatBoostClassifier(
        **config.CATBOOST,
        random_seed=config.SEED,
        cat_features=cat_features,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    logger.info(f"{model_name} Val AUC: {val_auc:.4f}")
    return model, val_auc


def pseudo_label_rejects_raw(teacher, X_reject, threshold=None):
    """
    Generate pseudo-labels for D_R using the teacher model.
    NO filtering in Route B (classical approach).

    Args:
        teacher: Trained CatBoost teacher model
        X_reject: Reject features
        threshold: Probability threshold for positive class

    Returns:
        labels: Binary pseudo-labels
        proba: Predicted probabilities
    """
    if threshold is None:
        threshold = config.PSEUDO_LABEL_THRESHOLD

    proba = teacher.predict_proba(X_reject)[:, 1]
    labels = (proba >= threshold).astype(int)

    logger.info(f"Pseudo-labels: {labels.sum()}/{len(labels)} defaults ({labels.mean():.2%})")
    logger.info("(No Isolation Forest filtering in Route B)")
    return labels, proba


def run_route_b_pipeline():
    """
    Execute full Route B classical reject inference pipeline.

    Steps:
        5.1 Train Teacher CatBoost on D_L_train raw features
        5.2 Pseudo-label D_R (NO filtering)
        5.3 Train Final model on D_L_train + ALL D_R
        5.4 Evaluate on D_L_test

    Returns:
        results: Dict with metrics and statistics
    """
    logger.info("=" * 60)
    logger.info("Route B — Classical Reject Inference Pipeline")
    logger.info("=" * 60)
    logger.info("Uses raw features + CatBoost (NO VIME, NO IF filtering)")
    logger.info("")

    # Load data
    X_train, X_val, X_test, X_reject, y_train, y_val, y_test, cat_features = load_raw_data()

    # 5.1 Train Teacher
    logger.info("\n[5.1] Training Teacher Model...")
    teacher, teacher_val_auc = train_catboost_raw(X_train, y_train, X_val, y_val, cat_features, "Teacher")

    # 5.2 Pseudo-label D_R (NO filtering)
    logger.info("\n[5.2] Pseudo-labeling D_R (no filtering)...")
    pseudo_labels, pseudo_proba = pseudo_label_rejects_raw(teacher, X_reject)

    # 5.3 Train Final Model on D_L + ALL D_R
    logger.info("\n[5.3] Training Final Model (D_L + ALL D_R)...")
    X_augmented = pd.concat([X_train, X_reject], ignore_index=True)
    y_augmented = np.concatenate([y_train, pseudo_labels])

    logger.info(f"  Augmented data: {len(X_augmented)} samples "
                f"(D_L={len(X_train)}, D_R={len(X_reject)})")
    logger.info(f"  Augmented default rate: {y_augmented.mean():.2%}")

    final_model, final_val_auc = train_catboost_raw(X_augmented, y_augmented, X_val, y_val, cat_features, "Final")

    # 5.4 Evaluate on D_L_test
    logger.info("\n[5.4] Evaluating on D_L_test...")
    test_pred = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    logger.info(f"Final Model Test AUC (D_L_test): {test_auc:.4f}")

    # Also evaluate teacher on test for comparison
    teacher_test_pred = teacher.predict_proba(X_test)[:, 1]
    teacher_test_auc = roc_auc_score(y_test, teacher_test_pred)
    logger.info(f"Teacher Test AUC (D_L_test): {teacher_test_auc:.4f}")

    # Compile results
    results = {
        "teacher_val_auc": float(teacher_val_auc),
        "teacher_test_auc": float(teacher_test_auc),
        "final_val_auc": float(final_val_auc),
        "final_test_auc": float(test_auc),
        "pseudo_label_default_rate": float(pseudo_labels.mean()),
        "augmented_samples": int(len(X_augmented)),
        "augmented_default_rate": float(y_augmented.mean()),
        "n_features": int(X_train.shape[1]),
        "n_cat_features": len(cat_features),
    }

    # Save models
    logger.info("\nSaving models...")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    teacher.save_model(str(config.MODEL_DIR / "route_b_teacher.cbm"))
    final_model.save_model(str(config.MODEL_DIR / "route_b_final.cbm"))
    logger.info(f"  Saved to {config.MODEL_DIR}")

    # Save metrics
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULT_DIR / "route_b_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Metrics saved to {config.RESULT_DIR / 'route_b_metrics.json'}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Route B Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Teacher Val AUC:  {teacher_val_auc:.4f}")
    logger.info(f"Teacher Test AUC: {teacher_test_auc:.4f}")
    logger.info(f"Final Val AUC:    {final_val_auc:.4f}")
    logger.info(f"Final Test AUC:   {test_auc:.4f}")
    logger.info(f"Test AUC Change:  {(test_auc - teacher_test_auc):+.4f}")

    return results


if __name__ == "__main__":
    results = run_route_b_pipeline()
    print("\n" + json.dumps(results, indent=2))
