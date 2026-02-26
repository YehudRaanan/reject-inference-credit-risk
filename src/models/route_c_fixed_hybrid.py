"""
Route C-Fixed — Raw Features + Fixed VIME Embeddings (Step 1.4B)

Combines fixed VIME embeddings (128-dim) with raw tabular features (132)
for a total of ~260 features. Uses CatBoost with native categorical handling.

Fixed preprocessing improvements applied to the VIME encoder:
  - RobustScaler (704x income variance improvement)
  - Partial Reconstruction Loss (masked MSE)
  - Target Encoding (328→253 features before VIME)

Pipeline (mirrors Route C but with fixed artifacts):
    1.4B.1 Load and combine fixed embeddings + raw features
    1.4B.2 Train Teacher on D_L combined features
    1.4B.3 Pseudo-label D_R (no IF filtering)
    1.4B.4 Train Student on D_L + ALL D_R
    1.4B.5 Evaluate on D_L_test

Baselines: Route B = 0.7391, Route C (original) = 0.7331

Reference: WORKING_PLAN.md Phase 5.6, Step 1.4B
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


def load_combined_fixed_features():
    """
    Load and concatenate:
    - VIME-Fixed embeddings (128-dim): vime_emb_*_fixed.parquet
    - Raw features (132): D_L_*.parquet / D_R_features.parquet

    Returns:
        X_train, X_val, X_test, X_reject: Combined feature DataFrames
        y_train, y_val, y_test: Label arrays
        cat_feature_indices: List of categorical column indices for CatBoost
    """
    logger.info("Loading combined features (fixed embeddings + raw)...")

    # Load VIME-Fixed embeddings
    emb_train = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_train_fixed.parquet")
    emb_val = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_val_fixed.parquet")
    emb_test = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_test_fixed.parquet")
    emb_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_reject_fixed.parquet")

    # Rename embedding columns to avoid conflicts
    emb_cols = [f"emb_{i}" for i in range(emb_train.shape[1])]
    emb_train.columns = emb_cols
    emb_val.columns = emb_cols
    emb_test.columns = emb_cols
    emb_reject.columns = emb_cols

    # Load raw features
    df_train = pd.read_parquet(config.SPLIT_DL_TRAIN)
    df_val = pd.read_parquet(config.SPLIT_DL_VAL)
    df_test = pd.read_parquet(config.SPLIT_DL_TEST)
    df_reject = pd.read_parquet(config.SPLIT_DR_FEATURES)

    # Separate labels
    y_train = df_train[config.TARGET_COL].values
    y_val = df_val[config.TARGET_COL].values
    y_test = df_test[config.TARGET_COL].values

    # Get raw features (without target)
    X_raw_train = df_train.drop(columns=[config.TARGET_COL]).reset_index(drop=True)
    X_raw_val = df_val.drop(columns=[config.TARGET_COL]).reset_index(drop=True)
    X_raw_test = df_test.drop(columns=[config.TARGET_COL]).reset_index(drop=True)
    X_raw_reject = df_reject.reset_index(drop=True)

    # Identify categorical columns
    cat_features = [col for col in config.CATEGORICAL_COLS if col in X_raw_train.columns]

    # Fill NaN in categorical columns (CatBoost requires non-null for cat features)
    for col in cat_features:
        X_raw_train[col] = X_raw_train[col].fillna("_MISSING_")
        X_raw_val[col] = X_raw_val[col].fillna("_MISSING_")
        X_raw_test[col] = X_raw_test[col].fillna("_MISSING_")
        X_raw_reject[col] = X_raw_reject[col].fillna("_MISSING_")

    # Concatenate raw + fixed embeddings
    X_train = pd.concat([X_raw_train, emb_train.reset_index(drop=True)], axis=1)
    X_val = pd.concat([X_raw_val, emb_val.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_raw_test, emb_test.reset_index(drop=True)], axis=1)
    X_reject = pd.concat([X_raw_reject, emb_reject.reset_index(drop=True)], axis=1)

    # Get categorical indices for CatBoost
    cat_indices = [X_train.columns.get_loc(col) for col in cat_features]

    logger.info(f"  Raw features: {X_raw_train.shape[1]}")
    logger.info(f"  VIME-Fixed embeddings: {emb_train.shape[1]}")
    logger.info(f"  Combined: {X_train.shape[1]} features")
    logger.info(f"  Categorical features: {len(cat_features)}")
    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Reject: {X_reject.shape}")

    return X_train, X_val, X_test, X_reject, y_train, y_val, y_test, cat_indices


def train_catboost(X_train, y_train, X_val, y_val, cat_features, model_name="model"):
    """Train CatBoost classifier with native categorical handling."""
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


def pseudo_label_rejects(teacher, X_reject, threshold=None):
    """Generate pseudo-labels for D_R (no IF filtering in Route C-Fixed)."""
    if threshold is None:
        threshold = config.PSEUDO_LABEL_THRESHOLD

    proba = teacher.predict_proba(X_reject)[:, 1]
    labels = (proba >= threshold).astype(int)

    logger.info(f"Pseudo-labels: {labels.sum()}/{len(labels)} defaults ({labels.mean():.2%})")
    return labels, proba


def run_route_c_fixed_pipeline():
    """
    Execute full Route C-Fixed hybrid pipeline.

    Steps:
        1.4B.1 Load combined features (fixed embeddings + raw)
        1.4B.2 Train Teacher on D_L
        1.4B.3 Pseudo-label D_R (no filtering)
        1.4B.4 Train Student on D_L + ALL D_R
        1.4B.5 Evaluate on D_L_test

    Returns:
        results: Dict with metrics and statistics
    """
    logger.info("=" * 60)
    logger.info("Route C-Fixed — Raw + Fixed Embeddings (Step 1.4B)")
    logger.info("=" * 60)
    logger.info("Combines 128-dim fixed embeddings with 132 raw features = 260 total")

    # 1.4B.1 Load data
    X_train, X_val, X_test, X_reject, y_train, y_val, y_test, cat_features = load_combined_fixed_features()

    # 1.4B.2 Train Teacher
    logger.info("\n[1.4B.2] Training Teacher Model...")
    teacher, teacher_val_auc = train_catboost(X_train, y_train, X_val, y_val, cat_features, "Teacher")

    # 1.4B.3 Pseudo-label D_R (no filtering)
    logger.info("\n[1.4B.3] Pseudo-labeling D_R (no filtering)...")
    pseudo_labels, pseudo_proba = pseudo_label_rejects(teacher, X_reject)

    # 1.4B.4 Train Student on D_L + ALL D_R
    logger.info("\n[1.4B.4] Training Student Model (D_L + ALL D_R)...")
    X_augmented = pd.concat([X_train, X_reject], ignore_index=True)
    y_augmented = np.concatenate([y_train, pseudo_labels])

    logger.info(f"  Augmented data: {len(X_augmented)} samples "
                f"(D_L={len(X_train)}, D_R={len(X_reject)})")
    logger.info(f"  Augmented default rate: {y_augmented.mean():.2%}")

    final_model, final_val_auc = train_catboost(X_augmented, y_augmented, X_val, y_val, cat_features, "Student")

    # 1.4B.5 Evaluate on D_L_test
    logger.info("\n[1.4B.5] Evaluating on D_L_test...")
    test_pred = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    logger.info(f"Student Model Test AUC (D_L_test): {test_auc:.4f}")

    # Also evaluate teacher on test
    teacher_test_pred = teacher.predict_proba(X_test)[:, 1]
    teacher_test_auc = roc_auc_score(y_test, teacher_test_pred)
    logger.info(f"Teacher Test AUC (D_L_test): {teacher_test_auc:.4f}")

    # Compile results
    results = {
        "route": "C-Fixed",
        "step": "1.4B",
        "description": "Raw features + Fixed VIME embeddings",
        "teacher_val_auc": float(teacher_val_auc),
        "teacher_test_auc": float(teacher_test_auc),
        "final_val_auc": float(final_val_auc),
        "final_test_auc": float(test_auc),
        "pseudo_label_default_rate": float(pseudo_labels.mean()),
        "augmented_samples": int(len(X_augmented)),
        "augmented_default_rate": float(y_augmented.mean()),
        "n_raw_features": int(X_train.shape[1] - 128),
        "n_embedding_features": 128,
        "n_total_features": int(X_train.shape[1]),
        "n_cat_features": len(cat_features),
        "baseline_route_b_test_auc": 0.7391,
        "baseline_route_c_test_auc": 0.7331,
        "improvement_over_route_c": float(test_auc - 0.7331),
        "improvement_over_route_b": float(test_auc - 0.7391),
    }

    # Save models
    logger.info("\nSaving models...")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    teacher.save_model(str(config.MODEL_DIR / "route_c_fixed_teacher.cbm"))
    final_model.save_model(str(config.MODEL_DIR / "route_c_fixed_final.cbm"))
    logger.info(f"  Saved to {config.MODEL_DIR}")

    # Save metrics
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULT_DIR / "route_c_fixed_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Metrics saved to {config.RESULT_DIR / 'route_c_fixed_metrics.json'}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Route C-Fixed Pipeline Complete (Step 1.4B)")
    logger.info("=" * 60)
    logger.info(f"Teacher Val AUC:  {teacher_val_auc:.4f}")
    logger.info(f"Teacher Test AUC: {teacher_test_auc:.4f}")
    logger.info(f"Student Val AUC:  {final_val_auc:.4f}")
    logger.info(f"Student Test AUC: {test_auc:.4f}")
    logger.info(f"Baseline (Route B): 0.7391")
    logger.info(f"Baseline (Route C): 0.7331")
    logger.info(f"Improvement vs Route C: {test_auc - 0.7331:+.4f}")
    logger.info(f"Improvement vs Route B: {test_auc - 0.7391:+.4f}")

    return results


if __name__ == "__main__":
    results = run_route_c_fixed_pipeline()
    print("\n" + json.dumps(results, indent=2))
