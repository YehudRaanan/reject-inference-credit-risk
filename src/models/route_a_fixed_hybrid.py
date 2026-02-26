"""
Route A-Fixed — Hybrid Reject Inference with Fixed VIME Embeddings (Step 1.4A)

Uses FIXED VIME embeddings (128-dim) from improved preprocessing:
  - RobustScaler (704x income variance improvement)
  - Partial Reconstruction Loss (masked MSE)
  - Target Encoding (328→253 features before VIME)

Pipeline (mirrors Route A but with fixed artifacts):
    1.4A.1 Train Teacher on D_L_train fixed embeddings
    1.4A.2 Pseudo-label D_R using teacher
    1.4A.3 Filter D_R with Isolation Forest (fit on D_L embeddings)
    1.4A.4 Train Student on D_L_train + filtered D_R
    1.4A.5 Evaluate on D_L_test

Baseline to beat: Route A (original) Test AUC = 0.6912

Reference: WORKING_PLAN.md Phase 5.6, Step 1.4A
"""
import logging
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_fixed_embeddings_and_labels():
    """
    Load VIME-Fixed embeddings and labels for Route A-Fixed pipeline.

    Returns:
        X_train, X_val, X_test, X_reject: Fixed embedding arrays (128-dim)
        y_train, y_val, y_test: Label arrays
    """
    logger.info("Loading VIME-Fixed embeddings and labels...")

    X_train = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_train_fixed.parquet").values
    X_val = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_val_fixed.parquet").values
    X_test = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_test_fixed.parquet").values
    X_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_reject_fixed.parquet").values

    y_train = pd.read_parquet(config.DATA_PROCESSED / "vime_y_train.parquet")["TARGET"].values
    y_val = pd.read_parquet(config.DATA_PROCESSED / "vime_y_val.parquet")["TARGET"].values
    y_test = pd.read_parquet(config.DATA_PROCESSED / "vime_y_test.parquet")["TARGET"].values

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Reject: {X_reject.shape}")
    return X_train, X_val, X_test, X_reject, y_train, y_val, y_test


def train_catboost(X_train, y_train, X_val, y_val, model_name="model"):
    """
    Train CatBoost classifier with early stopping.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data for early stopping
        model_name: Name for logging

    Returns:
        model: Trained CatBoostClassifier
        val_auc: Validation AUC-ROC
    """
    model = CatBoostClassifier(**config.CATBOOST, random_seed=config.SEED)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    logger.info(f"{model_name} Val AUC: {val_auc:.4f}")
    return model, val_auc


def pseudo_label_rejects(teacher, X_reject, threshold=None):
    """
    Generate pseudo-labels for D_R using the teacher model.

    Args:
        teacher: Trained CatBoost teacher model
        X_reject: Reject embeddings
        threshold: Probability threshold (default: config.PSEUDO_LABEL_THRESHOLD)

    Returns:
        labels: Binary pseudo-labels
        proba: Predicted probabilities
    """
    if threshold is None:
        threshold = config.PSEUDO_LABEL_THRESHOLD

    proba = teacher.predict_proba(X_reject)[:, 1]
    labels = (proba >= threshold).astype(int)

    logger.info(f"Pseudo-labels: {labels.sum()}/{len(labels)} defaults ({labels.mean():.2%})")
    return labels, proba


def filter_with_isolation_forest(X_train, X_reject):
    """
    Use Isolation Forest to filter anomalous samples from D_R.

    Fits on D_L embeddings (approved population) to define "normal".
    Predicts inliers in D_R — samples similar to approved population.

    Args:
        X_train: D_L embeddings (fit data)
        X_reject: D_R embeddings (predict data)

    Returns:
        inlier_mask: Boolean array, True = keep (inlier)
    """
    logger.info("Fitting Isolation Forest on D_L fixed embeddings...")
    iso = IsolationForest(**config.ISOLATION_FOREST)
    iso.fit(X_train)

    predictions = iso.predict(X_reject)
    inlier_mask = (predictions == 1)

    n_inliers = inlier_mask.sum()
    logger.info(f"IF Filtering: {n_inliers}/{len(inlier_mask)} inliers kept ({n_inliers/len(inlier_mask):.1%})")
    return inlier_mask


def run_route_a_fixed_pipeline():
    """
    Execute full Route A-Fixed hybrid reject inference pipeline.

    Steps:
        1.4A.1 Train Teacher CatBoost on D_L_train fixed embeddings
        1.4A.2 Pseudo-label D_R using teacher
        1.4A.3 Filter D_R with Isolation Forest (fit on D_L)
        1.4A.4 Train Student on D_L_train + filtered D_R
        1.4A.5 Evaluate on D_L_test

    Returns:
        results: Dict with metrics and statistics
    """
    logger.info("=" * 60)
    logger.info("Route A-Fixed — Embeddings Only (Step 1.4A)")
    logger.info("=" * 60)
    logger.info("Using VIME-Fixed embeddings (RobustScaler + Partial Loss + Target Enc)")

    # Load data
    X_train, X_val, X_test, X_reject, y_train, y_val, y_test = load_fixed_embeddings_and_labels()

    # 1.4A.1 Train Teacher
    logger.info("\n[1.4A.1] Training Teacher Model on fixed embeddings...")
    teacher, teacher_val_auc = train_catboost(X_train, y_train, X_val, y_val, "Teacher")

    # 1.4A.2 Pseudo-label D_R
    logger.info("\n[1.4A.2] Pseudo-labeling D_R...")
    pseudo_labels, pseudo_proba = pseudo_label_rejects(teacher, X_reject)

    # 1.4A.3 Filter with Isolation Forest
    logger.info("\n[1.4A.3] Filtering with Isolation Forest...")
    inlier_mask = filter_with_isolation_forest(X_train, X_reject)
    n_inliers = inlier_mask.sum()

    # 1.4A.4 Train Student on D_L + filtered D_R
    logger.info("\n[1.4A.4] Training Student Model (D_L + filtered D_R)...")
    X_augmented = np.vstack([X_train, X_reject[inlier_mask]])
    y_augmented = np.concatenate([y_train, pseudo_labels[inlier_mask]])

    logger.info(f"  Augmented data: {X_augmented.shape[0]} samples "
                f"(D_L={len(X_train)}, D_R_filtered={n_inliers})")
    logger.info(f"  Augmented default rate: {y_augmented.mean():.2%}")

    final_model, final_val_auc = train_catboost(X_augmented, y_augmented, X_val, y_val, "Student")

    # 1.4A.5 Evaluate on D_L_test
    logger.info("\n[1.4A.5] Evaluating on D_L_test...")
    test_pred = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    logger.info(f"Student Model Test AUC (D_L_test): {test_auc:.4f}")

    # Also evaluate teacher on test
    teacher_test_pred = teacher.predict_proba(X_test)[:, 1]
    teacher_test_auc = roc_auc_score(y_test, teacher_test_pred)
    logger.info(f"Teacher Test AUC (D_L_test): {teacher_test_auc:.4f}")

    # Compile results
    results = {
        "route": "A-Fixed",
        "step": "1.4A",
        "description": "Embeddings only (Fixed VIME)",
        "teacher_val_auc": float(teacher_val_auc),
        "teacher_test_auc": float(teacher_test_auc),
        "final_val_auc": float(final_val_auc),
        "final_test_auc": float(test_auc),
        "pseudo_label_default_rate": float(pseudo_labels.mean()),
        "pseudo_label_filtered_default_rate": float(pseudo_labels[inlier_mask].mean()),
        "inliers_kept": int(n_inliers),
        "inliers_total": int(len(inlier_mask)),
        "inlier_ratio": float(n_inliers / len(inlier_mask)),
        "augmented_samples": int(len(X_augmented)),
        "augmented_default_rate": float(y_augmented.mean()),
        "baseline_route_a_test_auc": 0.6912,
        "improvement_over_baseline": float(test_auc - 0.6912),
    }

    # Save models
    logger.info("\nSaving models...")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    teacher.save_model(str(config.MODEL_DIR / "route_a_fixed_teacher.cbm"))
    final_model.save_model(str(config.MODEL_DIR / "route_a_fixed_final.cbm"))
    logger.info(f"  Saved to {config.MODEL_DIR}")

    # Save metrics
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULT_DIR / "route_a_fixed_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Metrics saved to {config.RESULT_DIR / 'route_a_fixed_metrics.json'}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Route A-Fixed Pipeline Complete (Step 1.4A)")
    logger.info("=" * 60)
    logger.info(f"Teacher Val AUC:  {teacher_val_auc:.4f}")
    logger.info(f"Teacher Test AUC: {teacher_test_auc:.4f}")
    logger.info(f"Student Val AUC:  {final_val_auc:.4f}")
    logger.info(f"Student Test AUC: {test_auc:.4f}")
    logger.info(f"Baseline (Route A original): 0.6912")
    logger.info(f"Improvement: {test_auc - 0.6912:+.4f}")

    return results


if __name__ == "__main__":
    results = run_route_a_fixed_pipeline()
    print("\n" + json.dumps(results, indent=2))
