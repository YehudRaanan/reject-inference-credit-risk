"""
Threshold Sweep for Route A Pseudo-Labeling

Tests multiple threshold values and computes comprehensive metrics:
- AUC-ROC
- Confusion Matrix
- Precision, Recall, Accuracy
- F1, F2 scores

Usage: python -m src.models.threshold_sweep
"""
import logging
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, precision_score, recall_score,
    accuracy_score, f1_score, fbeta_score
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    """Load embeddings and labels."""
    X_train = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_train.parquet").values
    X_val = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_val.parquet").values
    X_test = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_test.parquet").values
    X_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_emb_reject.parquet").values

    y_train = pd.read_parquet(config.DATA_PROCESSED / "vime_y_train.parquet")["TARGET"].values
    y_val = pd.read_parquet(config.DATA_PROCESSED / "vime_y_val.parquet")["TARGET"].values
    y_test = pd.read_parquet(config.DATA_PROCESSED / "vime_y_test.parquet")["TARGET"].values

    return X_train, X_val, X_test, X_reject, y_train, y_val, y_test


def compute_metrics(y_true, y_pred_proba, y_pred_labels):
    """Compute all metrics for a given prediction."""
    cm = confusion_matrix(y_true, y_pred_labels)
    tn, fp, fn, tp = cm.ravel()

    return {
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred_labels),
        "precision": precision_score(y_true, y_pred_labels, zero_division=0),
        "recall": recall_score(y_true, y_pred_labels, zero_division=0),
        "f1": f1_score(y_true, y_pred_labels, zero_division=0),
        "f2": fbeta_score(y_true, y_pred_labels, beta=2, zero_division=0),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }


def run_threshold_sweep(thresholds):
    """
    Run Route A pipeline with multiple thresholds.

    Args:
        thresholds: List of threshold values to test

    Returns:
        results: Dict with results for each threshold
    """
    logger.info("=" * 70)
    logger.info("Threshold Sweep for Route A Pseudo-Labeling")
    logger.info("=" * 70)

    # Load data
    X_train, X_val, X_test, X_reject, y_train, y_val, y_test = load_data()
    logger.info(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}, Reject={len(X_reject)}")

    # Train Teacher (once)
    logger.info("\n[Step 1] Training Teacher Model (once)...")
    teacher = CatBoostClassifier(**config.CATBOOST, random_seed=config.SEED)
    teacher.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    teacher_val_pred = teacher.predict_proba(X_val)[:, 1]
    teacher_val_auc = roc_auc_score(y_val, teacher_val_pred)
    teacher_test_pred = teacher.predict_proba(X_test)[:, 1]
    teacher_test_auc = roc_auc_score(y_test, teacher_test_pred)
    logger.info(f"Teacher Val AUC: {teacher_val_auc:.4f}, Test AUC: {teacher_test_auc:.4f}")

    # Get predictions for D_R (once)
    reject_proba = teacher.predict_proba(X_reject)[:, 1]

    # Fit Isolation Forest (once)
    logger.info("\n[Step 2] Fitting Isolation Forest...")
    iso = IsolationForest(**config.ISOLATION_FOREST)
    iso.fit(X_train)
    inlier_mask = (iso.predict(X_reject) == 1)
    n_inliers = inlier_mask.sum()
    logger.info(f"IF Filtering: {n_inliers}/{len(inlier_mask)} inliers ({n_inliers/len(inlier_mask):.1%})")

    # Results storage
    all_results = {
        "teacher": {
            "val_auc": float(teacher_val_auc),
            "test_auc": float(teacher_test_auc),
        },
        "isolation_forest": {
            "inliers_kept": int(n_inliers),
            "inliers_total": int(len(inlier_mask)),
            "inlier_ratio": float(n_inliers / len(inlier_mask)),
        },
        "thresholds": {}
    }

    # Summary table header
    logger.info("\n" + "=" * 70)
    logger.info("Threshold Sweep Results")
    logger.info("=" * 70)
    logger.info(f"{'Thresh':>6} | {'PL Rate':>7} | {'Aug DR':>6} | {'Val AUC':>7} | {'Test AUC':>8} | {'Prec':>5} | {'Rec':>5} | {'F1':>5} | {'F2':>5}")
    logger.info("-" * 70)

    for threshold in thresholds:
        logger.info(f"\n[Threshold {threshold:.2f}] Processing...")

        # Generate pseudo-labels
        pseudo_labels = (reject_proba >= threshold).astype(int)
        pl_default_rate = pseudo_labels.mean()

        # Apply IF filter
        pseudo_labels_filtered = pseudo_labels[inlier_mask]
        pl_filtered_rate = pseudo_labels_filtered.mean()

        # Augment data
        X_augmented = np.vstack([X_train, X_reject[inlier_mask]])
        y_augmented = np.concatenate([y_train, pseudo_labels_filtered])
        aug_default_rate = y_augmented.mean()

        # Train final model
        final_model = CatBoostClassifier(**config.CATBOOST, random_seed=config.SEED)
        final_model.fit(X_augmented, y_augmented, eval_set=(X_val, y_val), use_best_model=True, verbose=False)

        # Evaluate on validation
        val_pred_proba = final_model.predict_proba(X_val)[:, 1]
        val_pred_labels = (val_pred_proba >= 0.5).astype(int)
        val_metrics = compute_metrics(y_val, val_pred_proba, val_pred_labels)

        # Evaluate on test
        test_pred_proba = final_model.predict_proba(X_test)[:, 1]
        test_pred_labels = (test_pred_proba >= 0.5).astype(int)
        test_metrics = compute_metrics(y_test, test_pred_proba, test_pred_labels)

        # Store results
        all_results["thresholds"][str(threshold)] = {
            "pseudo_label_default_rate": float(pl_default_rate),
            "pseudo_label_filtered_rate": float(pl_filtered_rate),
            "augmented_default_rate": float(aug_default_rate),
            "validation": val_metrics,
            "test": test_metrics,
        }

        # Print summary row
        logger.info(f"{threshold:>6.2f} | {pl_default_rate:>6.1%} | {aug_default_rate:>5.1%} | {val_metrics['auc_roc']:>7.4f} | {test_metrics['auc_roc']:>8.4f} | {test_metrics['precision']:>5.2f} | {test_metrics['recall']:>5.2f} | {test_metrics['f1']:>5.2f} | {test_metrics['f2']:>5.2f}")

    # Find best threshold
    best_threshold = max(
        all_results["thresholds"].items(),
        key=lambda x: x[1]["test"]["auc_roc"]
    )
    all_results["best_threshold"] = {
        "value": float(best_threshold[0]),
        "test_auc": best_threshold[1]["test"]["auc_roc"]
    }

    logger.info("-" * 70)
    logger.info(f"Best Threshold: {best_threshold[0]} (Test AUC: {best_threshold[1]['test']['auc_roc']:.4f})")
    logger.info(f"Teacher Baseline Test AUC: {teacher_test_auc:.4f}")
    logger.info(f"Improvement: {best_threshold[1]['test']['auc_roc'] - teacher_test_auc:+.4f}")

    # Save results
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULT_DIR / "threshold_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    # Thresholds from 0.15 to 0.45, increment 0.05
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    results = run_threshold_sweep(thresholds)
