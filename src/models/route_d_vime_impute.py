"""
Route D — VIME-based Income Reconstruction

Tests VIME's ability to predict AMT_INCOME_TOTAL for the rejected population (D_R).
We mask the income column and use VIME's feature reconstructor to predict it.

This demonstrates:
1. VIME's learned ability to predict income from other features
2. Whether VIME-reconstructed income improves classification

Pipeline:
    D.1 Load VIME preprocessed data (vime_X_*.parquet)
    D.2 For D_R: mask income column, use VIME to reconstruct
    D.3 Compare reconstructed income vs actual income (MSE, correlation)
    D.4 Add reconstructed income as a new feature
    D.5 Train CatBoost with VIME-enhanced features
"""
import logging
import json
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import pearsonr

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

# Add VIME to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "vime"))
from vime_model import VIME

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_vime_model():
    """Load the trained VIME encoder."""
    model_path = config.MODEL_DIR / "vime_encoder.pth"

    # Get input dimension from data
    sample = pd.read_parquet(config.DATA_PROCESSED / "vime_X_train.parquet")
    input_dim = sample.shape[1]

    model = VIME(
        input_dim=input_dim,
        hidden_dim=config.VIME_HIDDEN_DIM,
        embedding_dim=config.VIME_EMBEDDING_DIM,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    logger.info(f"Loaded VIME model from {model_path}")
    logger.info(f"  Input dim: {input_dim}, Embedding dim: {config.VIME_EMBEDDING_DIM}")

    return model, input_dim


def get_income_column_index(columns):
    """Find the index of AMT_INCOME_TOTAL in the column list."""
    for i, col in enumerate(columns):
        if col == "AMT_INCOME_TOTAL":
            return i
    raise ValueError("AMT_INCOME_TOTAL not found in columns")


def reconstruct_income_with_vime(X, income_idx, vime_model):
    """
    Mask the income column and use VIME to reconstruct it.

    Args:
        X: numpy array of features (scaled 0-1)
        income_idx: column index of AMT_INCOME_TOTAL
        vime_model: trained VIME model

    Returns:
        reconstructed_income: array of VIME-predicted income values
    """
    # Create a copy and mask income column (set to median = 0.5 for scaled data)
    X_masked = X.copy()
    X_masked[:, income_idx] = 0.5  # Median value for scaled data

    # Convert to tensor and run through VIME
    X_tensor = torch.tensor(X_masked, dtype=torch.float32)

    with torch.no_grad():
        # Get reconstruction through encoder + reconstructor
        z = vime_model.encode(X_tensor)  # Latent embedding
        x_recon = vime_model.recon_fc(z)  # Reconstructed features

    # Extract reconstructed income
    reconstructed_income = x_recon[:, income_idx].numpy()

    return reconstructed_income


def evaluate_reconstruction_quality(actual, predicted, name="Income"):
    """Evaluate how well VIME reconstructed the income."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    correlation, p_value = pearsonr(actual, predicted)

    # Mean absolute error
    mae = np.mean(np.abs(actual - predicted))

    logger.info(f"\n{name} Reconstruction Quality:")
    logger.info(f"  MSE:  {mse:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  MAE:  {mae:.6f}")
    logger.info(f"  Correlation: {correlation:.4f} (p={p_value:.2e})")

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": float(correlation),
        "p_value": float(p_value),
    }


def run_route_d_pipeline():
    """
    Execute Route D: VIME income reconstruction pipeline.

    Tests VIME's ability to predict income from other features,
    then uses this as an additional signal for classification.
    """
    logger.info("=" * 60)
    logger.info("Route D — VIME Income Reconstruction")
    logger.info("=" * 60)
    logger.info("Testing VIME's ability to predict income for D_R")
    logger.info("")

    # D.1 Load VIME model and data
    vime_model, input_dim = load_vime_model()

    # Load VIME preprocessed data
    X_train = pd.read_parquet(config.DATA_PROCESSED / "vime_X_train.parquet")
    X_val = pd.read_parquet(config.DATA_PROCESSED / "vime_X_val.parquet")
    X_test = pd.read_parquet(config.DATA_PROCESSED / "vime_X_test.parquet")
    X_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_X_reject.parquet")

    y_train = pd.read_parquet(config.DATA_PROCESSED / "vime_y_train.parquet")["TARGET"].values
    y_val = pd.read_parquet(config.DATA_PROCESSED / "vime_y_val.parquet")["TARGET"].values
    y_test = pd.read_parquet(config.DATA_PROCESSED / "vime_y_test.parquet")["TARGET"].values

    columns = X_train.columns.tolist()
    income_idx = get_income_column_index(columns)
    logger.info(f"AMT_INCOME_TOTAL column index: {income_idx}")

    # Convert to numpy
    X_train_np = X_train.values
    X_val_np = X_val.values
    X_test_np = X_test.values
    X_reject_np = X_reject.values

    # D.2 For D_R: reconstruct income using VIME
    logger.info("\n[D.2] Reconstructing income for D_R using VIME...")
    actual_income_reject = X_reject_np[:, income_idx].copy()
    predicted_income_reject = reconstruct_income_with_vime(X_reject_np, income_idx, vime_model)

    # D.3 Evaluate reconstruction quality
    logger.info("\n[D.3] Evaluating reconstruction quality...")
    recon_metrics_reject = evaluate_reconstruction_quality(
        actual_income_reject, predicted_income_reject, "D_R Income"
    )

    # Also test on D_L_test for comparison
    actual_income_test = X_test_np[:, income_idx].copy()
    predicted_income_test = reconstruct_income_with_vime(X_test_np, income_idx, vime_model)
    recon_metrics_test = evaluate_reconstruction_quality(
        actual_income_test, predicted_income_test, "D_L_test Income"
    )

    # D.4 Create enhanced features (add income_error as feature)
    logger.info("\n[D.4] Creating enhanced features with income reconstruction error...")

    def add_income_error_feature(X_np, income_idx, vime_model):
        """Add income reconstruction error as a new feature."""
        actual = X_np[:, income_idx]
        predicted = reconstruct_income_with_vime(X_np, income_idx, vime_model)
        error = actual - predicted  # Positive = higher income than predicted
        return np.column_stack([X_np, error, predicted])

    X_train_enhanced = add_income_error_feature(X_train_np, income_idx, vime_model)
    X_val_enhanced = add_income_error_feature(X_val_np, income_idx, vime_model)
    X_test_enhanced = add_income_error_feature(X_test_np, income_idx, vime_model)
    X_reject_enhanced = add_income_error_feature(X_reject_np, income_idx, vime_model)

    logger.info(f"  Original features: {X_train_np.shape[1]}")
    logger.info(f"  Enhanced features: {X_train_enhanced.shape[1]} (+2: income_error, predicted_income)")

    # D.5 Train CatBoost on enhanced features
    logger.info("\n[D.5] Training CatBoost on VIME-enhanced features...")

    # Train Teacher
    teacher = CatBoostClassifier(**config.CATBOOST, random_seed=config.SEED)
    teacher.fit(X_train_enhanced, y_train, eval_set=(X_val_enhanced, y_val), use_best_model=True)

    teacher_val_pred = teacher.predict_proba(X_val_enhanced)[:, 1]
    teacher_val_auc = roc_auc_score(y_val, teacher_val_pred)
    logger.info(f"Teacher Val AUC: {teacher_val_auc:.4f}")

    # Pseudo-label D_R
    pseudo_proba = teacher.predict_proba(X_reject_enhanced)[:, 1]
    pseudo_labels = (pseudo_proba >= config.PSEUDO_LABEL_THRESHOLD).astype(int)
    logger.info(f"Pseudo-labels: {pseudo_labels.sum()}/{len(pseudo_labels)} defaults ({pseudo_labels.mean():.2%})")

    # Train Final model
    X_augmented = np.vstack([X_train_enhanced, X_reject_enhanced])
    y_augmented = np.concatenate([y_train, pseudo_labels])

    final_model = CatBoostClassifier(**config.CATBOOST, random_seed=config.SEED)
    final_model.fit(X_augmented, y_augmented, eval_set=(X_val_enhanced, y_val), use_best_model=True)

    final_val_pred = final_model.predict_proba(X_val_enhanced)[:, 1]
    final_val_auc = roc_auc_score(y_val, final_val_pred)
    logger.info(f"Final Val AUC: {final_val_auc:.4f}")

    # Evaluate on test
    test_pred = final_model.predict_proba(X_test_enhanced)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    logger.info(f"Final Test AUC (D_L_test): {test_auc:.4f}")

    teacher_test_pred = teacher.predict_proba(X_test_enhanced)[:, 1]
    teacher_test_auc = roc_auc_score(y_test, teacher_test_pred)
    logger.info(f"Teacher Test AUC (D_L_test): {teacher_test_auc:.4f}")

    # Compile results
    results = {
        "reconstruction_quality": {
            "d_r": recon_metrics_reject,
            "d_l_test": recon_metrics_test,
        },
        "teacher_val_auc": float(teacher_val_auc),
        "teacher_test_auc": float(teacher_test_auc),
        "final_val_auc": float(final_val_auc),
        "final_test_auc": float(test_auc),
        "pseudo_label_default_rate": float(pseudo_labels.mean()),
        "n_original_features": int(X_train_np.shape[1]),
        "n_enhanced_features": int(X_train_enhanced.shape[1]),
    }

    # Save models
    logger.info("\nSaving models...")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    teacher.save_model(str(config.MODEL_DIR / "route_d_teacher.cbm"))
    final_model.save_model(str(config.MODEL_DIR / "route_d_final.cbm"))

    # Save metrics
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULT_DIR / "route_d_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {config.RESULT_DIR / 'route_d_metrics.json'}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Route D Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Income Reconstruction Correlation (D_R): {recon_metrics_reject['correlation']:.4f}")
    logger.info(f"Teacher Val AUC:  {teacher_val_auc:.4f}")
    logger.info(f"Teacher Test AUC: {teacher_test_auc:.4f}")
    logger.info(f"Final Val AUC:    {final_val_auc:.4f}")
    logger.info(f"Final Test AUC:   {test_auc:.4f}")
    logger.info(f"Test AUC Change:  {(test_auc - teacher_test_auc):+.4f}")

    return results


if __name__ == "__main__":
    results = run_route_d_pipeline()
    print("\n" + json.dumps(results, indent=2))
