"""
Script to evaluate the trained VIME encoder on the test set.
Computes:
1. Reconstruction MSE: How well the Autoencoder recovers original features.
2. Mask Prediction AUC: How well the Mask Estimator detects corrupted values.
Also performs a visual inspection of sample reconstructions.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error

import config
from src.models.vime.vime_model import VIME
from src.models.vime.vime_utils import compute_mask_probabilities, vime_corruption
from src.models.vime.column_manager import identify_feature_groups

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_vime():
    """
    Runs the full evaluation pipeline on `vime_X_test.parquet`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running VIME Evaluation on {device}...")

    # 1. Load Data (Test Set)
    test_path = config.DATA_PROCESSED / "vime_X_test.parquet"
    if not test_path.exists():
        logger.error(f"Test data not found at {test_path}")
        return

    df_test = pd.read_parquet(test_path)
    X_test = torch.tensor(df_test.values, dtype=torch.float32).to(device)
    logger.info(f"Loaded Test Data: {X_test.shape}")

    # 2. Load Model
    model_path = config.MODEL_DIR / "vime_encoder.pth"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    input_dim = X_test.shape[1]
    model = VIME(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully.")

    # 3. Prepare Masking (Same logic as training)
    # We need to compute the probability vector to generate consistent corruption
    feature_groups = identify_feature_groups(
        columns=df_test.columns.tolist(),
        high_missing_rate_threshold=0.5,
        df_for_stats=df_test # Use test stats or just rely on columns? Use test for independence.
    )
    
    p_mask = compute_mask_probabilities(
        df_test,
        critical_cols=feature_groups['critical'],
        high_missing_cols=feature_groups['high_missing'],
        p_critical=0.5,
        p_high_missing=0.1,
        p_regular=0.3
    ).to(device)

    # 4. Evaluation Loop
    # We corrupt the test set and see how well it recovers
    with torch.no_grad():
        # Corrupt
        x_tilde, m_true = vime_corruption(X_test, p_mask)
        
        # Predict
        m_pred, x_hat = model(x_tilde)
        
        # Metrics
        # A. Reconstruction (MSE)
        mse = nn.MSELoss()(x_hat, X_test).item()
        
        # B. Mask Prediction (AUC) - Flatten for calculation
        m_true_flat = m_true.cpu().numpy().flatten()
        m_pred_flat = m_pred.cpu().numpy().flatten()
        mask_auc = roc_auc_score(m_true_flat, m_pred_flat)
        
        logger.info("-" * 40)
        logger.info(f"Validation Results:")
        logger.info(f"Reconstruction MSE: {mse:.6f} (Lower is better)")
        logger.info(f"Mask Prediction AUC: {mask_auc:.4f} (1.0 is perfect)")
        logger.info("-" * 40)

        # 5. Visual Inspection (First 5 samples, Critical Feature)
        # Find index of AMT_INCOME_TOTAL
        if 'AMT_INCOME_TOTAL' in df_test.columns:
            idx_income = df_test.columns.get_loc('AMT_INCOME_TOTAL')
            logger.info(f"Inspection: AMT_INCOME_TOTAL (Column {idx_income})")
            logger.info(f"{'Original':<10} | {'Corrupted':<10} | {'Reconstructed':<15} | {'Masked?':<10}")
            logger.info("-" * 55)
            
            for i in range(10):
                orig = X_test[i, idx_income].item()
                corr = x_tilde[i, idx_income].item()
                rec = x_hat[i, idx_income].item()
                is_masked = int(m_true[i, idx_income].item())
                
                # Highlight if masked
                masked_str = "YES" if is_masked else "No"
                logger.info(f"{orig:.4f}     | {corr:.4f}     | {rec:.4f}          | {masked_str}")

if __name__ == "__main__":
    evaluate_vime()
