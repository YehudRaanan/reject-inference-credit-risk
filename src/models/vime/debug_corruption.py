import torch
import pandas as pd
import numpy as np
import logging
import config
from src.models.vime.vime_utils import compute_mask_probabilities, vime_corruption
from src.models.vime.column_manager import identify_feature_groups

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_corruption")

def check_corruption_quality():
    # Load Test Data
    df_test = pd.read_parquet(config.DATA_PROCESSED / "vime_X_test.parquet")
    X_test = torch.tensor(df_test.values, dtype=torch.float32)
    
    # Compute Probabilities
    feature_groups = identify_feature_groups(
        columns=df_test.columns.tolist(),
        high_missing_rate_threshold=0.5,
        df_for_stats=df_test
    )
    p_mask = compute_mask_probabilities(
        df_test,
        critical_cols=feature_groups['critical'],
        high_missing_cols=feature_groups['high_missing'],
        p_critical=0.5,
        p_high_missing=0.1,
        p_regular=0.3
    )
    
    # Run Corruption
    x_tilde, m = vime_corruption(X_test, p_mask)
    
    # Analysis
    # 1. Total Mask Rate
    actual_mask_rate = m.mean().item()
    logger.info(f"Target Mask Rate (Avg): {p_mask.mean().item():.4f}")
    logger.info(f"Actual Mask Rate: {actual_mask_rate:.4f}")
    
    # 2. Effective Corruption
    # How often is x_tilde != X_test given m=1?
    # We look at indices where m=1
    mask_indices = (m == 1)
    n_masked = mask_indices.sum().item()
    
    if n_masked == 0:
        logger.warning("No values were masked!")
        return

    diff = torch.abs(x_tilde - X_test)
    # Using a small epsilon for float comparison
    is_different = (diff > 1e-6)
    
    # Count how many masked values are actually different
    n_effective = (is_different & mask_indices).sum().item()
    effective_rate = n_effective / n_masked
    
    logger.info(f"Total Masked Values: {n_masked}")
    logger.info(f"Effective Changes (Value Changed): {n_effective}")
    logger.info(f"Effective Corruption Rate: {effective_rate:.4f}")
    
    if effective_rate < 0.5:
        logger.warning("MAJOR ISSUE: More than 50% of 'corrupted' values are identical to the original!")
        logger.warning("Reason: Sparse data (many 0s) or small batch shuffling?")
        
        # Check sparsity
        sparsity = (X_test == 0).float().mean().item()
        logger.info(f"Dataset Sparsity (% zeros): {sparsity:.4f}")

if __name__ == "__main__":
    check_corruption_quality()
