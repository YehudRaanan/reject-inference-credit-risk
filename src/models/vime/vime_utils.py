import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_mask_probabilities(
    df, 
    critical_cols=None, 
    high_missing_cols=None,
    low_mask_threshold=0.5, 
    p_critical=0.5, 
    p_high_missing=0.1, 
    p_regular=0.3
):
    """
    Computes a probability vector P for masking.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        critical_cols (list): Columns for p_critical (Level 1).
        high_missing_cols (list): Columns for p_high_missing (Level 2).
        low_mask_threshold (float): Missing rate above which use p_high_missing.
        p_critical (float): Mask rate for critical columns (default 0.5).
        p_high_missing (float): Mask rate for columns with high missingness (default 0.1).
        p_regular (float): Default mask rate for all other columns (default 0.3).
    """
    n_features = df.shape[1]
    p_vector = torch.full((n_features,), p_regular) # Default: Level 3 (Regular)
    
    # 2. High Missing Columns (Level 2)
    if high_missing_cols:
        indices = [df.columns.get_loc(c) for c in high_missing_cols if c in df.columns]
        p_vector[indices] = p_high_missing
        logger.info(f"Set high-missing mask rate ({p_high_missing}) for {len(indices)} columns.")

    # 1. Critical Columns (Level 1) - Overrides Level 2 if overlap
    if critical_cols:
        indices = [df.columns.get_loc(c) for c in critical_cols if c in df.columns]
        p_vector[indices] = p_critical
        logger.info(f"Set critical mask rate ({p_critical}) for {len(indices)} columns.")

    return p_vector

def vime_corruption(x: torch.Tensor, p_mask: torch.Tensor):
    """
    Generates corrupted VIME inputs using a per-feature mask probability.
    
    Args:
        x (Tensor): Clean batch [batch_size, features]
        p_mask (Tensor): Probability vector [features] (derived from compute_mask_probabilities)
        
    Returns:
        x_corrupted (Tensor): The corrupted input.
        mask (Tensor): The binary mask (1=corrupted).
    """
    batch_size, n_features = x.shape
    
    # 1. Generate Mask (Bernoulli with per-column probabilities)
    # p_mask is (features,), we broadcast to (batch, features)
    p_matrix = p_mask.repeat(batch_size, 1).to(x.device)
    mask = torch.bernoulli(p_matrix)
    
    # 2. Generate Shuffled X (feature-wise shuffle)
    # Generate random noise for permutation
    noise = torch.rand_like(x)
    perm_indices = torch.argsort(noise, dim=0)
    
    # Gather shuffled values
    x_shuffled = torch.gather(x, 0, perm_indices)
    
    # 3. Combine
    x_corrupted = x * (1 - mask) + x_shuffled * mask
    
    # 4. Refine Mask (Smart Labeling)
    # If the corrupted value happens to be the same as the original (common in sparse data),
    # we should NOT label it as corrupted. This confuses the model.
    # We only keep m=1 if the value actually changed.
    is_changed = (torch.abs(x - x_corrupted) > 1e-5).float()
    mask = mask * is_changed
    
    return x_corrupted, mask
