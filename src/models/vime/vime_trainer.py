
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import config
from src.models.vime.vime_model import VIME
from src.models.vime.vime_utils import compute_mask_probabilities, vime_corruption

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Device configuration (CPU is fine for this scale, but scalable to GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_vime(
    batch_size=config.VIME_BATCH_SIZE,
    epochs=config.VIME_EPOCHS,
    learning_rate=config.VIME_LR,
    p_critical=config.VIME_P_CRITICAL,
    p_high_missing=config.VIME_P_HIGH_MISSING,
    p_regular=config.VIME_P_REGULAR,
    alpha=config.VIME_ALPHA,
    sample_size=None
):
    """
    Train VIME Encoder using Self-Supervised Learning.

    Args:
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.
        learning_rate (float): Adam optimizer learning rate.
        p_critical (float): Mask rate for critical columns.
        p_high_missing (float): Mask rate for high-missing columns.
        p_regular (float): Default mask rate for regular columns.
        alpha (float): Weight for mask loss (total = alpha*mask + recon).
        sample_size (int, optional): Subsample for debugging.
    """
    logger.info(f"Starting VIME Training on {device}...")
    
    # 1. Load Data
    # ---------------------------------------------------------
    
    logger.info("Loading datasets...")
    X_train = pd.read_parquet(config.DATA_PROCESSED / "vime_X_train.parquet")
    X_val = pd.read_parquet(config.DATA_PROCESSED / "vime_X_val.parquet")
    X_test = pd.read_parquet(config.DATA_PROCESSED / "vime_X_test.parquet")
    X_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_X_reject.parquet")
    
    # Concatenate for SSL training
    X_all_df = pd.concat([X_train, X_val, X_test, X_reject], axis=0)
    
    if sample_size:
        logger.info(f"Subsampling {sample_size} rows for debugging...")
        X_all_df = X_all_df.sample(n=sample_size, random_state=42)
        
    logger.info(f"Full Unlabeled Corpus: {X_all_df.shape}")
    
    # Convert to Tensor
    X_all_tensor = torch.tensor(X_all_df.values, dtype=torch.float32)

    # 2. Compute Custom Mask Probabilities
    # ---------------------------------------------------------
    from src.models.vime.column_manager import identify_feature_groups
    
    # Identify groups dynamically
    feature_groups = identify_feature_groups(
        columns=X_all_df.columns.tolist(),
        high_missing_rate_threshold=0.5,
        df_for_stats=X_all_df
    )
    
    logger.info(f"Feature Groups Identified:")
    logger.info(f"  - Critical (p={p_critical}): {len(feature_groups['critical'])} cols (e.g. {feature_groups['critical'][:3]})")
    logger.info(f"  - High Missing (p={p_high_missing}): {len(feature_groups['high_missing'])} cols")
    logger.info(f"  - Regular (p={p_regular}): {len(feature_groups['regular'])} cols")

    p_mask = compute_mask_probabilities(
        X_all_df, 
        critical_cols=feature_groups['critical'], 
        high_missing_cols=feature_groups['high_missing'],
        low_mask_threshold=0.5,
        p_critical=p_critical,
        p_high_missing=p_high_missing,
        p_regular=p_regular
    ).to(device)
    logger.info("Computed custom mask probability vector.")

    # 3. Create DataLoader
    # ---------------------------------------------------------
    dataset = TensorDataset(X_all_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Initialize Model
    # ---------------------------------------------------------
    input_dim = X_all_tensor.shape[1]
    model = VIME(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion_mask = nn.BCELoss() # Binary Cross Entropy for mask prediction
    criterion_recon = nn.MSELoss() # Mean Squared Error for reconstruction (or BCE if sigmoided)
    
    # 5. Training Loop
    # ---------------------------------------------------------
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_recon_loss = 0.0
        start_time = time.time()
        
        for batch in dataloader:
            x_clean = batch[0].to(device)
            
            # A. Generate Corrupted Input
            # ---------------------------
            # x_tilde: corrupted input
            # m: binary mask (1 = corrupted)
            x_tilde, m = vime_corruption(x_clean, p_mask)
            
            # B. Forward Pass
            # ---------------
            # m_hat: predicted mask (probability)
            # x_hat: reconstructed features
            m_hat, x_hat = model(x_tilde)
            
            # C. Compute Loss
            # ---------------
            loss_m = criterion_mask(m_hat, m)
            loss_r = criterion_recon(x_hat, x_clean)
            
            # Total Loss = alpha * Mask_Loss + Recon_Loss
            loss = alpha * loss_m + loss_r
            
            # D. Backward Pass
            # ----------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mask_loss += loss_m.item()
            epoch_recon_loss += loss_r.item()
            
        # Logging
        avg_loss = epoch_loss / len(dataloader)
        avg_m = epoch_mask_loss / len(dataloader)
        avg_r = epoch_recon_loss / len(dataloader)
        elapsed = time.time() - start_time
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} (Mask: {avg_m:.4f}, Recon: {avg_r:.4f}) - {elapsed:.1f}s")

    # 6. Save Model
    # ---------------------------------------------------------
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.MODEL_DIR / "vime_encoder.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

def masked_mse_loss(x_hat: torch.Tensor, x_clean: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE only on positions where data was NOT originally missing.

    This is the Partial Reconstruction Loss from Phase 1.2.

    Args:
        x_hat: Reconstructed values [batch, features]
        x_clean: Original (imputed) values [batch, features]
        missing_mask: Binary mask [batch, features] (1 = was missing, 0 = was present)

    Returns:
        Scalar loss averaged over valid (non-missing) positions
    """
    # Only compute loss where data was originally present
    valid_mask = 1.0 - missing_mask
    squared_error = (x_hat - x_clean) ** 2
    masked_squared_error = squared_error * valid_mask

    # Average over valid positions only
    num_valid = valid_mask.sum()
    if num_valid > 0:
        loss = masked_squared_error.sum() / num_valid
    else:
        loss = torch.tensor(0.0, device=x_hat.device)

    return loss


def train_vime_fixed(
    batch_size=config.VIME_BATCH_SIZE,
    epochs=config.VIME_EPOCHS,
    learning_rate=config.VIME_LR,
    p_critical=config.VIME_P_CRITICAL,
    p_high_missing=config.VIME_P_HIGH_MISSING,
    p_regular=config.VIME_P_REGULAR,
    alpha=config.VIME_ALPHA,
    sample_size=None,
    use_partial_loss=True
):
    """
    Train VIME Encoder with Phase 1 Improvements (Route A-Fixed).

    Improvements:
    1. Uses vime_X_*_fixed.parquet (RobustScaler + Target Encoding)
    2. Uses partial reconstruction loss (masked MSE) when use_partial_loss=True
    3. Loads missing masks from vime_missing_mask_*.parquet

    Args:
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.
        learning_rate (float): Adam optimizer learning rate.
        p_critical (float): Mask rate for critical columns.
        p_high_missing (float): Mask rate for high-missing columns.
        p_regular (float): Default mask rate for regular columns.
        alpha (float): Weight for mask loss (total = alpha*mask + recon).
        sample_size (int, optional): Subsample for debugging.
        use_partial_loss (bool): Use masked MSE loss (True) or standard MSE (False).
    """
    logger.info(f"Starting VIME-Fixed Training on {device}...")
    logger.info(f"Using partial reconstruction loss: {use_partial_loss}")

    # 1. Load Data (Fixed preprocessing version)
    # ---------------------------------------------------------
    logger.info("Loading fixed-preprocessed datasets...")
    X_train = pd.read_parquet(config.DATA_PROCESSED / "vime_X_train_fixed.parquet")
    X_val = pd.read_parquet(config.DATA_PROCESSED / "vime_X_val_fixed.parquet")
    X_test = pd.read_parquet(config.DATA_PROCESSED / "vime_X_test_fixed.parquet")
    X_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_X_reject_fixed.parquet")

    # Load missing masks (for partial reconstruction loss)
    if use_partial_loss:
        logger.info("Loading missing masks...")
        mask_train = pd.read_parquet(config.DATA_PROCESSED / "vime_missing_mask_train.parquet")
        mask_val = pd.read_parquet(config.DATA_PROCESSED / "vime_missing_mask_val.parquet")
        mask_test = pd.read_parquet(config.DATA_PROCESSED / "vime_missing_mask_test.parquet")
        mask_reject = pd.read_parquet(config.DATA_PROCESSED / "vime_missing_mask_reject.parquet")

    # Concatenate for SSL training (reset index for proper alignment)
    X_all_df = pd.concat([X_train, X_val, X_test, X_reject], axis=0, ignore_index=True)

    if use_partial_loss:
        mask_all_df = pd.concat([mask_train, mask_val, mask_test, mask_reject], axis=0, ignore_index=True)

    if sample_size and sample_size < len(X_all_df):
        logger.info(f"Subsampling {sample_size} rows for debugging...")
        sample_idx = np.random.RandomState(42).choice(len(X_all_df), size=sample_size, replace=False)
        X_all_df = X_all_df.iloc[sample_idx].reset_index(drop=True)
        if use_partial_loss:
            mask_all_df = mask_all_df.iloc[sample_idx].reset_index(drop=True)

    logger.info(f"Full Unlabeled Corpus: {X_all_df.shape}")

    # Convert to Tensor
    X_all_tensor = torch.tensor(X_all_df.values, dtype=torch.float32)

    # Create expanded missing mask tensor (match feature dimensions)
    # The mask only covers original numeric columns, need to align with full feature set
    if use_partial_loss:
        # Create full mask tensor with same shape as features
        # For non-numeric columns (target encoded, one-hot), use 0 (not missing)
        full_mask = torch.zeros_like(X_all_tensor)

        # Map original numeric columns to their positions in the processed features
        feature_names = X_all_df.columns.tolist()
        mask_col_names = mask_all_df.columns.tolist()

        for i, mask_col in enumerate(mask_col_names):
            # mask_col is like "missing_AMT_INCOME_TOTAL"
            original_col = mask_col.replace("missing_", "")
            if original_col in feature_names:
                feat_idx = feature_names.index(original_col)
                full_mask[:, feat_idx] = torch.tensor(mask_all_df.iloc[:, i].values, dtype=torch.float32)

        logger.info(f"Missing mask covers {(full_mask.sum(dim=0) > 0).sum().item()} features")

    # 2. Compute Custom Mask Probabilities
    # ---------------------------------------------------------
    from src.models.vime.column_manager import identify_feature_groups

    feature_groups = identify_feature_groups(
        columns=X_all_df.columns.tolist(),
        high_missing_rate_threshold=0.5,
        df_for_stats=X_all_df
    )

    logger.info(f"Feature Groups Identified:")
    logger.info(f"  - Critical (p={p_critical}): {len(feature_groups['critical'])} cols")
    logger.info(f"  - High Missing (p={p_high_missing}): {len(feature_groups['high_missing'])} cols")
    logger.info(f"  - Regular (p={p_regular}): {len(feature_groups['regular'])} cols")

    p_mask = compute_mask_probabilities(
        X_all_df,
        critical_cols=feature_groups['critical'],
        high_missing_cols=feature_groups['high_missing'],
        low_mask_threshold=0.5,
        p_critical=p_critical,
        p_high_missing=p_high_missing,
        p_regular=p_regular
    ).to(device)

    # 3. Create DataLoader
    # ---------------------------------------------------------
    if use_partial_loss:
        dataset = TensorDataset(X_all_tensor, full_mask)
    else:
        dataset = TensorDataset(X_all_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Initialize Model
    # ---------------------------------------------------------
    input_dim = X_all_tensor.shape[1]
    model = VIME(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion_mask = nn.BCELoss()
    criterion_recon = nn.MSELoss()  # Used only when use_partial_loss=False

    # 5. Training Loop
    # ---------------------------------------------------------
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_mask_loss = 0.0
        epoch_recon_loss = 0.0
        start_time = time.time()

        for batch in dataloader:
            if use_partial_loss:
                x_clean, missing_mask_batch = batch
                x_clean = x_clean.to(device)
                missing_mask_batch = missing_mask_batch.to(device)
            else:
                x_clean = batch[0].to(device)
                missing_mask_batch = None

            # A. Generate Corrupted Input
            x_tilde, m = vime_corruption(x_clean, p_mask)

            # B. Forward Pass
            m_hat, x_hat = model(x_tilde)

            # C. Compute Loss
            loss_m = criterion_mask(m_hat, m)

            if use_partial_loss and missing_mask_batch is not None:
                # Partial reconstruction loss: only on non-missing positions
                loss_r = masked_mse_loss(x_hat, x_clean, missing_mask_batch)
            else:
                # Standard MSE on all positions
                loss_r = criterion_recon(x_hat, x_clean)

            # Total Loss = alpha * Mask_Loss + Recon_Loss
            loss = alpha * loss_m + loss_r

            # D. Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mask_loss += loss_m.item()
            epoch_recon_loss += loss_r.item()

        # Logging
        avg_loss = epoch_loss / len(dataloader)
        avg_m = epoch_mask_loss / len(dataloader)
        avg_r = epoch_recon_loss / len(dataloader)
        elapsed = time.time() - start_time

        logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} (Mask: {avg_m:.4f}, Recon: {avg_r:.4f}) - {elapsed:.1f}s")

    # 6. Save Model
    # ---------------------------------------------------------
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.MODEL_DIR / "vime_encoder_fixed.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

    return model


if __name__ == "__main__":
    # Default to fixed training
    train_vime_fixed()
